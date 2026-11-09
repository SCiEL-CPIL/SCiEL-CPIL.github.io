"""Rewritten symbolic-regression fitter: GP + least-squares with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import operator
import math

# --- GP operators and primitives ---
# Terminal types: 'x' (input index), 'const' (learned scalar)
# We will do least-squares over the constants after tree structure is fixed.

def _safe_div(a, b):
    b = np.where(np.abs(b) < 1e-12, 1e-12, b)
    return a / b

def _safe_log(a):
    return np.log(np.clip(a, 1e-12, None))

def _safe_sqrt(a):
    return np.sqrt(np.clip(a, 0, None))

def _safe_exp(a):
    return np.exp(np.clip(a, -50, 50))

def _safe_tanh(a):
    return np.tanh(a)

def _safe_sin(a):
    return np.sin(a)

def _safe_cos(a):
    return np.cos(a)

def _safe_abs(a):
    return np.abs(a)

# Binary operators
BIN_OPS = {
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'div': _safe_div,
    'pow': lambda a, b: np.power(np.clip(a, 1e-8, None), np.clip(b, -5, 5)),
}
BIN_OP_NAMES = list(BIN_OPS.keys())

# Unary operators
UNARY_OPS = {
    'neg': lambda a: -a,
    'sin': _safe_sin,
    'cos': _safe_cos,
    'tanh': _safe_tanh,
    'exp': _safe_exp,
    'log': _safe_log,
    'sqrt': _safe_sqrt,
    'abs': _safe_abs,
}
UNARY_OP_NAMES = list(UNARY_OPS.keys())

class Node:
    __slots__ = ('op', 'left', 'right', 'term_idx')
    def __init__(self, op=None, left=None, right=None, term_idx=None):
        self.op = op
        self.left = left
        self.right = right
        self.term_idx = term_idx

def make_leaf(d):
    # Random leaf: either an input variable or a constant placeholder
    if random.random() < 0.5:
        idx = random.randint(0, d - 1)
        return Node(op='x', term_idx=idx)
    else:
        return Node(op='const', term_idx=random.randint(0, 15))  # up to 16 constants

def make_internal(d, depth):
    if depth <= 0 or random.random() < 0.15:
        return make_leaf(d)
    if random.random() < 0.4:
        # Unary
        op = random.choice(UNARY_OP_NAMES)
        left = make_internal(d, depth - 1)
        return Node(op=op, left=left, right=None)
    else:
        # Binary
        op = random.choice(BIN_OP_NAMES)
        left = make_internal(d, depth - 1)
        right = make_internal(d, depth - 1)
        return Node(op=op, left=left, right=right)

def tree_depth(node):
    if node is None:
        return 0
    if node.op in ('x', 'const'):
        return 1
    if node.op in UNARY_OPS:
        return 1 + tree_depth(node.left)
    return 1 + max(tree_depth(node.left), tree_depth(node.right))

def evaluate_tree(node, X, const_values):
    """Evaluate tree on X (n, d) with constant values (k,). Returns (n,)"""
    if node is None:
        return np.zeros(len(X))
    if node.op == 'x':
        return X[:, node.term_idx]
    if node.op == 'const':
        return np.full(len(X), const_values[node.term_idx])
    if node.op in UNARY_OPS:
        child = evaluate_tree(node.left, X, const_values)
        return UNARY_OPS[node.op](child)
    if node.op in BIN_OPS:
        left = evaluate_tree(node.left, X, const_values)
        right = evaluate_tree(node.right, X, const_values)
        return BIN_OPS[node.op](left, right)
    raise ValueError(f"Unknown op: {node.op}")

def collect_constants(node, const_set):
    """Collect all constant indices used in the tree."""
    if node is None:
        return
    if node.op == 'const':
        const_set.add(node.term_idx)
    elif node.op in UNARY_OPS:
        collect_constants(node.left, const_set)
    elif node.op in BIN_OPS:
        collect_constants(node.left, const_set)
        collect_constants(node.right, const_set)

def fit_constants(tree, X, y):
    """Fit constant values via fast optimization. Returns (const_dict, predictions, mse)."""
    const_set = set()
    collect_constants(tree, const_set)
    if not const_set:
        # No constants, just evaluate with zeros
        pred = evaluate_tree(tree, X, np.zeros(16))
        if not np.all(np.isfinite(pred)):
            return {}, pred, np.inf
        mse = np.mean((pred - y) ** 2)
        return {}, pred, mse
    
    from scipy.optimize import minimize
    
    const_indices = sorted(const_set)
    n_consts = len(const_indices)
    
    best_const = np.zeros(n_consts)
    best_mse = np.inf
    best_pred = None
    
    # Try a few random initializations with fast local search
    for init in range(2):
        if init == 0:
            x0 = np.zeros(n_consts)
        else:
            x0 = np.random.uniform(-1, 1, n_consts)
        
        def objective(c):
            full_c = np.zeros(16)
            for i, idx in enumerate(const_indices):
                full_c[idx] = c[i]
            pred = evaluate_tree(tree, X, full_c)
            if not np.all(np.isfinite(pred)):
                return 1e10
            return np.mean((pred - y) ** 2)
        
        try:
            # Nelder-Mead is robust and fast for small dimensions
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 20, 'xatol': 1e-4, 'fatol': 1e-6})
            c_opt = res.x
            full_c = np.zeros(16)
            for i, idx in enumerate(const_indices):
                full_c[idx] = c_opt[i]
            pred = evaluate_tree(tree, X, full_c)
            if np.all(np.isfinite(pred)):
                mse = np.mean((pred - y) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_const = c_opt
                    best_pred = pred
        except:
            pass
    
    if best_pred is None:
        return {}, np.zeros(len(X)), np.inf
    
    const_dict = {}
    for i, idx in enumerate(const_indices):
        const_dict[idx] = best_const[i]
    return const_dict, best_pred, best_mse

def mutate_tree(node, d, depth_limit=6):
    """Mutate a tree by replacing a random subtree."""
    if node is None:
        return make_internal(d, 3)
    
    # Collect all nodes
    nodes = []
    def collect(n, parent, is_left):
        if n is None:
            return
        nodes.append((n, parent, is_left))
        if n.op in UNARY_OPS:
            collect(n.left, n, True)
        elif n.op in BIN_OPS:
            collect(n.left, n, True)
            collect(n.right, n, False)
    
    collect(node, None, False)
    
    if not nodes:
        return node
    
    # Pick a random node to replace
    idx = random.randint(0, len(nodes) - 1)
    target, parent, is_left = nodes[idx]
    
    # Create a new subtree
    new_subtree = make_internal(d, random.randint(1, 3))
    
    # Replace
    if parent is None:
        return new_subtree
    if is_left:
        parent.left = new_subtree
    else:
        parent.right = new_subtree
    
    return node

def crossover(t1, t2, d, depth_limit=6):
    """Crossover: swap subtrees between two trees."""
    def collect_nodes(n, parent, is_left):
        nodes = []
        if n is None:
            return nodes
        nodes.append((n, parent, is_left))
        if n.op in UNARY_OPS:
            nodes.extend(collect_nodes(n.left, n, True))
        elif n.op in BIN_OPS:
            nodes.extend(collect_nodes(n.left, n, True))
            nodes.extend(collect_nodes(n.right, n, False))
        return nodes
    
    nodes1 = collect_nodes(t1, None, False)
    nodes2 = collect_nodes(t2, None, False)
    
    if not nodes1 or not nodes2:
        return t1.copy() if hasattr(t1, 'copy') else t1
    
    # Pick a node from t1 to replace with a subtree from t2
    idx1 = random.randint(0, len(nodes1) - 1)
    idx2 = random.randint(0, len(nodes2) - 1)
    
    target1, parent1, is_left1 = nodes1[idx1]
    donor2, parent2, is_left2 = nodes2[idx2]
    
    # Deep copy the donor subtree
    def deep_copy(n):
        if n is None:
            return None
        if n.op in ('x', 'const'):
            return Node(op=n.op, term_idx=n.term_idx)
        if n.op in UNARY_OPS:
            return Node(op=n.op, left=deep_copy(n.left))
        return Node(op=n.op, left=deep_copy(n.left), right=deep_copy(n.right))
    
    new_subtree = deep_copy(donor2)
    
    if parent1 is None:
        return new_subtree
    if is_left1:
        parent1.left = new_subtree
    else:
        parent1.right = new_subtree
    
    return t1

def _deep_copy_tree(n):
    if n is None:
        return None
    if n.op in ('x', 'const'):
        return Node(op=n.op, term_idx=n.term_idx)
    if n.op in UNARY_OPS:
        return Node(op=n.op, left=_deep_copy_tree(n.left))
    return Node(op=n.op, left=_deep_copy_tree(n.left), right=_deep_copy_tree(n.right))


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290  # Leave 10s safety margin
    
    d = X_train.shape[1]
    n = len(X_train)
    
    # Normalize X for better numerical behavior in GP
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_std[X_std < 1e-8] = 1.0
    X_norm = (X_train - X_mean) / X_std
    
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    y_std = max(y_std, 1e-8)
    y_norm = (y_train - y_mean) / y_std
    
    # Population-based GP search - Reduced size and generations for speed
    pop_size = 15
    max_generations = 20
    
    # Initialize population
    population = []
    for i in range(pop_size):
        tree = make_internal(d, random.randint(2, 4))
        population.append(tree)
    
    best_tree = None
    best_const = None
    best_mse = np.inf
    best_pred_train = None
    
    # Evaluate initial population
    for tree in population:
        if time.time() > deadline - 10:
            break
        const_dict, pred, mse = fit_constants(tree, X_norm, y_norm)
        if mse < best_mse:
            best_mse = mse
            best_tree = _deep_copy_tree(tree)
            best_const = dict(const_dict)
            best_pred_train = pred
    
    # Evolution
    generation = 0
    while generation < max_generations:
        if time.time() > deadline - 10:
            break
        
        generation += 1
        
        # Evaluate current population
        scored = []
        for tree in population:
            if time.time() > deadline - 10:
                break
            const_dict, pred, mse = fit_constants(tree, X_norm, y_norm)
            scored.append((mse, tree, const_dict, pred))
        
        if not scored:
            break
        
        scored.sort(key=lambda x: x[0])
        
        # Update best
        if scored[0][0] < best_mse:
            best_mse = scored[0][0]
            best_tree = _deep_copy_tree(scored[0][1])
            best_const = dict(scored[0][2])
            best_pred_train = scored[0][3]
        
        # Selection: keep top half
        elite_size = max(3, pop_size // 2)
        new_population = [_deep_copy_tree(scored[i][1]) for i in range(elite_size)]
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 10:
                break
            
            # Pick two parents
            p1 = scored[random.randint(0, min(elite_size, len(scored)) - 1)][1]
            p2 = scored[random.randint(0, min(elite_size, len(scored)) - 1)][1]
            
            r = random.random()
            if r < 0.5:
                child = crossover(_deep_copy_tree(p1), _deep_copy_tree(p2), d)
            elif r < 0.9:
                child = mutate_tree(_deep_copy_tree(p1), d)
            else:
                child = make_internal(d, random.randint(2, 4))
            
            # Depth check
            if tree_depth(child) > 8:
                child = make_internal(d, 3)
            
            new_population.append(child)
        
        population = new_population
    
    # Fallback: Linear model if GP failed or didn't find anything good
    def _linear_fallback():
        def features(X):
            cols = [np.ones(len(X))] + [X[:, j] for j in range(X.shape[1])]
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
        return lambda X: features(X) @ w
    
    if best_tree is None or best_mse > np.var(y_train) * 1.5:
        # If GP is worse than a simple baseline, use baseline
        # Calculate baseline MSE
        pred_lin = _linear_fallback()(X_train)
        mse_lin = np.mean((pred_lin - y_train)**2)
        if mse_lin < best_mse:
            return _linear_fallback()
    
    # Fit final constants on original scale
    # Re-evaluate on normalized X, then convert predictions back
    const_dict, pred_norm, mse_norm = fit_constants(best_tree, X_norm, y_norm)
    
    # Convert back to original scale
    pred_original = pred_norm * y_std + y_mean
    
    # Define predict function
    def predict(X):
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        X_n = (X_arr - X_mean) / X_std
        full_c = np.zeros(16)
        for k, v in const_dict.items():
            full_c[k] = v
        p_norm = evaluate_tree(best_tree, X_n, full_c)
        p = p_norm * y_std + y_mean
        # Ensure finite
        p = np.where(np.isfinite(p), p, np.nanmean(y_train))
        if p.ndim == 1 and len(p) == 1:
            return p[0]
        return p
    
    return predict
# EVOLVE-BLOCK-END
