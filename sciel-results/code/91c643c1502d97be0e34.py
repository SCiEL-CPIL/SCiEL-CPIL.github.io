"""Genetic Programming based symbolic regression with robust evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import operator
import math

# Deadline guard: 330s wall-clock limit, use 300s for safety
_DEADLINE = time.time() + 300.0

def _safe_div(x, y):
    """Safe division avoiding zero and large values."""
    # Use epsilon to avoid division by zero
    eps = 1e-8
    # Clip denominator to avoid huge values
    y_safe = np.where(np.abs(y) < eps, np.sign(y) * eps, y)
    # Handle case where y is exactly 0 -> sign is 0, so we need to be careful
    # Actually np.sign(0) is 0. Let's just replace 0 with eps.
    y_safe = np.where(y == 0, eps, y)
    # Also clip the result to prevent overflow in subsequent operations
    result = x / y_safe
    return np.clip(result, -1e6, 1e6)

def _safe_log(x):
    """Safe log."""
    return np.log(np.clip(x, 1e-8, 1e6))

def _safe_sqrt(x):
    """Safe sqrt."""
    return np.sqrt(np.clip(x, 0, 1e6))

def _safe_exp(x):
    """Safe exp."""
    return np.exp(np.clip(x, -100, 100))

def _safe_sin(x):
    return np.sin(x)

def _safe_cos(x):
    return np.cos(x)

def _safe_tan(x):
    # Tan has asymptotes. Clip output.
    return np.clip(np.tan(x), -1e6, 1e6)

def _safe_abs(x):
    return np.abs(x)

def _safe_pow(x, y):
    """Safe power. Only allow integer powers or positive bases to avoid complex/NaN."""
    # For simplicity and stability, we might restrict this or use a simplified version.
    # A common trick: if y is not integer, use exp(y*log(x)) with safe log/exp.
    # But GP trees can generate any y.
    # Let's just use np.power with clipping and nan_to_num.
    with np.errstate(all='ignore'):
        result = np.power(np.clip(x, -1e6, 1e6), np.clip(y, -10, 10))
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)

# Terminal and Function sets
TERMINALS = ['x']
FUNCTIONS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': _safe_div,
    'sin': _safe_sin,
    'cos': _safe_cos,
    'tan': _safe_tan,
    'abs': _safe_abs,
    'sqrt': _safe_sqrt,
    'exp': _safe_exp,
    'log': _safe_log,
    'pow': _safe_pow
}

class GPNode:
    def __init__(self, name, *args):
        self.name = name
        self.args = args

    def evaluate(self, X):
        """
        Evaluate the tree on input X (array of shape (n, d) or (n,)).
        Since we only have 1D inputs in the problem description, X is effectively 1D vector per row.
        We assume X is passed as a 2D array (n, 1) or 1D array (n,).
        """
        if self.name == 'x':
            # If X is 2D (n,1), take first column
            if X.ndim == 2:
                return X[:, 0]
            else:
                return X
        elif self.name in FUNCTIONS:
            func = FUNCTIONS[self.name]
            # Evaluate children
            child_vals = [child.evaluate(X) for child in self.args]
            # Apply function
            return func(*child_vals)
        else:
            # Fallback: constant? We don't have constants in terminals, 
            # but we could add them. For now, if unknown, return 0.
            return np.zeros_like(X[:, 0] if X.ndim == 2 else X)

    def get_size(self):
        if not self.args:
            return 1
        return 1 + sum(child.get_size() for child in self.args)

    def get_subtrees(self):
        """Get all subtree indices (0-based in a list representation? No, let's use a helper)."""
        # This is tricky with the object structure. 
        # Better to implement crossover/mutation on the tree structure directly.
        pass

def _create_random_tree(max_depth, depth=0, p_func=0.7):
    """
    Create a random tree using full method (grow) or half-and-half.
    """
    if depth >= max_depth or (depth > 0 and random.random() > p_func):
        # Terminal
        return GPNode('x')
    
    # Function
    func_name = random.choice(list(FUNCTIONS.keys()))
    n_args = 2 if func_name in ['+', '-', '*', '/', 'pow'] else 1
    
    children = []
    for _ in range(n_args):
        child = _create_random_tree(max_depth, depth + 1, p_func)
        children.append(child)
    
    return GPNode(func_name, *children)

def _mutate_tree(tree, mutation_rate=0.1):
    """
    Mutate a tree by replacing a random subtree with a new random tree.
    """
    if random.random() > mutation_rate:
        return tree
    
    # We need a way to pick a random node.
    # Let's flatten the tree into a list of nodes to pick an index.
    nodes = []
    parents = [] # (parent, index_in_parent)
    
    def _flatten(node, parent=None, idx=None):
        nodes.append(node)
        if parent is not None:
            parents.append((parent, idx))
        for i, child in enumerate(node.args):
            _flatten(child, node, i)
    
    _flatten(tree)
    
    if not nodes:
        return tree
    
    # Pick a random node to replace
    idx = random.randint(0, len(nodes) - 1)
    target_node = nodes[idx]
    
    # Generate a new random subtree
    new_subtree = _create_random_tree(max_depth=3, depth=0)
    
    # Replace
    if idx == 0:
        return new_subtree
    else:
        parent, child_idx = parents[idx]
        parent.args[child_idx] = new_subtree
        return tree

def _crossover(tree1, tree2, crossover_rate=0.7):
    """
    Crossover: pick a random subtree from tree1 and replace it with a random subtree from tree2.
    """
    if random.random() > crossover_rate:
        return tree1
    
    # Flatten tree1
    nodes1 = []
    parents1 = []
    def _flatten1(node, parent=None, idx=None):
        nodes1.append(node)
        if parent is not None:
            parents1.append((parent, idx))
        for i, child in enumerate(node.args):
            _flatten1(child, node, i)
    _flatten1(tree1)
    
    if not nodes1:
        return tree1
    
    idx1 = random.randint(0, len(nodes1) - 1)
    
    # Flatten tree2 to get a random subtree
    nodes2 = []
    def _flatten2(node):
        nodes2.append(node)
        for child in node.args:
            _flatten2(child)
    _flatten2(tree2)
    
    if not nodes2:
        return tree1
    
    idx2 = random.randint(0, len(nodes2) - 1)
    donor_subtree = nodes2[idx2]
    
    # Replace
    if idx1 == 0:
        return donor_subtree
    else:
        parent, child_idx = parents1[idx1]
        parent.args[child_idx] = donor_subtree
        return tree1

def _fit_gp(X, y, time_limit):
    """
    Run GP to fit X to y.
    Returns the best tree.
    """
    n, d = X.shape
    # X is (n, d). We assume d=1 for now based on problem description.
    x_input = X[:, 0]
    
    population_size = 200
    max_generations = 1000 # Will be limited by time
    max_depth = 6
    
    # Initialize population
    population = []
    for _ in range(population_size):
        # Half-and-half
        if random.random() < 0.5:
            # Full
            tree = _create_random_tree(max_depth, 0, 1.0)
        else:
            # Grow
            tree = _create_random_tree(max_depth, 0, 0.5)
        population.append(tree)
    
    best_tree = None
    best_fitness = np.inf
    
    start_time = time.time()
    
    for gen in range(max_generations):
        if time.time() > time_limit:
            break
        
        # Evaluate fitness
        fitnesses = []
        for tree in population:
            try:
                pred = tree.evaluate(X)
                # Check for NaN/Inf
                if not np.all(np.isfinite(pred)):
                    mse = 1e10
                else:
                    mse = np.mean((pred - y) ** 2)
            except Exception:
                mse = 1e10
            fitnesses.append(mse)
            
            if mse < best_fitness:
                best_fitness = mse
                best_tree = tree
        
        # Selection: Tournament selection
        # Sort by fitness
        sorted_indices = np.argsort(fitnesses)
        
        new_population = []
        
        # Elitism: keep top 5
        for i in range(min(5, len(sorted_indices))):
            new_population.append(population[sorted_indices[i]])
        
        # Crossover and Mutation
        while len(new_population) < population_size:
            # Pick two parents
            idx1 = sorted_indices[random.randint(0, 10)] # Top 10
            idx2 = sorted_indices[random.randint(0, 10)]
            
            parent1 = population[idx1]
            parent2 = population[idx2]
            
            child = _crossover(parent1, parent2)
            child = _mutate_tree(child)
            
            new_population.append(child)
        
        population = new_population
        
        # Check convergence
        if best_fitness < 1e-8:
            break
            
    return best_tree

def fit(X_train, y_train):
    global _DEADLINE
    
    start_time = time.time()
    n, d = X_train.shape
    
    # Time allocation
    # We have ~300s total.
    # Let's give GP 250s.
    gp_time_limit = min(_DEADLINE - start_time - 10, 250.0)
    
    # Run GP
    best_tree = _fit_gp(X_train, y_train, gp_time_limit)
    
    # If GP failed or didn't find anything, fallback to polynomial
    if best_tree is None:
        # Fallback: Linear regression
        try:
            X_poly = np.column_stack([np.ones(n), X_train[:, 0], X_train[:, 0]**2, X_train[:, 0]**3])
            w, *_ = np.linalg.lstsq(X_poly, y_train, rcond=None)
            def predict(X):
                x = X[:, 0]
                Xp = np.column_stack([np.ones(len(x)), x, x**2, x**3])
                return Xp @ w
            return predict
        except Exception:
            y_mean = np.mean(y_train)
            def predict(X):
                return np.full(X.shape[0], y_mean)
            return predict
    
    # Create predict function
    def predict(X):
        try:
            # Ensure X is 2D
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            result = best_tree.evaluate(X)
            # Ensure finite
            result = np.where(np.isfinite(result), result, np.mean(y_train))
            return result
        except Exception:
            return np.full(X.shape[0], np.mean(y_train))
    
    return predict
# EVOLVE-BLOCK-END
