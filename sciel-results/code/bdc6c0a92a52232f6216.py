"""Genetic Programming based symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import operator

# Deadline guard: 330s wall-clock limit, use 300s for safety
_DEADLINE = time.time() + 300.0
_TIME_PER_EQUATION = 70.0  # roughly 4 equations * 70s = 280s, leaving margin

# GP Parameters
POP_SIZE = 200
MAX_DEPTH = 6
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
ELITE_SIZE = 5

# Primitives
# Terminal: x (input)
# Functions: +, -, *, /, sin, cos, exp, log, sqrt, abs
# Note: We need to handle division by zero and log/sqrt domain issues safely.

def _safe_div(x, y):
    # Avoid division by zero
    y_safe = np.where(np.abs(y) < 1e-12, np.sign(y) * 1e-12 + (y == 0), y)
    return x / y_safe

def _safe_log(x):
    return np.log(np.abs(x) + 1e-12)

def _safe_sqrt(x):
    return np.sqrt(np.abs(x))

def _safe_exp(x):
    # Clamp to avoid overflow
    x_clamped = np.clip(x, -700, 700)
    return np.exp(x_clamped)

GP_FUNCS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': _safe_div,
    'sin': np.sin,
    'cos': np.cos,
    'exp': _safe_exp,
    'log': _safe_log,
    'sqrt': _safe_sqrt,
    'abs': np.abs
}

GP_FUNC_NAMES = list(GP_FUNCS.keys())
GP_ARITY = {
    '+': 2, '-': 2, '*': 2, '/': 2,
    'sin': 1, 'cos': 1, 'exp': 1, 'log': 1, 'sqrt': 1, 'abs': 1
}

class GPNode:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children or []
        
    def evaluate(self, X):
        """
        Evaluate the tree for input X (shape n, d).
        Since we have 1D inputs in the problem description (d=1 for all), 
        we assume X is (n, 1) or (n,). We'll flatten to 1D for evaluation.
        """
        if self.name == 'x':
            return X.flatten()
        
        func = GP_FUNCS[self.name]
        arity = GP_ARITY[self.name]
        
        if arity == 1:
            arg = self.children[0].evaluate(X)
            return func(arg)
        elif arity == 2:
            left = self.children[0].evaluate(X)
            right = self.children[1].evaluate(X)
            return func(left, right)
        else:
            raise ValueError(f"Unknown arity for {self.name}")

def _create_random_tree(depth=0, max_depth=6, force_leaf=False):
    if depth >= max_depth or (force_leaf and depth > 0):
        return GPNode('x')
    
    # Probability of stopping
    if depth > 0 and random.random() < 0.3:
        return GPNode('x')
    
    func_name = random.choice(GP_FUNC_NAMES)
    arity = GP_ARITY[func_name]
    children = []
    for _ in range(arity):
        child = _create_random_tree(depth + 1, max_depth)
        children.append(child)
    
    return GPNode(func_name, children)

def _get_subtree(root, path):
    node = root
    for idx in path:
        node = node.children[idx]
    return node

def _crossover(tree1, tree2):
    """
    Subtree crossover.
    Returns a new tree.
    """
    # Get random path in tree1
    path1 = []
    node = tree1
    while node.name != 'x':
        idx = random.randint(0, len(node.children) - 1)
        path1.append(idx)
        node = node.children[idx]
    
    # Get random path in tree2
    path2 = []
    node = tree2
    while node.name != 'x':
        idx = random.randint(0, len(node.children) - 1)
        path2.append(idx)
        node = node.children[idx]
    
    # Swap subtrees
    # We need to modify the trees in place or create copies. 
    # For simplicity and safety, we'll implement a swap that works on the structure.
    # Note: This implementation assumes we can swap the child pointers.
    
    # Find the parent of the subtree in tree1
    if not path1:
        # Swap root
        return tree2 # Conceptually, we return a new tree structure. 
                     # In a real GP, we'd manage a population of trees.
                     # Here, we'll just return tree2 as the offspring for simplicity 
                     # if we don't implement deep copy/swap logic properly.
                     # Let's implement a proper swap.
    
    # Proper Swap Logic:
    # 1. Get subtree A from tree1 at path1
    # 2. Get subtree B from tree2 at path2
    # 3. Replace subtree A in tree1 with subtree B (conceptually)
    # 4. Replace subtree B in tree2 with subtree A (conceptually)
    
    # Since we want to return ONE offspring, we can:
    # Create a copy of tree1, and replace the subtree at path1 with the subtree from tree2.
    # To avoid complex copying, we can just return the subtree from tree2 if we treat 
    # the "offspring" as the result of the operation. But standard GP returns a full tree.
    
    # Let's implement a helper to get the subtree object.
    sub1 = _get_subtree(tree1, path1)
    sub2 = _get_subtree(tree2, path2)
    
    # We will modify tree1 to contain sub2 at the location of sub1.
    # And we return tree1.
    # This mutates tree1. To avoid side effects, we should copy.
    # Given time constraints, let's assume we can mutate or just use a simple heuristic:
    # If path1 is empty, return tree2.
    # Otherwise, we need to traverse tree1 and replace the child.
    
    # Simplified Crossover for this implementation:
    # We'll just pick a random subtree from tree2 and attach it to a random spot in tree1.
    # This is a "graft" operation.
    
    # Find parent in tree1
    if path1:
        parent = tree1
        for idx in path1[:-1]:
            parent = parent.children[idx]
        last_idx = path1[-1]
        parent.children[last_idx] = sub2
        return tree1
    else:
        return tree2

def _mutate(tree):
    """
    Point mutation: replace a random subtree with a new random tree.
    """
    # Pick a random node
    # We need a way to traverse all nodes.
    # For simplicity, we'll just mutate the root or a random child.
    # Better: Randomly pick a path.
    path = []
    node = tree
    while node.name != 'x' and random.random() > 0.2:
        idx = random.randint(0, len(node.children) - 1)
        path.append(idx)
        node = node.children[idx]
        
    # Create new subtree
    new_subtree = _create_random_tree(depth=0, max_depth=3)
    
    # Replace
    if path:
        parent = tree
        for idx in path[:-1]:
            parent = parent.children[idx]
        last_idx = path[-1]
        parent.children[last_idx] = new_subtree
    else:
        # Replace root
        return new_subtree
        
    return tree

def _fit_gp(X_train, y_train):
    """
    Run Genetic Programming to find the best symbolic expression.
    """
    n = X_train.shape[0]
    X_flat = X_train.flatten()
    
    # Initialize population
    population = [_create_random_tree() for _ in range(POP_SIZE)]
    
    best_tree = None
    best_error = np.inf
    
    # Evaluate initial population
    for tree in population:
        try:
            pred = tree.evaluate(X_flat)
            # Check for non-finite values
            if not np.all(np.isfinite(pred)):
                error = np.inf
            else:
                error = np.mean((pred - y_train) ** 2)
            
            if error < best_error:
                best_error = error
                best_tree = tree
        except Exception:
            error = np.inf
    
    # Evolution loop
    start_time = time.time()
    timeout = _TIME_PER_EQUATION - 5  # Leave 5s margin
    
    generation = 0
    while time.time() - start_time < timeout:
        generation += 1
        
        # Sort population by error
        scored_pop = []
        for tree in population:
            try:
                pred = tree.evaluate(X_flat)
                if not np.all(np.isfinite(pred)):
                    error = np.inf
                else:
                    error = np.mean((pred - y_train) ** 2)
            except Exception:
                error = np.inf
            scored_pop.append((error, tree))
        
        scored_pop.sort(key=lambda x: x[0])
        
        # Update best
        if scored_pop[0][0] < best_error:
            best_error = scored_pop[0][0]
            best_tree = scored_pop[0][1]
            
        # Create new population
        new_population = []
        
        # Elitism: keep top ELITE_SIZE
        for i in range(min(ELITE_SIZE, len(scored_pop))):
            new_population.append(scored_pop[i][1])
            
        # Fill rest with offspring
        while len(new_population) < POP_SIZE:
            # Selection: Tournament selection
            t1 = random.choice(scored_pop)
            t2 = random.choice(scored_pop)
            parent1 = t1[1] if t1[0] < t2[0] else t2[1]
            
            t1 = random.choice(scored_pop)
            t2 = random.choice(scored_pop)
            parent2 = t1[1] if t1[0] < t2[0] else t2[1]
            
            if random.random() < CROSSOVER_RATE:
                # Crossover
                # To avoid mutating parents in the population, we should ideally copy.
                # For speed, we'll do a simplified crossover that returns a new tree structure
                # by deep copying the path.
                # Given complexity, let's use a simpler mutation-heavy approach for speed
                # or implement a safe crossover.
                
                # Safe Crossover Implementation:
                # We need to create a copy of parent1, then graft parent2's subtree.
                # Let's implement a helper to copy a tree.
                def copy_tree(node):
                    if node.name == 'x':
                        return GPNode('x')
                    children = [copy_tree(c) for c in node.children]
                    return GPNode(node.name, children)
                
                offspring = copy_tree(parent1)
                
                # Get random subtree from parent2
                path2 = []
                node = parent2
                while node.name != 'x':
                    idx = random.randint(0, len(node.children) - 1)
                    path2.append(idx)
                    node = node.children[idx]
                sub2 = _get_subtree(parent2, path2)
                
                # Graft into offspring
                if path2: # Wait, we need a path in offspring too.
                    # Pick random spot in offspring
                    path1 = []
                    node = offspring
                    while node.name != 'x' and random.random() > 0.3:
                        idx = random.randint(0, len(node.children) - 1)
                        path1.append(idx)
                        node = node.children[idx]
                    
                    if path1:
                        parent_node = offspring
                        for idx in path1[:-1]:
                            parent_node = parent_node.children[idx]
                        last_idx = path1[-1]
                        parent_node.children[last_idx] = copy_tree(sub2)
                    else:
                        offspring = copy_tree(sub2)
                else:
                    offspring = copy_tree(sub2)
                
                new_population.append(offspring)
            else:
                # Mutation
                offspring = copy_tree(parent1)
                # Mutate
                path = []
                node = offspring
                while node.name != 'x' and random.random() > 0.3:
                    idx = random.randint(0, len(node.children) - 1)
                    path.append(idx)
                    node = node.children[idx]
                
                if path:
                    parent_node = offspring
                    for idx in path[:-1]:
                        parent_node = parent_node.children[idx]
                    last_idx = path[-1]
                    parent_node.children[last_idx] = _create_random_tree(depth=0, max_depth=3)
                else:
                    offspring = _create_random_tree()
                    
                new_population.append(offspring)
        
        population = new_population
        
        # Early stopping if error is very low
        if best_error < 1e-10:
            break
            
    return best_tree, best_error

def fit(X_train, y_train):
    global _DEADLINE
    start_time = time.time()
    
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    if y_var < 1e-12:
        y_var = 1e-12
    
    # Fallback predict
    best_predict = lambda X: np.full(X.shape[0], y_mean)
    best_mse = np.inf
    
    # Try GP
    try:
        tree, mse = _fit_gp(X_train, y_train)
        if tree is not None and mse < best_mse:
            best_mse = mse
            def gp_predict(X):
                try:
                    x_flat = X.flatten()
                    pred = tree.evaluate(x_flat)
                    # Ensure finite
                    pred = np.where(np.isfinite(pred), pred, y_mean)
                    return pred
                except Exception:
                    return np.full(X.shape[0], y_mean)
            best_predict = gp_predict
    except Exception as e:
        pass
    
    # If GP failed or didn't find a good solution, try a simple polynomial fallback
    # to ensure we don't return a bad mean prediction if a simple fit works better.
    if best_mse > 1e-3:
        try:
            # Simple linear regression fallback
            X_poly = np.column_stack([np.ones(n), X_train.flatten()])
            w, *_ = np.linalg.lstsq(X_poly, y_train, rcond=None)
            pred = X_poly @ w
            mse = np.mean((pred - y_train) ** 2)
            if mse < best_mse:
                best_mse = mse
                def poly_predict(X):
                    X_p = np.column_stack([np.ones(X.shape[0]), X.flatten()])
                    return X_p @ w
                best_predict = poly_predict
        except Exception:
            pass
            
    # Final safety check
    def safe_predict(X):
        try:
            result = best_predict(X)
            result = np.where(np.isfinite(result), result, y_mean)
            return result
        except Exception:
            return np.full(X.shape[0], y_mean)
    
    return safe_predict
# EVOLVE-BLOCK-END
