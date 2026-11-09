"""Genetic Programming Symbolic Regression with Time Budget."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import operator
from typing import List, Union

# Constants
TERMINALS = ['x0', 'x1', 'x2']  # We will filter based on actual d
FUNCTIONS = [
    ('+', operator.add, 2),
    ('-', operator.sub, 2),
    ('*', operator.mul, 2),
    ('/', lambda a, b: np.where(np.abs(b) > 1e-12, a / np.where(np.abs(b) > 1e-12, b, 1e-12), 0.0), 2),
    ('sin', np.sin, 1),
    ('cos', np.cos, 1),
    ('sqrt', lambda a: np.sqrt(np.maximum(a, 0)), 1),
    ('abs', np.abs, 1),
    ('exp', lambda a: np.exp(np.clip(a, -10, 10)), 1),
    ('log', lambda a: np.log(np.maximum(a, 1e-12)), 1),
]

class Node:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children if children else []
        
    def __repr__(self):
        if not self.children:
            return self.name
        return f"{self.name}({', '.join(map(str, self.children))})"

    def depth(self):
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def size(self):
        if not self.children:
            return 1
        return 1 + sum(c.size() for c in self.children)

def get_function(name):
    for f in FUNCTIONS:
        if f[0] == name:
            return f[1]
    return None

def get_arity(name):
    for f in FUNCTIONS:
        if f[0] == name:
            return f[2]
    return 0

def evaluate_tree(node, X):
    """
    Recursively evaluate the tree.
    X: numpy array of shape (n, d)
    """
    if not node.children:
        # Terminal
        try:
            idx = int(node.name[1:])
            if idx < X.shape[1]:
                return X[:, idx]
            else:
                return np.zeros(X.shape[0])
        except:
            return np.zeros(X.shape[0])
    else:
        func_name = node.name
        func = get_function(func_name)
        if func is None:
            return np.zeros(X.shape[0])
        
        args = []
        for child in node.children:
            arg_val = evaluate_tree(child, X)
            args.append(arg_val)
        
        try:
            res = func(*args)
            # Handle NaNs and Infs
            res = np.nan_to_num(res, nan=0.0, posinf=1e6, neginf=-1e6)
            return res
        except:
            return np.zeros(X.shape[0])

def random_tree(depth, max_depth, d):
    """Generate a random tree with half-and-half method."""
    if depth == max_depth or (depth < max_depth and random.random() < 0.1):
        # Terminal
        return Node(random.choice([f"x{i}" for i in range(d)]))
    
    # Function node
    func_name, _, arity = random.choice(FUNCTIONS)
    children = [random_tree(depth + 1, max_depth, d) for _ in range(arity)]
    return Node(func_name, children)

def random_subtree(max_depth, d):
    return random_tree(0, max_depth, d)

def crossover(tree1, tree2, d):
    """
    Subtree crossover.
    """
    # Get all nodes in tree1
    nodes1 = []
    def get_nodes(node):
        nodes1.append(node)
        for c in node.children:
            get_nodes(c)
    get_nodes(tree1)
    
    if not nodes1:
        return tree1, tree2
        
    # Pick a random node in tree1 to replace
    idx = random.randint(0, len(nodes1) - 1)
    target_node = nodes1[idx]
    
    # Create a new tree by replacing target_node with a copy of tree2 (or a random subtree)
    # To keep it simple, we replace with a new random subtree of limited depth
    # Or we can swap with a node from tree2. Let's do subtree replacement with a random subtree.
    new_subtree = random_subtree(3, d)
    
    # Deep copy tree1
    def copy_tree(node):
        if not node.children:
            return Node(node.name)
        children = [copy_tree(c) for c in node.children]
        return Node(node.name, children)
    
    new_tree1 = copy_tree(tree1)
    
    # Now we need to replace the specific node. Since we copied, we need to find the path or just re-construct.
    # Easier: modify the copy in place.
    def replace_node(node, target, new_node):
        if node is target: # This won't work with copies
            pass
        
    # Let's do a proper recursive replacement
    def replace_in_tree(node, target_idx, new_node, current_path, path_to_target):
        if current_path == path_to_target:
            return new_node
        if not node.children:
            return node
        new_children = []
        for i, c in enumerate(node.children):
            new_children.append(replace_in_tree(c, target_idx, new_node, current_path + [i], path_to_target))
        return Node(node.name, new_children)
    
    # Find path to target in original tree1
    def find_path(node, target, path):
        if node is target:
            return path
        for i, c in enumerate(node.children):
            p = find_path(c, target, path + [i])
            if p is not None:
                return p
        return None
    
    path = find_path(tree1, target_node, [])
    if path is not None:
        new_tree1 = replace_in_tree(new_tree1, target_idx, new_subtree, [], path)
        
    # For tree2, we just return a copy of tree2 for now (or mutate it)
    new_tree2 = copy_tree(tree2)
    
    return new_tree1, new_tree2

def mutation(tree, d):
    """
    Point mutation or subtree mutation.
    """
    def copy_tree(node):
        if not node.children:
            return Node(node.name)
        children = [copy_tree(c) for c in node.children]
        return Node(node.name, children)
        
    new_tree = copy_tree(tree)
    
    # Get all nodes
    nodes = []
    def get_nodes(node):
        nodes.append(node)
        for c in node.children:
            get_nodes(c)
    get_nodes(new_tree)
    
    if not nodes:
        return new_tree
        
    idx = random.randint(0, len(nodes) - 1)
    target_node = nodes[idx]
    
    if random.random() < 0.5:
        # Point mutation: change function name or terminal
        if target_node.children:
            # Change function
            target_node.name = random.choice([f[0] for f in FUNCTIONS])
            # Ensure arity matches? No, simple mutation might break arity.
            # For safety, if we change a function, we should ideally restructure or just stick to same arity.
            # Let's only mutate to a function with the same arity.
            current_arity = len(target_node.children)
            valid_funcs = [f[0] for f in FUNCTIONS if f[2] == current_arity]
            if valid_funcs:
                target_node.name = random.choice(valid_funcs)
        else:
            # Change terminal
            target_node.name = random.choice([f"x{i}" for i in range(d)])
    else:
        # Subtree mutation: replace with random subtree
        new_subtree = random_subtree(3, d)
        # Replace target_node in new_tree with new_subtree
        # We need a helper to replace a specific node object reference in the tree structure
        # Since we are in a copy, we can just do a recursive walk and replace the first match by identity?
        # No, we copied it, so identities are different.
        # We need to replace by position.
        
        # Find path again
        def find_path_in_copy(node, target, path):
            if node is target:
                return path
            for i, c in enumerate(node.children):
                p = find_path_in_copy(c, target, path + [i])
                if p is not None:
                    return p
            return None
            
        path = find_path_in_copy(new_tree, target_node, [])
        if path is not None:
             def replace_by_path(node, path, new_node):
                 if not path:
                     return new_node
                 if not node.children:
                     return node
                 next_path = path[1:]
                 new_children = []
                 for i, c in enumerate(node.children):
                     if i == path[0]:
                         new_children.append(replace_by_path(c, next_path, new_node))
                     else:
                         new_children.append(c)
                 return Node(node.name, new_children)
             
             new_tree = replace_by_path(new_tree, path, new_subtree)
             
    return new_tree

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin
    
    d = X_train.shape[1]
    n = len(X_train)
    y_var = np.var(y_train)
    y_mean = np.mean(y_train)
    
    if y_var < 1e-12:
        return lambda X: np.full(X.shape[0], y_mean)
    
    # Baseline: Mean
    best_pred = np.full(n, y_mean)
    best_nrmse = np.mean((best_pred - y_train)**2) / y_var
    best_tree = None
    
    # Baseline: Linear fit (simple)
    try:
        # Add bias
        X_ext = np.column_stack([np.ones(n), X_train])
        w, *_ = np.linalg.lstsq(X_ext, y_train, rcond=None)
        pred_lin = X_ext @ w
        nrmse_lin = np.mean((pred_lin - y_train)**2) / y_var
        if nrmse_lin < best_nrmse:
            best_nrmse = nrmse_lin
            # Represent linear as a tree? Or just keep the model type separate.
            # To keep it simple, let's just let GP try to find it.
            # But GP might be slow. Let's keep a separate "linear" flag.
            pass
    except:
        pass
    
    # GP Initialization
    population_size = 50
    max_depth = 4
    
    # Initialize population
    population = [random_tree(0, max_depth, d) for _ in range(population_size)]
    
    # Evaluate initial population
    def eval_tree(tree):
        try:
            pred = evaluate_tree(tree, X_train)
            mse = np.mean((pred - y_train)**2)
            nrmse = mse / y_var
            return nrmse
        except:
            return float('inf')
    
    scores = [eval_tree(t) for t in population]
    
    # Track best
    min_idx = np.argmin(scores)
    if scores[min_idx] < best_nrmse:
        best_nrmse = scores[min_idx]
        best_tree = population[min_idx]
    
    # GP Loop
    while time.time() < deadline - 10:
        # Selection: Tournament
        def select():
            a = random.choice(population)
            b = random.choice(population)
            sa = eval_tree(a)
            sb = eval_tree(b)
            return a if sa < sb else b
        
        new_population = []
        
        for _ in range(population_size):
            op = random.random()
            if op < 0.1:
                # Elitism: keep best from previous gen? 
                # Simple: just copy best
                new_population.append(copy_tree(population[min_idx]))
            elif op < 0.7:
                # Crossover
                p1 = select()
                p2 = select()
                c1, c2 = crossover(p1, p2, d)
                new_population.append(c1)
                if len(new_population) < population_size:
                    new_population.append(c2)
            else:
                # Mutation
                p1 = select()
                m1 = mutation(p1, d)
                new_population.append(m1)
                
        # Ensure population size
        while len(new_population) < population_size:
            new_population.append(random_tree(0, max_depth, d))
        
        population = new_population[:population_size]
        
        # Evaluate new population
        scores = [eval_tree(t) for t in population]
        min_idx = np.argmin(scores)
        if scores[min_idx] < best_nrmse:
            best_nrmse = scores[min_idx]
            best_tree = population[min_idx]
            
            # Optional: If we get very good, we could stop early?
            # But let's keep searching to see if we can improve.
            if best_nrmse < 1e-6:
                break

    # If GP found a good tree, use it.
    if best_tree is not None:
        def predict_gp(X):
            try:
                return evaluate_tree(best_tree, X)
            except:
                return np.full(X.shape[0], y_mean)
        return predict_gp
    else:
        # Fallback to linear or mean
        try:
            X_ext = np.column_stack([np.ones(n), X_train])
            w, *_ = np.linalg.lstsq(X_ext, y_train, rcond=None)
            return lambda X: np.column_stack([np.ones(X.shape[0]), X]) @ w
        except:
            return lambda X: np.full(X.shape[0], y_mean)

# Helper for copy_tree used in fit
def copy_tree(node):
    if not node.children:
        return Node(node.name)
    children = [copy_tree(c) for c in node.children]
    return Node(node.name, children)

# EVOLVE-BLOCK-END
