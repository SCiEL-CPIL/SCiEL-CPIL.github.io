import numpy as np
import time
import random
import math

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

def protected_div(x, y):
    """Protected division to avoid division by zero."""
    return np.where(np.abs(y) > 1e-12, x / y, 0.0)

def protected_log(x):
    """Protected log to avoid log of non-positive numbers."""
    return np.log(np.maximum(x, 1e-12))

def protected_sqrt(x):
    """Protected sqrt to avoid sqrt of negative numbers."""
    return np.sqrt(np.maximum(x, 0.0))

def protected_arcsin(x):
    return np.arcsin(np.clip(x, -1.0, 1.0))

def protected_arccos(x):
    return np.arccos(np.clip(x, -1.0, 1.0))

def protected_arctan(x):
    return np.arctan(x)

class Individual:
    def __init__(self, expr, X_train, y_train):
        self.expr = expr
        self.X_train = X_train
        self.y_train = y_train
        self.fitness = self._calculate_fitness()
        
    def _calculate_fitness(self):
        try:
            pred = self.expr(self.X_train)
            # Check for non-finite values
            if not np.all(np.isfinite(pred)):
                return np.inf
            mse = np.mean((pred - self.y_train) ** 2)
            return mse
        except Exception:
            return np.inf

def generate_random_expr(depth=0, max_depth=3, X_train=None):
    """
    Generates a random expression as a callable function.
    We represent the expression as a nested lambda or function tree.
    For simplicity and speed, we'll build a list of operations and evaluate it.
    However, for GP, it's easier to store the tree structure.
    Let's use a simple tree representation:
    Node: ('op', left, right) or ('const', value) or ('var', index)
    """
    if depth == max_depth:
        return _generate_terminal(X_train)
    
    if random.random() < 0.2: # 20% chance to stop early
        return _generate_terminal(X_train)
    
    op = random.choice(['+', '-', '*', '/', 'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs'])
    left = generate_random_expr(depth + 1, max_depth, X_train)
    right = generate_random_expr(depth + 1, max_depth, X_train)
    return (op, left, right)

def _generate_terminal(X_train):
    """Generates a terminal node: variable or constant."""
    if X_train is not None and random.random() < 0.8: # 80% chance to pick a variable
        idx = random.randint(0, X_train.shape[1] - 1)
        return ('var', idx)
    else:
        # Constant in range [-5, 5] or [0.1, 5]
        val = random.uniform(-5, 5)
        return ('const', val)

def evaluate_expr(node, X):
    """Recursively evaluates the expression tree."""
    if node[0] == 'var':
        return X[:, node[1]]
    elif node[0] == 'const':
        return np.full(X.shape[0], node[1])
    elif node[0] == '+':
        return evaluate_expr(node[1], X) + evaluate_expr(node[2], X)
    elif node[0] == '-':
        return evaluate_expr(node[1], X) - evaluate_expr(node[2], X)
    elif node[0] == '*':
        return evaluate_expr(node[1], X) * evaluate_expr(node[2], X)
    elif node[0] == '/':
        return protected_div(evaluate_expr(node[1], X), evaluate_expr(node[2], X))
    elif node[0] == 'sin':
        return np.sin(evaluate_expr(node[1], X))
    elif node[0] == 'cos':
        return np.cos(evaluate_expr(node[1], X))
    elif node[0] == 'tan':
        return np.tan(evaluate_expr(node[1], X))
    elif node[0] == 'exp':
        # Clip to avoid overflow
        val = np.clip(evaluate_expr(node[1], X), -50, 50)
        return np.exp(val)
    elif node[0] == 'log':
        return protected_log(evaluate_expr(node[1], X))
    elif node[0] == 'sqrt':
        return protected_sqrt(evaluate_expr(node[1], X))
    elif node[0] == 'abs':
        return np.abs(evaluate_expr(node[1], X))
    else:
        raise ValueError(f"Unknown op: {node[0]}")

def get_subtrees(node, parent=None):
    """Yields (parent, child_index, subtree) for all subtrees."""
    if node[0] in ['var', 'const']:
        return
    yield parent, node
    for i in [1, 2]:
        if node[0] not in ['var', 'const']:
            yield from get_subtrees(node[i], node)

def mutate(node, X_train, max_depth=3):
    """Performs subtree replacement mutation."""
    # Get a random subtree
    subtrees = list(get_subtrees(node))
    if not subtrees:
        return node
    
    # Randomly select a subtree to replace
    # Note: get_subtrees returns (parent, subtree_node). We need to modify the parent's reference.
    # This is tricky with tuples. Let's use a mutable structure or copy.
    # For simplicity, let's just replace the whole root with a new random expr if mutation fails or is complex.
    # Better approach: Deep copy and modify.
    
    # Simpler mutation: Point mutation of constants or ops, or replace root with random.
    # Let's do subtree replacement properly by using a list-based tree or careful copying.
    
    # Since tuples are immutable, we'll rebuild the tree.
    # Alternative: Use a class for Node. But for speed, let's try a simpler mutation strategy:
    # 1. With prob 0.5, replace a random constant with a new constant.
    # 2. With prob 0.3, replace a random op with a new op.
    # 3. With prob 0.2, replace a random subtree with a new random subtree.
    
    # Let's implement a helper to find a random path to a node.
    def find_random_path(node):
        if node[0] in ['var', 'const']:
            return []
        # Randomly choose left or right
        side = random.randint(1, 2)
        return [side] + find_random_path(node[side])
    
    path = find_random_path(node)
    
    # Now we need to traverse and replace the node at the end of the path.
    # We'll do a recursive copy-and-replace.
    def copy_and_replace(node, path, depth=0):
        if depth == len(path):
            # This is the node to replace
            if random.random() < 0.5:
                # Replace with new random subtree
                return generate_random_expr(0, max_depth, X_train)
            else:
                # Point mutation: change op or const
                if node[0] == 'const':
                    return ('const', random.uniform(-5, 5))
                elif node[0] == 'var':
                    # Change to another variable
                    new_idx = random.randint(0, X_train.shape[1] - 1)
                    return ('var', new_idx)
                else:
                    # Change op
                    new_op = random.choice(['+', '-', '*', '/', 'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs'])
                    return (new_op, node[1], node[2])
        
        new_left = copy_and_replace(node[1], path, depth + 1) if path[0] == 1 else node[1]
        new_right = copy_and_replace(node[2], path, depth + 1) if path[0] == 2 else node[2]
        
        # If the node itself is a leaf, it shouldn't be reached by a path of length > 0 unless it's the target
        # But our path logic stops at the target.
        # If we are not at the target, we just recurse down the path.
        # Wait, the logic above replaces the node at depth == len(path).
        # But if we are at a node that is NOT the target, we need to recurse into the child specified by path[0].
        
        # Let's re-structure:
        if depth == len(path):
             # Return the modified node (handled above)
             pass # The logic above returned directly? No, it fell through.
        
        # Correct logic:
        if depth == len(path):
            # We are at the node to mutate
            if random.random() < 0.5:
                return generate_random_expr(0, max_depth, X_train)
            else:
                if node[0] == 'const':
                    return ('const', random.uniform(-5, 5))
                elif node[0] == 'var':
                    return ('var', random.randint(0, X_train.shape[1] - 1))
                else:
                    new_op = random.choice(['+', '-', '*', '/', 'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs'])
                    return (new_op, node[1], node[2])
        
        # Recurse
        if path[0] == 1:
            new_left = copy_and_replace(node[1], path[1:], depth + 1)
            new_right = node[2]
        else:
            new_left = node[1]
            new_right = copy_and_replace(node[2], path[1:], depth + 1)
            
        return (node[0], new_left, new_right)
        
    return copy_and_replace(node, path)

def crossover(node1, node2, X_train, max_depth=3):
    """Performs subtree crossover."""
    # Find a random subtree in node1
    def find_random_path(node):
        if node[0] in ['var', 'const']:
            return []
        side = random.randint(1, 2)
        return [side] + find_random_path(node[side])
        
    path1 = find_random_path(node1)
    
    # Extract subtree from node2
    def extract_subtree(node, path):
        if not path:
            return node
        return extract_subtree(node[path[0]], path[1:])
        
    subtree_from_2 = extract_subtree(node2, find_random_path(node2))
    
    # Replace subtree in node1 with subtree_from_2
    def replace_subtree(node, path, new_subtree, depth=0):
        if depth == len(path):
            return new_subtree
        
        if path[0] == 1:
            new_left = replace_subtree(node[1], path[1:], new_subtree, depth + 1)
            new_right = node[2]
        else:
            new_left = node[1]
            new_right = replace_subtree(node[2], path[1:], new_subtree, depth + 1)
            
        return (node[0], new_left, new_right)
        
    new_node1 = replace_subtree(node1, path1, subtree_from_2)
    
    # Symmetric crossover: replace subtree in node2 with subtree from node1
    path2 = find_random_path(node2)
    subtree_from_1 = extract_subtree(node1, path1)
    new_node2 = replace_subtree(node2, path2, subtree_from_1)
    
    return new_node1, new_node2

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 25.0 # Leave 5s margin for other equations overhead if any, but we have 330s total / 10 = 33s. 25s is safe.
    
    # Initial population
    pop_size = 50
    max_depth = 4
    population = []
    for _ in range(pop_size):
        expr = generate_random_expr(0, max_depth, X_train)
        ind = Individual(expr, X_train, y_train)
        population.append(ind)
        
    best_ind = min(population, key=lambda x: x.fitness)
    
    # GP Loop
    generation = 0
    while time.time() < deadline:
        # Sort by fitness
        population.sort(key=lambda x: x.fitness)
        best_ind = population[0]
        
        # Check for perfect fit
        if best_ind.fitness < 1e-8:
            break
            
        # Selection: Tournament
        def tournament_select(pop):
            k = 5
            candidates = random.sample(pop, min(k, len(pop)))
            return min(candidates, key=lambda x: x.fitness)
            
        new_population = [best_ind] # Elitism
        
        # Crossover and Mutation
        while len(new_population) < pop_size:
            p1 = tournament_select(population)
            p2 = tournament_select(population)
            
            r = random.random()
            if r < 0.6: # Crossover
                child1_expr, child2_expr = crossover(p1.expr, p2.expr, X_train, max_depth)
                child1 = Individual(child1_expr, X_train, y_train)
                child2 = Individual(child2_expr, X_train, y_train)
                new_population.append(child1)
                if len(new_population) < pop_size:
                    new_population.append(child2)
            elif r < 0.9: # Mutation
                child_expr = mutate(p1.expr, X_train, max_depth)
                child = Individual(child_expr, X_train, y_train)
                new_population.append(child)
            else: # Random restart
                child_expr = generate_random_expr(0, max_depth, X_train)
                child = Individual(child_expr, X_train, y_train)
                new_population.append(child)
                
        population = new_population
        generation += 1
        
    # Final Prediction Function
    def predict(X):
        return evaluate_expr(best_ind.expr, X)
        
    return predict
# EVOLVE-BLOCK-END
