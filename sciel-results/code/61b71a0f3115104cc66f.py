import numpy as np
import random
import time

# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def fit(X_train, y_train):
    """
    Symbolic regression using Genetic Programming.
    Falls back to high-order polynomial if GP fails to find a near-perfect fit.
    """
    d = X_train.shape[1]
    n_samples = X_train.shape[0]
    
    # Define operators and terminals
    # Operators take two arguments
    operators = ['+', '-', '*', '/', 'sin', 'cos', 'exp', 'log', 'abs', 'sqrt', 'pow2', 'pow3']
    # Terminals are variables or constants
    constants = [1, 2, 3, 4, 5, 0.5, -1, -2, -3, -4, -5]
    
    def create_node():
        """Create a random node. Returns (type, value_or_op, left, right)."""
        if random.random() < 0.5:
            # Terminal
            if random.random() < 0.7:
                # Variable
                return ('var', random.randint(0, d-1), None, None)
            else:
                # Constant
                return ('const', random.choice(constants), None, None)
        else:
            # Operator
            op = random.choice(operators)
            left = create_node()
            right = create_node()
            return (op, None, left, right)

    def tree_depth(node):
        if node[0] in ['var', 'const']:
            return 0
        return 1 + max(tree_depth(node[2]), tree_depth(node[3]))

    def create_tree(max_depth=5):
        """Create a random tree with depth <= max_depth."""
        # Simple method: grow until max depth, then shrink? 
        # Let's just use full growth with probability of stopping.
        def grow(depth):
            if depth >= max_depth:
                return create_node_terminal_only()
            if random.random() < 0.2: # Probability of stopping
                return create_node_terminal_only()
            return create_node()
        
        def create_node_terminal_only():
            if random.random() < 0.7:
                return ('var', random.randint(0, d-1), None, None)
            else:
                return ('const', random.choice(constants), None, None)

        return grow(0)

    def evaluate_tree(tree, X):
        """
        Evaluate tree on dataset X.
        Returns array of predictions.
        """
        if tree[0] == 'var':
            return X[:, tree[1]]
        elif tree[0] == 'const':
            return np.full(X.shape[0], tree[1])
        else:
            op = tree[0]
            left_val = evaluate_tree(tree[2], X)
            right_val = evaluate_tree(tree[3], X)
            
            # Handle division and log safely
            if op == '+':
                return left_val + right_val
            elif op == '-':
                return left_val - right_val
            elif op == '*':
                return left_val * right_val
            elif op == '/':
                # Avoid division by zero
                mask = right_val != 0
                res = np.zeros_like(left_val)
                res[mask] = left_val[mask] / right_val[mask]
                return res
            elif op == 'sin':
                return np.sin(left_val)
            elif op == 'cos':
                return np.cos(left_val)
            elif op == 'exp':
                # Clip to avoid overflow
                return np.exp(np.clip(left_val, -100, 100))
            elif op == 'log':
                # Log of negative is nan, clip to small positive
                return np.log(np.clip(left_val, 1e-10, None))
            elif op == 'abs':
                return np.abs(left_val)
            elif op == 'sqrt':
                return np.sqrt(np.clip(left_val, 0, None))
            elif op == 'pow2':
                return left_val ** 2
            elif op == 'pow3':
                return left_val ** 3
            else:
                return left_val + right_val # Fallback

    def calculate_fitness(tree, X, y):
        try:
            pred = evaluate_tree(tree, X)
            if not np.all(np.isfinite(pred)):
                return float('inf')
            mse = np.mean((pred - y) ** 2)
            var_y = np.var(y)
            if var_y == 0:
                return 0 if mse == 0 else float('inf')
            nmse = mse / var_y
            return nmse
        except:
            return float('inf')

    # Initial Population
    pop_size = 100
    population = [create_tree(max_depth=6) for _ in range(pop_size)]
    
    best_tree = None
    best_fitness = float('inf')
    
    # Precompute y stats
    y_var = np.var(y_train)
    
    start_time = time.time()
    time_limit = 25.0 # Leave some time for 10 equations total 300s
    
    # Initial Evaluation
    for i, tree in enumerate(population):
        f = calculate_fitness(tree, X_train, y_train)
        if f < best_fitness:
            best_fitness = f
            best_tree = tree
        
        # Early stop if perfect
        if best_fitness < 1e-10:
            break
            
    generations = 0
    max_generations = 200
    
    while generations < max_generations:
        if time.time() - start_time > time_limit:
            break
            
        # Check for perfect fit
        if best_fitness < 1e-10:
            break
            
        # Selection: Tournament
        def tournament(k=3):
            indices = random.sample(range(len(population)), k)
            best_idx = indices[0]
            best_f = calculate_fitness(population[best_idx], X_train, y_train)
            for idx in indices[1:]:
                f = calculate_fitness(population[idx], X_train, y_train)
                if f < best_f:
                    best_f = f
                    best_idx = idx
            return population[best_idx]
        
        new_population = []
        
        # Elitism: Keep best
        new_population.append(best_tree)
        
        while len(new_population) < pop_size:
            parent1 = tournament()
            parent2 = tournament()
            
            if random.random() < 0.7:
                # Crossover
                child = crossover(parent1, parent2)
            else:
                # Mutation
                child = mutate(parent1)
            
            # Simplification/Shrinking
            if random.random() < 0.2:
                child = shrink(child)
                
            new_population.append(child)
            
        population = new_population
        
        # Evaluate new population
        current_best = float('inf')
        for tree in population:
            f = calculate_fitness(tree, X_train, y_train)
            if f < current_best:
                current_best = f
                if f < best_fitness:
                    best_fitness = f
                    best_tree = tree
        
        generations += 1

    # Check if we found a good solution
    if best_fitness < 1e-5:
        # Return the symbolic solution
        final_tree = best_tree
        def predict(X):
            return evaluate_tree(final_tree, X)
        return predict
    else:
        # Fallback: High-order polynomial fit
        # This ensures we don't return a bad symbolic fit if GP failed
        def features_poly(X):
            cols = [np.ones(len(X))]
            for j in range(X.shape[1]):
                cols.append(X[:, j])
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    cols.append(X[:, i] * X[:, j])
            # Add cubic terms if dim is small
            if X.shape[1] <= 3:
                for i in range(X.shape[1]):
                    for j in range(i, X.shape[1]):
                        for k in range(j, X.shape[1]):
                            cols.append(X[:, i] * X[:, j] * X[:, k])
            return np.column_stack(cols)
        
        w, *_ = np.linalg.lstsq(features_poly(X_train), y_train, rcond=None)
        def predict_poly(X):
            return features_poly(X) @ w
        return predict_poly

def crossover(t1, t2):
    """Subtree crossover."""
    # Copy t1
    def copy_tree(t):
        if t[0] in ['var', 'const']:
            return t
        return (t[0], None, copy_tree(t[2]), copy_tree(t[3]))
    
    c1 = copy_tree(t1)
    c2 = copy_tree(t2)
    
    # Get random node from t2 to swap with random node in t1
    def get_random_node_indices(tree, indices=None):
        if indices is None:
            indices = []
        if tree[0] not in ['var', 'const']:
            indices.append(('left', tree[2], tree[3], tree[0])) # Placeholder, need to return the node itself or index
            # Actually, let's return the node object reference is tricky. 
            # Let's do a simpler recursive swap.
        return indices

    # Simpler crossover: Pick a random node in t1, replace with a random node from t2
    def pick_random_node(tree):
        if tree[0] in ['var', 'const']:
            return tree
        if random.random() < 0.5:
            return pick_random_node(tree[2])
        else:
            return pick_random_node(tree[3])

    def replace_node(tree, target_node, new_node):
        # This is tricky with shared references. 
        # Let's just do a depth-first search and replace the first match.
        # Since we copied, we can modify in place.
        if tree[0] in ['var', 'const']:
            if tree == target_node:
                return new_node
            return tree
        else:
            left = replace_node(tree[2], target_node, new_node)
            right = replace_node(tree[3], target_node, new_node)
            return (tree[0], None, left, right)
            
    # The above replace logic is flawed because tree equality might not work if we have nested structures or if we pick the same node twice.
    # Let's use a simpler approach: Random subtree mutation on a copy.
    
    def mutate_tree(tree):
        if tree[0] in ['var', 'const']:
            return tree
        if random.random() < 0.3:
            # Change operator
            new_op = random.choice(operators)
            return (new_op, None, tree[2], tree[3])
        # Recurse
        if random.random() < 0.5:
            new_left = mutate_tree(tree[2])
            return (tree[0], None, new_left, tree[3])
        else:
            new_right = mutate_tree(tree[3])
            return (tree[0], None, tree[2], new_right)
            
    # Let's implement a proper crossover by swapping subtrees
    def get_subtree_index(tree, path):
        # path is a list of 'L' or 'R'
        if not path:
            return tree
        if path[0] == 'L':
            return get_subtree_index(tree[2], path[1:])
        else:
            return get_subtree_index(tree[3], path[1:])
            
    def set_subtree(tree, path, new_subtree):
        if not path:
            return new_subtree
        if path[0] == 'L':
            new_left = set_subtree(tree[2], path[1:], new_subtree)
            return (tree[0], None, new_left, tree[3])
        else:
            new_right = set_subtree(tree[3], path[1:], new_subtree)
            return (tree[0], None, tree[2], new_right)
            
    # Generate random paths
    def random_path(tree, depth=0, max_depth=10):
        if tree[0] in ['var', 'const'] or depth >= max_depth:
            return []
        if random.random() < 0.5:
            return ['L'] + random_path(tree[2], depth+1, max_depth)
        else:
            return ['R'] + random_path(tree[3], depth+1, max_depth)
            
    path1 = random_path(t1)
    path2 = random_path(t2)
    
    subtree1 = get_subtree_index(t1, path1)
    subtree2 = get_subtree_index(t2, path2)
    
    c1 = set_subtree(t1, path1, subtree2)
    c2 = set_subtree(t2, path2, subtree1)
    
    return random.choice([c1, c2])

def mutate(tree):
    """Point mutation or subtree replacement."""
    def random_path(tree, depth=0, max_depth=10):
        if tree[0] in ['var', 'const'] or depth >= max_depth:
            return []
        if random.random() < 0.5:
            return ['L'] + random_path(tree[2], depth+1, max_depth)
        else:
            return ['R'] + random_path(tree[3], depth+1, max_depth)
            
    def get_subtree_index(tree, path):
        if not path:
            return tree
        if path[0] == 'L':
            return get_subtree_index(tree[2], path[1:])
        else:
            return get_subtree_index(tree[3], path[1:])
            
    def set_subtree(tree, path, new_subtree):
        if not path:
            return new_subtree
        if path[0] == 'L':
            new_left = set_subtree(tree[2], path[1:], new_subtree)
            return (tree[0], None, new_left, tree[3])
        else:
            new_right = set_subtree(tree[3], path[1:], new_subtree)
            return (tree[0], None, tree[2], new_right)

    if random.random() < 0.5:
        # Point mutation: Change a node
        path = random_path(tree)
        node = get_subtree_index(tree, path)
        
        if node[0] in ['var', 'const']:
            # Change to a different terminal or an operator?
            # If terminal, change to another terminal or a simple operator with random children?
            # Let's just change the value if it's a const/var, or replace with a new random subtree
            if node[0] == 'const':
                new_node = ('const', random.choice(constants), None, None)
            else:
                # Change var to another var or const
                if random.random() < 0.5:
                    new_node = ('var', random.randint(0, d-1), None, None)
                else:
                    new_node = ('const', random.choice(constants), None, None)
        else:
            # Change operator
            new_node = (random.choice(operators), None, node[2], node[3])
            
        return set_subtree(tree, path, new_node)
    else:
        # Subtree mutation: Replace a subtree with a new random one
        path = random_path(tree)
        new_subtree = create_tree(max_depth=4)
        return set_subtree(tree, path, new_subtree)

def shrink(tree):
    """Simplify tree by replacing subtrees with their equivalent if possible, or just pruning."""
    # Simple shrink: if a node is 'x + 0' or 'x * 1', replace with 'x'
    # This is complex to implement generally. 
    # Instead, let's just return the tree. The mutation/crossover will handle diversity.
    return tree

# EVOLVE-BLOCK-END
