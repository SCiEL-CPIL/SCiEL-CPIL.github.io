"""Symbolic Regression via Genetic Programming."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import operator
import math

# --- Symbolic Regression Engine ---

# Define the operator set
# Terminal nodes: x0, x1, x2 (inputs), constants (randomized during evolution)
# Function nodes: +, -, *, /, sin, cos, exp, log, abs, sqrt, pow

class Node:
    def __init__(self, op, children=None, value=None):
        self.op = op
        self.children = children if children else []
        self.value = value

    def evaluate(self, x):
        if self.op == 'const':
            return self.value
        if self.op in ['x0', 'x1', 'x2']:
            return x[self.op[1]]
        
        if self.op == 'add':
            return self.children[0].evaluate(x) + self.children[1].evaluate(x)
        if self.op == 'sub':
            return self.children[0].evaluate(x) - self.children[1].evaluate(x)
        if self.op == 'mul':
            return self.children[0].evaluate(x) * self.children[1].evaluate(x)
        if self.op == 'div':
            a = self.children[0].evaluate(x)
            b = self.children[1].evaluate(x)
            if np.abs(b) < 1e-8:
                return 1e8 * np.sign(a) if a != 0 else 0.0
            return a / b
        if self.op == 'sin':
            return np.sin(self.children[0].evaluate(x))
        if self.op == 'cos':
            return np.cos(self.children[0].evaluate(x))
        if self.op == 'exp':
            val = self.children[0].evaluate(x)
            return np.exp(np.clip(val, -50, 50))
        if self.op == 'log':
            val = self.children[0].evaluate(x)
            return np.log(np.abs(val) + 1e-8)
        if self.op == 'abs':
            return np.abs(self.children[0].evaluate(x))
        if self.op == 'sqrt':
            val = self.children[0].evaluate(x)
            return np.sqrt(np.abs(val))
        if self.op == 'pow':
            a = self.children[0].evaluate(x)
            b = self.children[1].evaluate(x)
            # Safety for pow
            b_safe = np.clip(b, -2, 2)
            a_safe = np.clip(a, -10, 10)
            with np.errstate(invalid='ignore'):
                res = np.power(a_safe, b_safe)
            res[~np.isfinite(res)] = 1e8
            return res
        return 0.0

    def depth(self):
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def copy(self):
        new_children = [c.copy() for c in self.children]
        return Node(self.op, new_children, self.value)

    def __str__(self):
        if self.op == 'const':
            return f"{self.value:.3f}"
        if self.op in ['x0', 'x1', 'x2']:
            return self.op
        if len(self.children) == 1:
            return f"{self.op}({self.children[0]})"
        if len(self.children) == 2:
            return f"({self.children[0]} {self.op} {self.children[1]})"
        return self.op

# Operators
UNARY_OPS = ['sin', 'cos', 'exp', 'log', 'abs', 'sqrt']
BINARY_OPS = ['add', 'sub', 'mul', 'div', 'pow']
TERMINALS = ['x0', 'x1', 'x2']

def random_tree(depth, max_depth, d):
    """Generate a random tree up to max_depth."""
    if depth == max_depth:
        # Must be a terminal
        term = random.choice(TERMINALS[:d])
        return Node(term)
    
    # Probabilities to encourage simpler trees
    r = random.random()
    if r < 0.3:
        # Terminal
        term = random.choice(TERMINALS[:d])
        return Node(term)
    elif r < 0.4:
        # Constant
        val = random.uniform(-5, 5)
        return Node('const', value=val)
    elif r < 0.7:
        # Unary
        op = random.choice(UNARY_OPS)
        child = random_tree(depth + 1, max_depth, d)
        return Node(op, [child])
    else:
        # Binary
        op = random.choice(BINARY_OPS)
        c1 = random_tree(depth + 1, max_depth, d)
        c2 = random_tree(depth + 1, max_depth, d)
        return Node(op, [c1, c2])

def mutate(tree, d):
    """Randomly mutate a tree."""
    new_tree = tree.copy()
    
    # Pick a random node to mutate
    # Flatten tree for easy access
    nodes = []
    def collect(node, parent=None):
        nodes.append((node, parent))
        for c in node.children:
            collect(c, node)
    collect(new_tree)
    
    if not nodes:
        return new_tree
        
    idx = random.randint(0, len(nodes) - 1)
    node, parent = nodes[idx]
    
    r = random.random()
    if r < 0.2:
        # Replace with new random subtree
        new_subtree = random_tree(0, 3, d)
        if parent is None:
            return new_subtree
        else:
            # Replace in parent
            for i, c in enumerate(parent.children):
                if c is node:
                    parent.children[i] = new_subtree
            return new_tree
    elif r < 0.4 and node.op == 'const':
        # Change constant value
        node.value = random.uniform(-5, 5)
        return new_tree
    elif r < 0.6:
        # Change operator
        if node.op in UNARY_OPS:
            node.op = random.choice(UNARY_OPS)
        elif node.op in BINARY_OPS:
            node.op = random.choice(BINARY_OPS)
        return new_tree
    else:
        # Swap children if binary
        if len(node.children) == 2:
            node.children[0], node.children[1] = node.children[1], node.children[0]
        return new_tree

def crossover(t1, t2, d):
    """Crossover between two trees."""
    t1 = t1.copy()
    t2 = t2.copy()
    
    # Flatten
    nodes1 = []
    def collect1(node):
        nodes1.append(node)
        for c in node.children:
            collect1(c)
    collect1(t1)
    
    nodes2 = []
    def collect2(node):
        nodes2.append(node)
        for c in node.children:
            collect2(c)
    collect2(t2)
    
    if not nodes1 or not nodes2:
        return t1, t2
        
    idx1 = random.randint(0, len(nodes1) - 1)
    idx2 = random.randint(0, len(nodes2) - 1)
    
    n1 = nodes1[idx1]
    n2 = nodes2[idx2]
    
    # Swap
    new_n1 = n2.copy()
    new_n2 = n1.copy()
    
    # Re-insert
    # This is tricky because we need to update references.
    # Simple approach: replace root if root was swapped, else we need a pointer update.
    # For simplicity and speed, we'll just do a "subtree replacement" crossover:
    # Replace a random subtree in t1 with a random subtree from t2.
    
    # Let's restart with a simpler crossover logic for robustness
    # We will just return copies and let mutation do the work, or implement a proper pointer swap.
    
    # Proper swap:
    def replace_node(tree, target, new_node):
        if tree is target:
            return new_node
        for i, c in enumerate(tree.children):
            tree.children[i] = replace_node(c, target, new_node)
        return tree

    t1_new = replace_node(t1, n1, new_n1)
    t2_new = replace_node(t2, n2, new_n2)
    
    return t1_new, t2_new

def evaluate_population(pop, X, y):
    """Evaluate fitness of all individuals."""
    scores = []
    for tree in pop:
        try:
            pred = np.array([tree.evaluate(x) for x in X])
            if not np.all(np.isfinite(pred)):
                scores.append(np.inf)
                continue
            mse = np.mean((pred - y) ** 2)
            scores.append(mse)
        except:
            scores.append(np.inf)
    return scores

def genetic_programming(X, y, d, time_budget):
    pop_size = 100
    max_depth = 5
    elite_size = 10
    
    # Initialize population
    pop = [random_tree(0, max_depth, d) for _ in range(pop_size)]
    scores = evaluate_population(pop, X, y)
    
    best_idx = np.argmin(scores)
    best_tree = pop[best_idx].copy()
    best_score = scores[best_idx]
    
    start_time = time.time()
    generation = 0
    
    while time.time() - start_time < time_budget:
        generation += 1
        
        # Sort by score
        sorted_indices = np.argsort(scores)
        new_pop = []
        
        # Elitism
        for i in range(elite_size):
            new_pop.append(pop[sorted_indices[i]].copy())
        
        # Fill rest with children
        while len(new_pop) < pop_size:
            r = random.random()
            if r < 0.1:
                # Mutation
                parent = pop[sorted_indices[random.randint(0, len(sorted_indices)-1)]]
                child = mutate(parent, d)
                new_pop.append(child)
            elif r < 0.4:
                # Crossover
                p1 = pop[sorted_indices[random.randint(0, len(sorted_indices)-1)]]
                p2 = pop[sorted_indices[random.randint(0, len(sorted_indices)-1)]]
                c1, c2 = crossover(p1, p2, d)
                new_pop.append(c1)
                if len(new_pop) < pop_size:
                    new_pop.append(c2)
            else:
                # Random restart
                new_pop.append(random_tree(0, max_depth, d))
        
        pop = new_pop
        scores = evaluate_population(pop, X, y)
        
        # Update best
        curr_best_idx = np.argmin(scores)
        if scores[curr_best_idx] < best_score:
            best_score = scores[curr_best_idx]
            best_tree = pop[curr_best_idx].copy()
            
            # If we get a very low error, we might be close to the solution
            # For noiseless data, MSE should approach 0.
            # If MSE < 1e-6, we can stop early or tighten search.
            if best_score < 1e-6:
                break

    return best_tree, best_score

def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = 330.0 - 15.0  # safety margin
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).ravel()
    n, d = X.shape
    
    # Normalize y for better scaling? GP usually works fine with raw scale if constants are adaptive.
    # But let's keep y as is.
    
    # Allocate time for GP
    time_budget = DEADLINE - (time.time() - t0) - 10.0  # 10s for prediction setup
    
    if time_budget < 5.0:
        # Fallback
        mean_y = np.mean(y)
        return lambda X: np.full(len(np.asarray(X)), mean_y, dtype=float)

    best_tree, best_score = genetic_programming(X, y, d, time_budget)
    
    # Fallback if GP failed completely
    if best_score == np.inf:
        mean_y = np.mean(y)
        return lambda X: np.full(len(np.asarray(X)), mean_y, dtype=float)

    def predict(Xq):
        Xq = np.asarray(Xq, dtype=float)
        preds = []
        for x in Xq:
            try:
                val = best_tree.evaluate(x)
                if not np.isfinite(val):
                    val = 0.0
                preds.append(val)
            except:
                preds.append(0.0)
        return np.array(preds, dtype=float)

    return predict
# EVOLVE-BLOCK-END
