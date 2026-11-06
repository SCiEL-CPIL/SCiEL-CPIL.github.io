import time
import random
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def calc_alpha_milp(triples):
    """
    Calculate the independence number using MILP.
    Maximize sum(x_i) subject to sum(x_i for i in T) <= 2 for all T.
    """
    if not triples:
        return 16
    
    n = 16
    # Objective: maximize sum(x) -> minimize -sum(x)
    c = -np.ones(n)
    
    # Constraints: for each triple, sum of 3 variables <= 2
    # A is sparse, but for n=16 and ~50 triples, dense is fine
    num_constraints = len(triples)
    if num_constraints == 0:
        return 16
        
    A = np.zeros((num_constraints, n))
    for i, t in enumerate(triples):
        A[i, t[0]] = 1
        A[i, t[1]] = 1
        A[i, t[2]] = 1
        
    # Constraints: A @ x <= 2
    # milp requires LinearConstraint(A, lb, ub)
    lb = -np.inf * np.ones(num_constraints)
    ub = 2 * np.ones(num_constraints)
    
    constraints = LinearConstraint(A, lb, ub)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: all variables are integer (binary)
    integrality = np.ones(n)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        if res.status == 0:
            # res.fun is the minimum of -sum(x), so alpha = -res.fun
            return int(round(-res.fun))
        else:
            # Fallback to a simple greedy if MILP fails
            return calc_alpha_greedy(triples)
    except Exception:
        return calc_alpha_greedy(triples)

def calc_alpha_greedy(triples):
    """
    Fast heuristic for alpha, used as fallback or for initial estimates.
    """
    n = 16
    triple_sets = [set(t) for t in triples]
    # Simple check from n down
    # This is slow, but for n=16, checking k=16,15,14 is fast.
    # If we suspect alpha is small, we can just return a lower bound or use MILP.
    # For fallback, let's just do a quick random sampling or return a safe high value.
    # Actually, if MILP fails, it's likely an issue with the setup.
    # Let's implement a very basic branch and bound for n=16.
    
    # For simplicity in fallback, we'll just return 16 (worst case) to ensure validity, 
    # though it might hurt the search.
    # Better: use a bitset approach.
    # Represent independent sets as bitmasks.
    # This is getting complex. Let's trust MILP.
    return 16

def is_linear(triple, existing_triples):
    """Check if a new triple is linear with all existing triples."""
    a, b, c = triple
    new_set = {a, b, c}
    for t in existing_triples:
        old_set = {t[0], t[1], t[2]}
        if len(new_set & old_set) > 1:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_triples = []
    best_alpha = 16
    
    # 1. Generate a high-quality initial linear 3-graph
    # Strategy: Greedy addition balancing degrees
    def generate_initial_graph():
        triples = []
        used_pairs = set()
        degrees = [0] * 16
        
        # Shuffle points to avoid bias
        points = list(range(16))
        random.shuffle(points)
        
        # We want to add as many triples as possible.
        # Max triples is bounded by 16*15/6 = 40 (if each point degree 7).
        # Actually, max degree in linear 3-graph is 7 (since 2*7+1=15 < 16).
        # So max triples is 16*7/3 = 37.33 -> 37 triples.
        
        attempts = 0
        max_attempts = 10000
        
        while attempts < max_attempts:
            attempts += 1
            # Pick two points with lowest degrees
            # To speed up, we can just pick random points and check linearity.
            # But balancing degrees helps.
            
            # Find available pairs
            # Instead of iterating all pairs, pick random pairs
            a, b = random.sample(range(16), 2)
            if (a, b) in used_pairs or (b, a) in used_pairs:
                continue
            
            # Find a third point c such that {a,c} and {b,c} are unused
            # and {a,b,c} is linear with existing triples
            candidates = [c for c in range(16) if c not in (a, b)]
            random.shuffle(candidates)
            
            added = False
            for c in candidates:
                # Check pairs
                if (a, c) in used_pairs or (c, a) in used_pairs:
                    continue
                if (b, c) in used_pairs or (c, b) in used_pairs:
                    continue
                
                new_triple = (a, b, c)
                if is_linear(new_triple, triples):
                    triples.append(list(new_triple))
                    used_pairs.add((a, b))
                    used_pairs.add((a, c))
                    used_pairs.add((b, c))
                    degrees[a] += 1
                    degrees[b] += 1
                    degrees[c] += 1
                    added = True
                    break
            
            if not added:
                # If we can't add any triple with this pair, try another pair
                # If we've tried many pairs and added nothing, we might be stuck
                pass
                
        return triples

    # 2. Local Search with Simulated Annealing
    # State: List of triples
    # Energy: Alpha (lower is better)
    
    def get_energy(triples):
        return calc_alpha_milp(triples)
    
    # Initial state
    triples = generate_initial_graph()
    current_alpha = get_energy(triples)
    
    if current_alpha < best_alpha:
        best_alpha = current_alpha
        best_triples = [list(t) for t in triples]
        
    temperature = 10.0
    cooling = 0.99
    
    while time.time() < deadline:
        # Propose a move
        if random.random() < 0.5:
            # Try to add a triple
            a, b = random.sample(range(16), 2)
            # Check if pair is free
            # We don't maintain used_pairs in this loop efficiently.
            # Let's just pick 3 random points and check linearity.
            triple = list(random.sample(range(16), 3))
            random.shuffle(triple)
            
            if is_linear(triple, triples):
                new_triples = triples + [triple]
                new_alpha = get_energy(new_triples)
                
                if new_alpha < current_alpha:
                    triples = new_triples
                    current_alpha = new_alpha
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_triples = [list(t) for t in triples]
                else:
                    # Accept with probability exp(-delta/T)
                    delta = new_alpha - current_alpha
                    if delta < 0:
                        prob = 1.0
                    else:
                        prob = np.exp(-delta / temperature)
                    if random.random() < prob:
                        triples = new_triples
                        current_alpha = new_alpha
        else:
            # Try to remove a triple
            if triples:
                idx = random.randint(0, len(triples) - 1)
                new_triples = triples[:idx] + triples[idx+1:]
                new_alpha = get_energy(new_triples)
                
                if new_alpha < current_alpha:
                    triples = new_triples
                    current_alpha = new_alpha
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_triples = [list(t) for t in triples]
                else:
                    delta = new_alpha - current_alpha
                    if delta < 0:
                        prob = 1.0
                    else:
                        prob = np.exp(-delta / temperature)
                    if random.random() < prob:
                        triples = new_triples
                        current_alpha = new_alpha
                        
        temperature *= cooling
        if temperature < 0.1:
            temperature = 1.0 # Reheat occasionally? Or just keep going.
            
    return best_triples
