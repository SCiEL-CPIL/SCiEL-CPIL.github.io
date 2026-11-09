import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Leave 40s buffer for 330s limit
    
    # 1. Base Result: Paley Graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    base_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    best_n = q
    best_edges = base_edges
    
    def is_5_5_free(n, S):
        # S is a set of offsets
        # Create adjacency matrix as a set of frozensets for O(1) lookup
        # Or better, a boolean matrix for even faster access?
        # n is small (<= 46). A set of tuples is fine.
        edges = set()
        for i in range(n):
            for s in S:
                j = (i + s) % n
                if i < j:
                    edges.add((i, j))
                elif j < i:
                    edges.add((j, i))
        
        vertices = list(range(n))
        
        # Check for K5
        for combo in combinations(vertices, 5):
            is_clique = True
            for i in range(5):
                for j in range(i+1, 5):
                    u, v = combo[i], combo[j]
                    if u > v: u, v = v, u
                    if (u, v) not in edges:
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique:
                return False
            
            # Check if independent set
            is_indep = True
            for i in range(5):
                for j in range(i+1, 5):
                    u, v = combo[i], combo[j]
                    if u > v: u, v = v, u
                    if (u, v) in edges:
                        is_indep = False
                        break
                if not is_indep:
                    break
            if is_indep:
                return False
                
        return True

    def generate_edges(n, S):
        edges = []
        for i in range(n):
            for s in S:
                j = (i + s) % n
                if i < j:
                    edges.append((i, j))
        return edges

    # Search parameters
    # Known best is 42 (Exoo). We know R(5,5) <= 46, so we can try up to 46.
    # However, random search for 43-46 is extremely unlikely to find a solution quickly.
    # But let's try 42 first, then 43, 44, 45, 46 if time permits.
    # The seed only goes up to 42.
    
    # We will search n in descending order from 46 down to 38.
    # If we find a valid graph for n, we return it immediately (since we go down, first found is max).
    
    max_n_to_try = 46
    
    for n in range(max_n_to_try, best_n, -1):
        if time.time() > deadline - 5:
            break
            
        # Number of random trials per n
        # For larger n, probability of success is lower, but checking is slower.
        # Let's do a fixed number of trials, or time-based.
        # Let's do 1000 trials per n, but break if time runs out.
        trials = 1000
        
        for _ in range(trials):
            if time.time() > deadline - 2:
                break
                
            # Generate random symmetric set S
            # Reps: {1, ..., n-1} such that i < n-i.
            # If n is even, n/2 is a self-inverse.
            
            reps = []
            for i in range(1, n):
                if i < n - i:
                    reps.append(i)
            
            # Optional self-inverse
            self_inv = None
            if n % 2 == 0:
                self_inv = n // 2
            
            # Target size of S is roughly n/2 (50% density)
            # |S| = 2 * |X| + (1 if self_inv included)
            # We want 2*|X| + delta ~= n/2
            # |X| ~= n/4
            
            k = n // 4
            # Add variance
            k = max(1, k + random.randint(-3, 3))
            
            # Shuffle reps and pick k
            random.shuffle(reps)
            X = set(reps[:k])
            
            S = set()
            for x in X:
                S.add(x)
                S.add(n - x)
            
            if self_inv is not None and random.random() < 0.5:
                S.add(self_inv)
            
            if not S:
                continue
                
            # Check validity
            if is_5_5_free(n, S):
                # Found a valid graph
                best_n = n
                best_edges = generate_edges(n, S)
                return best_n, best_edges

    # If no better graph found, return seed
    return best_n, best_edges
