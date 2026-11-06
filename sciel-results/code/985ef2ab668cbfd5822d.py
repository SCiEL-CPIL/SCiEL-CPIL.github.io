import numpy as np
import random
import time
from itertools import combinations

def count_violations(n, neighbors):
    """
    Counts the number of K5s and independent K5s (bar-K5s).
    neighbors: list of ints, where neighbors[i] has bit j set if (i,j) is an edge.
    """
    # Precompute all 5-combinations
    # This is static for a given n
    if not hasattr(count_violations, "combos"):
        count_violations.combos = list(combinations(range(n), 5))
        count_violations.n_cache = n
    
    if count_violations.n_cache != n:
        count_violations.combos = list(combinations(range(n), 5))
        count_violations.n_cache = n

    k5_count = 0
    bar_k5_count = 0
    
    # To speed up, we can check K5 first.
    # If we find a K5, we don't need to check if it's a bar-K5? 
    # Actually a set cannot be both a K5 and a bar-K5 in a simple graph.
    
    for c in count_violations.combos:
        v0, v1, v2, v3, v4 = c
        # Check K5
        # v0-v1, v0-v2, v0-v3, v0-v4
        # v1-v2, v1-v3, v1-v4
        # v2-v3, v2-v4
        # v3-v4
        
        # Using bitmasks
        m0 = neighbors[v0]
        m1 = neighbors[v1]
        m2 = neighbors[v2]
        m3 = neighbors[v3]
        m4 = neighbors[v4]
        
        if (m0 & (1 << v1)) and (m0 & (1 << v2)) and (m0 & (1 << v3)) and (m0 & (1 << v4)) and \
           (m1 & (1 << v2)) and (m1 & (1 << v3)) and (m1 & (1 << v4)) and \
           (m2 & (1 << v3)) and (m2 & (1 << v4)) and \
           (m3 & (1 << v4)):
            k5_count += 1
        else:
            # Check bar-K5 (Independent Set)
            # This means NO edges between any pair.
            if not (m0 & (1 << v1)) and not (m0 & (1 << v2)) and not (m0 & (1 << v3)) and not (m0 & (1 << v4)) and \
               not (m1 & (1 << v2)) and not (m1 & (1 << v3)) and not (m1 & (1 << v4)) and \
               not (m2 & (1 << v3)) and not (m2 & (1 << v4)) and \
               not (m3 & (1 << v4)):
                bar_k5_count += 1

    return k5_count + bar_k5_count

def construct():
    start_time = time.time()
    time_limit = 290  # Leave 10s margin
    
    # Fallback: Paley 37
    def get_paley_37():
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges

    # Try to find a better graph using Circulant SA
    for n in range(42, 37, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize random circulant graph
        # Jumps are 1 to n//2
        half_n = n // 2
        # Randomly select jumps
        # We want roughly half the possible jumps to be present?
        # Degree should be around n/2 for balanced graph
        # Number of jumps k. Degree = 2k (if n is even and no n/2 jump? or n/2 jump counts 1?)
        # For n even, jump n/2 is a single edge per vertex? No, it's a matching.
        # Let's just pick a random subset of {1, ..., n//2}
        
        best_energy = float('inf')
        best_state = None
        
        # Simulated Annealing
        T = 1.0
        T_min = 0.01
        cooling = 0.95
        steps = 200 # Limit steps to keep time low
        
        # Initial state: random
        current_state = [random.randint(0, 1) for _ in range(half_n)]
        
        def build_neighbors(state):
            neighbors = [0] * n
            for i in range(n):
                mask = 0
                for j_idx, present in enumerate(state):
                    if present:
                        jump = j_idx + 1
                        # Add i+jump and i-jump
                        v1 = (i + jump) % n
                        v2 = (i - jump) % n
                        mask |= (1 << v1)
                        if v1 != v2:
                            mask |= (1 << v2)
                neighbors[i] = mask
            return neighbors

        current_neighbors = build_neighbors(current_state)
        current_energy = count_violations(n, current_neighbors)
        
        if current_energy == 0:
            # Found a valid graph immediately? Unlikely but possible
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if current_neighbors[i] & (1 << j):
                        edges.append((i, j))
            return n, edges

        for step in range(steps):
            if time.time() - start_time > time_limit:
                break
                
            # Pick a random jump to flip
            idx = random.randint(0, half_n - 1)
            new_state = current_state.copy()
            new_state[idx] = 1 - new_state[idx]
            
            new_neighbors = build_neighbors(new_state)
            new_energy = count_violations(n, new_neighbors)
            
            delta = new_energy - current_energy
            
            if delta < 0 or (T > 0 and random.random() < pow(2.71828, -delta / T)):
                current_state = new_state
                current_neighbors = new_neighbors
                current_energy = new_energy
                
                if current_energy == 0:
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if current_neighbors[i] & (1 << j):
                                edges.append((i, j))
                    return n, edges
            
            T *= cooling
            if T < T_min:
                T = T_min

    # If no better graph found, return seed
    return get_paley_37()
