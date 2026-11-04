import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    D = 11
    
    # We focus on Weight 6 vectors.
    # Norm^2 = 6.
    # Condition for compatibility of all sign vectors:
    # Supports S, T must satisfy |S intersection T| <= 3.
    # Because max dot product is |S intersection T|.
    # If |S intersection T| <= 3, max dot product <= 3.
    # Required: u . v <= (Norm^2) / 2 = 6 / 2 = 3.
    # So |S intersection T| <= 3 is sufficient for all sign pairs.
    
    # Generate all 6-subsets
    supports = list(combinations(range(D), 6))
    N_sup = len(supports)
    
    # Convert to bitmasks for fast intersection
    sup_masks = []
    for s in supports:
        mask = 0
        for idx in s:
            mask |= (1 << idx)
        sup_masks.append(mask)
        
    # Precompute compatibility
    # compat[i] is a set of j such that supports[i] and supports[j] are compatible
    # Compatible if intersection size <= 3
    compat = [set() for _ in range(N_sup)]
    
    for i in range(N_sup):
        if time.time() > deadline - 30.0:
            break
        for j in range(i + 1, N_sup):
            inter_count = bin(sup_masks[i] & sup_masks[j]).count('1')
            if inter_count <= 3:
                compat[i].add(j)
                compat[j].add(i)
                
    # Find a large independent set in the incompatibility graph
    # Which is a clique in the compatibility graph.
    # We use a greedy heuristic with random restarts.
    
    best_clique = set()
    best_size = 0
    
    import random
    rng = random.Random(42)
    
    max_restarts = 1000
    for r in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Greedy selection
        order = list(range(N_sup))
        rng.shuffle(order)
        
        current_clique = set()
        
        for idx in order:
            # Check if idx is compatible with all in current_clique
            # Since current_clique is a clique, all members are mutually compatible.
            # We just need idx to be compatible with all members.
            compatible = True
            for member in current_clique:
                if member not in compat[idx]:
                    compatible = False
                    break
            
            if compatible:
                current_clique.add(idx)
                
        if len(current_clique) > best_size:
            best_size = len(current_clique)
            best_clique = set(current_clique)
            
            # If we found a very good one, we can stop early?
            # Or continue to try to improve.
            
    # If the greedy didn't find anything (unlikely), fallback to a single support
    if not best_clique:
        best_clique = {0}
        
    # Generate vectors
    vectors = []
    for i in best_clique:
        sup = supports[i]
        # Generate all 2^6 sign combinations
        for signs in product([1, -1], repeat=6):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)
            
    if not vectors:
        # Fallback
        v = np.zeros(D, dtype=np.int8)
        v[0:6] = 1
        vectors.append(v)
        
    return np.array(vectors, dtype=np.int8)

# Helper for product
from itertools import product
