import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Target n
    n = 34
    
    # Generate all triples
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    
    # Map triple to index
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute for each triple, which other triples form a K4 (red) or K5 (blue)
    # For a triple t=(a,b,c), a red K4 is formed if we add a vertex d such that
    # (a,b,d), (a,c,d), (b,c,d) are also red.
    # It is more efficient to store the neighbors for each triple.
    
    # Let's store for each triple index i, the list of other triple indices that share 2 vertices with it.
    # Actually, for K4 detection, we need to check if adding a 4th vertex creates 3 new red edges.
    # Let's precompute for each pair of vertices (a,b), the list of vertices c such that (a,b,c) is a triple.
    
    # This might be overkill. Let's just iterate.
    # n=34, num_triples ~ 6000.
    # Checking all K4s: C(34,4) = 46376.
    # Checking all K5s: C(34,5) = 278256.
    # Total checks per evaluation: ~324k.
    # In Python, 324k checks per iteration might be slow.
    # 110 seconds / 324k checks ~ 3.4e-4 s/check.
    # Python can do ~1e6 simple ops/sec. So 324k checks might take 0.1-0.3s.
    # So we can do ~300-1000 iterations. Might not be enough.
    
    # We need a faster cost function.
    # Let's maintain counts.
    # For each vertex subset of size 4, count red edges. If count==4, it's a violation.
    # For each vertex subset of size 5, count blue edges. If count==10, it's a violation.
    
    # Let's precompute the list of all K4s and K5s.
    k4s = list(combinations(range(n), 4))
    k5s = list(combinations(range(n), 5))
    
    # Map each K4 to its 4 constituent triple indices
    k4_triples = []
    for k4 in k4s:
        t_indices = []
        for i in range(4):
            # The 4 triples in a K4 are formed by omitting one vertex
            # e.g., for (a,b,c,d), triples are (a,b,c), (a,b,d), (a,c,d), (b,c,d)
            # These are all combinations of 3 from the 4 vertices.
            for t in combinations(k4, 3):
                t_indices.append(triple_to_idx[t])
        k4_triples.append(t_indices)
        
    k5_triples = []
    for k5 in k5s:
        t_indices = []
        for t in combinations(k5, 3):
            t_indices.append(triple_to_idx[t])
        k5_triples.append(t_indices)
        
    # Initialize colors
    # Try a few initial strategies
    best_cost = float('inf')
    best_colors = None
    
    # Strategy 1: Random
    random.seed(42)
    colors = np.random.randint(0, 2, num_triples).astype(np.int8)
    
    # Strategy 2: Algebraic (Sum mod 2)
    # colors2 = np.array([1 if (a+b+c)%2==0 else 0 for a,b,c in triples], dtype=np.int8)
    
    # Let's run local search on the random start
    # We can also try multiple random starts if time permits.
    
    # Function to calculate cost
    # cost = num_red_K4 + num_blue_K5
    
    def calc_cost(colors):
        red_k4s = 0
        blue_k5s = 0
        
        # Check K4s
        # For each K4, check if all 4 triples are red (0)
        # Using numpy might be faster if we batch, but lists are okay.
        for t_indices in k4_triples:
            # Check if all are 0
            # sum of colors in these indices should be 0
            if np.sum(colors[t_indices]) == 0:
                red_k4s += 1
                
        # Check K5s
        # For each K5, check if all 10 triples are blue (1)
        # Sum of (1 - colors) should be 10, or sum of colors should be 0?
        # Blue is 1. So all 10 should be 1.
        # Sum of colors should be 10.
        for t_indices in k5_triples:
            if np.sum(colors[t_indices]) == 10:
                blue_k5s += 1
                
        return red_k4s + blue_k5s

    current_cost = calc_cost(colors)
    if current_cost == 0:
        return {'n': n, 'colors': colors.tolist()}
        
    best_cost = current_cost
    best_colors = colors.copy()
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Pick random triple
        idx = random.randint(0, num_triples - 1)
        
        # Flip color
        old_color = colors[idx]
        new_color = 1 - old_color
        colors[idx] = new_color
        
        # Calculate delta cost
        # This is expensive to do from scratch.
        # We can update incrementally, but it's complex.
        # Given n=34, full recalculation is ~0.2s?
        # If so, we can only do ~500 steps.
        # Let's optimize calc_cost using numpy vectorization if possible.
        
        # Actually, let's just use the full calc_cost but try to make it faster.
        # Or, we can precompute the incidence matrix?
        # Incidence matrix M (num_hypergraphs x num_triples)
        # M[k4_idx, t_idx] = 1 if t_idx is in k4_idx else 0
        # Then red_K4_count = sum( (M_k4 @ colors == 0) )
        # This is a matrix-vector multiplication.
        
        # Let's build the matrices.
        # This is a one-time cost.
        
        break # Placeholder, will implement matrix version below
        
    # Re-implementation with Matrix Incidence for speed
    
    # Build M4 (num_k4s x num_triples)
    M4 = np.zeros((len(k4_triples), num_triples), dtype=np.int8)
    for i, t_indices in enumerate(k4_triples):
        M4[i, t_indices] = 1
        
    # Build M5 (num_k5s x num_triples)
    M5 = np.zeros((len(k5_triples), num_triples), dtype=np.int8)
    for i, t_indices in enumerate(k5_triples):
        M5[i, t_indices] = 1
        
    def calc_cost_fast(colors):
        # Red K4: sum of colors in each row of M4 is 0
        sums4 = M4.dot(colors)
        red_k4s = np.sum(sums4 == 0)
        
        # Blue K5: sum of colors in each row of M5 is 10 (since blue=1)
        sums5 = M5.dot(colors)
        blue_k5s = np.sum(sums5 == 10)
        
        return red_k4s + blue_k5s

    current_cost = calc_cost_fast(colors)
    if current_cost == 0:
        return {'n': n, 'colors': colors.tolist()}
        
    best_cost = current_cost
    best_colors = colors.copy()
    
    T = 1.0
    cooling_rate = 0.9995
    
    while time.time() < deadline:
        idx = random.randint(0, num_triples - 1)
        
        old_color = colors[idx]
        new_color = 1 - old_color
        colors[idx] = new_color
        
        new_cost = calc_cost_fast(colors)
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
                if best_cost == 0:
                    break
        else:
            colors[idx] = old_color # Revert
            
        T *= cooling_rate
        if T < 0.01:
            T = 0.01

    if best_cost == 0:
        return {'n': n, 'colors': best_colors.tolist()}
    
    # If we didn't find a perfect coloring for n=34, try n=33?
    # Or just return the best found? The task says "Maximize n".
    # If we can't make n=34 valid, we should try to make n=33 valid.
    # But the evaluator scores based on the raw value (n).
    # If we return an invalid coloring for n=34, the score is 0.
    # We MUST return a valid coloring.
    
    # Fallback: Try smaller n if 34 fails.
    # However, 34 is the known bound. It is very likely solvable.
    # If we run out of time, we might have a non-zero cost.
    # In that case, we should reduce n and re-solve?
    # Or just return the best valid n we found?
    # We didn't track validity for smaller n.
    
    # Let's assume we can find 0 cost for 34.
    # If not, we might return an invalid solution which scores 0.
    # To be safe, let's try to ensure we return a valid one.
    # If best_cost > 0, we can try to find a valid coloring for n=33.
    
    if best_cost > 0:
        # Try n=33
        n = 33
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        
        k4_triples = []
        for k4 in k4s:
            t_indices = []
            for t in combinations(k4, 3):
                t_indices.append(triple_to_idx[t])
            k4_triples.append(t_indices)
            
        k5_triples = []
        for k5 in k5s:
            t_indices = []
            for t in combinations(k5, 3):
                t_indices.append(triple_to_idx[t])
            k5_triples.append(t_indices)
            
        M4 = np.zeros((len(k4_triples), num_triples), dtype=np.int8)
        for i, t_indices in enumerate(k4_triples):
            M4[i, t_indices] = 1
            
        M5 = np.zeros((len(k5_triples), num_triples), dtype=np.int8)
        for i, t_indices in enumerate(k5_triples):
            M5[i, t_indices] = 1
            
        def calc_cost_fast_33(colors):
            sums4 = M4.dot(colors)
            red_k4s = np.sum(sums4 == 0)
            sums5 = M5.dot(colors)
            blue_k5s = np.sum(sums5 == 10)
            return red_k4s + blue_k5s

        random.seed(42)
        colors33 = np.random.randint(0, 2, num_triples).astype(np.int8)
        current_cost33 = calc_cost_fast_33(colors33)
        best_cost33 = current_cost33
        best_colors33 = colors33.copy()
        
        T = 1.0
        while time.time() < deadline:
            idx = random.randint(0, num_triples - 1)
            old_color = colors33[idx]
            new_color = 1 - old_color
            colors33[idx] = new_color
            
            new_cost33 = calc_cost_fast_33(colors33)
            delta = new_cost33 - current_cost33
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_cost33 = new_cost33
                if current_cost33 < best_cost33:
                    best_cost33 = current_cost33
                    best_colors33 = colors33.copy()
                    if best_cost33 == 0:
                        break
            else:
                colors33[idx] = old_color
            T *= cooling_rate
            if T < 0.01:
                T = 0.01

        if best_cost33 == 0:
            return {'n': 33, 'colors': best_colors33.tolist()}
            
    # If both fail, return the seed solution (n=9) to at least get a valid score, 
    # though it's worse. But the task implies maximizing n.
    # Actually, if we return invalid, score is 0.
    # The seed is valid for n=9.
    # Let's return the seed if we fail.
    
    # Seed solution
    n_seed = 9
    return {'n': n_seed, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_seed), 3)]}

# EVOLVE-BLOCK-END
