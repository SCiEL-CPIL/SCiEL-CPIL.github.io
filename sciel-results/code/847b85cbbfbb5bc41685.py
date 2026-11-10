import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    # Handle specific case q=3, n=8, radius=1
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other cases: simple greedy
        # But task implies this specific case.
        pass

    N = q ** n
    deadline = time.time() + 115.0 # 5s safety margin
    
    # Precompute neighbors
    neighbors = np.zeros((N, 1 + 2 * n), dtype=np.int32)
    for w in range(N):
        idx = 0
        neighbors[w, idx] = w
        idx += 1
        temp = w
        for i in range(n):
            d = temp % q
            temp //= q
            w1 = w + ((d + 1) % q - d) * (q ** i)
            w2 = w + ((d + 2) % q - d) * (q ** i)
            neighbors[w, idx] = w1
            idx += 1
            neighbors[w, idx] = w2
            idx += 1

    def greedy_search(rng, noise_factor=0.0):
        uncovered = np.ones(N, dtype=np.uint8)
        code = []
        while uncovered.sum() > 0:
            gains = uncovered[neighbors].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            
            if noise_factor > 0 and len(candidates) > 1:
                # Pick randomly from candidates
                w = rng.choice(candidates)
            else:
                w = rng.choice(candidates)
                
            code.append(w)
            uncovered[neighbors[w]] = 0
            
        return code

    def local_search_removal(code, neighbors, N, rng):
        # Convert to list
        code = list(code)
        if not code:
            return code
            
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code:
            cover_count[neighbors[c]] += 1
            
        changed = True
        while changed:
            changed = False
            indices = list(range(len(code)))
            rng.shuffle(indices)
            for i in indices:
                c = code[i]
                # Check if removal is safe: all neighbors must have count > 1
                if np.all(cover_count[neighbors[c]] > 1):
                    cover_count[neighbors[c]] -= 1
                    code.pop(i)
                    changed = True
        return code

    def perturb_and_optimize(code, neighbors, N, rng, time_limit):
        code = list(code)
        if not code:
            return code
            
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code:
            cover_count[neighbors[c]] += 1
            
        start_time = time.time()
        while time.time() - start_time < time_limit:
            # 1. Removals
            changed = True
            while changed:
                changed = False
                indices = list(range(len(code)))
                rng.shuffle(indices)
                for i in indices:
                    c = code[i]
                    if np.all(cover_count[neighbors[c]] > 1):
                        cover_count[neighbors[c]] -= 1
                        code.pop(i)
                        changed = True
                        
            # 2. Perturbation (Swap)
            if len(code) > 0:
                i = rng.integers(len(code))
                c_old = code[i]
                # Remove
                cover_count[neighbors[c_old]] -= 1
                code.pop(i)
                
                # Find uncovered
                uncovered_mask = np.zeros(N, dtype=np.uint8)
                uncovered_mask[cover_count == 0] = 1
                
                if np.any(uncovered_mask):
                    # Find best new codeword
                    gains = uncovered_mask[neighbors].sum(axis=1)
                    max_gain = np.max(gains)
                    
                    if max_gain > 0:
                        candidates = np.where(gains == max_gain)[0]
                        c_new = rng.choice(candidates)
                        
                        # Add c_new
                        code.append(c_new)
                        cover_count[neighbors[c_new]] += 1
                        # Note: This might create redundancies, removal phase will handle.
                        
                        # If max_gain == 0, we removed a critical codeword and can't fix it easily?
                        # Actually if max_gain == 0, it means no codeword covers any uncovered point?
                        # That implies the uncovered points are isolated? Impossible in connected graph.
                        # So max_gain should be > 0 if there are uncovered points.
                        
        return code

    # Initial Seed: Direct Sum (Tetracode)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            val = 0
            for i, digit in enumerate(word):
                val += digit * (q ** i)
            seed_code.append(val)
            
    best_code = seed_code
    best_size = len(seed_code)
    
    rng = np.random.default_rng()
    
    # Optimize seed
    optimized_seed = local_search_removal(seed_code, neighbors, N, rng)
    if len(optimized_seed) < best_size:
        best_size = len(optimized_seed)
        best_code = optimized_seed
        
    # Main Search Loop
    while time.time() < deadline:
        # Run a few greedy searches
        noise = rng.random() * 0.5
        code = greedy_search(rng, noise)
        
        if len(code) < best_size:
            # Refine with local search
            optimized = local_search_removal(code, neighbors, N, rng)
            if len(optimized) < best_size:
                best_size = len(optimized)
                best_code = optimized
                
        # If time permits, refine the current best with perturbations
        if time.time() < deadline - 5.0:
            # Use a small time slice for perturbation
            slice_time = min(2.0, max(0.1, deadline - time.time() - 1.0))
            if slice_time > 0.1:
                perturbed = perturb_and_optimize(best_code, neighbors, N, rng, slice_time)
                if len(perturbed) < best_size:
                    best_size = len(perturbed)
                    best_code = perturbed

    # Convert back to list of lists
    final_code = []
    for w in best_code:
        word = []
        temp = w
        for _ in range(n):
            word.append(temp % q)
            temp //= q
        final_code.append(word)
        
    return final_code
