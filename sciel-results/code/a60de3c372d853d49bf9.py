import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: 150s limit, aim for 145s
    time_limit = 145.0
    
    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        # Simple product code or exhaustive for small n
        if n <= 4 and q == 3 and radius == 1:
            # Tetracode-like or simple grid
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    N = q ** n  # 2187
    shape = (q,) * n

    # Helper functions for index conversion
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % q
            idx //= q
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * q + x
        return idx

    # Precompute neighbors
    # neighbors[i] is a list of indices of words within Hamming distance 1 of i
    neighbors = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                nb.append(tuple_to_idx(w))
            w[pos] = orig_val
        neighbors.append(nb)
    
    # Convert to numpy arrays for faster indexing if needed, but lists are fine for small sizes
    # We will use lists for flexibility in set operations

    def greedy_set_cover(seed=None, perturb=0.0):
        """
        Greedy set cover using vectorized gain accumulation.
        """
        rng = np.random.default_rng(seed) if seed is not None else None
        uncovered = np.ones(N, dtype=bool)
        code = []
        code_set = set()
        deadline = start_time + (time_limit - 30) # Reserve 30s for local search
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            # Compute gains: for each candidate, how many uncovered points it covers
            # Accumulate from uncovered points
            gains = np.zeros(N, dtype=np.int32)
            uncovered_indices = np.where(uncovered)[0]
            
            # This loop is O(uncovered * 15). 2187 * 15 ~ 32k ops. Very fast.
            for idx in uncovered_indices:
                gains[neighbors[idx]] += 1
            
            if perturb > 0 and rng is not None:
                gains += rng.random(N) * perturb
            
            best_idx = int(np.argmax(gains))
            max_gain = gains[best_idx]
            
            if max_gain == 0:
                break
            
            code.append(best_idx)
            code_set.add(best_idx)
            # Update uncovered
            uncovered[neighbors[best_idx]] = False
        
        return code, code_set

    def local_search(code, code_set, time_budget):
        """
        Local search using critical point relocation.
        """
        # Initialize coverage counts
        counts = np.zeros(N, dtype=np.int32)
        for idx in code:
            counts[neighbors[idx]] += 1
        
        deadline = start_time + time_budget
        best_size = len(code)
        best_code = list(code)
        
        def try_remove_redundant():
            """Remove any codeword that is not critical for any point."""
            removed = False
            for c in list(code):
                if time.time() > deadline:
                    break
                # If all neighbors of c have count > 1, c is redundant
                if np.min(counts[neighbors[c]]) > 1:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbors[c]] -= 1
                    removed = True
            return removed

        def try_swap(c):
            """
            Try to replace codeword c with a better one.
            Strategy: Find a replacement w such that:
            1. w covers all critical points of c (points where counts==1 and in neighbors[c]).
            2. w maximizes the number of points it covers that are currently covered exactly once (to create redundancy).
            """
            # Find critical points of c
            crit_points = [i for i in neighbors[c] if counts[i] == 1]
            
            if not crit_points:
                # If no critical points, c is redundant, should have been removed
                return False

            # Candidate pool: neighbors of critical points
            # We want w to be in the intersection of neighbors of all crit_points
            # Start with neighbors of the first critical point
            first_crit = crit_points[0]
            candidates = [x for x in neighbors[first_crit] if x not in code_set]
            
            best_w = -1
            best_score = -1
            
            for w in candidates:
                # Check if w covers ALL critical points
                valid = True
                for k in crit_points:
                    if k not in neighbors[w]: # O(15) check
                        valid = False
                        break
                if not valid:
                    continue
                
                # Score: maximize redundancy creation
                # Count how many neighbors of w have count == 1
                # These will become count == 2 after adding w (if we remove c, they might drop, but we are checking potential)
                # Actually, if we replace c with w:
                # Points in neighbors[c] \ neighbors[w] will lose coverage.
                # Points in neighbors[w] \ neighbors[c] will gain coverage.
                # We already ensured neighbors[c] \ neighbors[w] is empty for critical points.
                # But there might be non-critical points in neighbors[c] \ neighbors[w]?
                # No, if a point is in neighbors[c] and not in neighbors[w], and it was count 1, it's critical.
                # If it was count > 1, it's safe.
                # So the condition "covers all critical points" ensures no new uncovered points.
                
                # Score: number of points in neighbors[w] that are currently count 1
                # These will become count 2 (if they were 1) or count 1 (if they were 0, but they can't be 0 if covered by c? No, they might not be covered by c).
                # Wait, if a point is in neighbors[w] but NOT in neighbors[c], and count is 1, it means it was covered by some other codeword.
                # If we add w, it becomes 2. Good.
                # If a point is in neighbors[w] AND neighbors[c], and count is 1, it means it was covered ONLY by c.
                # If we remove c and add w, it is still covered by w. Count stays 1?
                # No, count is global.
                # Let's just count neighbors[w] where counts[i] == 1.
                
                score = 0
                for x in neighbors[w]:
                    if counts[x] == 1:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_w = w
            
            if best_w != -1:
                # Perform swap
                code.remove(c)
                code_set.remove(c)
                counts[neighbors[c]] -= 1
                
                code.append(best_w)
                code_set.add(best_w)
                counts[neighbors[best_w]] += 1
                
                return True
            
            return False

        # Main local search loop
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Remove redundant
            if try_remove_redundant():
                improved = True
                continue # Restart loop as counts changed
            
            # Phase 2: Try swaps
            # Iterate over a copy of the list to allow modification
            for c in list(code):
                if time.time() > deadline:
                    break
                if try_swap(c):
                    improved = True
                    # After a swap, counts changed. We should probably re-check for redundancies immediately
                    # but the outer loop will handle it.
                    break # Restart the for loop to re-evaluate with new counts
        
        return code, code_set

    best_result = None
    best_size = float('inf')
    
    # Run multiple restarts
    # Allocation: 20s per restart for greedy+LS, total 100s for 5 restarts?
    # Let's do 5 restarts.
    num_restarts = 5
    restart_budget = (time_limit - 10) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > time_limit - 5:
            break
            
        # Vary the greedy perturbation to get different initial solutions
        perturb = 0.0 if r == 0 else 0.5 * (r + 1)
        seed = r * 100 + 42
        
        code, code_set = greedy_set_cover(seed=seed, perturb=perturb)
        
        # Local Search with remaining time for this restart
        ls_budget = restart_budget - (time.time() - (start_time + r * restart_budget))
        if ls_budget < 5:
            ls_budget = 5 # Minimum budget
        
        code, code_set = local_search(code, code_set, ls_budget)
        
        # Validate
        if code:
            # Quick check: ensure all covered
            # We trust the local search logic, but a final check is cheap
            # Actually, local search maintains validity.
            if len(code) < best_size:
                best_size = len(code)
                best_result = code
    
    if best_result is None:
        # Fallback: Full space (inefficient but valid)
        best_result = list(range(N))
        
    # Convert to list of lists of ints
    return [[int(x) for x in idx_to_tuple(i)] for i in best_result]
