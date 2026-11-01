import itertools
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import milp, LinearConstraint

def construct(q, n, radius):
    """
    Construct a q-ary covering code of length n and radius r.
    For the target case (q=3, n=8, r=1) uses a combination of:
    1. Precomputed structural initialization (symmetric subspaces).
    2. Local search with aggressive pruning and time limits.
    3. ILP refinement if time permits.
    4. Fallback to greedy set cover if necessary.
    """
    # Fallback for non-target parameters
    if q != 3 or n != 8 or radius != 1:
        if n >= 4 and q == 3:
            # Tetracode construction for general n
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            # Trivial upper bound: cover every point
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=8, radius=1
    N = q ** n  # 6561
    start_time = time.time()
    TIME_LIMIT = 115  # Leave buffer for overhead

    # Precompute neighbors: for each word index, list of indices within distance 1
    # We represent words as integers 0..N-1.
    # Digit at position p (0-indexed from right) is (w // (3**p)) % 3
    
    # Precompute powers of 3
    pow3 = [1] * n
    for i in range(1, n):
        pow3[i] = pow3[i-1] * 3

    # neighbors[i] = set of indices j such that dist(i, j) <= 1
    # To save memory and time, we store as a list of lists or a sparse structure.
    # Since we need fast "is this neighbor uncovered" checks, and we'll do local search,
    # a flat array of neighbor lists is good.
    
    neighbors = [[] for _ in range(N)]
    for i in range(N):
        neighbors[i].append(i)  # self
        w = i
        for p in range(n):
            p3 = pow3[p]
            digit = (w // p3) % 3
            base = w - digit * p3
            for new_d in range(3):
                if new_d == digit:
                    continue
                j = base + new_d * p3
                neighbors[i].append(j)
                
    # Convert to numpy arrays for faster processing? 
    # Lists are okay for 6561 * 17.
    
    # --- Strategy 1: Structural Initialization ---
    # We try to build a code based on the "Tetracode product" structure but optimized.
    # The tetracode T(4) has 9 words in 4 digits.
    # A product code C = T x T has 81 words in 8 digits.
    # Does T x T cover radius 1?
    # Distance in T x T is sum of distances in components.
    # Max distance in T is 4 (e.g., 0000 vs 2222? No, max dist in T is 4? 
    # T = {0000, 0112, 0221, 1022, 1101, 1210, 2011, 2120, 2202}
    # Dist(0000, 1101) = 3. Dist(0000, 2222) is not in T.
    # Max weight in T is 4? 1210 has weight 4. 2202 has weight 4.
    # So max distance in T is 4.
    # In product, max distance is 8.
    # We need radius 1. So T x T is NOT a covering code of radius 1. 
    # It leaves many holes.
    
    # Instead, let's use a greedy initialization, then refine.
    # Or, use a known good starting point.
    # The "football pool" problem solution often involves codes close to 486.
    
    # Let's try a randomized greedy initialization to get a valid cover quickly.
    uncovered = np.ones(N, dtype=bool)
    code_indices = []
    
    # Greedy Set Cover for initial feasible solution
    # To speed up, we maintain a count of how many uncovered neighbors each candidate has.
    # gain[c] = number of currently uncovered words in neighbors[c]
    
    gain = np.zeros(N, dtype=np.int32)
    for c in range(N):
        for j in neighbors[c]:
            if uncovered[j]:
                gain[c] += 1
                
    while np.any(uncovered):
        # Find candidate with max gain
        best_c = np.argmax(gain)
        best_gain = gain[best_c]
        if best_gain == 0:
            break # Should not happen
        
        code_indices.append(best_c)
        
        # Update uncovered and gains
        for j in neighbors[best_c]:
            if uncovered[j]:
                uncovered[j] = False
                # This word j is now covered.
                # Any codeword c' that has j in its neighborhood had its gain reduced by 1.
                # We need to update gain[c'] for all c' such that j in neighbors[c'].
                # This is the reverse neighbor lookup.
                # Precompute reverse neighbors? 
                # rev_neighbors[j] = list of c such that j in neighbors[c]
                # Since the relation is symmetric (distance <= 1), rev_neighbors[j] == neighbors[j]
                for c_prime in neighbors[j]:
                    gain[c_prime] -= 1
                    
        # Optimization: if we are running out of time, break and return what we have?
        # No, we need a valid cover.
        if time.time() - start_time > TIME_LIMIT - 5:
            break

    # If we broke early, we might not have a full cover.
    # Check validity
    if not np.all(uncovered == False):
        # Force cover remaining by adding all remaining uncovered words?
        # That would blow up the size.
        # Better: complete the cover greedily in a tight loop, ignoring gain updates for speed?
        # Or just accept the risk if time is very tight? 
        # The ILP step below can fix it if it starts from a near-feasible point?
        # No, ILP needs a feasible start or finds one from scratch.
        # Let's just add the uncovered words themselves.
        for i in range(N):
            if uncovered[i]:
                code_indices.append(i)
                # Update coverage? We just return, so we can just build the set.
        
        # Re-evaluate uncovered status for the final check
        covered_final = np.zeros(N, dtype=bool)
        for c in code_indices:
            for j in neighbors[c]:
                covered_final[j] = True
        if not np.all(covered_final):
            # Last resort: full space
            return [list(itertools.product(range(q), repeat=n))]
            
    current_size = len(code_indices)
    
    # --- Strategy 2: Local Search (Removal) ---
    # Try to remove words from the code while maintaining coverage.
    # A word c can be removed if all its neighbors are covered by OTHER words in the code.
    # i.e., for all j in neighbors[c], there exists c' in code, c' != c, such that j in neighbors[c'].
    
    # To check this efficiently:
    # count_cover[j] = number of words in code that cover j.
    # c can be removed if min(count_cover[j] for j in neighbors[c]) >= 1 (excluding c's own contribution).
    # Actually, if we remove c, count_cover[j] decreases by 1 for j in neighbors[c].
    # So c is removable if count_cover[j] >= 2 for all j in neighbors[c].
    
    count_cover = np.zeros(N, dtype=np.int32)
    for c in code_indices:
        for j in neighbors[c]:
            count_cover[j] += 1
            
    # Convert code_indices to a set for fast membership check? 
    # We need to iterate and potentially remove.
    # Let's keep a list and a boolean array in_code.
    in_code = np.zeros(N, dtype=bool)
    for c in code_indices:
        in_code[c] = True
        
    removed = False
    # Multiple passes
    max_passes = 50
    for _ in range(max_passes):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        any_removed = False
        # Iterate through a copy of the current code indices?
        # Or iterate 0..N-1 and check in_code?
        # N=6561 is small.
        for c in range(N):
            if not in_code[c]:
                continue
            # Check if c is removable
            # Condition: for all j in neighbors[c], count_cover[j] >= 2
            # Note: c contributes 1 to count_cover[j] for all j in neighbors[c].
            # So we need count_cover[j] - 1 >= 1 => count_cover[j] >= 2.
            removable = True
            for j in neighbors[c]:
                if count_cover[j] < 2:
                    removable = False
                    break
            if removable:
                # Remove c
                in_code[c] = False
                for j in neighbors[c]:
                    count_cover[j] -= 1
                any_removed = True
        if not any_removed:
            break
        # Shuffle order in next pass? 
        # For now, linear scan.
        
    # Reconstruct code_indices from in_code
    code_indices = [i for i in range(N) if in_code[i]]
    current_size = len(code_indices)
    
    # --- Strategy 3: Swap/Exchange Local Search ---
    # Try to swap one word out and another in to reduce size? 
    # Removing one and adding one keeps size same.
    # We want to remove one without adding (handled above).
    # What about removing 2 and adding 1?
    # This is harder to check locally.
    # Let's stick to removal for now.
    
    # --- Strategy 4: ILP Refinement ---
    # If we have time left, use ILP to minimize further.
    # We can fix the current solution as an upper bound (hint).
    # Or just solve from scratch with a time limit.
    
    if time.time() - start_time < TIME_LIMIT - 30:
        # Build A matrix for ILP
        row_indices = []
        col_indices = []
        data = []
        
        # A[i, j] = 1 if j covers i (i.e., j in neighbors[i])
        for i in range(N):
            for j in neighbors[i]:
                row_indices.append(i)
                col_indices.append(j)
                data.append(1)
                
        A = csr_matrix((data, (row_indices, col_indices)), shape=(N, N))
        c_obj = np.ones(N)
        constraints = [LinearConstraint(A, 1, np.inf)]
        integrality = np.ones(N)
        bounds = [(0, 1)] * N
        
        # Use current solution as an initial guess? 
        # scipy.milp doesn't take initial guess directly in the same way, 
        # but we can provide it via the `x0` parameter? No, it doesn't.
        # We can just let it search. The fact that we have a good bound helps pruning.
        
        try:
            # Set time limit for ILP
            ilp_time_limit = max(5, int(TIME_LIMIT - (time.time() - start_time) - 5))
            
            res = milp(c_obj, constraints=constraints, integrality=integrality, bounds=bounds,
                       options={"time_limit": ilp_time_limit, "mip_rel_gap": 0.0, "mip_abs_gap": 0.0})
            
            if res.x is not None:
                # Check if the ILP solution is valid
                x = np.round(res.x).astype(int)
                # Verify
                covered_ilp = np.zeros(N, dtype=bool)
                for j in range(N):
                    if x[j] == 1:
                        for i in neighbors[j]:
                            covered_ilp[i] = True
                if np.all(covered_ilp):
                    ilp_size = np.sum(x)
                    if ilp_size < current_size:
                        code_indices = [i for i in range(N) if x[i] == 1]
                        current_size = ilp_size
        except Exception:
            pass

    # Final construction
    return [list(itertools.product(range(3), repeat=8))[i] for i in code_indices]
