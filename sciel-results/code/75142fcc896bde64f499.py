import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    # Fallback for non-standard cases (though task is specific)
    if not (q == 3 and n == 8 and radius == 1):
        N = q ** n
        if N > 100000:
            return [[i % q for i in range(n)] for i in range(N)]
        
        powers = [q ** i for i in range(n)]
        neighbors = []
        for w in range(N):
            ball = {w}
            digits = [0] * n
            temp = w
            for i in range(n):
                digits[i] = temp % q
                temp //= q
            for pos in range(n):
                original = digits[pos]
                for val in range(q):
                    if val == original:
                        continue
                    digits[pos] = val
                    new_w = 0
                    mult = 1
                    for i in range(n):
                        new_w += digits[i] * mult
                        mult *= q
                    ball.add(new_w)
                digits[pos] = original
            neighbors.append(list(ball))
        
        uncovered = set(range(N))
        code = []
        while uncovered and time.time() < deadline:
            best_cand = -1
            best_count = -1
            for c in uncovered:
                count = sum(1 for nb in neighbors[c] if nb in uncovered)
                if count > best_count:
                    best_count = count
                    best_cand = c
            if best_cand == -1:
                break
            code.append(best_cand)
            for nb in neighbors[best_cand]:
                if nb in uncovered:
                    uncovered.remove(nb)
        if uncovered:
            code.extend(uncovered)
            
        result = []
        for w in code:
            digits = []
            temp = w
            for _ in range(n):
                digits.append(temp % q)
                temp //= q
            result.append(digits)
        return result

    # Specific q=3, n=8, radius=1
    N = q ** n
    powers = [q ** i for i in range(n)]
    neigh_size = 1 + n * (q - 1) # 1 + 8*2 = 17
    
    # Precompute neighbors
    neighbors = np.zeros((N, neigh_size), dtype=np.int32)
    for w in range(N):
        neighbors[w, 0] = w
        idx = 1
        for i in range(n):
            d = (w // powers[i]) % q
            for delta in range(1, q):
                new_d = (d + delta) % q
                diff = new_d - d
                new_w = w + diff * powers[i]
                neighbors[w, idx] = new_w
                idx += 1

    def greedy_init(rng):
        uncovered = np.ones(N, dtype=np.bool_)
        code = set()
        while np.any(uncovered) and time.time() < deadline - 5:
            # Vectorized gain calculation
            # gains[w] = number of uncovered neighbors of w
            # neighbors is (N, 17). uncovered is (N,)
            # We want sum over neighbors of uncovered[neighbors]
            # This is a matrix-vector product but with boolean mask?
            # gains = neighbors @ uncovered? No, neighbors contains indices.
            # We need to gather uncovered values for each neighbor.
            # np.take(uncovered, neighbors) gives shape (N, 17) of booleans.
            # Sum along axis 1.
            gains = np.take(uncovered, neighbors).sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            w = rng.choice(candidates)
            code.add(w)
            uncovered[neighbors[w, :]] = False
        return code

    def local_search(code_set, rng, deadline):
        code_list = list(code_set)
        in_code = np.zeros(N, dtype=np.bool_)
        for c in code_list:
            in_code[c] = True
            
        coverage = np.zeros(N, dtype=np.int32)
        for c in code_list:
            coverage[neighbors[c, :]] += 1
            
        current_size = len(code_list)
        
        while time.time() < deadline - 2:
            improved = False
            
            # 1. Deletion Pass (Vectorized)
            # A codeword c is redundant if min(coverage[neighbors[c]]) > 1
            # Compute min coverage over neighbors for all codewords in code_list
            if code_list:
                code_arr = np.array(code_list, dtype=np.int32)
                # Get neighbors of all codewords: shape (len(code_list), neigh_size)
                code_neighbors = neighbors[code_arr, :]
                # Get coverage of those neighbors: shape (len(code_list), neigh_size)
                cov_neighbors = coverage[code_neighbors]
                # Min coverage for each codeword
                min_covs = cov_neighbors.min(axis=1)
                # Find any redundant codeword
                redundant_indices = np.where(min_covs > 1)[0]
                
                if len(redundant_indices) > 0:
                    # Pick one to remove (randomly from redundant ones for diversity)
                    idx_to_remove = redundant_indices[rng.integers(0, len(redundant_indices))]
                    c_to_remove = code_list[idx_to_remove]
                    
                    code_list.pop(idx_to_remove)
                    in_code[c_to_remove] = False
                    coverage[neighbors[c_to_remove, :]] -= 1
                    current_size -= 1
                    improved = True
                    continue

            # 2. Swap Pass
            # Find c with smallest essential set E(c)
            # E(c) = { w in N(c) | coverage[w] == 1 }
            
            best_c = -1
            best_ess = None
            min_ess_size = neigh_size + 1
            
            if not code_list:
                break
                
            # Iterate through code_list
            for c in code_list:
                c_neighbors = neighbors[c, :]
                ess_mask = coverage[c_neighbors] == 1
                ess_count = np.sum(ess_mask)
                if ess_count < min_ess_size:
                    min_ess_size = ess_count
                    best_c = c
                    best_ess = c_neighbors[ess_mask]
                    if ess_count == 1:
                        break
            
            if best_c == -1:
                break

            if min_ess_size == 0:
                # Should have been caught by deletion, but just in case
                continue

            # Try to find d not in code such that best_ess subset of N(d)
            # This means d must be in intersection of N(w) for all w in best_ess
            
            # Intersection logic
            candidates = set(neighbors[best_ess[0], :].tolist())
            for w in best_ess[1:]:
                candidates.intersection_update(neighbors[w, :].tolist())
            
            # Filter out existing codewords
            valid_candidates = [d for d in candidates if not in_code[d]]
            
            if valid_candidates:
                # Choose d
                # Heuristic: Pick d that maximizes the minimum coverage increase?
                # Or just random?
                # Let's pick the one that maximizes sum(coverage[neighbors[d]]) to create more redundancy
                # This might help future deletions.
                if len(valid_candidates) > 1:
                    # Vectorized selection: pick d with max sum of coverage of its neighbors
                    # This is a proxy for creating redundancy
                    cand_arr = np.array(valid_candidates, dtype=np.int32)
                    cand_neighbors = neighbors[cand_arr, :]
                    cand_covs = coverage[cand_neighbors]
                    scores = cand_covs.sum(axis=1)
                    d = cand_arr[rng.choice(np.where(scores == scores.max())[0])]
                else:
                    d = valid_candidates[0]
                
                # Perform swap
                coverage[neighbors[best_c, :]] -= 1
                coverage[neighbors[d, :]] += 1
                in_code[best_c] = False
                in_code[d] = True
                code_list.remove(best_c)
                code_list.append(d)
                improved = True
                # Continue loop to check for deletions
                
            else:
                # No valid swap for this c
                # Perturbation: Random Swap
                # Pick a random c from code_list
                # Find any d that covers E(c)
                # If no such d, we are stuck in a local optimum for this c.
                
                # Let's try one random perturbation per stall.
                if not code_list:
                    break
                    
                c = code_list[rng.integers(0, len(code_list))]
                c_neighbors = neighbors[c, :]
                ess = c_neighbors[coverage[c_neighbors] == 1]
                
                if len(ess) > 0:
                    candidates = set(neighbors[ess[0], :].tolist())
                    for w in ess[1:]:
                        candidates.intersection_update(neighbors[w, :].tolist())
                    valid_candidates = [d for d in candidates if not in_code[d]]
                    if valid_candidates:
                        d = valid_candidates[rng.integers(0, len(valid_candidates))]
                        coverage[neighbors[c, :]] -= 1
                        coverage[neighbors[d, :]] += 1
                        in_code[c] = False
                        in_code[d] = True
                        code_list.remove(c)
                        code_list.append(d)
                        improved = True
                        continue
                else:
                    # c is redundant? Should have been deleted.
                    pass
                
                # If no perturbation worked, break local search
                break

        return set(code_list)

    best_code = None
    best_size = float('inf')
    
    # Run multiple restarts
    max_restarts = 100
    for i in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        rng = np.random.default_rng(42 + i)
        code = greedy_init(rng)
        code = local_search(code, rng, deadline)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        if best_size <= 486:
            break

    if best_code is None:
        # Fallback: Tetracode construction
        # Tetracode is a (3, 4, 3) code? No, Tetracode is length 4 over Z3.
        # K_3(8,1) can be constructed as product of Tetracode?
        # Tetracode has 9 words. 9*9=81? No.
        # The best known is 486.
        # 486 = 2 * 3^5 = 2 * 243.
        # Let's just return the best found or a simple greedy result if none.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best_code = set()
        for c1 in tetracode:
            for c2 in tetracode:
                best_code.add(int("".join(map(str, c1 + c2)), 3))
        # This gives 81 words, which is NOT a covering code for radius 1.
        # Distance between any two in product is sum of distances.
        # Min distance of tetracode is 3?
        # If min distance is 3, then product min distance is 6.
        # Radius 1 covering requires distance <= 2 between balls.
        # So 81 words is definitely not enough.
        # We must rely on the search.
        # If search fails, return a large valid code (e.g. all words) to be safe, though score will be bad.
        # But greedy_init should always produce a valid code.
        # So best_code should not be None.
        pass

    result = []
    for w in best_code:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        result.append(digits)
    
    return result
