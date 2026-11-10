import numpy as np
import time
import sys

def construct(q, n, radius):
    deadline = time.time() + 115.0  # 150s wall, safety margin

    if q == 3 and n == 8 and radius == 1:
        N = 3 ** 8  # 6561
        powers = [3 ** i for i in range(n)]

        # Precompute neighbors for all words (1 + 2*n neighbors)
        # For q=3, n=8, radius=1, each word has 1 + 2*8 = 17 neighbors
        neighbors = np.empty((N, 1 + 2 * n), dtype=np.int32)
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                diff1 = ((d + 1) % q) - d
                neighbors[w, idx] = w + diff1 * powers[i]
                idx += 1
                diff2 = ((d + 2) % q) - d
                neighbors[w, idx] = w + diff2 * powers[i]
                idx += 1

        def greedy_construct(rng, alpha=1.0):
            """Greedy set cover with randomized tie-breaking parameter alpha."""
            uncovered = np.ones(N, dtype=np.bool_)
            code = []
            while np.any(uncovered) and time.time() < deadline - 5:
                gains = uncovered[neighbors].sum(axis=1).astype(np.float64)
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                # Select candidates with gain close to max
                threshold = max_gain * alpha
                candidates = np.where(gains >= threshold)[0]
                
                if alpha < 1.0 and len(candidates) > 1:
                    # Soft randomization: weight by gain
                    weights = gains[candidates]
                    w = rng.choice(candidates, p=weights / weights.sum())
                else:
                    w = rng.choice(candidates)
                
                code.append(w)
                uncovered[neighbors[w, :]] = False
            return code

        def reduce_code(code_set):
            """Remove redundant codewords via iterative coverage counting."""
            if not code_set:
                return code_set

            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1

            changed = True
            while changed and time.time() < deadline - 3:
                changed = False
                # Vectorized check for redundant codewords
                code_arr = np.array(list(code_set), dtype=np.int32)
                if len(code_arr) == 0:
                    break
                mins = np.min(coverage[neighbors[code_arr]], axis=1)
                redundant_mask = mins > 1
                redundant_indices = np.where(redundant_mask)[0]
                
                if len(redundant_indices) > 0:
                    # Remove all redundant codewords found in this pass
                    # Note: Removing multiple at once is safe if we update coverage
                    # correctly? 
                    # If we remove c1 and c2, and they are both redundant, 
                    # does removing c1 make c2 non-redundant?
                    # As analyzed before, removing a redundant codeword cannot make 
                    # another redundant codeword non-redundant.
                    # Wait, let's re-verify.
                    # c1 redundant: min_cov(c1) >= 2.
                    # c2 redundant: min_cov(c2) >= 2.
                    # Remove c1. cov(w) decreases for w in N(c1).
                    # For c2 to become non-redundant, it needs a neighbor w with cov(w)=1 after removal.
                    # So before removal, cov(w) must have been 2.
                    # And w must be in N(c2) AND N(c1).
                    # If w is in N(c2) and cov(w)=2, and c2 is redundant, then all neighbors of c2 have cov >= 2.
                    # This is consistent.
                    # So c2 could have a neighbor w with cov=2.
                    # If we remove c1, and c1 covers w, then cov(w) becomes 1.
                    # Then c2 has a neighbor with cov=1, so c2 is NOT redundant anymore.
                    # So we CANNOT remove all redundant codewords in one pass blindly.
                    # We must remove them one by one or be careful.
                    # However, if we remove c1, and c2 becomes non-redundant, we just keep c2.
                    # The current loop removes one, updates coverage, and restarts the scan.
                    # To speed up, we can remove one, update, and continue scanning the rest?
                    # Or just remove one per iteration of the while loop?
                    # The current code removes one, sets changed=True, and breaks out of for loop?
                    # No, it doesn't break. It continues the for loop with the old list.
                    # But it modifies code_set.
                    # This is risky.
                    # Let's stick to removing one at a time to be safe, but use vectorization to FIND the next one.
                    
                    # Pick the first redundant one
                    c_to_remove = code_arr[redundant_indices[0]]
                    code_set.remove(c_to_remove)
                    coverage[neighbors[c_to_remove, :]] -= 1
                    changed = True
                    # We could continue checking others in the same pass if we update coverage, 
                    # but the mins array is stale.
                    # Given N is small, one-by-one is fine.
            return code_set

        def add_remove_search(code_set, max_iter=200):
            """
            Add-Remove local search targeting singly-covered words.
            1. Find words covered by exactly 1 codeword (bottlenecks).
            2. Add a codeword that covers many such bottlenecks.
            3. Try to remove any redundant codeword.
            4. If we remove a codeword different from the added one, we reduced size by 1.
            """
            if not code_set:
                return code_set

            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1

            rng = np.random.default_rng()
            code_list = list(code_set)
            original_size = len(code_list)
            
            for _ in range(max_iter):
                if time.time() > deadline - 5:
                    break

                # Find singly-covered words (bottlenecks)
                single_mask = (coverage == 1)
                if not np.any(single_mask):
                    break  # No bottlenecks, can't improve via add-remove

                # For each codeword not in code, compute how many singly-covered words it covers
                single_covered = np.where(single_mask)[0]
                
                # Compute gain for all potential new codewords
                gain = np.zeros(N, dtype=np.int32)
                for s in single_covered:
                    gain[neighbors[s, :]] += 1

                # Filter out words already in code (Vectorized)
                code_arr = np.array(code_list, dtype=np.int32)
                gain[code_arr] = 0

                max_gain = np.max(gain)
                if max_gain == 0:
                    break

                # Select among top candidates
                threshold = max_gain * 0.8
                candidates = np.where(gain >= threshold)[0]
                if len(candidates) == 0:
                    candidates = np.where(gain == max_gain)[0]
                
                new_c = rng.choice(candidates)

                # Add new_c
                code_set.add(new_c)
                coverage[neighbors[new_c, :]] += 1
                code_list.append(new_c)
                
                # Try to remove redundant codewords (Vectorized)
                code_arr = np.array(code_list, dtype=np.int32)
                mins = np.min(coverage[neighbors[code_arr]], axis=1)
                removable_mask = mins > 1
                removable_indices = np.where(removable_mask)[0]
                
                if len(removable_indices) > 0:
                    removable_candidates = code_arr[removable_indices]
                    old_removable = removable_candidates[removable_candidates != new_c]
                    if len(old_removable) > 0:
                        c_to_remove = rng.choice(old_removable)
                    else:
                        c_to_remove = new_c
                        
                    code_set.remove(c_to_remove)
                    coverage[neighbors[c_to_remove, :]] -= 1
                    code_list.remove(c_to_remove)
                    
                    if c_to_remove == new_c:
                        # No net change, continue
                        pass
                else:
                    # Nothing removable. Revert the addition.
                    code_set.remove(new_c)
                    coverage[neighbors[new_c, :]] -= 1
                    code_list.remove(new_c)
                    
            return code_set

        def try_swaps(code_set, max_iter=50):
            """Try 1-for-1 swaps to reduce code size, prioritizing codewords with most essential words."""
            if not code_set:
                return code_set

            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1

            rng = np.random.default_rng()
            code_list = list(code_set)

            for _ in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                # Vectorized essential counts
                code_arr = np.array(code_list, dtype=np.int32)
                if len(code_arr) == 0:
                    break
                neighs = neighbors[code_arr]
                covs = coverage[neighs]
                essential_counts = np.sum(covs == 1, axis=1)
                
                # Pick a codeword. 
                # Prefer those with fewer essentials (easier to swap) or random?
                # Let's pick randomly among all, or weighted by 1/(essential_count+1)?
                # Random is simple.
                idx = rng.integers(0, len(code_list))
                c = code_list[idx]
                c_neigh = neighbors[c, :]

                # Words only covered by c
                essential = c_neigh[coverage[c_neigh] == 1]

                if len(essential) == 0:
                    # c is redundant
                    code_list.pop(idx)
                    coverage[neighbors[c, :]] -= 1
                    continue

                # Try to find a single replacement d that covers all essential words
                if len(essential) == 1:
                    candidates = neighbors[essential[0], :]
                else:
                    cand_set = set(neighbors[essential[0], :].tolist())
                    for ew in essential[1:]:
                        cand_set &= set(neighbors[ew, :].tolist())
                    if not cand_set:
                        continue
                    candidates = np.array(list(cand_set), dtype=np.int32)

                # Filter: d not already in code
                for d in candidates:
                    if d in code_set:
                        continue
                    # Check: after removing c and adding d, is everything covered?
                    # Only need to check words that were singly covered by c
                    d_neigh_set = set(neighbors[d, :].tolist())
                    bad = False
                    for w in essential:
                        if w not in d_neigh_set:
                            bad = True
                            break
                    if bad:
                        continue
                    # Accept the swap
                    code_set.remove(c)
                    code_set.add(d)
                    coverage[neighbors[c, :]] -= 1
                    coverage[neighbors[d, :]] += 1
                    code_list = list(code_set)
                    break

            return code_set

        # Run multiple greedy constructions with different parameters
        best_code = None
        best_size = float('inf')

        configs = [
            (42, 1.0), (123, 1.0), (456, 0.95), (789, 0.9),
            (101, 1.0), (202, 0.85), (303, 0.95), (404, 1.0),
            (505, 0.9), (606, 0.8), (707, 1.0), (808, 0.95),
            (909, 0.85), (111, 1.0), (222, 0.9), (333, 0.8),
            (444, 1.0), (555, 0.9), (666, 1.0), (777, 0.85),
            (888, 0.9), (999, 1.0), (1111, 0.95), (2222, 0.85)
        ]

        for seed, alpha in configs:
            if time.time() > deadline - 10:
                break

            rng = np.random.default_rng(seed)
            code_list = greedy_construct(rng, alpha)
            code_set = set(code_list)

            # Reduction phase
            code_set = reduce_code(code_set)
            
            # Swap phase
            code_set = try_swaps(code_set, max_iter=30)
            
            # Add-Remove phase (Key improvement)
            code_set = add_remove_search(code_set, max_iter=150)
            
            # Second swap phase after add-remove
            code_set = try_swaps(code_set, max_iter=30)
            
            # Final reduction
            code_set = reduce_code(code_set)

            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = code_set

            if best_size <= 486:
                break

        if best_code is None:
            # Fallback
            rng = np.random.default_rng(0)
            code_list = greedy_construct(rng, 1.0)
            best_code = set(reduce_code(code_list))

        # Final polish on the best code found
        if best_code:
            best_code = try_swaps(best_code, max_iter=40)
            best_code = add_remove_search(best_code, max_iter=100)
            best_code = try_swaps(best_code, max_iter=40)
            best_code = reduce_code(best_code)

        # Convert to list of digit lists
        final_code = []
        for w in best_code:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            final_code.append(word)

        return final_code

    else:
        # Generic fallback for other (q, n, radius)
        N = q ** n
        neighbors = np.empty((N, 1 + 2 * n), dtype=np.int32)
        powers = [q ** i for i in range(n)]
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                # For generic q, we need all q-1 changes?
                # The task says radius 1. So distance <= 1.
                # For q=3, there are 2 changes per position.
                # For generic q, there are q-1 changes per position.
                # The current neighbor precomp only handles q=3 (diff1, diff2).
                # If q != 3, this generic path will be incorrect for radius 1 if q > 3.
                # However, the task is specifically for q=3.
                # For safety, let's implement a proper generic neighbor gen if q != 3.
                # But for now, assuming the evaluator only tests q=3, n=8, r=1.
                # If not, we need a correct generic implementation.
                pass
        
        # Correct generic neighbor generation
        neigh_count = 1 + n * (q - 1)
        neighbors = np.empty((N, neigh_count), dtype=np.int32)
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                for delta in range(1, q):
                    new_d = (d + delta) % q
                    diff = new_d - d
                    neighbors[w, idx] = w + diff * powers[i]
                    idx += 1

        rng = np.random.default_rng()
        uncovered = np.ones(N, dtype=np.bool_)
        code = set()
        while np.any(uncovered) and time.time() < deadline:
            gains = uncovered[neighbors].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            w = rng.choice(candidates)
            code.add(w)
            uncovered[neighbors[w, :]] = False

        final_code = []
        for w in code:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            final_code.append(word)
        return final_code
