import numpy as np
import time
import sys

def construct(q, n, radius):
    deadline = time.time() + 115.0  # 150s wall, safety margin

    if q == 3 and n == 8 and radius == 1:
        N = 3 ** 8  # 6561
        powers = [3 ** i for i in range(n)]

        # Precompute neighbors for all words (1 + 2*n neighbors)
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
                # Iterate over a copy since we modify the set
                for c in list(code_set):
                    # Check if removing c leaves any word uncovered
                    # i.e., is there any neighbor of c with coverage == 1?
                    if np.min(coverage[neighbors[c, :]]) > 1:
                        code_set.remove(c)
                        coverage[neighbors[c, :]] -= 1
                        changed = True
            return code_set

        def add_remove_search(code_set, max_iter=200):
            """
            Optimized Add-Remove local search.
            Only checks redundancy for codewords that might be affected by the addition.
            """
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

                # Step 1: Add a candidate codeword
                new_c = None
                # Try a few random candidates
                for _ in range(5):
                    candidate = rng.integers(0, N)
                    if candidate not in code_set:
                        new_c = candidate
                        break
                
                if new_c is None:
                    continue

                # Add new_c
                code_set.add(new_c)
                coverage[neighbors[new_c, :]] += 1
                code_list.append(new_c)
                
                # Step 2: Try to remove ANY redundant codeword
                # Optimization: Only check codewords c where neighbors[c] intersects neighbors[new_c]
                # Precompute intersection logic? 
                # Simple check: if c is in the "vicinity" of new_c.
                # For now, just check all, but break early if found?
                # Or better: identify potential candidates.
                
                # A codeword c is redundant if min(coverage[neighbors[c]]) > 1.
                # Coverage only changed for neighbors of new_c.
                # So only c such that neighbors[c] overlaps with neighbors[new_c] can change status.
                # This is a small set.
                
                # Heuristic: Check new_c first, then its neighbors that are in code_set.
                # Actually, just checking all is ~8000 ops. It's fast enough.
                # Let's stick to checking all for simplicity and robustness, 
                # but we can optimize by checking if min(coverage[neighbors[c]]) > 1.
                
                removable_candidates = []
                for c in code_list:
                    if np.min(coverage[neighbors[c, :]]) > 1:
                        removable_candidates.append(c)
                
                if removable_candidates:
                    # Prioritize removing an old one
                    old_removable = [c for c in removable_candidates if c != new_c]
                    if old_removable:
                        c_to_remove = rng.choice(old_removable)
                    else:
                        c_to_remove = new_c
                        
                    code_set.remove(c_to_remove)
                    coverage[neighbors[c_to_remove, :]] -= 1
                    code_list.remove(c_to_remove)
                    
                else:
                    # Nothing removable. Revert the addition.
                    code_set.remove(new_c)
                    coverage[neighbors[new_c, :]] -= 1
                    code_list.remove(new_c)
                    
            return code_set

        def try_swaps(code_set, max_iter=50):
            """Try 1-for-1 swaps to reduce code size."""
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
                # Pick a random codeword to try removing
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
                    # Intersection of neighborhoods of all essential words
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
                    bad = False
                    for w in c_neigh:
                        if coverage[w] == 1 and w not in neighbors[d, :]:
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

        # Increased configurations to leverage any speedups
        configs = [
            (42, 1.0), (123, 1.0), (456, 0.95), (789, 0.9),
            (101, 1.0), (202, 0.85), (303, 0.95), (404, 1.0),
            (505, 0.9), (606, 0.8), (707, 1.0), (808, 0.95),
            (909, 0.85), (111, 1.0), (222, 0.9), (333, 0.8),
            (444, 1.0), (555, 0.9), (666, 1.0), (777, 0.85),
            (888, 0.9), (999, 1.0), (121, 0.95), (131, 0.85)
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
            code_set = try_swaps(code_set, max_iter=20)
            
            # Add-Remove phase
            code_set = add_remove_search(code_set, max_iter=100)
            
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
            best_code = try_swaps(best_code, max_iter=20)
            best_code = add_remove_search(best_code, max_iter=50)
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
                diff1 = ((d + 1) % q) - d
                neighbors[w, idx] = w + diff1 * powers[i]
                idx += 1
                diff2 = ((d + 2) % q) - d
                neighbors[w, idx] = w + diff2 * powers[i]
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
