import numpy as np
import time
import sys

def construct(q, n, radius):
    deadline = time.time() + 115.0  # safety margin before 150s wall limit

    if q == 3 and n == 8 and radius == 1:
        N = 3 ** 8  # 6561
        powers = [3 ** i for i in range(n)]
        neigh_count = 1 + 2 * n  # 17 neighbors per word

        # Precompute neighbors for all words
        neighbors = np.empty((N, neigh_count), dtype=np.int32)
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
            """Greedy set cover with randomized tie-breaking."""
            uncovered = np.ones(N, dtype=np.bool_)
            code = []
            while np.any(uncovered) and time.time() < deadline - 8:
                gains = uncovered[neighbors].sum(axis=1).astype(np.float64)
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                threshold = max_gain * alpha
                candidates = np.where(gains >= threshold)[0]
                if alpha < 1.0 and len(candidates) > 1:
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
                code_arr = np.array(list(code_set), dtype=np.int32)
                if len(code_arr) == 0:
                    break
                mins = np.min(coverage[neighbors[code_arr]], axis=1)
                redundant_mask = mins > 1
                redundant_indices = np.where(redundant_mask)[0]
                if len(redundant_indices) > 0:
                    c_to_remove = code_arr[redundant_indices[0]]
                    code_set.remove(c_to_remove)
                    coverage[neighbors[c_to_remove, :]] -= 1
                    changed = True
            return code_set

        def add_remove_search(code_set, max_iter=200):
            """Add-Remove local search targeting singly-covered words."""
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
                single_mask = (coverage == 1)
                if not np.any(single_mask):
                    break
                single_covered = np.where(single_mask)[0]
                gain = np.zeros(N, dtype=np.int32)
                for s in single_covered:
                    gain[neighbors[s, :]] += 1
                code_arr = np.array(code_list, dtype=np.int32)
                gain[code_arr] = 0
                max_gain = np.max(gain)
                if max_gain == 0:
                    break
                threshold = max_gain * 0.8
                candidates = np.where(gain >= threshold)[0]
                if len(candidates) == 0:
                    candidates = np.where(gain == max_gain)[0]
                new_c = rng.choice(candidates)
                code_set.add(new_c)
                coverage[neighbors[new_c, :]] += 1
                code_list.append(new_c)
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
                else:
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
                code_arr = np.array(code_list, dtype=np.int32)
                if len(code_arr) == 0:
                    break
                idx = rng.integers(0, len(code_list))
                c = code_list[idx]
                c_neigh = neighbors[c, :]
                essential = c_neigh[coverage[c_neigh] == 1]
                if len(essential) == 0:
                    code_list.pop(idx)
                    coverage[neighbors[c, :]] -= 1
                    continue
                if len(essential) == 1:
                    candidates = neighbors[essential[0], :]
                else:
                    cand_set = set(neighbors[essential[0], :].tolist())
                    for ew in essential[1:]:
                        cand_set &= set(neighbors[ew, :].tolist())
                    if not cand_set:
                        continue
                    candidates = np.array(list(cand_set), dtype=np.int32)
                for d in candidates:
                    if d in code_set:
                        continue
                    d_neigh_set = set(neighbors[d, :].tolist())
                    bad = False
                    for w in essential:
                        if w not in d_neigh_set:
                            bad = True
                            break
                    if bad:
                        continue
                    code_set.remove(c)
                    code_set.add(d)
                    coverage[neighbors[c, :]] -= 1
                    coverage[neighbors[d, :]] += 1
                    code_list = list(code_set)
                    break
            return code_set

        def swap2_search(code_set, max_iter=30):
            """Try 2-for-1 swaps: remove two codewords, add one that covers their essential words."""
            if len(code_set) < 3:
                return code_set
            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1
            rng = np.random.default_rng()
            code_list = list(code_set)
            for _ in range(max_iter):
                if time.time() > deadline - 5:
                    break
                # Pick two random codewords
                if len(code_list) < 2:
                    break
                i1, i2 = rng.choice(len(code_list), size=2, replace=False)
                c1, c2 = code_list[i1], code_list[i2]
                n1 = neighbors[c1, :]
                n2 = neighbors[c2, :]
                # Essential words: covered only by c1 or only by c2
                ess1 = n1[coverage[n1] == 1]
                ess2 = n2[coverage[n2] == 1]
                if len(ess1) == 0 and len(ess2) == 0:
                    # Both redundant, just remove one
                    code_set.remove(c1)
                    coverage[n1] -= 1
                    code_list.remove(c1)
                    continue
                # Find candidate d that covers all essential words from both
                all_ess = np.concatenate([ess1, ess2])
                if len(all_ess) == 0:
                    continue
                if len(all_ess) == 1:
                    candidates = neighbors[all_ess[0], :]
                else:
                    cand_set = set(neighbors[all_ess[0], :].tolist())
                    for ew in all_ess[1:]:
                        cand_set &= set(neighbors[ew, :].tolist())
                    if not cand_set:
                        continue
                    candidates = np.array(list(cand_set), dtype=np.int32)
                # Filter: d not in code, and d must cover all words that become uncovered
                for d in candidates:
                    if d in code_set:
                        continue
                    d_neigh = neighbors[d, :]
                    d_neigh_set = set(d_neigh.tolist())
                    # Check all essential words are covered by d
                    bad = False
                    for w in all_ess:
                        if w not in d_neigh_set:
                            bad = True
                            break
                    if bad:
                        continue
                    # Check words in n1∩n2 with coverage==2
                    shared = np.intersect1d(n1, n2)
                    shared_with_cov2 = shared[coverage[shared] == 2]
                    for w in shared_with_cov2:
                        if w not in d_neigh_set:
                            bad = True
                            break
                    if bad:
                        continue
                    # Accept the swap
                    code_set.remove(c1)
                    code_set.remove(c2)
                    code_set.add(d)
                    coverage[n1] -= 1
                    coverage[n2] -= 1
                    coverage[d_neigh] += 1
                    code_list = list(code_set)
                    break
            return code_set

        def swap22_search(code_set, max_iter=20):
            """Try 2-for-2 swaps: remove two codewords, add two new ones that cover all essential words."""
            if len(code_set) < 4:
                return code_set
            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1
            rng = np.random.default_rng()
            code_list = list(code_set)
            for _ in range(max_iter):
                if time.time() > deadline - 5:
                    break
                if len(code_list) < 2:
                    break
                i1, i2 = rng.choice(len(code_list), size=2, replace=False)
                c1, c2 = code_list[i1], code_list[i2]
                n1 = neighbors[c1, :]
                n2 = neighbors[c2, :]
                # Words that would become uncovered if we remove c1 and c2
                # A word w becomes uncovered if coverage[w] == 1 and w is in n1 or n2,
                # OR coverage[w] == 2 and w is in n1 and n2
                ess1 = n1[coverage[n1] == 1]
                ess2 = n2[coverage[n2] == 1]
                shared = np.intersect1d(n1, n2)
                shared_cov2 = shared[coverage[shared] == 2]
                all_ess = np.concatenate([ess1, ess2, shared_cov2])
                if len(all_ess) == 0:
                    # Both redundant
                    code_set.remove(c1)
                    coverage[n1] -= 1
                    code_list.remove(c1)
                    continue
                # Find two new codewords d1, d2 such that their union covers all_ess
                # Strategy: for each essential word, find a candidate that covers it
                # Then try to pair up candidates
                # Simpler: pick a random essential word, find a candidate d1 covering it,
                # then find d2 covering the remaining essential words not covered by d1
                ess_list = list(all_ess)
                rng.shuffle(ess_list)
                found = False
                for ew in ess_list[:5]:  # Try up to 5 starting points
                    d1_candidates = neighbors[ew, :]
                    for d1 in d1_candidates:
                        if d1 in code_set:
                            continue
                        if d1 == c1 or d1 == c2:
                            continue
                        d1_neigh = neighbors[d1, :]
                        d1_set = set(d1_neigh.tolist())
                        # Remaining essential words not covered by d1
                        remaining = [w for w in all_ess if w not in d1_set]
                        if not remaining:
                            # d1 alone covers everything, just do 2-for-1
                            code_set.remove(c1)
                            code_set.remove(c2)
                            code_set.add(d1)
                            coverage[n1] -= 1
                            coverage[n2] -= 1
                            coverage[d1_neigh] += 1
                            code_list = list(code_set)
                            found = True
                            break
                        # Find d2 that covers all remaining
                        if len(remaining) == 1:
                            d2_candidates = neighbors[remaining[0], :]
                        else:
                            cand_set = set(neighbors[remaining[0], :].tolist())
                            for rw in remaining[1:]:
                                cand_set &= set(neighbors[rw, :].tolist())
                            if not cand_set:
                                continue
                            d2_candidates = np.array(list(cand_set), dtype=np.int32)
                        for d2 in d2_candidates:
                            if d2 in code_set:
                                continue
                            if d2 == c1 or d2 == c2 or d2 == d1:
                                continue
                            d2_neigh = neighbors[d2, :]
                            d2_set = set(d2_neigh.tolist())
                            # Verify all essential words are covered
                            bad = False
                            for w in all_ess:
                                if w not in d1_set and w not in d2_set:
                                    bad = True
                                    break
                            if bad:
                                continue
                            # Accept the 2-for-2 swap
                            code_set.remove(c1)
                            code_set.remove(c2)
                            code_set.add(d1)
                            code_set.add(d2)
                            coverage[n1] -= 1
                            coverage[n2] -= 1
                            coverage[d1_neigh] += 1
                            coverage[d2_neigh] += 1
                            code_list = list(code_set)
                            found = True
                            break
                        if found:
                            break
                    if found:
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
            if time.time() > deadline - 15:
                break

            rng = np.random.default_rng(seed)
            code_list = greedy_construct(rng, alpha)
            code_set = set(code_list)

            # Reduction phase
            code_set = reduce_code(code_set)
            # Swap phase
            code_set = try_swaps(code_set, max_iter=30)
            # Add-Remove phase
            code_set = add_remove_search(code_set, max_iter=150)
            # 2-for-1 swap phase
            code_set = swap2_search(code_set, max_iter=20)
            # 2-for-2 swap phase
            code_set = swap22_search(code_set, max_iter=15)
            # Second swap phase
            code_set = try_swaps(code_set, max_iter=30)
            # Final reduction
            code_set = reduce_code(code_set)

            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = code_set

            if best_size <= 486:
                break

        if best_code is None:
            rng = np.random.default_rng(0)
            code_list = greedy_construct(rng, 1.0)
            best_code = set(reduce_code(code_list))

        # Final polish on the best code found
        if best_code:
            best_code = try_swaps(best_code, max_iter=40)
            best_code = add_remove_search(best_code, max_iter=100)
            best_code = swap2_search(best_code, max_iter=20)
            best_code = swap22_search(best_code, max_iter=15)
            best_code = try_swaps(best_code, max_iter=40)
            best_code = reduce_code(best_code)
            # One more round
            best_code = add_remove_search(best_code, max_iter=50)
            best_code = swap2_search(best_code, max_iter=10)
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
        powers = [q ** i for i in range(n)]
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
