import numpy as np
import time
import random
import heapq
import itertools


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    # Fallback for non-specific parameters
    if q != 3 or n != 7 or radius != 1:
        if n <= 4 and q == 3 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbor lists as numpy arrays and bitsets
    neighbors_list = []
    neighbor_bitsets = []
    neighbor_sets = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(nb, dtype=np.int32))
        bs = 0
        for x in nb:
            bs |= (1 << x)
        neighbor_bitsets.append(bs)
        neighbor_sets.append(set(nb))

    # ---------- Phase 1: Vectorized Greedy Set-Cover ----------
    uncovered = np.ones(N, dtype=bool)
    code = []
    code_set = set()

    while np.any(uncovered):
        if time.time() > deadline - 40:
            break

        # Compute gains: for each candidate, how many uncovered points does it cover?
        gains = np.zeros(N, dtype=np.int32)
        uncovered_indices = np.where(uncovered)[0]
        for idx in uncovered_indices:
            gains[neighbors_list[idx]] += 1

        best_idx = int(np.argmax(gains))
        max_gain = gains[best_idx]

        if max_gain == 0:
            break

        code.append(best_idx)
        code_set.add(best_idx)
        uncovered[neighbors_list[best_idx]] = False

    # ---------- Phase 2: Local Search ----------
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbors_list[idx]] += 1

    best_size = len(code)
    best_code = list(code)

    def try_remove_redundant():
        """Remove all redundant codewords."""
        removed = False
        for c in list(code):
            if np.min(counts[neighbors_list[c]]) > 1:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1
                removed = True
        return removed

    def find_replacement(c, crit_points, exclude_set=None):
        """Find best replacement for c that covers all critical points."""
        candidate_bitset = (1 << N) - 1
        for p in crit_points:
            candidate_bitset &= neighbor_bitsets[p]

        if exclude_set:
            for cw in exclude_set:
                candidate_bitset &= ~(1 << cw)

        if candidate_bitset == 0:
            return -1

        candidates = []
        temp = candidate_bitset
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            candidates.append(idx)
            temp ^= lsb

        if len(candidates) > 300:
            random.shuffle(candidates)
            candidates = candidates[:300]

        best_w = -1
        best_score = -1
        for w in candidates:
            valid = True
            for k in crit_points:
                if k not in neighbor_sets[w]:
                    valid = False
                    break
            if not valid:
                continue
            score = int(np.sum(counts[neighbors_list[w]] == 1))
            if score > best_score:
                best_score = score
                best_w = w
        return best_w

    def try_swap(c):
        """Try to swap codeword c with a better-positioned codeword."""
        if np.min(counts[neighbors_list[c]]) > 1:
            return False

        crit_points = [i for i in neighbors_list[c] if counts[i] == 1]
        if not crit_points:
            return False

        best_w = find_replacement(c, crit_points, exclude_set=code_set)

        if best_w != -1:
            code.remove(c)
            code_set.remove(c)
            counts[neighbors_list[c]] -= 1

            code.append(best_w)
            code_set.add(best_w)
            counts[neighbors_list[best_w]] += 1
            return True

        # Strategy 2: Relocation
        temp_counts = counts.copy()
        temp_counts[neighbors_list[c]] -= 1
        would_be_uncovered = [i for i in neighbors_list[c] if temp_counts[i] == 0]

        if not would_be_uncovered:
            return False

        all_needed = set(crit_points) | set(would_be_uncovered)

        candidate_bitset = (1 << N) - 1
        for p in all_needed:
            candidate_bitset &= neighbor_bitsets[p]

        for cw in code:
            candidate_bitset &= ~(1 << cw)

        if candidate_bitset == 0:
            return False

        candidates = []
        temp = candidate_bitset
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            candidates.append(idx)
            temp ^= lsb

        if len(candidates) > 300:
            random.shuffle(candidates)
            candidates = candidates[:300]

        best_w = -1
        best_score = -1
        for w in candidates:
            valid = True
            for k in all_needed:
                if k not in neighbor_sets[w]:
                    valid = False
                    break
            if not valid:
                continue
            score = int(np.sum(counts[neighbors_list[w]] == 1))
            if score > best_score:
                best_score = score
                best_w = w

        if best_w != -1:
            code.remove(c)
            code_set.remove(c)
            counts[neighbors_list[c]] -= 1

            code.append(best_w)
            code_set.add(best_w)
            counts[neighbors_list[best_w]] += 1
            return True

        return False

    def try_pair_replace():
        """Try to replace two codewords with one that covers all their unique points."""
        if len(code) < 2:
            return False

        # Sample pairs to avoid O(k^2) explosion
        k = len(code)
        pairs = []
        if k <= 40:
            for i in range(k):
                for j in range(i + 1, k):
                    pairs.append((i, j))
        else:
            # Sample a subset of pairs
            for _ in range(200):
                i, j = random.sample(range(k), 2)
                if i > j:
                    i, j = j, i
                pairs.append((i, j))

        for i, j in pairs:
            if time.time() > deadline - 15:
                return False
            if i >= len(code) or j >= len(code):
                continue
            c1 = code[i]
            c2 = code[j]

            # Find points uniquely covered by c1 or c2
            temp_counts = counts.copy()
            temp_counts[neighbors_list[c1]] -= 1
            temp_counts[neighbors_list[c2]] -= 1
            would_be_uncovered = [x for x in set(neighbors_list[c1].tolist()) | set(neighbors_list[c2].tolist())
                                  if temp_counts[x] == 0]

            if not would_be_uncovered:
                # Both can be removed
                code.remove(c1)
                code_set.remove(c1)
                counts[neighbors_list[c1]] -= 1
                code.remove(c2)
                code_set.remove(c2)
                counts[neighbors_list[c2]] -= 1
                return True

            # Find a single codeword that covers all would_be_uncovered
            candidate_bitset = (1 << N) - 1
            for p in would_be_uncovered:
                candidate_bitset &= neighbor_bitsets[p]

            for cw in code:
                if cw != c1 and cw != c2:
                    candidate_bitset &= ~(1 << cw)

            if candidate_bitset == 0:
                continue

            candidates = []
            temp = candidate_bitset
            while temp:
                lsb = temp & (-temp)
                idx = lsb.bit_length() - 1
                candidates.append(idx)
                temp ^= lsb

            if len(candidates) > 100:
                random.shuffle(candidates)
                candidates = candidates[:100]

            best_w = -1
            best_score = -1
            for w in candidates:
                valid = True
                for k in would_be_uncovered:
                    if k not in neighbor_sets[w]:
                        valid = False
                        break
                if not valid:
                    continue
                score = int(np.sum(counts[neighbors_list[w]] == 1))
                if score > best_score:
                    best_score = score
                    best_w = w

            if best_w != -1:
                code.remove(c1)
                code_set.remove(c1)
                counts[neighbors_list[c1]] -= 1
                code.remove(c2)
                code_set.remove(c2)
                counts[neighbors_list[c2]] -= 1

                code.append(best_w)
                code_set.add(best_w)
                counts[neighbors_list[best_w]] += 1
                return True

        return False

    # Iterative local search with multiple passes
    improved = True
    pass_count = 0
    while improved and time.time() < deadline - 10:
        improved = False
        pass_count += 1

        # Removal pass
        if try_remove_redundant():
            improved = True

        # Swap pass
        code_indices = list(range(len(code)))
        random.shuffle(code_indices)
        for idx in code_indices:
            if time.time() > deadline - 10:
                break
            if idx >= len(code):
                continue
            if try_swap(code[idx]):
                improved = True
                break  # Restart outer loop

        # Pair replace pass (2-for-1)
        if not improved and time.time() < deadline - 10:
            if try_pair_replace():
                improved = True

        # Update best
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

    # Final cleanup
    try_remove_redundant()

    # Use best found
    if len(code) > best_size:
        code = best_code

    return [idx_to_tuple(c) for c in code]
