import numpy as np
import time
import itertools
import random
from collections import deque

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        deadline = time.time() + 115.0
        N = q ** n  # 6561
        powers = [q ** i for i in range(n)]
        neigh_size = 1 + n * (q - 1)  # 17

        # Precompute neighbors
        neighbors = np.empty((N, neigh_size), dtype=np.int32)
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

        def compute_coverage(code_set):
            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1
            return coverage

        def greedy_init(rng, mode=0):
            uncovered = np.ones(N, dtype=np.bool_)
            gains = np.full(N, neigh_size, dtype=np.int32)
            code = set()

            if mode == 2:
                w = int(rng.integers(0, N))
                code.add(w)
                uncovered[neighbors[w, :]] = False
                gains[neighbors[w, :]] -= 1

            while np.any(uncovered) and time.time() < deadline - 5:
                max_gain = int(np.max(gains))
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]

                if mode == 1 and len(candidates) > 1:
                    top_k = min(10, len(candidates))
                    w = int(rng.choice(candidates[:top_k]))
                elif mode == 0:
                    w = int(candidates[0])
                else:
                    w = int(rng.choice(candidates))

                code.add(w)
                uncovered[neighbors[w, :]] = False
                gains[neighbors[w, :]] -= 1

            return code

        def reduce_code(code_set, deadline):
            if not code_set:
                return code_set
            coverage = compute_coverage(code_set)
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for c in list(code_set):
                    if np.min(coverage[neighbors[c, :]]) > 1:
                        code_set.remove(c)
                        coverage[neighbors[c, :]] -= 1
                        changed = True
            return code_set

        def find_replacement(c, essential_words, code_set, rng):
            if len(essential_words) == 0:
                return None
            ew0 = int(essential_words[0])
            candidates = set(neighbors[ew0, :].tolist())
            for ew in essential_words[1:]:
                candidates.intersection_update(neighbors[int(ew), :].tolist())
            candidates = [d for d in candidates if d not in code_set]
            if not candidates:
                return None
            return candidates[int(rng.integers(0, len(candidates)))]

        def try_single_swap(code_set, coverage, deadline, rng):
            if not code_set:
                return code_set, coverage, False
            code_list = list(code_set)
            rng.shuffle(code_list)
            for c in code_list[:25]:
                c_neighbors = neighbors[c, :]
                essential_words = c_neighbors[coverage[c_neighbors] == 1]
                if len(essential_words) == 0:
                    code_set.remove(c)
                    coverage[neighbors[c, :]] -= 1
                    return code_set, coverage, True
                d = find_replacement(c, essential_words, code_set, rng)
                if d is not None:
                    d_neighbors_set = set(neighbors[d, :].tolist())
                    if all(int(ew) in d_neighbors_set for ew in essential_words):
                        code_set.remove(c)
                        code_set.add(d)
                        coverage[neighbors[c, :]] -= 1
                        coverage[neighbors[d, :]] += 1
                        code_set = reduce_code(code_set, deadline)
                        coverage = compute_coverage(code_set)
                        return code_set, coverage, True
            return code_set, coverage, False

        def try_double_swap(code_set, coverage, deadline, rng):
            if len(code_set) < 3:
                return code_set, coverage, False
            code_list = list(code_set)
            rng.shuffle(code_list)
            for i in range(min(12, len(code_list) - 1)):
                c1 = code_list[i]
                c2 = code_list[i + 1]
                combined_cov = coverage.copy()
                combined_cov[neighbors[c1, :]] -= 1
                combined_cov[neighbors[c2, :]] -= 1
                uncovered_words = np.where(combined_cov == 0)[0]
                if len(uncovered_words) > 4:
                    continue
                if len(uncovered_words) == 0:
                    code_set.remove(c1)
                    code_set.remove(c2)
                    coverage[neighbors[c1, :]] -= 1
                    coverage[neighbors[c2, :]] -= 1
                    return code_set, coverage, True
                uw = uncovered_words.tolist()
                if len(uw) <= 3:
                    candidates = set(neighbors[uw[0], :].tolist())
                    for w in uw[1:]:
                        candidates.intersection_update(neighbors[w, :].tolist())
                    candidates = [d for d in candidates if d not in code_set]
                    if candidates:
                        d = candidates[int(rng.integers(0, len(candidates)))]
                        code_set.remove(c1)
                        code_set.remove(c2)
                        code_set.add(d)
                        coverage[neighbors[c1, :]] -= 1
                        coverage[neighbors[c2, :]] -= 1
                        coverage[neighbors[d, :]] += 1
                        code_set = reduce_code(code_set, deadline)
                        coverage = compute_coverage(code_set)
                        return code_set, coverage, True
                if len(uw) <= 6:
                    for split in range(1, len(uw)):
                        group1 = uw[:split]
                        group2 = uw[split:]
                        cand1 = set(neighbors[group1[0], :].tolist())
                        for w in group1[1:]:
                            cand1.intersection_update(neighbors[w, :].tolist())
                        cand1 = [d for d in cand1 if d not in code_set]
                        if not cand1:
                            continue
                        cand2 = set(neighbors[group2[0], :].tolist())
                        for w in group2[1:]:
                            cand2.intersection_update(neighbors[w, :].tolist())
                        cand2 = [d for d in cand2 if d not in code_set and d not in cand1]
                        if not cand2:
                            continue
                        d1 = cand1[int(rng.integers(0, len(cand1)))]
                        d2 = cand2[int(rng.integers(0, len(cand2)))]
                        code_set.remove(c1)
                        code_set.remove(c2)
                        code_set.add(d1)
                        code_set.add(d2)
                        coverage[neighbors[c1, :]] -= 1
                        coverage[neighbors[c2, :]] -= 1
                        coverage[neighbors[d1, :]] += 1
                        coverage[neighbors[d2, :]] += 1
                        code_set = reduce_code(code_set, deadline)
                        coverage = compute_coverage(code_set)
                        return code_set, coverage, True
            return code_set, coverage, False

        def add_reduce(code_set, coverage, deadline, rng):
            if not code_set:
                return code_set, coverage, False
            w = int(rng.integers(0, N))
            if w in code_set:
                return code_set, coverage, False
            new_set = set(code_set)
            new_set.add(w)
            new_set = reduce_code(new_set, deadline)
            if len(new_set) < len(code_set):
                new_cov = compute_coverage(new_set)
                return new_set, new_cov, True
            return code_set, coverage, False

        def local_search(code_set, rng, deadline):
            # Deletion + swap loop; recompute coverage after each change.
            while time.time() < deadline - 2:
                cov = compute_coverage(code_set)
                # Deletion pass
                redundant = [c for c in list(code_set)
                             if np.min(cov[neighbors[c, :]]) > 1]
                if redundant:
                    for c in redundant:
                        code_set.remove(c)
                    continue
                # Swap pass
                improved = False
                c_list = list(code_set)
                rng.shuffle(c_list)
                for c in c_list:
                    if time.time() > deadline - 2:
                        break
                    c_neighbors = neighbors[c, :]
                    essential_words = c_neighbors[cov[c_neighbors] == 1]
                    if len(essential_words) == 0:
                        continue
                    d = find_replacement(c, essential_words, code_set, rng)
                    if d is not None:
                        d_neighbors_set = set(neighbors[d, :].tolist())
                        if all(int(ew) in d_neighbors_set for ew in essential_words):
                            code_set.remove(c)
                            code_set.add(d)
                            improved = True
                            break
                if not improved:
                    if rng.random() < 0.3:
                        # Perturbation: add a random word and reduce
                        w = int(rng.integers(0, N))
                        if w not in code_set:
                            code_set.add(w)
                            code_set = reduce_code(code_set, deadline)
                    else:
                        break
            return code_set

        best_code = None
        best_size = float('inf')
        modes = [0, 1, 2, 1, 0, 2, 1, 0, 2, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1]

        for i in range(20):
            if time.time() > deadline - 10:
                break
            rng = np.random.default_rng(42 + i * 13)
            mode = modes[i % len(modes)]

            code_set = greedy_init(rng, mode)
            code_set = reduce_code(code_set, deadline)
            code_set = local_search(code_set, rng, deadline)

            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = set(code_set)

            if best_size <= 486:
                break

            # Second pass on the best code found so far (inspiration idea)
            if best_code is not None and i > 0 and time.time() < deadline - 15:
                code2 = local_search(set(best_code), rng, deadline)
                if len(code2) < best_size:
                    best_size = len(code2)
                    best_code = set(code2)
                    if best_size <= 486:
                        break

        if best_code is None:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            best_code = set()
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=n - 4):
                    word = h + list(tail)
                    val = 0
                    for i, d in enumerate(word):
                        val += d * (q ** i)
                    best_code.add(val)

        # Final high-precision polish pass
        if best_code is not None and time.time() < deadline - 3:
            code_set = set(best_code)
            coverage = compute_coverage(code_set)
            rng = np.random.default_rng(999)
            for _ in range(600):
                if time.time() > deadline - 3:
                    break
                r = rng.random()
                if r < 0.6:
                    _, coverage, was_improved = try_single_swap(
                        code_set, coverage, deadline, rng)
                    if was_improved:
                        coverage = compute_coverage(code_set)
                elif r < 0.85:
                    _, coverage, was_improved = try_double_swap(
                        code_set, coverage, deadline, rng)
                    if was_improved:
                        coverage = compute_coverage(code_set)
                else:
                    _, coverage, was_improved = add_reduce(
                        code_set, coverage, deadline, rng)
                    if was_improved:
                        coverage = compute_coverage(code_set)
            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = code_set

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
        N = q ** n
        deadline = time.time() + 110.0
        powers = [q ** i for i in range(n)]
        neigh_size = 1 + n * (q - 1)
        neighbors = np.empty((N, neigh_size), dtype=np.int32)
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
        gains = np.full(N, neigh_size, dtype=np.int32)
        code = set()
        while np.any(uncovered) and time.time() < deadline:
            max_gain = int(np.max(gains))
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            w = int(rng.choice(candidates))
            code.add(w)
            uncovered[neighbors[w, :]] = False
            gains[neighbors[w, :]] -= 1

        final_code = []
        for w in code:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            final_code.append(word)
        return final_code
