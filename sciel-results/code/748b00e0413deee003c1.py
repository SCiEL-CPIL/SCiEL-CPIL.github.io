import numpy as np
import time
from itertools import product

def construct(q, n, radius):
    deadline = time.time() + 118.0

    if q == 3 and n == 8 and radius == 1:
        N = 3 ** 8  # 6561
        powers = np.array([3 ** i for i in range(n)], dtype=np.int64)

        # Precompute neighbors: for each word, the 17 words within distance 1
        # neighbors[w] = array of 17 indices
        neighbors = np.empty((N, 17), dtype=np.int32)
        neighbors[:, 0] = np.arange(N)
        for i in range(n):
            for delta in [1, 2]:
                # For each word, compute the neighbor with digit i changed by delta
                d = (np.arange(N) // powers[i]) % q
                new_d = (d + delta) % q
                diff = (new_d - d).astype(np.int64)
                neighbors[:, 1 + 2 * i + (delta - 1)] = (np.arange(N) + diff * powers[i]).astype(np.int32)

        def to_digits(w):
            return [(w // powers[i]) % q for i in range(n)]

        def from_digits(digits):
            return sum(d * powers[i] for i, d in enumerate(digits))

        def build_initial_code(seed):
            """Build a structured initial code using a block-based approach."""
            rng = np.random.default_rng(seed)
            code = set()
            uncovered = np.ones(N, dtype=np.bool_)

            # Phase 1: Greedy with random tie-breaking
            while np.any(uncovered) and time.time() < deadline - 30:
                gains = uncovered[neighbors].sum(axis=1)
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                w = rng.choice(candidates)
                code.add(w)
                uncovered[neighbors[w]] = False

            return code

        def build_structured_code(seed):
            """Build code using a structured approach: split into 4+4 positions."""
            rng = np.random.default_rng(seed)
            code = set()

            # Use a 2-level construction:
            # Level 1: Cover positions 0-3 with a small code
            # Level 2: Cover positions 4-7 with a small code
            # But this gives a product which may not be optimal.
            # Instead, use a smarter approach.

            # Start with greedy but bias toward structured positions
            uncovered = np.ones(N, dtype=np.bool_)

            # First, add codewords that cover many "new" words
            # Use a priority based on how many uncovered words they cover
            for _ in range(200):
                if not np.any(uncovered):
                    break
                gains = uncovered[neighbors].sum(axis=1)
                if gains.max() == 0:
                    break
                # Pick from top candidates with some randomness
                top_k = min(50, N)
                top_idx = np.argsort(gains)[-top_k:]
                w = rng.choice(top_idx)
                code.add(w)
                uncovered[neighbors[w]] = False

            # Finish with pure greedy
            while np.any(uncovered) and time.time() < deadline - 30:
                gains = uncovered[neighbors].sum(axis=1)
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                w = rng.choice(candidates)
                code.add(w)
                uncovered[neighbors[w]] = False

            return code

        def reduce_code(code_set):
            """Remove redundant codewords."""
            if not code_set:
                return code_set

            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c]] += 1

            code_list = list(code_set)
            changed = True
            while changed and time.time() < deadline - 10:
                changed = False
                if not code_list:
                    break
                code_arr = np.array(code_list, dtype=np.int32)
                mins = np.min(coverage[neighbors[code_arr]], axis=1)
                redundant_mask = mins > 1
                redundant_indices = np.where(redundant_mask)[0]

                if len(redundant_indices) > 0:
                    # Remove all redundant at once for speed
                    for idx in redundant_indices:
                        c = code_arr[idx]
                        code_set.remove(c)
                        coverage[neighbors[c]] -= 1
                    code_list = list(code_set)
                    changed = True

            return code_set

        def add_remove_search(code_set, max_iter=200):
            """Add-Remove local search."""
            if not code_set:
                return code_set

            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c]] += 1

            rng = np.random.default_rng()
            code_list = list(code_set)
            improvements = 0

            for _ in range(max_iter):
                if time.time() > deadline - 12:
                    break

                # Find singly-covered words
                single_mask = (coverage == 1)
                if not np.any(single_mask):
                    break

                single_covered = np.where(single_mask)[0]

                # Compute gain: for each potential new codeword, how many singly-covered words it covers
                gain = np.zeros(N, dtype=np.int32)
                for s in single_covered:
                    gain[neighbors[s]] += 1

                # Filter out words already in code
                code_arr = np.array(code_list, dtype=np.int32)
                gain[code_arr] = 0

                max_gain = np.max(gain)
                if max_gain == 0:
                    break

                # Select among top candidates
                threshold = max_gain * 0.85
                candidates = np.where(gain >= threshold)[0]
                if len(candidates) == 0:
                    candidates = np.where(gain == max_gain)[0]

                new_c = int(rng.choice(candidates))

                # Add new_c
                code_set.add(new_c)
                coverage[neighbors[new_c]] += 1
                code_list.append(new_c)

                # Try to remove redundant codewords
                code_arr = np.array(code_list, dtype=np.int32)
                mins = np.min(coverage[neighbors[code_arr]], axis=1)
                redundant_mask = mins > 1
                redundant_indices = np.where(redundant_mask)[0]

                if len(redundant_indices) > 0:
                    # Remove the one with lowest "essentiality"
                    # Essentiality = number of words it uniquely covers
                    best_idx = -1
                    best_ess = float('inf')
                    for idx in redundant_indices:
                        c = code_arr[idx]
                        ess = np.sum(coverage[neighbors[c]] == 1)
                        if ess < best_ess:
                            best_ess = ess
                            best_idx = idx
                    c_to_remove = code_arr[best_idx]
                    code_set.remove(c_to_remove)
                    coverage[neighbors[c_to_remove]] -= 1
                    code_list.remove(c_to_remove)
                    improvements += 1
                else:
                    # Revert
                    code_set.remove(new_c)
                    coverage[neighbors[new_c]] -= 1
                    code_list.remove(new_c)

            return code_set

        def try_swaps(code_set, max_iter=40):
            """Try 1-for-1 swaps."""
            if not code_set:
                return code_set

            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c]] += 1

            rng = np.random.default_rng()
            swaps_done = 0

            for _ in range(max_iter):
                if time.time() > deadline - 12:
                    break

                code_list = list(code_set)
                if not code_list:
                    break

                # Pick a codeword to try replacing
                idx = rng.choice(len(code_list))
                c = code_list[idx]
                c_neigh = neighbors[c]

                # Words only covered by c
                essential = c_neigh[coverage[c_neigh] == 1]

                if len(essential) == 0:
                    # c is redundant
                    code_set.remove(c)
                    coverage[neighbors[c]] -= 1
                    continue

                # Find candidates that cover all essential words
                if len(essential) == 1:
                    candidates = neighbors[essential[0]]
                else:
                    # Intersection of neighbors of all essential words
                    cand_set = set(neighbors[essential[0]].tolist())
                    for ew in essential[1:]:
                        cand_set &= set(neighbors[ew].tolist())
                    if not cand_set:
                        continue
                    candidates = np.array(list(cand_set), dtype=np.int32)

                # Try each candidate
                found = False
                for d in candidates:
                    d = int(d)
                    if d in code_set:
                        continue
                    d_neigh_set = set(neighbors[d].tolist())
                    if not all(ew in d_neigh_set for ew in essential):
                        continue
                    # Check no gaps: words covered by c but not d must have other coverage
                    bad = False
                    for w in c_neigh:
                        if w not in d_neigh_set and coverage[w] <= 1:
                            bad = True
                            break
                    if not bad:
                        code_set.remove(c)
                        code_set.add(d)
                        coverage[neighbors[c]] -= 1
                        coverage[neighbors[d]] += 1
                        swaps_done += 1
                        found = True
                        break

            return code_set

        def two_for_one(code_set, max_iter=30):
            """Try 2-for-1 reduction."""
            if not code_set or len(code_set) < 2:
                return code_set

            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c]] += 1

            rng = np.random.default_rng()

            for _ in range(max_iter):
                if time.time() > deadline - 12:
                    break

                code_list = list(code_set)
                if len(code_list) < 2:
                    break

                idx1, idx2 = rng.choice(len(code_list), size=2, replace=False)
                c1, c2 = int(code_list[idx1]), int(code_list[idx2])

                n1_set = set(neighbors[c1].tolist())
                n2_set = set(neighbors[c2].tolist())

                # Words that would be uncovered if we remove both c1 and c2
                # and need to be covered by a single replacement
                all_ess = set()
                for w in n1_set | n2_set:
                    if coverage[w] <= 2:
                        all_ess.add(w)

                if not all_ess:
                    # Both redundant
                    code_set.remove(c1)
                    code_set.remove(c2)
                    coverage[neighbors[c1]] -= 1
                    coverage[neighbors[c2]] -= 1
                    continue

                if len(all_ess) > 5:
                    continue

                # Find a single word covering all_ess
                ess_list = list(all_ess)
                cand_set = set(neighbors[ess_list[0]].tolist())
                for ew in ess_list[1:]:
                    cand_set &= set(neighbors[ew].tolist())
                    if not cand_set:
                        break
                if not cand_set:
                    continue

                found = False
                for d in cand_set:
                    d = int(d)
                    if d in code_set:
                        continue
                    d_neigh_set = set(neighbors[d].tolist())
                    if not all_ess.issubset(d_neigh_set):
                        continue
                    # Check no other gaps
                    bad = False
                    for w in (n1_set | n2_set) - d_neigh_set:
                        if coverage[w] <= 2:
                            bad = True
                            break
                    if not bad:
                        code_set.remove(c1)
                        code_set.remove(c2)
                        code_set.add(d)
                        coverage[neighbors[c1]] -= 1
                        coverage[neighbors[c2]] -= 1
                        coverage[neighbors[d]] += 1
                        found = True
                        break

            return code_set

        def two_for_two(code_set, max_iter=30):
            """Try 2-for-2 swap: replace two codewords with two different ones."""
            if not code_set or len(code_set) < 2:
                return code_set

            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c]] += 1

            rng = np.random.default_rng()

            for _ in range(max_iter):
                if time.time() > deadline - 12:
                    break

                code_list = list(code_set)
                if len(code_list) < 2:
                    break

                idx1, idx2 = rng.choice(len(code_list), size=2, replace=False)
                c1, c2 = int(code_list[idx1]), int(code_list[idx2])

                n1_set = set(neighbors[c1].tolist())
                n2_set = set(neighbors[c2].tolist())
                union_set = n1_set | n2_set

                # Words that would be uncovered if we remove both
                uncovered_if_removed = set()
                for w in union_set:
                    if coverage[w] <= 2:
                        uncovered_if_removed.add(w)

                if not uncovered_if_removed:
                    # Both redundant
                    code_set.remove(c1)
                    code_set.remove(c2)
                    coverage[neighbors[c1]] -= 1
                    coverage[neighbors[c2]] -= 1
                    continue

                # Try to find two new codewords d1, d2 that cover all uncovered_if_removed
                # and don't create new gaps
                # Simple heuristic: try random pairs from the neighborhood of uncovered words
                if len(uncovered_if_removed) > 10:
                    continue

                # Get candidate pool: neighbors of uncovered words
                cand_pool = set()
                for w in uncovered_if_removed:
                    cand_pool.update(neighbors[w].tolist())
                cand_pool -= code_set
                cand_pool.discard(c1)
                cand_pool.discard(c2)

                if len(cand_pool) > 100:
                    cand_list = list(rng.choice(list(cand_pool), size=100, replace=False))
                else:
                    cand_list = list(cand_pool)

                if len(cand_list) < 2:
                    continue

                # Try random pairs
                found = False
                for _ in range(20):
                    d1, d2 = rng.choice(cand_list, size=2, replace=False)
                    d1, d2 = int(d1), int(d2)

                    d1_set = set(neighbors[d1].tolist())
                    d2_set = set(neighbors[d2].tolist())
                    new_union = d1_set | d2_set

                    # Check all uncovered_if_removed are covered
                    if not uncovered_if_removed.issubset(new_union):
                        continue

                    # Check no new gaps: words in union_set but not in new_union must have other coverage
                    bad = False
                    for w in union_set - new_union:
                        if coverage[w] <= 2:
                            bad = True
                            break
                    if bad:
                        continue

                    # Accept
                    code_set.remove(c1)
                    code_set.remove(c2)
                    code_set.add(d1)
                    code_set.add(d2)
                    coverage[neighbors[c1]] -= 1
                    coverage[neighbors[c2]] -= 1
                    coverage[neighbors[d1]] += 1
                    coverage[neighbors[d2]] += 1
                    found = True
                    break

            return code_set

        # Run multiple diverse constructions
        best_code = None
        best_size = float('inf')

        strategies = [
            (42, 'greedy'), (123, 'greedy'), (456, 'structured'), (789, 'greedy'),
            (101, 'structured'), (202, 'greedy'), (303, 'structured'), (404, 'greedy'),
            (505, 'structured'), (606, 'greedy'), (707, 'structured'), (808, 'greedy'),
            (909, 'structured'), (111, 'greedy'), (222, 'structured'), (333, 'greedy'),
        ]

        for seed, strategy in strategies:
            if time.time() > deadline - 25:
                break

            if strategy == 'greedy':
                code_set = build_initial_code(seed)
            else:
                code_set = build_structured_code(seed)

            # Local search pipeline
            code_set = reduce_code(code_set)
            code_set = try_swaps(code_set, max_iter=30)
            code_set = add_remove_search(code_set, max_iter=150)
            code_set = two_for_one(code_set, max_iter=25)
            code_set = two_for_two(code_set, max_iter=25)
            code_set = try_swaps(code_set, max_iter=30)
            code_set = reduce_code(code_set)

            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = code_set

            if best_size <= 486:
                break

        if best_code is None:
            best_code = build_initial_code(0)
            best_code = reduce_code(best_code)

        # Final polish
        if best_code:
            best_code = try_swaps(best_code, max_iter=40)
            best_code = add_remove_search(best_code, max_iter=100)
            best_code = two_for_one(best_code, max_iter=20)
            best_code = two_for_two(best_code, max_iter=20)
            best_code = try_swaps(best_code, max_iter=40)
            best_code = reduce_code(best_code)

        # Convert to list of digit lists
        final_code = []
        for w in best_code:
            final_code.append(to_digits(w))

        return final_code

    else:
        # Generic fallback
        N = q ** n
        powers = np.array([q ** i for i in range(n)], dtype=np.int64)
        neigh_count = 1 + n * (q - 1)
        neighbors = np.empty((N, neigh_count), dtype=np.int32)
        neighbors[:, 0] = np.arange(N)
        for i in range(n):
            for delta in range(1, q):
                d = (np.arange(N) // powers[i]) % q
                new_d = (d + delta) % q
                diff = (new_d - d).astype(np.int64)
                neighbors[:, 1 + (i * (q - 1)) + (delta - 1)] = (np.arange(N) + diff * powers[i]).astype(np.int32)

        rng = np.random.default_rng()
        uncovered = np.ones(N, dtype=np.bool_)
        code = set()
        while np.any(uncovered) and time.time() < deadline:
            gains = uncovered[neighbors].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            w = int(rng.choice(candidates))
            code.add(w)
            uncovered[neighbors[w]] = False

        final_code = []
        for w in code:
            final_code.append([(w // powers[i]) % q for i in range(n)])
        return final_code
