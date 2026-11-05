import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    def get_direct_sum():
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [[0] * n]

    if q == 3 and n == 6 and radius == 1:
        total = 3 ** n
        powers = [3 ** p for p in range(n)]

        # Precompute neighbors as numpy array (each word has 13 neighbors: itself + 12 at distance 1)
        neighbors = np.empty((total, 13), dtype=np.int32)
        for i in range(total):
            d = [(i // powers[p]) % 3 for p in range(n)]
            neighbors[i, 0] = i
            k = 1
            for p in range(n):
                for alt in range(3):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        neighbors[i, k] = new_i
                        k += 1

        # Precompute neighbor sets for fast membership checks
        neighbor_sets = [set(neighbors[i].tolist()) for i in range(total)]

        # Precompute: for each word, which codewords cover it (inverse neighbor map)
        # Actually we'll compute this on the fly using coverage_counts

        def idx_to_list(idx):
            return [(idx // powers[p]) % 3 for p in range(n)]

        # Phase 1: Greedy Set Cover using numpy for speed
        uncovered = np.ones(total, dtype=np.int8)
        codewords = []

        while uncovered.sum() > 0:
            if time.time() > deadline - 60:
                break
            # Vectorized: count how many uncovered neighbors each candidate covers
            counts = np.sum(uncovered[neighbors], axis=1)
            best_c = int(np.argmax(counts))
            if counts[best_c] == 0:
                break
            codewords.append(best_c)
            uncovered[neighbors[best_c]] = 0

        # Ensure full coverage
        if uncovered.sum() != 0:
            while uncovered.sum() > 0:
                c = int(np.random.randint(total))
                codewords.append(c)
                uncovered[neighbors[c]] = 0

        # Initialize coverage counts
        coverage_counts = np.zeros(total, dtype=np.int32)
        for c in codewords:
            coverage_counts[neighbors[c]] += 1

        in_set = set(codewords)
        best_codewords = list(codewords)
        best_size = len(codewords)

        random.seed(42)
        np.random.seed(42)

        # Phase 2: Local search with simulated annealing
        T = 1.0
        T_min = 0.005
        cooling = 0.9998
        no_improve = 0
        max_no_improve = 8000

        while time.time() < deadline - 5.0 and no_improve < max_no_improve:
            # Step 1: Try to remove redundant codewords
            redundant_indices = []
            for idx, c in enumerate(codewords):
                if coverage_counts[neighbors[c]].min() >= 2:
                    redundant_indices.append(idx)

            if redundant_indices:
                # Remove one redundant word - prefer the one with highest min coverage (most redundant)
                best_idx = redundant_indices[0]
                best_min = -1
                for idx in redundant_indices:
                    c = codewords[idx]
                    mc = int(coverage_counts[neighbors[c]].min())
                    if mc > best_min:
                        best_min = mc
                        best_idx = idx

                c_to_remove = codewords[best_idx]
                codewords.pop(best_idx)
                coverage_counts[neighbors[c_to_remove]] -= 1
                in_set.discard(c_to_remove)

                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                    no_improve = 0
                else:
                    no_improve += 1
                T = max(T * cooling, T_min)
                continue

            # Step 2: No redundant words. Try swaps and 2-for-1 moves.
            found_move = False
            num_tries = 8
            for _ in range(num_tries):
                # Try 1-for-1 swap first
                if random.random() < 0.7:
                    idx_to_remove = random.randint(0, len(codewords) - 1)
                    c_to_remove = codewords[idx_to_remove]

                    # Find critical words: neighbors of c_to_remove with coverage exactly 1
                    crit_mask = (coverage_counts[neighbors[c_to_remove]] == 1)
                    critical_words = neighbors[c_to_remove][crit_mask]

                    if len(critical_words) == 0:
                        continue

                    # Candidates must cover ALL critical words
                    w0 = int(critical_words[0])
                    candidates = neighbors[w0]

                    valid_candidates = []
                    for cand in candidates:
                        if cand == c_to_remove:
                            continue
                        # Check if cand covers all critical words
                        if all(w in neighbor_sets[int(cand)] for w in critical_words):
                            valid_candidates.append(int(cand))

                    if not valid_candidates:
                        continue

                    # Pick a candidate - prefer one that maximizes minimum coverage of its neighbors
                    best_cand = valid_candidates[0]
                    best_min_cov = -1
                    for cand in valid_candidates:
                        min_cov = int(coverage_counts[neighbors[cand]].min())
                        if min_cov > best_min_cov:
                            best_min_cov = min_cov
                            best_cand = cand

                    # Perform swap
                    codewords[idx_to_remove] = best_cand
                    coverage_counts[neighbors[c_to_remove]] -= 1
                    coverage_counts[neighbors[best_cand]] += 1
                    in_set.discard(c_to_remove)
                    in_set.add(best_cand)

                    found_move = True
                    break
                else:
                    # Try 2-for-1: remove 2 codewords, add 1
                    if len(codewords) < 3:
                        continue
                    i1 = random.randint(0, len(codewords) - 1)
                    i2 = random.randint(0, len(codewords) - 1)
                    if i2 == i1:
                        continue
                    c1 = codewords[i1]
                    c2 = codewords[i2]

                    # Critical words: those covered by exactly one of {c1, c2}
                    cov_c1 = coverage_counts[neighbors[c1]]
                    cov_c2 = coverage_counts[neighbors[c2]]
                    # A word is critical if it's in N(c1) xor N(c2) and has coverage 1 overall
                    # Actually: after removing both, a word w is uncovered if it was only covered by c1 or only by c2
                    # w is in N(c1) and not in N(c2) and coverage_counts[w] == 1 (only c1 covers it)
                    # OR w is in N(c2) and not in N(c1) and coverage_counts[w] == 1 (only c2 covers it)
                    # OR w is in both and coverage_counts[w] == 2 (both c1 and c2 cover it, and no one else)
                    
                    n1 = neighbors[c1]
                    n2 = neighbors[c2]
                    
                    # Words only covered by c1: in n1, not in n2, coverage == 1
                    in_n2_set = set(n2.tolist())
                    crit_from_c1 = [w for w in n1.tolist() if w not in in_n2_set and coverage_counts[w] == 1]
                    
                    # Words only covered by c2: in n2, not in n1, coverage == 1
                    in_n1_set = set(n1.tolist())
                    crit_from_c2 = [w for w in n2.tolist() if w not in in_n1_set and coverage_counts[w] == 1]
                    
                    # Words covered by both: in n1 and in n2, coverage == 2
                    crit_from_both = [w for w in n1.tolist() if w in in_n2_set and coverage_counts[w] == 2]
                    
                    all_critical = crit_from_c1 + crit_from_c2 + crit_from_both
                    
                    if not all_critical:
                        continue
                    
                    # Find a single codeword that covers all critical words
                    w0 = all_critical[0]
                    candidates = neighbors[w0]
                    
                    valid_candidates = []
                    for cand in candidates:
                        if cand == c1 or cand == c2:
                            continue
                        if all(w in neighbor_sets[int(cand)] for w in all_critical):
                            valid_candidates.append(int(cand))
                    
                    if not valid_candidates:
                        continue
                    
                    # Pick best candidate
                    best_cand = valid_candidates[0]
                    best_min_cov = -1
                    for cand in valid_candidates:
                        min_cov = int(coverage_counts[neighbors[cand]].min())
                        if min_cov > best_min_cov:
                            best_min_cov = min_cov
                            best_cand = cand
                    
                    # Perform 2-for-1: remove c1 and c2, add best_cand
                    # Remove the higher index first to keep indices valid
                    if i1 > i2:
                        i1, i2 = i2, i1
                    codewords.pop(i1)
                    codewords.pop(i2)
                    codewords.append(best_cand)
                    
                    coverage_counts[neighbors[c1]] -= 1
                    coverage_counts[neighbors[c2]] -= 1
                    coverage_counts[neighbors[best_cand]] += 1
                    in_set.discard(c1)
                    in_set.discard(c2)
                    in_set.add(best_cand)
                    
                    found_move = True
                    break

            if not found_move:
                no_improve += 1
            else:
                no_improve = 0

            T = max(T * cooling, T_min)

        # Final cleanup: iteratively remove redundant words
        improved = True
        while improved:
            improved = False
            for idx in range(len(codewords)):
                c = codewords[idx]
                if coverage_counts[neighbors[c]].min() >= 2:
                    codewords.pop(idx)
                    coverage_counts[neighbors[c]] -= 1
                    improved = True
                    if len(codewords) < best_size:
                        best_size = len(codewords)
                        best_codewords = list(codewords)
                    break

        # Use the best found
        if len(best_codewords) < len(codewords):
            codewords = best_codewords
        elif len(codewords) < len(best_codewords):
            best_codewords = list(codewords)

        # Validate
        covered = np.zeros(total, dtype=np.int8)
        for c in codewords:
            covered[neighbors[c]] = 1
        if covered.all():
            final_codewords = [idx_to_list(c) for c in codewords]
            ds = get_direct_sum()
            if len(final_codewords) <= len(ds):
                return final_codewords
            else:
                return ds

        ds = get_direct_sum()
        return ds

    return get_direct_sum()
