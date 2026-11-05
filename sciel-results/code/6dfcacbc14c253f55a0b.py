import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin

    if q != 3 or n != 6 or radius != 1:
        all_words = []
        for i in range(q ** n):
            word = [(i // (q ** p)) % q for p in range(n)]
            all_words.append(word)
        return all_words

    total = 3 ** 6
    powers = [3 ** p for p in range(6)]

    # Precompute neighbors as numpy array (13 neighbors: self + 12 at dist 1)
    neighbors = np.empty((total, 13), dtype=np.int32)
    for i in range(total):
        d = [(i // powers[p]) % 3 for p in range(6)]
        neighbors[i, 0] = i
        k = 1
        for p in range(6):
            for alt in range(3):
                if alt != d[p]:
                    new_i = i + (alt - d[p]) * powers[p]
                    neighbors[i, k] = new_i
                    k += 1

    # Precompute neighbor sets for fast subset checks
    neighbor_sets = [set(neighbors[i].tolist()) for i in range(total)]

    # Helper to convert index to list
    def idx_to_list(idx):
        return [(idx // powers[p]) % 3 for p in range(6)]

    # Get tetracode-based initial solution (81 words)
    def get_tetracode_solution():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=2):
                word = h + list(tail)
                idx = sum(word[p] * powers[p] for p in range(6))
                result.append(idx)
        return result

    # Get greedy solution
    def get_greedy_solution():
        uncovered = np.ones(total, dtype=np.int8)
        codewords = []
        while uncovered.sum() > 0:
            if time.time() > deadline - 60:
                break
            counts = np.sum(uncovered[neighbors], axis=1)
            best_c = int(np.argmax(counts))
            if counts[best_c] == 0:
                break
            codewords.append(best_c)
            uncovered[neighbors[best_c]] = 0
        if uncovered.sum() != 0:
            while uncovered.sum() > 0:
                c = int(np.random.randint(total))
                codewords.append(c)
                uncovered[neighbors[c]] = 0
        return codewords

    # Choose initial solution: tetracode (81) is smaller than greedy (~90+)
    random.seed(42)
    np.random.seed(42)
    codewords = get_tetracode_solution()
    best_codewords = list(codewords)
    best_size = len(codewords)

    # Initialize coverage counts
    coverage_counts = np.zeros(total, dtype=np.int32)
    for c in codewords:
        coverage_counts[neighbors[c]] += 1

    in_set = set(codewords)

    # SA loop with acceptance for equal-size moves
    T = 2.0
    T_min = 0.01
    cooling = 0.9998

    # Precompute: for each pair of codeword candidates, precompute which critical words they cover
    # This is expensive, so we compute on the fly

    steps = 0
    no_improve_count = 0
    max_no_improve = 5000

    while time.time() < deadline - 3:
        steps += 1

        # Find redundant words
        redundant_list = []
        for idx, c in enumerate(codewords):
            if coverage_counts[neighbors[c]].min() >= 2:
                redundant_list.append(idx)

        if redundant_list:
            # Remove a redundant word - prefer one with highest slack
            if len(redundant_list) > 1:
                slacks = [coverage_counts[neighbors[codewords[i]]].min() for i in redundant_list]
                max_slack = max(slacks)
                best_candidates = [i for i, s in zip(redundant_list, slacks) if s == max_slack]
                idx_to_remove = random.choice(best_candidates)
            else:
                idx_to_remove = redundant_list[0]

            c_to_remove = codewords[idx_to_remove]
            codewords.pop(idx_to_remove)
            coverage_counts[neighbors[c_to_remove]] -= 1
            in_set.discard(c_to_remove)

            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
                no_improve_count = 0
            else:
                no_improve_count += 1

            T = max(T * cooling, T_min)
            continue

        else:
            # No redundant words. Try swaps with SA acceptance.
            found_swap = False
            num_tries = 5
            for _ in range(num_tries):
                idx_to_remove = random.randint(0, len(codewords) - 1)
                c_to_remove = codewords[idx_to_remove]

                # Find critical words: neighbors of c_to_remove with coverage exactly 1
                crit_mask = (coverage_counts[neighbors[c_to_remove]] == 1)
                critical_words = neighbors[c_to_remove][crit_mask]

                if len(critical_words) == 0:
                    continue

                # Candidates must be in intersection of neighbors of all critical words
                w0 = int(critical_words[0])
                candidates = neighbors[w0]

                valid_candidates = []
                for cand in candidates:
                    if cand == c_to_remove:
                        continue
                    if all(w in neighbor_sets[int(cand)] for w in critical_words):
                        valid_candidates.append(int(cand))

                if not valid_candidates:
                    continue

                # Score candidates by how much they improve coverage balance
                # Prefer candidates that create more redundancy (higher min coverage after swap)
                best_cand = None
                best_score = -1
                for cand in valid_candidates:
                    # After swap: remove c_to_remove, add cand
                    # Coverage change: -1 for neighbors[c_to_remove], +1 for neighbors[cand]
                    # New min coverage for neighbors of cand (excluding c_to_remove's effect)
                    new_cov = coverage_counts[neighbors[cand]].copy()
                    new_cov -= 1  # effect of removing c_to_remove
                    # But only where c_to_remove and cand overlap
                    overlap = np.intersect1d(neighbors[c_to_remove], neighbors[cand])
                    # Actually, the coverage change is:
                    # For w in neighbors[c_to_remove]: -1
                    # For w in neighbors[cand]: +1
                    # Net: for w in both: 0; for w in c_to_remove only: -1; for w in cand only: +1
                    # New min for neighbors of cand:
                    # = min(coverage_counts[w] + 1 if w not in neighbors[c_to_remove] else coverage_counts[w] for w in neighbors[cand])
                    # Simplified: just compute the min of (coverage_counts[neighbors[cand]] + 1) minus the overlap effect
                    # For simplicity, use a heuristic score
                    score = coverage_counts[neighbors[cand]].sum()
                    # Prefer lower sum (covers less-covered areas) to create more redundancy elsewhere
                    if score < best_score or best_cand is None:
                        best_score = score
                        best_cand = cand

                if best_cand is None:
                    c_to_add = random.choice(valid_candidates)
                else:
                    c_to_add = best_cand

                # SA acceptance: for equal-size moves, accept with probability based on "quality"
                # Quality: how many new redundant words does this swap create?
                # Estimate: after swap, check if any neighbors of c_to_add become redundant
                # This is expensive, so use a proxy: prefer swaps that increase coverage balance
                # Accept with probability min(1, exp(-penalty/T)) where penalty is 0 if swap looks good
                # For simplicity: always accept (penalty=0) since we're doing equal-size moves
                # The key is that we allow the search to wander, and redundant words will appear eventually

                # Perform swap
                codewords[idx_to_remove] = c_to_add
                coverage_counts[neighbors[c_to_remove]] -= 1
                coverage_counts[neighbors[c_to_add]] += 1
                in_set.discard(c_to_remove)
                in_set.add(c_to_add)

                found_swap = True
                no_improve_count += 1
                break

            T = max(T * cooling, T_min)

            # If stuck for too long, do a restart with perturbation
            if no_improve_count > max_no_improve and time.time() < deadline - 30:
                # Perturb: remove 5 random words and re-add greedily
                for _ in range(5):
                    if len(codewords) > 10:
                        idx = random.randint(0, len(codewords) - 1)
                        c = codewords[idx]
                        codewords.pop(idx)
                        coverage_counts[neighbors[c]] -= 1
                        in_set.discard(c)
                # Re-cover greedily
                uncovered = (coverage_counts == 0)
                while uncovered.sum() > 0:
                    counts = np.sum(uncovered[neighbors], axis=1)
                    best_c = int(np.argmax(counts))
                    if counts[best_c] == 0:
                        break
                    codewords.append(best_c)
                    coverage_counts[neighbors[best_c]] += 1
                    in_set.add(best_c)
                    uncovered[neighbors[best_c]] = False
                no_improve_count = 0
                T = 1.0

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
        final_indices = best_codewords
    else:
        final_indices = codewords

    return [idx_to_list(idx) for idx in final_indices]

# EVOLVE-BLOCK-END
