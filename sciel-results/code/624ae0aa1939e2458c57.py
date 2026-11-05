import numpy as np
import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    if q != 3 or n != 6 or radius != 1:
        all_words = []
        for i in range(q ** n):
            word = [(i // (q ** p)) % q for p in range(n)]
            all_words.append(word)
        return all_words

    total = 3 ** 6
    powers = [3 ** p for p in range(6)]

    # Precompute neighbors: each word has 13 neighbors (itself + 12 at Hamming dist 1)
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

    # Build incidence matrix A: A[w, c] = 1 if w is in ball(c), i.e., w in neighbors[c]
    # Since neighbors is symmetric, A = neighbors.T essentially
    # A has shape (total, total): row w, col c -> 1 if c covers w
    A = np.zeros((total, total), dtype=np.int8)
    for c in range(total):
        A[neighbors[c], c] = 1

    def idx_to_list(idx):
        return [(idx // powers[p]) % 3 for p in range(6)]

    # Phase 1: Greedy set cover for initial solution
    uncovered = np.ones(total, dtype=np.int8)
    greedy_codewords = []
    
    while uncovered.sum() > 0:
        if time.time() > deadline - 60:
            break
        counts = A.T @ uncovered  # For each codeword, how many uncovered words it covers
        best_c = int(np.argmax(counts))
        if counts[best_c] == 0:
            break
        greedy_codewords.append(best_c)
        uncovered[neighbors[best_c]] = 0

    if uncovered.sum() != 0:
        while uncovered.sum() > 0:
            c = int(np.random.randint(total))
            greedy_codewords.append(c)
            uncovered[neighbors[c]] = 0

    best_size = len(greedy_codewords)
    best_codewords = list(greedy_codewords)

    # Phase 2: Try ILP for a better solution
    if time.time() < deadline - 40:
        try:
            # ILP: minimize sum(x_c) subject to A @ x >= 1 for all w, x_c in {0,1}
            # We can use scipy's milp with binary variables
            c_obj = np.ones(total)  # minimize sum of x
            # Constraints: A @ x >= 1 for all rows (words)
            # LinearConstraint(A, lb=np.ones(total), ub=np.inf)
            constraints = LinearConstraint(A, lb=np.ones(total), ub=np.inf)
            integrality = np.ones(total)  # all binary
            bounds = Bounds(lb=0, ub=1)
            
            result = milp(c=c_obj, constraints=constraints, integrality=integrality, bounds=bounds)
            
            if result.status == 0 and result.x is not None:
                ilp_solution = [int(c) for c in range(total) if result.x[c] > 0.5]
                
                # Verify validity
                cov = np.zeros(total, dtype=np.int8)
                for c in ilp_solution:
                    cov[neighbors[c]] = 1
                if cov.sum() == total:
                    if len(ilp_solution) < best_size:
                        best_size = len(ilp_solution)
                        best_codewords = ilp_solution
        except Exception:
            pass

    # Phase 3: Local search refinement on the best solution
    codewords = list(best_codewords)
    
    # Initialize coverage counts
    coverage_counts = np.zeros(total, dtype=np.int32)
    for c in codewords:
        coverage_counts[neighbors[c]] += 1

    in_set = set(codewords)
    random.seed(42)
    np.random.seed(42)

    # Precompute neighbor sets for fast subset checks
    neighbor_sets = [set(neighbors[i].tolist()) for i in range(total)]

    # Local search: iterated removal + swap
    T = 1.0
    T_min = 0.005
    cooling = 0.9998

    while time.time() < deadline - 3:
        # Try to remove redundant codewords
        removed_any = False
        for idx in range(len(codewords) - 1, -1, -1):
            c = codewords[idx]
            if coverage_counts[neighbors[c]].min() >= 2:
                codewords.pop(idx)
                coverage_counts[neighbors[c]] -= 1
                in_set.discard(c)
                removed_any = True
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
        
        if removed_any:
            T = max(T * cooling, T_min)
            continue

        # No redundant words. Try 1-swap moves.
        found_swap = False
        num_tries = 5
        for _ in range(num_tries):
            idx_to_remove = random.randint(0, len(codewords) - 1)
            c_to_remove = codewords[idx_to_remove]

            # Critical words: neighbors with coverage exactly 1
            crit_mask = (coverage_counts[neighbors[c_to_remove]] == 1)
            critical_words = neighbors[c_to_remove][crit_mask]

            if len(critical_words) == 0:
                continue

            # Find valid replacements: must cover all critical words
            w0 = int(critical_words[0])
            candidates = neighbors[w0]

            valid_candidates = []
            for cand in candidates:
                cand_int = int(cand)
                if cand_int == c_to_remove:
                    continue
                # Check if cand covers all critical words
                if all(w in neighbor_sets[cand_int] for w in critical_words):
                    valid_candidates.append(cand_int)

            if not valid_candidates:
                continue

            # Score candidates: prefer those that maximize minimum coverage after swap
            c_to_add = random.choice(valid_candidates)

            # Perform swap
            codewords[idx_to_remove] = c_to_add
            coverage_counts[neighbors[c_to_remove]] -= 1
            coverage_counts[neighbors[c_to_add]] += 1
            in_set.discard(c_to_remove)
            in_set.add(c_to_add)
            found_swap = True
            break

        T = max(T * cooling, T_min)

    # Final cleanup
    improved = True
    while improved:
        improved = False
        for idx in range(len(codewords) - 1, -1, -1):
            c = codewords[idx]
            if coverage_counts[neighbors[c]].min() >= 2:
                codewords.pop(idx)
                coverage_counts[neighbors[c]] -= 1
                improved = True
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                break

    final_indices = best_codewords if len(best_codewords) < len(codewords) else codewords
    return [idx_to_list(idx) for idx in final_indices]
