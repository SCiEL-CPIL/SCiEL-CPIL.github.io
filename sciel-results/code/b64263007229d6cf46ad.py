"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np
import random


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius
    'radius'. Uses greedy set-cover then fast local search with swap moves.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    if q == 3 and n == 7 and radius == 1:
        # Precompute all words
        all_words = list(itertools.product(range(3), repeat=7))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        total_words = len(all_words)

        # Precompute neighbors: for each word, the 15 words at distance <= 1
        neighbors = np.zeros((total_words, 15), dtype=np.int32)
        for i, w in enumerate(all_words):
            neighbors[i, 0] = i
            idx = 1
            for pos in range(n):
                for val in range(q):
                    if val != w[pos]:
                        nw = list(w)
                        nw[pos] = val
                        neighbors[i, idx] = word_to_idx[tuple(nw)]
                        idx += 1

        # Phase 1: Greedy set cover
        cov_counts = np.zeros(total_words, dtype=np.int32)
        codeword_list = []

        while time.time() < deadline - 40:
            uncovered = (cov_counts == 0)
            if not uncovered.any():
                break
            best_gain = -1
            best_cw = -1
            if len(codeword_list) < 50:
                candidates = list(range(total_words))
            else:
                candidates = random.sample(range(total_words), 500)
            for cw in candidates:
                gain = 0
                for nb in neighbors[cw]:
                    if cov_counts[nb] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_cw = cw
            if best_gain <= 0:
                break
            codeword_list.append(best_cw)
            for nb in neighbors[best_cw]:
                cov_counts[nb] += 1

        # Remove redundancies
        current = list(codeword_list)
        cur_cov = np.zeros(total_words, dtype=np.int32)
        for cw in current:
            for nb in neighbors[cw]:
                cur_cov[nb] += 1

        def try_remove_redundant(sol, cov_counts):
            removed = 0
            i = 0
            while i < len(sol):
                if time.time() > deadline - 25:
                    break
                cw = sol[i]
                can_remove = True
                for nb in neighbors[cw]:
                    if cov_counts[nb] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for nb in neighbors[cw]:
                        cov_counts[nb] -= 1
                    sol.pop(i)
                    removed += 1
                else:
                    i += 1
            return sol, cov_counts, removed

        current, cur_cov, _ = try_remove_redundant(current, cur_cov)
        best_solution = list(current)
        best_size = len(current)

        # Phase 2: Simulated Annealing
        temp = 5.0
        cooling = 0.999
        min_temp = 0.1
        no_improve = 0

        while time.time() < deadline - 10 and temp > min_temp:
            if no_improve > 300:
                break

            cur_size = len(current)
            move_type = random.random()
            improved = False

            if move_type < 0.3 and cur_size > 0:
                # Try remove
                idx = random.randint(0, cur_size - 1)
                cw = current[idx]
                can_remove = True
                for nb in neighbors[cw]:
                    if cur_cov[nb] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for nb in neighbors[cw]:
                        cur_cov[nb] -= 1
                    current.pop(idx)
                    improved = True
                    no_improve = 0
                    if len(current) < best_size:
                        best_size = len(current)
                        best_solution = list(current)
            elif move_type < 0.6:
                # Try add
                new_cw = random.randint(0, total_words - 1)
                if new_cw not in current:
                    current.append(new_cw)
                    for nb in neighbors[new_cw]:
                        cur_cov[nb] += 1
                    current, cur_cov, _ = try_remove_redundant(current, cur_cov)
                    if len(current) < best_size:
                        best_size = len(current)
                        best_solution = list(current)
                    no_improve = 0
                    improved = True
            else:
                # Swap
                if cur_size > 0:
                    idx = random.randint(0, cur_size - 1)
                    old_cw = current[idx]
                    critical = []
                    for nb in neighbors[old_cw]:
                        if cur_cov[nb] == 1:
                            critical.append(nb)
                    if not critical:
                        for nb in neighbors[old_cw]:
                            cur_cov[nb] -= 1
                        current.pop(idx)
                        improved = True
                        no_improve = 0
                        if len(current) < best_size:
                            best_size = len(current)
                            best_solution = list(current)
                    else:
                        for attempt in range(30):
                            if time.time() > deadline - 10:
                                break
                            new_cw = random.randint(0, total_words - 1)
                            if new_cw == old_cw or new_cw in current:
                                continue
                            new_covered = set(neighbors[new_cw].tolist())
                            if all(w in new_covered for w in critical):
                                for nb in neighbors[old_cw]:
                                    cur_cov[nb] -= 1
                                for nb in neighbors[new_cw]:
                                    cur_cov[nb] += 1
                                current[idx] = new_cw
                                improved = True
                                no_improve = 0
                                current, cur_cov, _ = try_remove_redundant(current, cur_cov)
                                if len(current) < best_size:
                                    best_size = len(current)
                                    best_solution = list(current)
                                break

            if not improved:
                no_improve += 1
            temp *= cooling

        # Final cleanup
        final = list(best_solution)
        final_cov = np.zeros(total_words, dtype=np.int32)
        for cw in final:
            for nb in neighbors[cw]:
                final_cov[nb] += 1
        final, final_cov, _ = try_remove_redundant(final, final_cov)

        if final_cov.min() >= 1:
            return [list(all_words[idx]) for idx in final]

        return [list(all_words[idx]) for idx in codeword_list]
    
    # General fallback: greedy set cover
    all_words = list(itertools.product(range(q), repeat=n))
    
    def get_coverage_general(codeword):
        covered = set()
        covered.add(codeword)
        for i in range(n):
            for val in range(q):
                if val != codeword[i]:
                    new_word = list(codeword)
                    new_word[i] = val
                    covered.add(tuple(new_word))
        return covered
    
    codewords = []
    covered = set()
    
    while time.time() < deadline - 10:
        best_cw = None
        best_count = -1
        best_new_covered = set()
        
        for candidate in all_words:
            new_covered = get_coverage_general(candidate) - covered
            count = len(new_covered)
            if count > best_count:
                best_count = count
                best_cw = candidate
                best_new_covered = new_covered
        
        if best_count <= 0:
            break
        
        codewords.append(best_cw)
        covered.update(best_new_covered)
        
        if len(covered) == len(all_words):
            break
    
    # Redundancy removal
    i = 0
    while i < len(codewords) and time.time() < deadline - 5:
        cw_to_remove = codewords[i]
        coverage_of_cw = get_coverage_general(cw_to_remove)
        
        can_remove = True
        for w in coverage_of_cw:
            is_covered_by_other = False
            for j, other_cw in enumerate(codewords):
                if i == j:
                    continue
                if w in get_coverage_general(other_cw):
                    is_covered_by_other = True
                    break
            if not is_covered_by_other:
                can_remove = False
                break
        
        if can_remove:
            codewords.pop(i)
        else:
            i += 1
    
    return [list(cw) for cw in codewords]
# EVOLVE-BLOCK-END
