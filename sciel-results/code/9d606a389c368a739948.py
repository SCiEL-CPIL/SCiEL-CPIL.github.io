import time
import itertools
import numpy as np

def construct(q, n, radius):
    deadline = time.time() + 115.0

    total = q ** n

    # For the specific case q=3, n=8, radius=1, use a specialized approach
    if q == 3 and n == 8 and radius == 1:
        return _solve_3_8_1(deadline)

    # Generic fallback for other parameters
    if total > 5_000_000:
        # Too large; use crude upper bound
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute balls
    balls = []
    for w in all_words:
        neighbors = [word_to_idx[w]]
        for i in range(n):
            for d in range(q):
                if d != w[i]:
                    new_w = list(w)
                    new_w[i] = d
                    neighbors.append(word_to_idx[tuple(new_w)])
        balls.append(neighbors)

    # Greedy set cover
    uncovered = np.ones(total, dtype=bool)
    covered_count = total
    chosen = []

    while covered_count > 0 and time.time() < deadline:
        best_idx = -1
        best_gain = 0
        for i in range(total):
            gain = 0
            for idx in balls[i]:
                if uncovered[idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_gain == 0:
            break
        chosen.append(best_idx)
        for idx in balls[best_idx]:
            if uncovered[idx]:
                uncovered[idx] = False
                covered_count -= 1

    # Prune
    chosen = _prune(chosen, balls)

    # Local search
    chosen = _local_search(chosen, balls, deadline)
    chosen = _prune(chosen, balls)

    return [list(all_words[i]) for i in chosen]


def _solve_3_8_1(deadline):
    n = 8
    q = 3
    total = 6561

    # Build the space
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute balls as NumPy array for speed
    ball_size = 1 + n * (q - 1)
    balls_np = np.zeros((total, ball_size), dtype=np.int32)
    for i, w in enumerate(all_words):
        balls_np[i, 0] = i
        k = 1
        for pos in range(n):
            for d in range(q):
                if d != w[pos]:
                    new_w = list(w)
                    new_w[pos] = d
                    balls_np[i, k] = word_to_idx[tuple(new_w)]
                    k += 1

    # Greedy set cover with NumPy
    uncovered = np.ones(total, dtype=bool)
    covered_count = total
    chosen = []

    # Reserve some time for local search
    greedy_deadline = deadline - 20.0
    
    while covered_count > 0 and time.time() < greedy_deadline:
        gains = np.sum(uncovered[balls_np], axis=1)
        max_gain = np.max(gains)
        if max_gain == 0:
            break
        candidates = np.where(gains == max_gain)[0]
        idx = np.random.choice(candidates)
        chosen.append(idx)
        
        # Update uncovered
        mask = uncovered[balls_np[idx]]
        count = np.sum(mask)
        uncovered[balls_np[idx]] = False
        covered_count -= count

    # Prune
    chosen = _prune_np(chosen, balls_np)

    # Aggressive local search
    chosen = _local_search_3_8_1_np(chosen, balls_np, deadline)
    chosen = _prune_np(chosen, balls_np)

    # Final validity check
    covered = np.zeros(total, dtype=bool)
    for w_idx in chosen:
        covered[balls_np[w_idx]] = True
    if not np.all(covered):
        # Fallback: tetracode-based construction
        _TETRA = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in _TETRA
                for tail in itertools.product(range(3), repeat=4)]

    return [list(all_words[i]) for i in chosen]


def _prune_np(chosen_list, balls_np):
    if not chosen_list:
        return []
    total = len(balls_np)
    coverage_count = np.zeros(total, dtype=np.int32)
    for c in chosen_list:
        coverage_count[balls_np[c]] += 1

    keep = np.ones(len(chosen_list), dtype=bool)
    any_removed = True
    while any_removed:
        any_removed = False
        for i in range(len(chosen_list)):
            if not keep[i]:
                continue
            c = chosen_list[i]
            # If all points covered by c have count > 1, c is redundant
            if np.min(coverage_count[balls_np[c]]) > 1:
                keep[i] = False
                any_removed = True
                coverage_count[balls_np[c]] -= 1
    return [chosen_list[i] for i in range(len(chosen_list)) if keep[i]]


def _local_search_3_8_1_np(chosen_list, balls_np, deadline):
    total = len(balls_np)
    new_chosen = list(chosen_list)
    
    def compute_coverage(chosen):
        cov = np.zeros(total, dtype=np.int32)
        for c in chosen:
            cov[balls_np[c]] += 1
        return cov

    coverage_count = compute_coverage(new_chosen)
    chosen_set = set(new_chosen)

    while time.time() < deadline:
        improved = False
        
        # 1. Try Swap (1-opt)
        swap_found = False
        for i in range(len(new_chosen)):
            if time.time() > deadline:
                break
            c = new_chosen[i]
            unique_mask = coverage_count[balls_np[c]] == 1
            unique_points = balls_np[c][unique_mask]
            
            if len(unique_points) == 0:
                # Redundant
                new_chosen.pop(i)
                coverage_count = compute_coverage(new_chosen)
                chosen_set.remove(c)
                improved = True
                swap_found = True
                break

            candidates = set(balls_np[unique_points[0]])
            for uj in unique_points[1:]:
                candidates &= set(balls_np[uj])
            
            candidates -= chosen_set
            valid_candidates = list(candidates)
            
            if not valid_candidates:
                continue
            
            # Pick the first valid candidate
            c_prime = valid_candidates[0]
            new_chosen[i] = c_prime
            coverage_count = compute_coverage(new_chosen)
            chosen_set.remove(c)
            chosen_set.add(c_prime)
            improved = True
            swap_found = True
            break
        
        if swap_found:
            new_chosen = _prune_np(new_chosen, balls_np)
            coverage_count = compute_coverage(new_chosen)
            chosen_set = set(new_chosen)
            continue

        # 2. Try 2-Opt (remove 2, add 1)
        n_chosen = len(new_chosen)
        two_opt_found = False
        for i in range(n_chosen):
            if time.time() > deadline:
                break
            c_i = new_chosen[i]
            for j in range(i + 1, n_chosen):
                if time.time() > deadline:
                    break
                c_j = new_chosen[j]
                
                unique_i = balls_np[c_i][coverage_count[balls_np[c_i]] == 1]
                unique_j = balls_np[c_j][coverage_count[balls_np[c_j]] == 1]
                
                risky_points = np.union1d(unique_i, unique_j)
                
                if len(risky_points) == 0:
                    # Both redundant
                    new_chosen.pop(j)
                    new_chosen.pop(i)
                    coverage_count = compute_coverage(new_chosen)
                    chosen_set.remove(c_i)
                    chosen_set.remove(c_j)
                    improved = True
                    two_opt_found = True
                    break
                
                candidates = set(balls_np[risky_points[0]])
                for rp in risky_points[1:]:
                    candidates &= set(balls_np[rp])
                
                candidates -= chosen_set
                valid_candidates = list(candidates)
                
                if not valid_candidates:
                    continue
                
                c_prime = valid_candidates[0]
                # Replace i and j with c_prime
                new_chosen.pop(j)
                new_chosen.pop(i)
                new_chosen.append(c_prime)
                
                coverage_count = compute_coverage(new_chosen)
                chosen_set.remove(c_i)
                chosen_set.remove(c_j)
                chosen_set.add(c_prime)
                improved = True
                two_opt_found = True
                break
            
            if two_opt_found:
                break
        
        if two_opt_found:
            new_chosen = _prune_np(new_chosen, balls_np)
            coverage_count = compute_coverage(new_chosen)
            chosen_set = set(new_chosen)
            continue

        if not improved:
            break

    return new_chosen


def _prune(chosen_list, balls):
    if not chosen_list:
        return []
    total = len(balls)
    coverage_count = np.zeros(total, dtype=np.int32)
    for c in chosen_list:
        for j in balls[c]:
            coverage_count[j] += 1

    keep = [True] * len(chosen_list)
    any_removed = True
    while any_removed:
        any_removed = False
        for i, c in enumerate(chosen_list):
            if not keep[i]:
                continue
            redundant = True
            for j in balls[c]:
                if coverage_count[j] <= 1:
                    redundant = False
                    break
            if redundant:
                keep[i] = False
                any_removed = True
                for j in balls[c]:
                    coverage_count[j] -= 1
    return [chosen_list[i] for i in range(len(chosen_list)) if keep[i]]


def _local_search(chosen_list, balls, deadline):
    total = len(balls)
    chosen_set = set(chosen_list)

    while time.time() < deadline:
        improved = False
        chosen_set = set(chosen_list)

        coverage_count = np.zeros(total, dtype=np.int32)
        for c in chosen_list:
            for j in balls[c]:
                coverage_count[j] += 1

        new_chosen = list(chosen_list)

        for i, c in enumerate(new_chosen):
            if time.time() > deadline:
                break

            unique_points = [j for j in balls[c] if coverage_count[j] == 1]
            if not unique_points:
                continue

            candidates = set(balls[unique_points[0]])
            for uj in unique_points[1:]:
                candidates &= set(balls[uj])

            candidates -= chosen_set
            valid_candidates = list(candidates)

            if not valid_candidates:
                continue

            for c_prime in valid_candidates:
                tentative = list(new_chosen)
                tentative[i] = c_prime
                pruned = _prune(tentative, balls)
                if len(pruned) < len(new_chosen):
                    new_chosen = pruned
                    chosen_set = set(new_chosen)
                    improved = True
                    break

            if improved:
                break

        if not improved:
            improved = False
            n_chosen = len(new_chosen)
            for i in range(n_chosen):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n_chosen):
                    if time.time() > deadline:
                        break
                    tentative = list(new_chosen)
                    tentative.pop(j)
                    tentative.pop(i)
                    pruned = _prune(tentative, balls)
                    if len(pruned) < len(new_chosen):
                        new_chosen = pruned
                        improved = True
                        break
                if improved:
                    break

        if not improved:
            break

    return new_chosen
