import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 118.0

    total = q ** n
    ball_size = 1 + n * (q - 1)

    # Precompute base
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute balls as boolean masks
    cov = np.zeros((total, total), dtype=np.bool_)
    cov_indices = [None] * total  # store list of covered indices for each codeword

    for c in range(total):
        temp = c
        digits = [0] * n
        for j in range(n - 1, -1, -1):
            digits[j] = temp % q
            temp //= q

        covered_words = [c]
        cov[c, c] = True

        for coord in range(n):
            orig = digits[coord]
            for v in range(q):
                if v == orig:
                    continue
                w = c + (v - orig) * base[coord]
                cov[c, w] = True
                covered_words.append(w)

        cov_indices[c] = np.array(covered_words, dtype=np.int32)

    # Precompute: for each word, which codewords cover it (inverse)
    # This helps find candidates that cover specific words
    # cov_inv[w] = list of c such that cov[c, w]
    # Actually, let's use the cov matrix directly for candidate finding

    # Greedy construction
    def greedy_build(seed=None):
        if seed is not None:
            random.seed(seed)

        covered = np.zeros(total, dtype=np.uint8)
        code = []
        uncovered = total

        while uncovered > 0:
            # Compute gains vectorized
            gains = cov @ (1 - covered)

            max_gain = np.max(gains)
            if max_gain == 0:
                break

            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1:
                best_c = random.choice(candidates)
            else:
                best_c = candidates[0]

            code.append(best_c)
            covered[cov[best_c]] = 1
            uncovered = np.count_nonzero(covered == 0)

        return code

    # Local search with 1-for-1 swaps, 2-for-1 improvements, and periodic greedy reduction
    def local_search(code_list, time_budget=5.0):
        in_code = np.zeros(total, dtype=np.bool_)
        for c in code_list:
            in_code[c] = True

        cnt = np.zeros(total, dtype=np.int32)
        for c in code_list:
            cnt[cov[c]] += 1

        t_start = time.time()
        best_size = len(code_list)
        best_code = list(code_list)
        current_code = list(code_list)

        def try_delete():
            nonlocal best_size, best_code, current_code
            changed = True
            while changed:
                changed = False
                to_remove = []
                for c in current_code:
                    if np.min(cnt[cov[c]]) >= 2:
                        to_remove.append(c)

                if to_remove:
                    for c in to_remove:
                        current_code.remove(c)
                        in_code[c] = False
                        cnt[cov[c]] -= 1
                        changed = True
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)

        try_delete()

        no_success = 0
        max_streak = 800
        greedy_reduction_interval = 200  # every 200 failed moves, try greedy reduction

        def greedy_reduction_pass():
            """Run a quick greedy reduction on the current code to find new deletions."""
            nonlocal best_size, best_code, current_code, in_code, cnt
            # Try to find a codeword whose removal can be compensated by greedy addition
            # Actually, let's try: for each codeword, check if removing it and greedily
            # covering the newly uncovered words with fewer than 1 additional word is possible.
            # This is complex. Instead, let's do a simpler thing:
            # Try to find two codewords that can be replaced by one.
            
            # Find pairs where union of their unique coverage can be covered by one codeword
            # This is O(k^2 * total) which might be too slow.
            # Instead, do a targeted greedy: pick a random codeword, remove it,
            # then greedily cover the gaps with minimal additions.
            
            if not current_code:
                return False
            
            # Try removing a random codeword and seeing if we can cover gaps with 0 or 1 additions
            for attempt in range(20):
                c_remove = random.choice(current_code)
                critical_mask = (cnt == 1) & cov[c_remove]
                critical_words = np.where(critical_mask)[0]
                
                if len(critical_words) == 0:
                    # Can delete for free
                    current_code.remove(c_remove)
                    in_code[c_remove] = False
                    cnt[cov[c_remove]] -= 1
                    try_delete()
                    return True
                
                # Find if there's a single codeword that covers all critical words
                candidates_mask = np.all(cov[:, critical_words], axis=1)
                candidates_mask &= ~in_code
                candidate_indices = np.where(candidates_mask)[0]
                
                if len(candidate_indices) > 0:
                    # 1-for-1 swap found
                    c_add = candidate_indices[0]
                    current_code.remove(c_remove)
                    in_code[c_remove] = False
                    current_code.append(c_add)
                    in_code[c_add] = True
                    cnt[cov[c_remove]] -= 1
                    cnt[cov[c_add]] += 1
                    try_delete()
                    return True
            
            return False

        while time.time() - t_start < time_budget:
            if len(current_code) == 0:
                break

            # 1. Try deletion
            deleted = False
            for c in current_code:
                if np.min(cnt[cov[c]]) >= 2:
                    current_code.remove(c)
                    in_code[c] = False
                    cnt[cov[c]] -= 1
                    deleted = True
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
                    break

            if deleted:
                continue

            # 2. Try 1-for-1 swap
            c_remove = random.choice(current_code)
            critical_mask = (cnt == 1) & cov[c_remove]

            if not np.any(critical_mask):
                current_code.remove(c_remove)
                in_code[c_remove] = False
                cnt[cov[c_remove]] -= 1
                continue

            critical_words = np.where(critical_mask)[0]
            candidates_mask = np.all(cov[:, critical_words], axis=1)
            candidates_mask &= ~in_code
            candidate_indices = np.where(candidates_mask)[0]

            if len(candidate_indices) > 0:
                c_add = random.choice(candidate_indices)
                current_code.remove(c_remove)
                in_code[c_remove] = False
                current_code.append(c_add)
                in_code[c_add] = True
                cnt[cov[c_remove]] -= 1
                cnt[cov[c_add]] += 1
                no_success = 0
                try_delete()
            else:
                no_success += 1

            # 3. Try 2-for-1 improvement (remove 1, add 2)
            if no_success > 300 and time.time() - t_start < time_budget - 0.3 and current_code:
                c_remove = random.choice(current_code)
                critical_mask = (cnt == 1) & cov[c_remove]
                critical_words = np.where(critical_mask)[0]

                if len(critical_words) > 0:
                    found = False
                    # Use structured search: find candidates that cover subsets of critical words
                    # First, find all codewords that cover at least one critical word
                    any_critical_mask = np.any(cov[:, critical_words], axis=1)
                    any_critical_mask &= ~in_code
                    any_critical_indices = np.where(any_critical_mask)[0]
                    
                    if len(any_critical_indices) > 0:
                        # Sample from these candidates
                        sample_size = min(100, len(any_critical_indices))
                        sampled = random.sample(list(any_critical_indices), sample_size)
                        
                        for i in range(len(sampled)):
                            c1 = sampled[i]
                            # What critical words does c1 cover?
                            c1_critical = cov[c1, critical_words]
                            remaining_critical = critical_words[~c1_critical]
                            
                            if len(remaining_critical) == 0:
                                # c1 alone covers all, but we already checked this in 1-for-1
                                continue
                            
                            # Find c2 that covers remaining_critical
                            c2_mask = np.all(cov[:, remaining_critical], axis=1)
                            c2_mask &= ~in_code
                            c2_mask[c1] = False
                            c2_indices = np.where(c2_mask)[0]
                            
                            if len(c2_indices) > 0:
                                c2 = c2_indices[0]
                                current_code.remove(c_remove)
                                in_code[c_remove] = False
                                current_code.append(c1)
                                in_code[c1] = True
                                current_code.append(c2)
                                in_code[c2] = True
                                cnt[cov[c_remove]] -= 1
                                cnt[cov[c1]] += 1
                                cnt[cov[c2]] += 1
                                no_success = 0
                                try_delete()
                                found = True
                                break
                    
                    if not found:
                        no_success += 1
                else:
                    no_success += 1

            # 4. Periodic greedy reduction pass
            if no_success > greedy_reduction_interval:
                if greedy_reduction_pass():
                    no_success = 0
                else:
                    no_success = 0  # Reset to avoid immediate re-trigger

            # 5. Perturb if stuck
            if no_success > max_streak:
                rand_c = random.randint(0, total - 1)
                if not in_code[rand_c]:
                    current_code.append(rand_c)
                    in_code[rand_c] = True
                    cnt[cov[rand_c]] += 1
                    try_delete()
                no_success = 0

        return best_code, best_size

    best_code_overall = []
    best_size_overall = total

    # First greedy + LS with generous time
    greedy_code = greedy_build(seed=42)
    best_code, best_size = local_search(greedy_code, time_budget=20.0)
    best_code_overall = best_code
    best_size_overall = best_size

    # Multi-restart with different seeds
    seed = 0
    while time.time() < deadline - 15.0 and best_size_overall > 165:
        seed += 1
        greedy_code = greedy_build(seed=seed)
        time_left = deadline - 15.0 - time.time()
        tb = min(12.0, max(3.0, time_left / 6.0))
        best_code, best_size = local_search(greedy_code, time_budget=tb)
        if best_size < best_size_overall:
            best_size_overall = best_size
            best_code_overall = best_code

    # Final polish: run local search on the best solution with remaining time
    if time.time() < deadline - 5.0 and best_code_overall:
        time_left = deadline - 5.0 - time.time()
        if time_left > 2.0:
            best_code, best_size = local_search(best_code_overall, time_budget=time_left)
            if best_size < best_size_overall:
                best_size_overall = best_size
                best_code_overall = best_code

    # Convert to list of lists
    result = []
    for idx in best_code_overall:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)

    return result
