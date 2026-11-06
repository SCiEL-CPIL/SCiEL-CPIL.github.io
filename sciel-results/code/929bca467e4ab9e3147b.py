import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 87.0

    T_min = 128
    T_max = 256
    N_max = 129
    R = T_max - T_min + 1  # 129

    best_A = None
    best_max = float('inf')

    def compute_counts(ind):
        """Compute unordered pair-sum representation counts for [T_min, T_max]."""
        conv = np.convolve(ind, ind, mode='full')
        counts = np.empty(R, dtype=np.int32)
        for i, s in enumerate(range(T_min, T_max + 1)):
            c_ord = conv[s]
            if s % 2 == 0:
                half = s // 2
                n_eq = 1 if (half < N_max and ind[half] == 1) else 0
            else:
                n_eq = 0
            counts[i] = (c_ord + n_eq) // 2
        return counts

    def max_count(counts):
        return int(np.max(counts))

    def is_valid(counts):
        return bool(np.all(counts > 0))

    def repair(A_set, ind, counts):
        """Repair gaps by adding elements with best gain."""
        for _ in range(200):
            if is_valid(counts):
                break
            gaps = np.where(counts == 0)[0]
            best_y = None
            best_gain = -1
            for y in range(N_max):
                if ind[y] == 1:
                    continue
                gain = 0
                for gi in gaps:
                    s = T_min + gi
                    a = s - y
                    if 0 <= a < N_max and (ind[a] == 1 or a == y):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_y = y
            if best_y is not None:
                A_set.add(best_y)
                ind[best_y] = 1
                counts = compute_counts(ind)
            else:
                break
        return is_valid(counts)

    def swap_counts(counts_cur, A_cur, x, y):
        """Efficiently compute new counts after swapping x out, y in."""
        c_new = counts_cur.copy()
        
        # Subtract contributions of x
        for a in A_cur:
            if a == x:
                continue
            s = a + x
            if T_min <= s <= T_max:
                c_new[s - T_min] -= 1
        s_xx = 2 * x
        if T_min <= s_xx <= T_max:
            c_new[s_xx - T_min] -= 1

        # Add contributions of y
        for a in A_cur:
            if a == x:
                continue
            s = a + y
            if T_min <= s <= T_max:
                c_new[s - T_min] += 1
        s_yy = 2 * y
        if T_min <= s_yy <= T_max:
            c_new[s_yy - T_min] += 1
            
        return c_new

    def find_elements_in_high_sums(counts_cur, A_cur, threshold):
        """Find elements that appear in many sums with count >= threshold."""
        # For each element a in A, count how many sums s in [T_min, T_max] have count >= threshold
        # and a is part of some pair summing to s
        elem_score = {}
        high_sums = np.where(counts_cur >= threshold)[0]
        for gi in high_sums:
            s = T_min + gi
            # Find all pairs (a,b) in A such that a+b=s, a<=b
            for a in A_cur:
                b = s - a
                if a <= b and b in A_cur:
                    if a not in elem_score:
                        elem_score[a] = 0
                    if b not in elem_score:
                        elem_score[b] = 0
                    elem_score[a] += 1
                    elem_score[b] += 1
        return elem_score

    def try_improve(A_initial, time_budget):
        nonlocal best_A, best_max
        A_set = set(A_initial)
        ind = np.zeros(N_max, dtype=np.int32)
        for x in A_set:
            ind[x] = 1
        counts = compute_counts(ind)

        if not is_valid(counts):
            if not repair(A_set, ind, counts):
                return

        current_max = max_count(counts)
        if current_max < best_max:
            best_max = current_max
            best_A = set(A_set)

        A_cur = set(A_set)
        ind_cur = ind.copy()
        counts_cur = counts.copy()
        max_cur = current_max

        # Phase 0: Greedy removal — try removing elements to reduce |A| and peak
        removal_deadline = start_time + time_budget * 0.2
        while time.time() < removal_deadline:
            improved = False
            for x in list(A_cur):
                if len(A_cur) <= 20:
                    break
                ind_test = ind_cur.copy()
                ind_test[x] = 0
                c_test = compute_counts(ind_test)
                if is_valid(c_test):
                    m_test = max_count(c_test)
                    if m_test <= max_cur:
                        A_cur.discard(x)
                        ind_cur = ind_test
                        counts_cur = c_test
                        max_cur = m_test
                        improved = True
                        if max_cur < best_max:
                            best_max = max_cur
                            best_A = set(A_cur)
                        break
            if not improved:
                break

        # Phase 1: Random swaps with incremental count update
        swap_deadline = start_time + time_budget * 0.6
        no_improve = 0

        while time.time() < swap_deadline:
            if len(A_cur) == 0:
                break
            A_list_cur = list(A_cur)
            x = random.choice(A_list_cur)
            y = random.randint(0, N_max - 1)
            if y == x or ind_cur[y] == 1:
                continue

            c_new = swap_counts(counts_cur, A_cur, x, y)

            if not is_valid(c_new):
                no_improve += 1
                if no_improve > 1500:
                    break
                continue

            m_new = max_count(c_new)

            if m_new < max_cur:
                A_cur.discard(x)
                A_cur.add(y)
                ind_cur[x] = 0
                ind_cur[y] = 1
                counts_cur = c_new
                max_cur = m_new
                no_improve = 0
                if max_cur < best_max:
                    best_max = max_cur
                    best_A = set(A_cur)
            elif m_new == max_cur:
                if random.random() < 0.2:
                    A_cur.discard(x)
                    A_cur.add(y)
                    ind_cur[x] = 0
                    ind_cur[y] = 1
                    counts_cur = c_new
            else:
                no_improve += 1
                if no_improve > 1500:
                    break

        # Phase 2: Targeted swaps focusing on reducing peak using CSP-inspired heuristic
        phase2_deadline = min(start_time + time_budget * 0.9, deadline - 5.0)
        no_improve2 = 0

        while time.time() < phase2_deadline:
            if len(A_cur) == 0:
                break

            max_c = max_count(counts_cur)
            if max_c <= 1:
                break

            # Find elements that contribute to high-multiplicity sums
            elem_score = find_elements_in_high_sums(counts_cur, A_cur, max_c)
            
            if not elem_score:
                no_improve2 += 1
                if no_improve2 > 300:
                    break
                # Fallback to random
                x = random.choice(list(A_cur))
            else:
                # Pick element with highest score (Most Constrained Variable heuristic)
                x = max(elem_score.keys(), key=lambda e: elem_score[e])

            best_y = None
            best_score = None

            # Try multiple candidates for y
            candidates = []
            for _ in range(15):
                y = random.randint(0, N_max - 1)
                if y == x or ind_cur[y] == 1:
                    continue
                if y not in candidates:
                    candidates.append(y)
            
            # Also try structured candidates based on high sums
            high_sums = np.where(counts_cur >= max_c)[0]
            for gi in high_sums[:10]:
                s = T_min + gi
                for a in A_cur:
                    if a == x:
                        continue
                    y = s - a
                    if 0 <= y < N_max and y != x and ind_cur[y] == 0 and y not in candidates:
                        candidates.append(y)
            
            for y in candidates[:20]:
                c_new = swap_counts(counts_cur, A_cur, x, y)

                if not is_valid(c_new):
                    continue

                m_new = max_count(c_new)
                n_high = int(np.sum(c_new >= m_new))
                score = (m_new, n_high)

                if best_score is None or score < best_score:
                    best_score = score
                    best_y = y

            if best_y is None:
                no_improve2 += 1
                if no_improve2 > 300:
                    break
                continue

            # Apply best swap
            ind_new = ind_cur.copy()
            ind_new[x] = 0
            ind_new[best_y] = 1
            c_new = compute_counts(ind_new)

            if not is_valid(c_new):
                no_improve2 += 1
                if no_improve2 > 300:
                    break
                continue

            m_new = max_count(c_new)

            if m_new < max_cur:
                A_cur.discard(x)
                A_cur.add(best_y)
                ind_cur = ind_new
                counts_cur = c_new
                max_cur = m_new
                no_improve2 = 0
                if max_cur < best_max:
                    best_max = max_cur
                    best_A = set(A_cur)
            elif m_new == max_cur:
                old_peak_count = int(np.sum(counts_cur >= max_cur))
                new_peak_count = int(np.sum(c_new >= max_cur))
                if new_peak_count < old_peak_count or random.random() < 0.15:
                    A_cur.discard(x)
                    A_cur.add(best_y)
                    ind_cur = ind_new
                    counts_cur = c_new
            else:
                no_improve2 += 1
                if no_improve2 > 300:
                    break

        # Phase 3: Final polish - try removals and targeted swaps
        polish_deadline = min(start_time + time_budget, deadline - 3.0)
        while time.time() < polish_deadline:
            # Try removals
            removed = False
            for x in list(A_cur):
                if len(A_cur) <= 15:
                    break
                ind_test = ind_cur.copy()
                ind_test[x] = 0
                c_test = compute_counts(ind_test)
                if is_valid(c_test):
                    m_test = max_count(c_test)
                    if m_test < max_cur:
                        A_cur.discard(x)
                        ind_cur = ind_test
                        counts_cur = c_test
                        max_cur = m_test
                        removed = True
                        if max_cur < best_max:
                            best_max = max_cur
                            best_A = set(A_cur)
                        break
            
            if removed:
                continue
            
            # Try targeted swap on peak sums
            max_c = max_count(counts_cur)
            if max_c <= 1:
                break
            
            high_sums = np.where(counts_cur >= max_c)[0]
            if len(high_sums) == 0:
                break
            
            # Pick a random high sum and try to reduce it
            gi = random.choice(high_sums)
            s = T_min + gi
            pairs_at_s = []
            for a in A_cur:
                b = s - a
                if a <= b and b in A_cur:
                    pairs_at_s.append((a, b))
            
            if not pairs_at_s:
                continue
            
            # Try swapping one element from a pair
            a, b = random.choice(pairs_at_s)
            x = random.choice([a, b])
            
            best_y = None
            best_score = None
            for _ in range(10):
                y = random.randint(0, N_max - 1)
                if y == x or ind_cur[y] == 1:
                    continue
                c_new = swap_counts(counts_cur, A_cur, x, y)
                if not is_valid(c_new):
                    continue
                m_new = max_count(c_new)
                n_high = int(np.sum(c_new >= m_new))
                score = (m_new, n_high)
                if best_score is None or score < best_score:
                    best_score = score
                    best_y = y
            
            if best_y is not None:
                ind_new = ind_cur.copy()
                ind_new[x] = 0
                ind_new[best_y] = 1
                c_new = compute_counts(ind_new)
                if is_valid(c_new):
                    m_new = max_count(c_new)
                    if m_new < max_cur:
                        A_cur.discard(x)
                        A_cur.add(best_y)
                        ind_cur = ind_new
                        counts_cur = c_new
                        max_cur = m_new
                        if max_cur < best_max:
                            best_max = max_cur
                            best_A = set(A_cur)

    # Try multiple diverse starting points
    starts = []
    # Seed: upper half
    starts.append(list(range(64, 129)))
    # Random sizes
    for sz in [30, 35, 40, 45, 50, 55, 60]:
        for trial in range(2):
            random.seed(42 + trial * 1000 + sz)
            starts.append(random.sample(range(0, N_max), sz))
    # Structured starts
    starts.append(list(range(0, 129, 2)))  # evens
    starts.append(list(range(1, 129, 2)))  # odds
    starts.append(list(range(32, 129, 2)))  # evens upper
    starts.append(list(range(33, 129, 2)))  # odds upper

    for i, s in enumerate(starts):
        remaining = deadline - time.time() - 3.0
        if remaining < 2.0:
            break
        time_budget = min(remaining * 0.7, 10.0)
        if i > 5:
            time_budget = min(time_budget, 5.0)
        try_improve(s, time_budget)

    if best_A is None:
        return list(range(64, 129))
    return list(best_A)
