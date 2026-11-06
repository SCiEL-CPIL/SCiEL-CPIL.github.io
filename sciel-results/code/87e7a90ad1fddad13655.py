import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for 90s limit

    T_min = 128
    T_max = 256
    N_max = 129
    R = T_max - T_min + 1  # 129

    best_A = None
    best_max = float('inf')
    best_sum_sq = float('inf')

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

    def get_metrics(counts):
        max_c = int(np.max(counts))
        sum_sq = int(np.sum(counts * counts))
        return max_c, sum_sq

    def is_valid(counts):
        return np.all(counts > 0)

    def repair(A_set, ind, counts):
        """Attempt to repair gaps by adding elements."""
        for _ in range(50):
            gaps = np.where(counts == 0)[0]
            if len(gaps) == 0:
                break
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
        return A_set, ind, counts

    def try_improve(A_initial, time_limit=10.0):
        nonlocal best_A, best_max, best_sum_sq
        
        A_set = set(A_initial)
        ind = np.zeros(N_max, dtype=np.int32)
        for x in A_set:
            ind[x] = 1
        counts = compute_counts(ind)

        if not is_valid(counts):
            A_set, ind, counts = repair(A_set, ind, counts)
            if not is_valid(counts):
                return

        current_max, current_sq = get_metrics(counts)
        
        # Update global best
        if current_max < best_max or (current_max == best_max and current_sq < best_sum_sq):
            best_max = current_max
            best_sum_sq = current_sq
            best_A = set(A_set)

        A_cur = set(A_set)
        ind_cur = ind.copy()
        counts_cur = counts.copy()
        max_cur, sq_cur = get_metrics(counts_cur)
        
        # Simulated Annealing
        T = 5.0
        T_min_temp = 0.01
        cooling = 0.995
        no_improve = 0
        max_no_improve = 100
        
        end_time = time.time() + time_limit
        while time.time() < end_time:
            if len(A_cur) == 0:
                break
            
            x = random.choice(list(A_cur))
            y = random.randint(0, N_max - 1)
            
            if y == x or ind_cur[y] == 1:
                continue

            ind_new = ind_cur.copy()
            ind_new[x] = 0
            ind_new[y] = 1
            c_new = compute_counts(ind_new)

            if not is_valid(c_new):
                no_improve += 1
                if no_improve > max_no_improve:
                    # Reheat
                    T = max(T_min_temp, T * 1.2)
                    no_improve = 0
                    # Optional: small kick
                    continue
            else:
                m_new, sq_new = get_metrics(c_new)
                delta_max = m_new - max_cur
                delta_sq = sq_new - sq_cur
                
                accept = False
                if delta_max < 0:
                    accept = True
                elif delta_max == 0:
                    if delta_sq < 0:
                        accept = True
                    elif T > 0.1:
                        if random.random() < np.exp(-delta_sq / (50.0 * T)):
                            accept = True
                else:
                    if T > 0.1:
                        if random.random() < np.exp(-delta_max / T):
                            accept = True

                if accept:
                    A_cur = A_cur - {x}
                    A_cur.add(y)
                    ind_cur = ind_new
                    counts_cur = c_new
                    max_cur, sq_cur = m_new, sq_new
                    no_improve = 0
                    
                    if max_cur < best_max or (max_cur == best_max and sq_cur < best_sum_sq):
                        best_max = max_cur
                        best_sum_sq = sq_cur
                        best_A = set(A_cur)
                else:
                    no_improve += 1
                    if no_improve > max_no_improve:
                        T = max(T_min_temp, T * 1.5)
                        no_improve = 0

            T = max(T_min_temp, T * cooling)

    # Generate starts
    starts = []
    
    # 1. Random sets
    random.seed(42)
    for sz in [45, 55]:
        for _ in range(3):
            starts.append(random.sample(range(N_max), sz))

    # 2. Structured
    starts.append(list(range(0, N_max, 2)))
    starts.append(list(range(1, N_max, 2)))

    # Run search
    time_per_start = 12.0
    for s in starts:
        if time.time() > deadline - 5.0:
            break
        try_improve(s, time_limit=time_per_start)

    if best_A is None:
        return list(range(64, 129))
    return list(best_A)
