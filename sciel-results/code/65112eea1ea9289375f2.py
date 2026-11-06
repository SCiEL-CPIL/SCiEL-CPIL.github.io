import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82.0

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

    def try_improve(A_initial):
        nonlocal best_A, best_max
        A_set = set(A_initial)
        ind = np.zeros(N_max, dtype=np.int32)
        for x in A_set:
            ind[x] = 1
        counts = compute_counts(ind)

        if not is_valid(counts):
            # Repair by adding elements; exhaustive candidate scan for best gain
            for _ in range(80):
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

            if not is_valid(counts):
                return

        current_max = max_count(counts)
        if current_max < best_max:
            best_max = current_max
            best_A = set(A_set)

        # Local search with swaps (fast numpy counts)
        A_cur = set(A_set)
        ind_cur = ind.copy()
        counts_cur = counts.copy()
        max_cur = current_max

        no_improve = 0
        while time.time() < deadline - 2.0:
            if len(A_cur) == 0:
                break
            A_list_cur = list(A_cur)
            x = random.choice(A_list_cur)
            y = random.randint(0, N_max - 1)
            if y == x or ind_cur[y] == 1:
                continue

            ind_new = ind_cur.copy()
            ind_new[x] = 0
            ind_new[y] = 1
            c_new = compute_counts(ind_new)

            if not is_valid(c_new):
                no_improve += 1
                if no_improve > 300:
                    break
                continue

            m_new = max_count(c_new)

            if m_new < max_cur:
                A_cur = A_cur - {x}
                A_cur.add(y)
                ind_cur = ind_new
                counts_cur = c_new
                max_cur = m_new
                no_improve = 0
                if max_cur < best_max:
                    best_max = max_cur
                    best_A = set(A_cur)
            elif m_new == max_cur:
                if random.random() < 0.15:
                    A_cur = A_cur - {x}
                    A_cur.add(y)
                    ind_cur = ind_new
                    counts_cur = c_new
            else:
                no_improve += 1
                if no_improve > 300:
                    break

    # Diverse starting points: structured + random sizes
    starts = []
    starts.append(list(range(64, 129)))                    # upper half (seed)
    starts.append(list(range(0, 129, 2)))                  # evens
    starts.append(list(range(0, 129, 2)) + [128])          # evens + 128
    starts.append(list(range(1, 129, 2)))                  # odds
    starts.append(list(range(0, 129, 3)) + list(range(2, 129, 3)))  # all mod-3 classes

    for sz in [35, 40, 45, 50, 55, 60]:
        for trial in range(3):
            random.seed(42 + trial * 1000 + sz)
            starts.append(random.sample(range(0, N_max), sz))

    for s in starts:
        if time.time() > deadline - 3.0:
            break
        try_improve(s)

    if best_A is None:
        return list(range(64, 129))
    return list(best_A)
