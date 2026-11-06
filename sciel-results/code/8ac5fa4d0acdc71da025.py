import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    T_min = 128
    T_max = 256
    N_max = 129
    R = T_max - T_min + 1  # 129

    def compute_counts(ind):
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

    def is_valid(counts):
        return bool(np.all(counts > 0))

    def get_metrics(counts):
        return int(np.max(counts)), float(np.sum(counts * counts))

    def try_swap(x, y, A_cur, ind_cur, counts_cur):
        if ind_cur[y] == 1:
            return None
        
        new_counts = counts_cur.copy()
        
        # Remove x
        for a in A_cur:
            s = x + a
            if T_min <= s <= T_max:
                idx = s - T_min
                new_counts[idx] -= 1
        
        # Add y
        # Partners are A_cur \ {x}
        for a in A_cur:
            if a == x:
                continue
            s = y + a
            if T_min <= s <= T_max:
                idx = s - T_min
                new_counts[idx] += 1
        
        s_2y = 2 * y
        if T_min <= s_2y <= T_max:
            idx = s_2y - T_min
            new_counts[idx] += 1

        if not is_valid(new_counts):
            return None

        new_max = int(np.max(new_counts))
        new_sum_sq = float(np.sum(new_counts * new_counts))
        return new_counts, new_max, new_sum_sq

    def try_delete(x, A_cur, ind_cur, counts_cur):
        new_counts = counts_cur.copy()
        for a in A_cur:
            s = x + a
            if T_min <= s <= T_max:
                idx = s - T_min
                new_counts[idx] -= 1
        
        if not is_valid(new_counts):
            return None

        new_max = int(np.max(new_counts))
        new_sum_sq = float(np.sum(new_counts * new_counts))
        return new_counts, new_max, new_sum_sq

    def try_add(y, A_cur, ind_cur, counts_cur):
        if ind_cur[y] == 1:
            return None
        new_counts = counts_cur.copy()
        for a in A_cur:
            s = y + a
            if T_min <= s <= T_max:
                idx = s - T_min
                new_counts[idx] += 1
        
        s_2y = 2 * y
        if T_min <= s_2y <= T_max:
            idx = s_2y - T_min
            new_counts[idx] += 1

        if not is_valid(new_counts):
            return None

        new_max = int(np.max(new_counts))
        new_sum_sq = float(np.sum(new_counts * new_counts))
        return new_counts, new_max, new_sum_sq

    def repair(A_set, ind, counts):
        for _ in range(100):
            if is_valid(counts):
                break
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

    best_A = None
    best_max = float('inf')
    best_sum_sq = float('inf')

    def update_best(A_set, counts):
        nonlocal best_A, best_max, best_sum_sq
        cur_max, cur_sum_sq = get_metrics(counts)
        if cur_max < best_max or (cur_max == best_max and cur_sum_sq < best_sum_sq):
            best_max = cur_max
            best_sum_sq = cur_sum_sq
            best_A = set(A_set)

    starts = []
    random.seed(42)
    for _ in range(5):
        sz = random.randint(40, 60)
        starts.append(set(random.sample(range(0, N_max), sz)))
        
    starts.append(set(range(64, 129)))
    starts.append(set(range(0, 129, 2)))
    starts.append(set(range(1, 129, 2)))
    starts.append(set(range(30, 100)))

    for A_init in starts:
        if time.time() > deadline - 5.0:
            break
            
        A_set = set(A_init)
        ind = np.zeros(N_max, dtype=np.int32)
        for x in A_set:
            ind[x] = 1
            
        counts = compute_counts(ind)
        A_set, ind, counts = repair(A_set, ind, counts)
        
        if not is_valid(counts):
            continue
            
        update_best(A_set, counts)
        cur_max, cur_sum_sq = get_metrics(counts)
        
        no_improve = 0
        max_no_improve = 200
        
        while time.time() < deadline - 2.0:
            if no_improve > max_no_improve:
                if len(A_set) > 0:
                    x = random.choice(list(A_set))
                    y = random.randint(0, N_max - 1)
                    if ind[y] == 0:
                        res = try_swap(x, y, A_set, ind, counts)
                        if res:
                            counts, cur_max, cur_sum_sq = res
                            A_set.remove(x)
                            A_set.add(y)
                            ind[x] = 0
                            ind[y] = 1
                            update_best(A_set, counts)
                no_improve = 0
                max_no_improve = int(max_no_improve * 1.5)
                continue

            s_max_idx = int(np.argmax(counts))
            s_max = T_min + s_max_idx
            max_count = counts[s_max_idx]
            
            if max_count <= 1:
                break
                
            pairs = []
            for a in A_set:
                b = s_max - a
                if b in A_set and a <= b:
                    pairs.append((a, b))
            
            improved = False
            random.shuffle(pairs)
            for a, b in pairs:
                res = try_delete(a, A_set, ind, counts)
                if res:
                    new_counts, new_max, new_sum_sq = res
                    if new_max < cur_max or (new_max == cur_max and new_sum_sq < cur_sum_sq):
                        A_set.remove(a)
                        ind[a] = 0
                        counts = new_counts
                        cur_max = new_max
                        cur_sum_sq = new_sum_sq
                        update_best(A_set, counts)
                        improved = True
                        break
                        
                for _ in range(10):
                    y = random.randint(0, N_max - 1)
                    if ind[y] == 1 or y == a:
                        continue
                    res = try_swap(a, y, A_set, ind, counts)
                    if res:
                        new_counts, new_max, new_sum_sq = res
                        if new_max < cur_max or (new_max == cur_max and new_sum_sq < cur_sum_sq):
                            A_set.remove(a)
                            A_set.add(y)
                            ind[a] = 0
                            ind[y] = 1
                            counts = new_counts
                            cur_max = new_max
                            cur_sum_sq = new_sum_sq
                            update_best(A_set, counts)
                            improved = True
                            break
                            
                if improved:
                    break
            
            if not improved:
                no_improve += 1
            else:
                no_improve = 0
                max_no_improve = 200

    if best_A is None:
        return list(range(64, 129))
    return list(best_A)
