import time

def _compute_table():
    """Pre-compute minimal antichain representations for targets 1..128."""
    start_time = time.time()
    DEADLINE = 80  # seconds, leave 10s safety margin
    
    smooth = set()
    for a in range(8):
        for b in range(5):
            for c in range(4):
                val = (2 ** a) * (3 ** b) * (5 ** c)
                if val <= 128:
                    smooth.add(val)
    
    smooth_list = sorted(smooth)
    n_smooth = len(smooth_list)
    val_to_idx = {v: i for i, v in enumerate(smooth_list)}
    smooth_set = smooth
    
    # Precompute incompatibility matrix
    incompatible = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            a_val = smooth_list[i]
            b_val = smooth_list[j]
            if a_val % b_val == 0 or b_val % a_val == 0:
                incompatible[i][j] = True
                incompatible[j][i] = True
    
    reps = [None] * 129
    
    # k=1: smooth numbers
    for t in range(1, 129):
        if t in smooth_set:
            reps[t] = (t,)
    
    # k=2: pairs
    for t in range(1, 129):
        if reps[t] is not None:
            continue
        for i in range(n_smooth):
            a = smooth_list[i]
            if a >= t:
                break
            b = t - a
            if b < a:
                continue
            if b in smooth_set:
                j = val_to_idx[b]
                if not incompatible[i][j]:
                    reps[t] = (a, b)
                    break
    
    # k=3: triples
    for t in range(1, 129):
        if reps[t] is not None:
            continue
        found = False
        for i in range(n_smooth):
            a = smooth_list[i]
            if a >= t // 3:
                break
            for j in range(i + 1, n_smooth):
                b = smooth_list[j]
                if a + b >= t:
                    break
                c = t - a - b
                if c <= b:
                    continue
                if c in smooth_set:
                    k_idx = val_to_idx[c]
                    if not incompatible[i][j] and not incompatible[i][k_idx] and not incompatible[j][k_idx]:
                        reps[t] = (a, b, c)
                        found = True
                        break
            if found:
                break
    
    # k=4 and above: DFS with precomputed incompatibility
    def dfs(rem, start_idx, current_indices, k_remaining):
        if time.time() - start_time > DEADLINE:
            return None
        if k_remaining == 0:
            return current_indices if rem == 0 else None
        if rem <= 0:
            return None
        for idx in range(start_idx, n_smooth):
            val = smooth_list[idx]
            if val > rem:
                break
            valid = True
            for existing in current_indices:
                if incompatible[idx][existing]:
                    valid = False
                    break
            if not valid:
                continue
            res = dfs(rem - val, idx + 1, current_indices + [idx], k_remaining - 1)
            if res is not None:
                return res
        return None
    
    for t in range(1, 129):
        if reps[t] is not None:
            continue
        for k in range(4, 11):
            if time.time() - start_time > DEADLINE:
                break
            res = dfs(t, 0, [], k)
            if res is not None:
                reps[t] = tuple(smooth_list[i] for i in res)
                break
    
    return reps

_TABLE = _compute_table()

def construct():
    return [list(_TABLE[t]) for t in range(1, 129)]
