import time
import random
import numpy as np

_START_TIME = None

def _all_aps(n=128):
    aps = []
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    return aps

_APS_ARR = None

def _get_aps():
    global _APS_ARR
    if _APS_ARR is None:
        _APS_ARR = np.array(_all_aps(128), dtype=np.int32)
    return _APS_ARR

def _violations(colors):
    arr = _get_aps()
    c0 = colors[arr[:, 0]]
    c1 = colors[arr[:, 1]]
    c2 = colors[arr[:, 2]]
    c3 = colors[arr[:, 3]]
    d1 = (c1 != c0).astype(np.int32)
    d2 = ((c2 != c0) & (c2 != c1)).astype(np.int32)
    d3 = ((c3 != c0) & (c3 != c1) & (c3 != c2)).astype(np.int32)
    distinct = 1 + d1 + d2 + d3
    return int(np.sum(distinct < 3))

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    deadline = _START_TIME + 81.0

    n = 128
    np.random.seed(123)
    random.seed(123)

    # Start with random 3-coloring
    cur = np.random.randint(0, 3, size=n, dtype=np.int32)
    cur_v = _violations(cur)
    cur_d = len(np.unique(cur))
    cur_s = cur_d * 100000 + cur_v

    best = cur.copy()
    best_v = cur_v
    best_d = cur_d
    best_s = cur_s

    T = 500.0
    cool = 0.9995

    while time.time() < deadline:
        # Random move
        if random.random() < 0.8:
            pos = random.randint(0, n - 1)
            new_c = random.randint(0, 2)
            if new_c == cur[pos]:
                continue
            old_c = cur[pos]
            cur[pos] = new_c
            new_v = _violations(cur)
            new_d = len(np.unique(cur))
            new_s = new_d * 100000 + new_v
            delta = new_s - cur_s
            if delta < 0 or random.random() < np.exp(-max(0, delta) / T):
                cur_s = new_s
                cur_v = new_v
                cur_d = new_d
            else:
                cur[pos] = old_c
        else:
            # Merge two colors
            present = np.unique(cur)
            if len(present) >= 2:
                x, y = random.sample(present.tolist(), 2)
                mask = (cur == x)
                if np.any(mask):
                    old_c = cur.copy()
                    cur[mask] = y
                    new_v = _violations(cur)
                    new_d = len(np.unique(cur))
                    new_s = new_d * 100000 + new_v
                    delta = new_s - cur_s
                    if delta < 0 or random.random() < np.exp(-max(0, delta) / T):
                        cur_s = new_s
                        cur_v = new_v
                        cur_d = new_d
                    else:
                        cur[:] = old_c

        if cur_s < best_s:
            best_s = cur_s
            best = cur.copy()
            best_v = cur_v
            best_d = cur_d

        if best_v == 0 and best_d <= 3:
            break

        T *= cool
        if T < 0.1:
            T = 0.1

    # If not valid, do a greedy repair to ensure validity
    if _violations(best) > 0:
        colors = best.copy()
        for _ in range(50000):
            if time.time() > deadline:
                break
            v = _violations(colors)
            if v == 0:
                break
            arr = _get_aps()
            c0 = colors[arr[:, 0]]
            c1 = colors[arr[:, 1]]
            c2 = colors[arr[:, 2]]
            c3 = colors[arr[:, 3]]
            d1 = (c1 != c0).astype(np.int32)
            d2 = ((c2 != c0) & (c2 != c1)).astype(np.int32)
            d3 = ((c3 != c0) & (c3 != c1) & (c3 != c2)).astype(np.int32)
            distinct = 1 + d1 + d2 + d3
            bad = np.where(distinct < 3)[0]
            if len(bad) == 0:
                break
            idx = bad[random.randint(0, len(bad) - 1)]
            a, b, c, d = arr[idx]
            pos = random.choice([a, b, c, d])
            # Assign a color that might help; try 0,1,2 first, then 3,4
            for new_c in [0, 1, 2, 3, 4]:
                colors[pos] = new_c
                if _violations(colors) < v:
                    v = _violations(colors)
                    break
            else:
                colors[pos] = random.randint(0, 4)
        best = colors

    return [int(x) for x in best]
