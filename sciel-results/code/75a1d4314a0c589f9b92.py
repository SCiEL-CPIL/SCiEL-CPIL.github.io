import numpy as np
import time

def ring_routing() -> np.ndarray:
    N = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]

    # CW incidence: demand i uses edge e if bit set.
    cw = np.zeros((D, N), dtype=np.int64)
    ccw = np.zeros((D, N), dtype=np.int64)
    w = np.zeros(D, dtype=np.int64)
    for i, (s, t, wi) in enumerate(demands):
        e = s
        while e != t:
            cw[i, e] = 1
            e = (e + 1) % N
        ccw[i] = 1 - cw[i]
        w[i] = wi

    # Vectorized max-load evaluation: routing r in {0,1}^D
    def eval_max(r):
        # r is shape (D,), need to reshape to (D, 1) for broadcasting against (D, N)
        r_col = r.astype(np.int64).reshape(-1, 1)
        # r_col is 1 for CCW (value 1 in best/result logic? No, let's check logic)
        # In local_search: trial[i] = 1 - trial[i].
        # In result: result[best == 1] = 1.0.
        # Contract: "An entry >= 0.5 routes the demand counter-clockwise".
        # So 1 -> CCW, 0 -> CW.
        # cw[i, e] is 1 if edge e is on CW path.
        # ccw[i, e] is 1 if edge e is on CCW path.
        # If r[i] == 1 (CCW), we want to use ccw[i].
        # If r[i] == 0 (CW), we want to use cw[i].
        # Current formula: (r * 2 - 1) * ccw + cw
        # If r=1: (2-1)*ccw + cw = ccw + cw. Since paths are disjoint (except maybe? No, ring paths partition edges), ccw+cw = all edges?
        # Wait. For a ring, CW path and CCW path are disjoint and cover all edges?
        # "Clockwise from s to t uses edges s, ..., t-1". "Counter-clockwise uses all other ring edges."
        # Yes, they are disjoint and cover all N edges.
        # So if r=1 (CCW), load should be on ccw edges.
        # Formula gives: 1*ccw + cw. This adds load to BOTH ccw and cw edges?
        # No, `cw` and `ccw` are 0/1 masks.
        # If r=1, we want mask = ccw.
        # If r=0, we want mask = cw.
        # Let's check the formula: `mask = (r * 2 - 1) * ccw + cw`
        # If r=1: (1)*ccw + cw = ccw + cw. This is wrong. It sums both.
        # If r=0: (-1)*ccw + cw = cw - ccw. This is wrong.
        
        # Correct logic:
        # We want `mask = r * ccw + (1-r) * cw`
        # Let's verify:
        # r=1 (CCW): 1*ccw + 0*cw = ccw. Correct.
        # r=0 (CW): 0*ccw + 1*cw = cw. Correct.
        
        # The original code had: `(r * 2 - 1) * ccw + cw`
        # If r=1: 1*ccw + cw. Wrong.
        # If r=0: -1*ccw + cw. Wrong.
        
        # It seems the original logic was flawed or I am misinterpreting `cw`/`ccw` definitions?
        # `cw[i, e] = 1` if e is in CW path.
        # `ccw[i, e] = 1` if e is in CCW path.
        # They are disjoint.
        
        # So the correct mask is:
        # mask = r_col * ccw + (1 - r_col) * cw
        
        mask = r_col * ccw + (1 - r_col) * cw
        loads = mask.T @ w
        return int(np.max(loads))

    start = time.time()
    deadline = start + 620.0

    def remaining():
        return deadline - time.time()

    # Baselines
    best = None
    best_ml = 10**9
    for init in (np.zeros(D, dtype=np.int8), np.ones(D, dtype=np.int8)):
        ml = eval_max(init)
        if ml < best_ml:
            best_ml = ml
            best = init.copy()

    rng = np.random.default_rng(12345)

    def local_search(cur, cur_ml, max_passes=8):
        nonlocal best, best_ml
        improved = True
        passes = 0
        while improved and passes < max_passes and remaining() > 2.0:
            improved = False
            passes += 1
            # single flips
            for i in range(D):
                if remaining() < 2.0:
                    break
                trial = cur.copy()
                trial[i] = 1 - trial[i]
                tml = eval_max(trial)
                if tml < cur_ml:
                    cur = trial; cur_ml = tml; improved = True
                    if cur_ml < best_ml:
                        best_ml = cur_ml; best = cur.copy()
            # pairwise 2-flips
            for i in range(D):
                for j in range(i + 1, D):
                    if remaining() < 2.0:
                        break
                    trial = cur.copy()
                    trial[i] = 1 - trial[i]
                    trial[j] = 1 - trial[j]
                    tml = eval_max(trial)
                    if tml < cur_ml:
                        cur = trial; cur_ml = tml; improved = True
                        if cur_ml < best_ml:
                            best_ml = cur_ml; best = cur.copy()
        return cur, cur_ml

    # Random restarts with local search
    n_restarts = 0
    while remaining() > 30.0 and n_restarts < 400:
        cur = rng.integers(0, 2, size=D, dtype=np.int8)
        cur_ml = eval_max(cur)
        cur, cur_ml = local_search(cur, cur_ml, max_passes=6)
        n_restarts += 1

    # Simulated annealing from best
    if remaining() > 10.0:
        cur = best.copy()
        cur_ml = eval_max(cur)
        T = 6.0
        T_min = 0.02
        cool = 0.999
        while T > T_min and remaining() > 5.0:
            for _ in range(60):
                if remaining() < 5.0:
                    break
                idx = rng.integers(0, D)
                trial = cur.copy()
                trial[idx] = 1 - trial[idx]
                tml = eval_max(trial)
                d = tml - cur_ml
                if d <= 0 or rng.random() < np.exp(-d / T):
                    cur = trial; cur_ml = tml
                    if cur_ml < best_ml:
                        best_ml = cur_ml; best = cur.copy()
            T *= cool

    # Final polish: local search from best
    if remaining() > 2.0:
        cur = best.copy()
        cur_ml = eval_max(cur)
        cur, cur_ml = local_search(cur, cur_ml, max_passes=10)

    result = np.zeros(D, dtype=float)
    result[best == 1] = 1.0
    return result
