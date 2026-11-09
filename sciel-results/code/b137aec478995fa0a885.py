"""Job-Shop Scheduling Dispatch Rule.
Structural Change: Replaces additive weighted scoring with a ratio-based
priority rule: Inverse-Weighted Shortest Processing Time.
Score = min_duration / (1 + remaining_work).
This prioritizes short operations, but amplifies the preference for
shortness on jobs with high remaining work (critical jobs).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_START_TS = _time.time()
DEADLINE_S = 110.0  # Safety margin against 120s/150s limit


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    n = len(ready_operations)

    if n == 1:
        op = ready_operations[0]
        mo = op.get("machine_options", {})
        if mo:
            best_machine = min(mo, key=mo.get)
            return (0, best_machine)
        return 0

    # Deadline guard: fall back to pure SPT (min min-duration) with best machine
    if _time.time() - _START_TS > DEADLINE_S:
        min_durs = np.empty(n, dtype=np.float64)
        best_m = [None] * n
        for i, op in enumerate(ready_operations):
            mo = op.get("machine_options", {})
            if mo:
                bm, bd = min(mo.items(), key=lambda kv: kv[1])
                min_durs[i] = bd
                best_m[i] = bm
            else:
                min_durs[i] = float("inf")
        idx = int(np.argmin(min_durs))
        if best_m[idx] is not None:
            return (idx, best_m[idx])
        return idx

    # ---- Extract features and compute ratio-based scores ----
    scores = np.empty(n, dtype=np.float64)
    best_mach = [None] * n
    min_dur = np.empty(n, dtype=np.float64)

    for i, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        if mo:
            bm, bd = min(mo.items(), key=lambda kv: kv[1])
            min_dur[i] = float(bd)
            best_mach[i] = bm
        else:
            min_dur[i] = float("inf")
            best_mach[i] = None

        rw = float(op.get("remaining_work", 0.0))
        
        # Structural Change: Ratio-based score
        # Lower score is better.
        # Short duration (min_dur) is good.
        # High remaining work (rw) in denominator makes score smaller 
        # if min_dur is small, but also makes score smaller if rw is large?
        # Wait: Score = d / (1 + rw).
        # If rw is large, denominator is large, score is small.
        # So we prefer ops with LARGE remaining work?
        # Yes, because they are critical. We want to schedule their short ops first.
        # If d is small, score is small.
        # So we prefer SMALL d AND LARGE rw.
        
        if np.isinf(min_dur[i]):
            scores[i] = float("inf")
        else:
            scores[i] = min_dur[i] / (1.0 + rw)

    # Select operation with minimum score
    best_idx = int(np.argmin(scores))
    
    bm = best_mach[best_idx]
    if bm is not None:
        return (best_idx, bm)
    return best_idx

# EVOLVE-BLOCK-END
