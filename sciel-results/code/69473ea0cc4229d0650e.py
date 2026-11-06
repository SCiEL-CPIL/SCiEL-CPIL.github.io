"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Wall-clock guard: the evaluator kills after 150s. Keep a safety margin.
_DEADLINE = _time.monotonic() + 130.0


def _fallback_index(ready_operations):
    # Cheap deterministic fallback: Most Work Remaining.
    if len(ready_operations) == 1:
        return 0
    return int(np.argmax([op["remaining_work"] for op in ready_operations]))


def dispatch(ready_operations, time):
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Safety margin: if we are dangerously close to the kill deadline,
    # return the cheap heuristic immediately.
    if _time.monotonic() > _DEADLINE:
        return _fallback_index(ready_operations)

    ops = ready_operations

    # --- Component 1: Most Work Remaining (higher = better) ---
    mwr = np.array([float(op["remaining_work"]) for op in ops])

    # --- Component 2: Shortest Processing Time (lower = better) ---
    # Use the minimum duration among the operation's machine options
    # as a proxy for the operation's processing time.
    spt = np.array([
        float(min(op["machine_options"].values()))
        for op in ops
    ])

    # --- Component 3: Machine congestion / conflict (lower = better) ---
    # For each candidate operation, estimate how much "conflict" it faces:
    # for each machine it can use, count how many *other* ready operations
    # also want that machine (weighted by their remaining work, as a proxy
    # for how busy that machine will be). Sum over the candidate's machines,
    # then normalize by the number of machines the candidate can use.
    # A higher congestion score means the candidate is likely to be blocked.
    congestion = np.zeros(n)
    for i in range(n):
        mi = ops[i]["machine_options"]
        total_conflict = 0.0
        for m in mi:
            # How many other ready ops also target machine m,
            # weighted by their remaining work.
            w = 0.0
            for j in range(n):
                if j == i:
                    continue
                mj = ops[j]["machine_options"]
                if m in mj:
                    w += float(ops[j]["remaining_work"])
            total_conflict += w
        # Average conflict across the candidate's machine options.
        congestion[i] = total_conflict / max(1, len(mi))

    # --- Normalize each component to [0, 1] ---
    def _norm(x, higher_is_better):
        x = np.asarray(x, dtype=float)
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-12:
            return np.full_like(x, 0.5)
        y = (x - lo) / (hi - lo)
        if higher_is_better:
            return y
        else:
            return 1.0 - y

    # MWR: higher is better -> normalize directly.
    score_mwr = _norm(mwr, higher_is_better=True)
    # SPT: lower is better -> invert.
    score_spt = _norm(spt, higher_is_better=False)
    # Congestion: lower is better -> invert.
    score_cong = _norm(congestion, higher_is_better=False)

    # --- Composite weighted score ---
    # Weights: MWR 0.4, SPT 0.35, Congestion 0.25
    w_mwr, w_spt, w_cong = 0.40, 0.35, 0.25
    total = (w_mwr * score_mwr
             + w_spt * score_spt
             + w_cong * score_cong)

    idx = int(np.argmax(total))

    # Final safety check before returning.
    if _time.monotonic() > _DEADLINE:
        return _fallback_index(ready_operations)

    return idx
# EVOLVE-BLOCK-END
