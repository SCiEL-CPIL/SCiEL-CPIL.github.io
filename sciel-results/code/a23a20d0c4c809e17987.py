"""Rewritten dispatch: regret-minimization centered priority within the smooth-feature family."""
# EVOLVE-BLOCK-START
import time as _time
import math
import numpy as np

_DEADLINE_START = _time.time()
_MAX_WALL = 138.0  # safety margin before the 150s kill


def _check_deadline(fallback):
    """If we're getting close to the wall-clock limit, return fallback immediately."""
    if _time.time() - _DEADLINE_START > _MAX_WALL:
        return fallback
    return None


def _norm(x):
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mn) / (mx - mn)


def dispatch(ready_operations, time):
    # Deadline guard: return the first available op if we're over budget.
    early = _check_deadline(0)
    if early is not None:
        return early

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # ---- Extract per-operation features ----------------------------------
    # durations[i] = array of machine durations for ready op i
    durations = []
    for op in ready_operations:
        mo = op.get("machine_options", {})
        vals = list(mo.values()) if mo else [0.0]
        durations.append(np.asarray(vals, dtype=np.float64))

    # Min duration: the load you CANNOT avoid (least-loaded / best machine).
    min_d = np.array([d.min() for d in durations], dtype=np.float64)
    # Soft-max (log-sum-exp) of durations: smooth surrogate for the worst-case
    # machine load this operation can impose (congestion spread).
    spread = np.array([
        float(np.log(np.sum(np.exp(d - d.max()))) + d.max()) if d.size else 0.0
        for d in durations
    ], dtype=np.float64)

    remaining_works = np.array([op.get("remaining_work", 0.0) for op in ready_operations], dtype=np.float64)
    remaining_ops = np.array([op.get("remaining_ops", 0.0) for op in ready_operations], dtype=np.float64)
    remaining_ops_safe = np.where(remaining_ops > 0, remaining_ops, 1.0)
    crit = remaining_works / remaining_ops_safe  # criticality / density

    # ---- Normalized features --------------------------------------------
    n_min = _norm(min_d)      # regret: longest unavoidable load -> delay is costly
    n_spread = _norm(spread)  # smooth worst-case congestion spread
    n_crit = _norm(crit)      # remaining work density

    # Regret-minimization priority: start the operation that, if delayed,
    # would create the most future congestion. The unavoidable (min) load
    # dominates; spread captures how much extra load it can push; crit keeps
    # the LRPT/density core so we still drain dense jobs.
    score = 3.0 * n_min + 1.0 * n_spread + 0.9 * n_crit

    best_idx = int(np.argmax(score))
    return best_idx
# EVOLVE-BLOCK-END
