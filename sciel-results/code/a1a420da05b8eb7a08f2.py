"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = 120.0
_START = _time.time()
_MARGIN = 5.0  # seconds to stay under the hard limit

# Lightweight bookkeeping: approximate machine load so we can balance.
# We do not receive completion events, so we accumulate a running estimate
# of how much work has been *assigned* to each machine. This is a proxy
# that helps spread load across machines within a single run.
_MACHINE_LOAD = {}
_LAST_TIME = -1.0

def _machine_load(m):
    return _MACHINE_LOAD.get(m, 0.0)

def _bump_machine_load(m, d):
    _MACHINE_LOAD[m] = _MACHINE_LOAD.get(m, 0.0) + d


def dispatch(ready_operations, time):
    global _LAST_TIME
    
    # Reset state if time goes backwards or resets to 0 (new instance)
    if time < _LAST_TIME or time == 0:
        _MACHINE_LOAD.clear()
    _LAST_TIME = time

    # Deadline guard: if we are about to exceed the hard limit, fall back
    # to the simple baseline (argmax remaining_work) which is O(n) and safe.
    if _time.time() - _START > _DEADLINE - _MARGIN:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    if not ready_operations:
        return 0

    n = len(ready_operations)

    # ---- Resource-contention priority ---------------------------------
    # Score each ready operation by how efficiently it can be placed:
    #   priority = 1 / (sum of min processing times across eligible machines)
    # Higher priority = operation can be executed quickly on available
    # resources, reducing congestion and tail latency.
    #
    # We also factor in job-level urgency (remaining work) so that long
    # jobs aren't starved, and a small flexibility penalty for operations
    # with narrow machine options (schedule constrained ops earlier).

    scores = np.zeros(n, dtype=float)
    min_durs = []  # store min duration per op for machine selection
    for idx, op in enumerate(ready_operations):
        mo = op.get("machine_options", {})
        rw = float(op.get("remaining_work", 0.0))
        ro = float(op.get("remaining_ops", 1)) or 1.0

        if mo:
            durs = [float(v) for v in mo.values()]
            min_d = min(durs)
            min_durs.append(min_d)
            # sum of all processing times across eligible machines
            # (proxy for how "expensive" this op is to place)
            sum_durs = sum(durs)
            # congestion: how loaded are the eligible machines?
            load_sum = sum(_machine_load(m) for m in mo)
            # efficiency: 1 / (sum_durs + load_sum + epsilon)
            # higher = less congested, faster to complete
            efficiency = 1.0 / (sum_durs + load_sum + 1e-9)
            # urgency: remaining work per remaining op
            urgency = rw / ro
            # flexibility: fewer options = more constrained = higher priority
            n_opts = len(mo)
            flex_boost = 1.0 / n_opts  # 1 for single option, 0.2 for 5
        else:
            min_d = rw / ro
            min_durs.append(min_d)
            efficiency = 1.0 / (min_d + 1e-9)
            urgency = rw / ro
            flex_boost = 1.0

        scores[idx] = efficiency + 0.3 * urgency + 0.2 * flex_boost

    best_idx = int(np.argmax(scores))

    # ---- Choose the machine -------------------------------------------
    # For the chosen operation, pick the machine that minimizes
    # (duration + current load), balancing speed and congestion.
    chosen_op = ready_operations[best_idx]
    mo = chosen_op.get("machine_options", {})
    if mo:
        best_m = None
        best_key = None
        for m, dur in mo.items():
            d = float(dur)
            key = d + _machine_load(m)
            if best_key is None or key < best_key:
                best_key = key
                best_m = m
                best_d = d
        if best_m is not None:
            _bump_machine_load(best_m, best_d)
            return (best_idx, best_m)

    # Fallback: return just the index.
    return best_idx
# EVOLVE-BLOCK-END
