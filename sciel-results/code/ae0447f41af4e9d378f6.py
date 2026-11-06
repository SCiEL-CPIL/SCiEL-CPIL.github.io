"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = 120.0
_START = _time.time()
_MARGIN = 5.0  # seconds to stay under the hard limit

# Lightweight bookkeeping: approximate machine load so we can balance.
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

    # Deadline guard
    if _time.time() - _START > _DEADLINE - _MARGIN:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    if not ready_operations:
        return 0

    n = len(ready_operations)

    # ---- Build per-operation features ---------------------------------
    urg = []
    inten = []
    own = []
    flex = []
    num_options = []
    for op in ready_operations:
        rw = float(op.get("remaining_work", 0.0))
        ro = float(op.get("remaining_ops", 1)) or 1.0
        mo = op.get("machine_options", {})
        if mo:
            durs = [float(v) for v in mo.values()]
            own_proc = min(durs)
            max_proc = max(durs)
            flex.append(max_proc - own_proc)
            num_options.append(len(mo))
        else:
            own_proc = rw / ro
            flex.append(0.0)
            num_options.append(1)
        urg.append(rw)
        inten.append(rw / ro)
        own.append(own_proc)

    urg = np.asarray(urg, dtype=float)
    inten = np.asarray(inten, dtype=float)
    own = np.asarray(own, dtype=float)
    flex = np.asarray(flex, dtype=float)
    num_options = np.asarray(num_options, dtype=float)

    # Normalise into [0,1]
    def _norm(x):
        rng = x.max() - x.min()
        if rng <= 1e-9:
            return np.zeros_like(x)
        return (x - x.min()) / rng

    u = _norm(urg)
    i = _norm(inten)
    o = _norm(own)
    f = _norm(flex)
    # Constraint tightness: fewer options = more constrained = higher priority
    # Invert num_options so fewer options -> higher score
    c = 1.0 - _norm(num_options)

    # ---- Choose the operation -----------------------------------------
    # Strategy: Prioritize constraint tightness (fewest machine options),
    # then urgency, intensity, and short processing time.
    w_u, w_i, w_o, w_f, w_c = 0.20, 0.10, 0.20, 0.10, 0.40
    op_score = w_u * u + w_i * i + w_o * (1.0 - o) + w_f * (1.0 - f) + w_c * c

    best_idx = int(np.argmax(op_score))

    # ---- Choose the machine -------------------------------------------
    chosen_op = ready_operations[best_idx]
    mo = chosen_op.get("machine_options", {})
    if mo:
        # Pick the machine with minimum projected completion time (load + duration)
        # This is a proxy for "when would this operation finish if started now?"
        best_m = None
        best_key = None
        best_d = None
        for m, dur in mo.items():
            key = _machine_load(m) + float(dur)
            if best_key is None or key < best_key:
                best_key = key
                best_m = m
                best_d = float(dur)
        if best_m is not None:
            _bump_machine_load(best_m, best_d)
            return (best_idx, best_m)

    return best_idx
# EVOLVE-BLOCK-END
