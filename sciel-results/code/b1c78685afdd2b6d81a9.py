"""Model-and-optimize dispatch for flexible job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 110.0  # safety margin below the 150s wall-clock kill
_T0 = None


def _elapsed():
    global _T0
    if _T0 is None:
        _T0 = _time.perf_counter()
    return _time.perf_counter() - _T0


def _build_context(ready_operations):
    """Precompute per-operation and per-machine features from the current ready set."""
    n = len(ready_operations)
    if n == 0:
        return None

    # For each op: list of (machine, duration), min/max duration, machine set
    min_dur = np.empty(n, dtype=float)
    max_dur = np.empty(n, dtype=float)
    rem_work = np.empty(n, dtype=float)
    rem_ops = np.empty(n, dtype=float)
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        durs = np.array([opts[m] for m in opts], dtype=float)
        min_dur[i] = durs.min() if durs.size else 0.0
        max_dur[i] = durs.max() if durs.size else 0.0
        rem_work[i] = float(op.get("remaining_work", 0.0))
        rem_ops[i] = float(op.get("remaining_ops", 0.0))

    # Machine contention: how many ready ops can use each machine
    machine_count = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_count[m] = machine_count.get(m, 0) + 1
    return {
        "n": n,
        "min_dur": min_dur,
        "max_dur": max_dur,
        "rem_work": rem_work,
        "rem_ops": rem_ops,
        "machine_count": machine_count,
        "ops": ready_operations,
    }


def _soft_min(durs, temperature=1.0):
    """Smooth minimum via log-sum-exp (negative log-sum-exp of negatives)."""
    if len(durs) == 0:
        return 0.0
    d = np.array(durs, dtype=float)
    d_min = d.min()
    # Stable computation: soft_min ≈ d_min - temperature * log(sum(exp(-(d - d_min)/temperature)))
    if temperature <= 0:
        return d_min
    scaled = -(d - d_min) / temperature
    max_s = scaled.max()
    log_sum = max_s + np.log(np.exp(scaled - max_s).sum())
    return d_min - temperature * log_sum


def _score_option(ctx, i, temperature=1.0):
    """Score the i-th ready op under current weight context. Higher = more attractive."""
    op = ctx["ops"][i]
    opts = op["machine_options"]
    if not opts:
        return -np.inf, None

    mc = ctx["machine_count"]
    # Use soft-min for a smooth surrogate of min duration
    durs = [float(opts[m]) for m in opts]
    soft_min_dur = _soft_min(durs, temperature)

    # Best (most attractive) machine for this op: prefer low contention, then short dur
    best_key = None
    best_machine = None
    for m, d in opts.items():
        contention = mc.get(m, 1)
        machine_score = -ctx["w_cont"] * contention - ctx["w_dur"] * float(d)
        if best_key is None or machine_score > best_key:
            best_key = machine_score
            best_machine = (m, float(d))

    # Job criticality using soft-min duration as a smooth proxy
    criticality = (
        ctx["w_work"] * ctx["rem_work"][i]
        + ctx["w_ops"] * ctx["rem_ops"][i]
        - ctx["w_dur"] * soft_min_dur
    )
    total = criticality + best_key
    return total, best_machine


def dispatch(ready_operations, time):
    if _elapsed() > _DEADLINE:
        # Hard safety: fall back to a simple deterministic rule
        if not ready_operations:
            return 0
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    if not ready_operations:
        return 0
    if len(ready_operations) == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        if opts:
            m = min(opts, key=lambda k: opts[k])
            return (0, m)
        return 0

    ctx = _build_context(ready_operations)
    if ctx is None:
        return 0

    # Expanded candidate weight settings with soft-min temperature parameter
    candidates = [
        # (w_work, w_ops, w_dur, w_cont, temperature)
        ("crit",    1.0,   0.3,   0.05, 0.2,  1.0),
        ("work",    1.0,   0.0,   0.10, 0.1,  1.0),
        ("spt",     0.3,   0.1,   1.0,  0.1,  0.5),
        ("bal",     1.0,   0.5,   0.30, 0.5,  1.0),
        ("cont",    0.5,   0.2,   0.20, 1.0,  1.0),
        ("ops",     0.4,   1.0,   0.20, 0.3,  1.0),
        ("crit2",   1.2,   0.4,   0.08, 0.3,  0.8),
        ("spt2",    0.2,   0.1,   1.5,  0.2,  0.3),
        ("bal2",    0.8,   0.6,   0.40, 0.6,  1.2),
        ("cont2",   0.6,   0.3,   0.15, 1.2,  0.8),
    ]

    best_quality = -np.inf
    best_index = 0
    best_machine = None

    for name, ww, wo, wd, wc, temp in candidates:
        ctx["w_work"], ctx["w_ops"], ctx["w_dur"], ctx["w_cont"] = ww, wo, wd, wc
        scores = np.empty(ctx["n"], dtype=float)
        machines = [None] * ctx["n"]
        for i in range(ctx["n"]):
            s, mb = _score_option(ctx, i, temp)
            scores[i] = s
            machines[i] = mb
        top = int(np.argmax(scores))
        m, d = machines[top]
        cont_after = ctx["machine_count"].get(m, 0) - 1
        # Improved quality: weight criticality more, penalize contention and duration
        quality = (
            1.0 * ctx["rem_work"][top]
            + 0.15 * ctx["rem_ops"][top]
            - 0.25 * cont_after
            - 0.08 * d
        )
        if quality > best_quality:
            best_quality = quality
            best_index = top
            best_machine = m

    # Deadline guard before returning a compound result
    if _elapsed() > _DEADLINE:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    if best_machine is not None:
        return (best_index, best_machine)
    return best_index
# EVOLVE-BLOCK-END
