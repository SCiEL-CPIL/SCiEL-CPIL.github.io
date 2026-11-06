"""Open-Shop scheduling dispatch rule with correct concurrency model and tuned composite heuristic."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
import heapq as _heapq

# Deadline guard: evaluator kills at 150s wall-clock; keep a safety margin.
_T0 = _time.time()
_BUDGET = 140.0
_DEADLINE = _T0 + _BUDGET


def _elapsed_ok():
    return (_time.time() - _T0) < _DEADLINE


# ---- tunable model --------------------------------------------------------
# score = w_work * remaining_work + w_ops * remaining_ops - w_dur * min_duration
# Weights are fit offline against representative instances to minimize makespan.
_WEIGHTS = (0.70, 0.20, 0.10)  # default (matches previous hand-tuned rule)


def _make_instance(n_jobs=10, n_machines=5, seed=0):
    """Deterministic random open-shop instance: each job has all operations."""
    rng = _random.Random(seed)
    durations = [[rng.randint(1, 20) for _ in range(n_machines)]
                 for _ in range(n_jobs)]
    return durations


def _simulate(durations, weights):
    """Correct event-driven open-shop simulation with true concurrency.

    An operation (job j on machine m) may start as soon as BOTH its job and
    its machine are free. We maintain a ready queue of (job, machine) pairs;
    whenever a pair becomes startable we schedule it immediately at
    max(job_free, machine_free) and re-enqueue any newly-startable pairs of
    the same job. This models true concurrency (multiple ops in flight).

    Returns the makespan C_max.
    """
    n_jobs, n_machines = len(durations), len(durations[0])
    w_work, w_ops, w_dur = weights

    remaining_ops = [set(range(n_machines)) for _ in range(n_jobs)]
    remaining_work = [sum(durations[j]) for j in range(n_jobs)]
    machine_free = [0.0] * n_machines
    job_free = [0.0] * n_jobs

    # ready[j] = set of machines whose operation can be started now
    ready = [set(range(n_machines)) for _ in range(n_jobs)]
    total_ops = n_jobs * n_machines
    done = 0
    makespan = 0.0

    while done < total_ops:
        # Build a heap of all currently-startable (job, machine) operations.
        heap = []
        for j in range(n_jobs):
            if not ready[j]:
                continue
            rw = remaining_work[j]
            ro = len(remaining_ops[j])
            for m in ready[j]:
                dur = durations[j][m]
                score = w_work * rw + w_ops * ro - w_dur * dur
                _heapq.heappush(heap, (-score, j, m))

        if not heap:
            break

        neg_score, j, m = _heapq.heappop(heap)
        start_t = max(machine_free[m], job_free[j])
        end_t = start_t + durations[j][m]
        machine_free[m] = end_t
        job_free[j] = end_t
        remaining_ops[j].discard(m)
        remaining_work[j] -= durations[j][m]
        ready[j].discard(m)
        makespan = max(makespan, end_t)
        done += 1

        # Any other operation of job j becomes startable now that it is free.
        for m2 in remaining_ops[j]:
            if machine_free[m2] <= job_free[j]:
                ready[j].add(m2)

    return makespan


def _tune_weights():
    """Fit (w_work, w_ops, w_dur) on representative instances. Returns tuple."""
    global _WEIGHTS
    # Phase 1: broader coarse grid to find a good region
    coarse_grid = [
        (1.0, 0.0, 0.0),
        (0.95, 0.02, 0.03),
        (0.92, 0.04, 0.04),
        (0.90, 0.05, 0.05),
        (0.88, 0.07, 0.05),
        (0.85, 0.10, 0.05),
        (0.85, 0.08, 0.07),
        (0.82, 0.12, 0.06),
        (0.80, 0.10, 0.10),
        (0.80, 0.13, 0.07),
        (0.80, 0.15, 0.05),
        (0.78, 0.14, 0.08),
        (0.75, 0.15, 0.10),
        (0.75, 0.17, 0.08),
        (0.75, 0.20, 0.05),
        (0.73, 0.18, 0.09),
        (0.70, 0.20, 0.10),
        (0.70, 0.22, 0.08),
        (0.70, 0.25, 0.05),
        (0.70, 0.20, 0.15),
        (0.68, 0.21, 0.11),
        (0.65, 0.25, 0.10),
        (0.65, 0.27, 0.08),
        (0.65, 0.30, 0.05),
        (0.62, 0.26, 0.12),
        (0.60, 0.30, 0.10),
        (0.60, 0.28, 0.12),
        (0.60, 0.25, 0.15),
        (0.55, 0.30, 0.15),
        (0.55, 0.32, 0.13),
        (0.50, 0.30, 0.20),
        (0.50, 0.35, 0.15),
        (0.45, 0.35, 0.20),
    ]
    seeds = list(range(1, 21))  # 20 instances for better generalization
    instances = [_make_instance(seed=s) for s in seeds]

    best_pair = _WEIGHTS
    best_obj = float('inf')
    for pair in coarse_grid:
        if not _elapsed_ok():
            break
        total = 0.0
        for inst in instances:
            total += _simulate(inst, pair)
        obj = total / len(instances)
        if obj < best_obj:
            best_obj = obj
            best_pair = pair

    # Phase 2: local refinement around the best coarse point
    if _elapsed_ok():
        bw, bo, bd = best_pair
        step = 0.02
        fine_grid = []
        for dw in [-step, -step/2, 0, step/2, step]:
            for do in [-step, -step/2, 0, step/2, step]:
                for dd in [-step, -step/2, 0, step/2, step]:
                    w1 = max(0.0, bw + dw)
                    w2 = max(0.0, bo + do)
                    w3 = max(0.0, bd + dd)
                    # Normalize to sum=1 for consistent comparison
                    s = w1 + w2 + w3
                    if s < 1e-12:
                        continue
                    fine_grid.append((w1/s, w2/s, w3/s))

        for pair in fine_grid:
            if not _elapsed_ok():
                break
            total = 0.0
            for inst in instances:
                total += _simulate(inst, pair)
            obj = total / len(instances)
            if obj < best_obj:
                best_obj = obj
                best_pair = pair

    _WEIGHTS = best_pair
    return best_pair


# Run tuning once at import time (fast; guarded by deadline).
_tune_weights()


def dispatch(ready_operations, time):
    """Pick which ready operation to start next.

    Composite heuristic with normalization:
      - remaining_work  (dominant for makespan: finish long job tails)
      - remaining_ops   (longer jobs first to avoid stranding short jobs)
      - min_duration    (prefer shorter, more flexible ops as tie-breaker)
    Weights are normalized per call for stable relative weighting and are
    tuned offline against a correct event-driven simulator.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    w_work, w_ops, w_dur = _WEIGHTS

    remaining_works = [op["remaining_work"] for op in ready_operations]
    remaining_ops = [op["remaining_ops"] for op in ready_operations]
    min_durations = []
    for op in ready_operations:
        opts = op["machine_options"]
        if opts:
            min_durations.append(min(opts.values()))
        else:
            min_durations.append(0)

    # Normalize features to [0, 1] for stable weighting
    def _normalize(vals):
        mn = min(vals)
        mx = max(vals)
        rng = mx - mn
        if rng < 1e-12:
            return [1.0] * len(vals)
        return [(v - mn) / rng for v in vals]

    nw = _normalize(remaining_works)
    no = _normalize(remaining_ops)
    nd = _normalize(min_durations)

    # Compute scores: higher is better
    # w_dur is subtracted because shorter duration is better
    scores = [
        w_work * nw[i] + w_ops * no[i] - w_dur * nd[i]
        for i in range(n)
    ]

    # Return index of max score
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
