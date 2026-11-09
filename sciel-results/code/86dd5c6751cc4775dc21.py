"""Rewritten program for Job-Shop Scheduling dispatch rule."""
# EVOLVE-BLOCK-START
import time as _time
import math

_START_TIME = _time.time()
_DEADLINE_SECONDS = 115.0  # Safety margin before 120s limit


def _soft_min(values, temperature=0.5):
    """Compute soft minimum using log-sum-exp trick."""
    if not values:
        return 0.0
    vmin = min(values)
    if vmin <= 1e-9:
        return vmin
    # Soft min: -T * log(sum(exp(-v/T)))
    scaled = [-v / temperature for v in values]
    max_scaled = max(scaled)
    log_sum = math.log(sum(math.exp(s - max_scaled) for s in scaled)) + max_scaled
    return -temperature * log_sum


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in a job-shop simulation.
    
    Strategy: Use a composite heuristic that balances:
    1. Critical Path Length (CPL) via remaining_work: higher is more urgent.
    2. Bottleneck Elimination (LPT-like): prioritize operations with longer min duration
       to prevent them from becoming critical path bottlenecks.
    3. Machine contention: prefer machines with less contention.
    4. Remaining operations count: more remaining ops means the job is in early stages.
    
    We compute a weighted score and pick the operation with the highest score.
    """
    # Deadline guard
    if _time.time() - _START_TIME > _DEADLINE_SECONDS:
        return 0

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Extract features
    remaining_works = [op["remaining_work"] for op in ready_operations]
    remaining_ops = [op["remaining_ops"] for op in ready_operations]
    
    # For each operation, compute min duration (bottleneck indicator)
    # and track machine contention
    min_durations = []
    machine_counts = {}
    
    for op in ready_operations:
        mo = op["machine_options"]
        if mo:
            durations = list(mo.values())
            machines = list(mo.keys())
            # Count machine contention
            for m in machines:
                machine_counts[m] = machine_counts.get(m, 0) + 1
            min_durations.append(min(durations))
        else:
            min_durations.append(1.0)
    
    # Compute contention score: for each op, average contention of its available machines
    # Lower contention = better (less likely to wait)
    contention_scores = []
    for op in ready_operations:
        mo = op["machine_options"]
        if mo:
            machines = list(mo.keys())
            avg_contention = sum(machine_counts[m] for m in machines) / len(machines)
        else:
            avg_contention = 1.0
        contention_scores.append(avg_contention)

    # Normalize remaining_work to [0, 1] - higher remaining work = higher priority
    max_rw = max(remaining_works) if remaining_works else 1.0
    min_rw = min(remaining_works) if remaining_works else 0.0
    rw_range = max_rw - min_rw
    if rw_range < 1e-9:
        norm_rw = [0.5] * n
    else:
        norm_rw = [(rw - min_rw) / rw_range for rw in remaining_works]

    # Bottleneck elimination: LONGER min duration gets higher score (LPT-like)
    # This prevents long operations from becoming bottlenecks later
    max_md = max(min_durations) if min_durations else 1.0
    min_md = min(min_durations) if min_durations else 0.0
    md_range = max_md - min_md
    if md_range < 1e-9:
        norm_md = [0.5] * n
    else:
        norm_md = [(md - min_md) / md_range for md in min_durations]
    
    # Inverse contention: lower contention is better
    max_cont = max(contention_scores) if contention_scores else 1.0
    min_cont = min(contention_scores) if contention_scores else 1.0
    cont_range = max_cont - min_cont
    if cont_range < 1e-9:
        norm_cont = [0.5] * n
    else:
        norm_cont = [1.0 - (c - min_cont) / cont_range for c in contention_scores]

    # Remaining ops: more remaining ops means the job has more future work to unlock
    # So we prioritize jobs with more remaining ops to keep the schedule flowing
    max_ro = max(remaining_ops) if remaining_ops else 1.0
    min_ro = min(remaining_ops) if remaining_ops else 0.0
    ro_range = max_ro - min_ro
    if ro_range < 1e-9:
        norm_ro = [0.5] * n
    else:
        norm_ro = [(ro - min_ro) / ro_range for ro in remaining_ops]

    # Composite score weights - tuned for job-shop scheduling
    # Higher weight on critical path (remaining_work) as it directly impacts makespan
    # Significant weight on bottleneck elimination (LPT) to schedule long ops early
    # Machine contention: prefer ops on less-contended machines
    # Lower weight on remaining ops as a tiebreaker
    w1 = 0.40  # remaining work (critical path) - dominant factor
    w2 = 0.35  # bottleneck elimination (longer min duration preferred)
    w3 = 0.15  # machine contention (inverse)
    w4 = 0.10  # remaining ops urgency

    scores = []
    for i in range(n):
        s = w1 * norm_rw[i] + w2 * norm_md[i] + w3 * norm_cont[i] + w4 * norm_ro[i]
        scores.append(s)

    best_idx = max(range(n), key=lambda i: scores[i])
    return int(best_idx)
# EVOLVE-BLOCK-END
