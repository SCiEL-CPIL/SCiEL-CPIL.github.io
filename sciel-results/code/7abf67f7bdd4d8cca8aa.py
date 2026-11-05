"""Job-shop dispatch rule: Ratio-based critical path with soft-min machine assignment."""
# EVOLVE-BLOCK-START
import time as _time
import math

_dispatch_start_time = _time.time()
_DEADLINE_MARGIN = 140.0


def _deadline_exceeded():
    return (_time.time() - _dispatch_start_time) > _DEADLINE_MARGIN


def _softmin(values, temperature=0.3):
    """Smooth min via log-sum-exp. Returns a value slightly above the true min."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    arr = [v / temperature for v in values]
    m = min(arr)
    # Guard against overflow
    shifted = [a - m for a in arr]
    s = math.exp(shifted[0])
    for a in shifted[1:]:
        s += math.exp(a)
    return (m + math.log(s)) * temperature


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.

    Strategy:
    1. Primary signal: Ratio of remaining work to estimated completion cost
       (RW / softmin_cost). This prioritizes jobs that are both deep AND
       have feasible machine options, balancing depth against feasibility.
    2. Soft-min over machine options with contention-aware cost gives a smooth
       surrogate for the nonsmooth min-duration selection.
    3. Small urgency bonus for jobs with fewer remaining operations to break
       ties and encourage parallelism.
    4. Returns (index, chosen_machine) for the selected operation.
    """
    if not ready_operations:
        return 0
    if _deadline_exceeded():
        return 0

    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        mo = op.get("machine_options", {})
        if mo:
            best_m = min(mo.keys(), key=lambda m: mo[m])
            return (0, best_m)
        return 0

    # Build machine contention counts: how many ready ops can use each machine
    machine_counts = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m in mo:
            machine_counts[m] = machine_counts.get(m, 0) + 1

    # Soft-min temperature: lower = sharper min, higher = smoother
    T = 0.25
    # Urgency weight: small bonus for fewer remaining ops
    W_URG = 0.05

    best_idx = 0
    best_machine = None
    best_score = float("-inf")

    for i in range(n):
        op = ready_operations[i]
        rw = float(op.get("remaining_work", 0) or 0)
        rem_ops = op.get("remaining_ops", 0) or 0

        mo = op.get("machine_options", {})
        if not mo:
            # Fallback: score based on RW alone
            score = rw
            if score > best_score:
                best_score = score
                best_idx = i
                best_machine = None
            continue

        # Build contention-adjusted costs for each machine option
        costs = []
        for m, d in mo.items():
            c = machine_counts.get(m, 1)
            # Cost model: duration scaled by contention factor
            # When multiple ops target the same machine, effective delay grows
            cost = d * (1.0 + 0.4 * (c - 1))
            costs.append(cost)

        # Soft-min gives a smooth estimate of the best achievable cost
        sm_cost = _softmin(costs, T)

        # Primary: ratio RW / cost (higher is better)
        if sm_cost > 0:
            ratio_score = rw / sm_cost
        else:
            ratio_score = rw  # degenerate case

        # Urgency bonus: fewer remaining ops → slight preference
        urgency_bonus = W_URG * (1.0 / (1.0 + rem_ops)) if rem_ops else 0.0

        score = ratio_score + urgency_bonus

        if score > best_score:
            best_score = score
            best_idx = i
            # Choose the machine that actually minimizes the cost (not the soft-min)
            min_cost_val = float("inf")
            chosen_m = None
            for m, d in mo.items():
                c = machine_counts.get(m, 1)
                cost = d * (1.0 + 0.4 * (c - 1))
                if cost < min_cost_val:
                    min_cost_val = cost
                    chosen_m = m
            best_machine = chosen_m

    if best_machine is not None:
        return (int(best_idx), best_machine)
    else:
        return int(best_idx)

# EVOLVE-BLOCK-END
