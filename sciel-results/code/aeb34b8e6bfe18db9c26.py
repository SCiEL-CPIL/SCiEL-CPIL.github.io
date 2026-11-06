"""Open-shop dispatch rule: Combined MWR job selection with SPT machine assignment."""
# EVOLVE-BLOCK-START
import time
import math

# Global time budget tracking
_START_TIME = time.time()
# Evaluator kills at 150s wall-clock. We reserve safety margin.
_DEADLINE = _START_TIME + 145.0


def _time_left():
    return _DEADLINE - time.time()


def _smooth_max(values, temperature=1.0):
    """
    Smooth max using log-sum-exp (softmax approximation).
    For a set of values, approximates max with a smooth function.
    """
    if not values:
        return 0.0
    max_val = max(values)
    if max_val < 0:
        # Avoid overflow in exp
        scaled = [v - max_val for v in values]
        sum_exp = sum(math.exp(v / temperature) for v in scaled)
        return max_val + temperature * math.log(sum_exp / len(values))
    else:
        scaled = [v / temperature for v in values]
        sum_exp = sum(math.exp(v - max(scaled)) for v in scaled)
        return temperature * (max(scaled) + math.log(sum_exp / len(values)))


def _combined_score(op, all_ops):
    """
    Compute a combined score for an operation that balances:
    1. Most Work Remaining (MWR) - prioritize jobs with most total remaining work
    2. Shortest Processing Time (SPT) - prefer ops that can complete quickly
    3. Load balancing - consider the spread of durations across machines
    
    Lower score is better (we want to pick the operation that best reduces makespan).
    """
    remaining_work = op.get("remaining_work", 0)
    remaining_ops = op.get("remaining_ops", 1)
    machine_options = op["machine_options"]
    
    if not machine_options:
        return float('inf')
    
    durations = list(machine_options.values())
    min_dur = min(durations)
    max_dur = max(durations)
    mean_dur = sum(durations) / len(durations)
    
    # Component 1: MWR - more remaining work means more urgent (higher urgency = lower score)
    # Normalize by typical remaining work across all ops
    max_rw = max(o.get("remaining_work", 0) for o in all_ops)
    if max_rw > 0:
        mwr_component = 1.0 - (remaining_work / max_rw)  # 0 if max, 1 if min
    else:
        mwr_component = 0.5
    
    # Component 2: SPT - shorter min duration is better
    # Normalize by max possible duration
    max_possible = max(max(o["machine_options"].values()) for o in all_ops if o["machine_options"])
    if max_possible > 0:
        spt_component = min_dur / max_possible  # 0 if shortest, 1 if longest
    else:
        spt_component = 0.5
    
    # Component 3: Load balancing - prefer ops where the best machine is much shorter than others
    # This indicates we can assign to a less-loaded machine
    if max_dur > 0:
        # Ratio: how much shorter is the best vs worst? Higher ratio = better balance potential
        balance_ratio = (max_dur - min_dur) / max_dur
        balance_component = 1.0 - balance_ratio  # 0 if no balance benefit, 1 if perfect
    else:
        balance_component = 0.5
    
    # Component 4: Remaining ops factor - jobs with more ops left need more attention
    max_ops = max(o.get("remaining_ops", 1) for o in all_ops)
    if max_ops > 0:
        ops_component = 1.0 - (remaining_ops / max_ops)
    else:
        ops_component = 0.5
    
    # Weighted combination: MWR is primary, SPT secondary, balance tertiary
    # Use smooth weighting to avoid hard thresholds
    score = (
        0.40 * mwr_component +
        0.35 * spt_component +
        0.15 * balance_component +
        0.10 * ops_component
    )
    
    return score


def dispatch(ready_operations, time):
    """
    Dispatch rule for open-shop scheduling.
    
    Strategy:
    1. Score each ready operation using a combined heuristic:
       - MWR (Most Work Remaining): prioritize jobs with most total remaining work
       - SPT (Shortest Processing Time): prefer ops that can complete quickly
       - Load balancing: consider machine availability
    2. Select the operation with the lowest combined score
    
    This approach uses a smooth surrogate to approximate the makespan objective,
    balancing urgency and efficiency to minimize overall completion time.
    """
    if not ready_operations:
        return 0
    
    # Time guard: if we're running low on time, use a simple fallback
    if _time_left() < 0.5:
        # Simple MWR fallback
        max_remaining = -1
        best_idx = 0
        for i, op in enumerate(ready_operations):
            rw = op.get("remaining_work", 0)
            if rw > max_remaining:
                max_remaining = rw
                best_idx = i
        return best_idx
    
    # Compute combined scores for all ready operations
    best_idx = 0
    best_score = float('inf')
    
    for i, op in enumerate(ready_operations):
        score = _combined_score(op, ready_operations)
        if score < best_score:
            best_score = score
            best_idx = i
    
    return best_idx
# EVOLVE-BLOCK-END
