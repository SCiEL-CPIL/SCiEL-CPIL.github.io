"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from collections import defaultdict

# Deadline guard
_dispatch_start_wall = _time.time()
_MAX_WALL_SECONDS = 110.0
_dispatch_call_count = 0


def _compute_slack_score(ready_operations):
    """
    Compute slack-based scores using bipartite matching intuition.
    For each ready operation, estimate its 'slack' - how much leeway it has
    before it becomes a bottleneck. We use a soft-min over machine options
    weighted by congestion, combined with job urgency.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    
    # Precompute machine congestion: for each machine, count how many ready ops can use it
    machine_congestion = defaultdict(int)
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_congestion[m] += 1
    
    # Also track total work per machine (weighted by duration)
    machine_load = defaultdict(float)
    for op in ready_operations:
        for m, dur in op["machine_options"].items():
            machine_load[m] += dur
    
    scores = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]
        machine_options = op["machine_options"]
        
        # Compute a soft-min over machine options using log-sum-exp trick
        # We want to find the machine that gives the best "slack"
        # Slack for machine m = -(congestion[m] * weight1 + load[m] * weight2 + duration * weight3)
        # But we want to maximize slack, so we minimize the negative
        
        # For each machine, compute a "cost" that represents how bad it is to assign here
        # Then use soft-min to get a smooth estimate of the best machine's cost
        
        costs = []
        for m, dur in machine_options.items():
            # Congestion penalty: more ops competing for this machine = worse
            cong_penalty = machine_congestion[m] * 1.0
            # Load penalty: more total work on this machine = worse
            load_penalty = machine_load[m] * 0.1
            # Duration penalty: longer processing = worse for makespan
            dur_penalty = dur * 0.5
            
            cost = cong_penalty + load_penalty + dur_penalty
            costs.append(cost)
        
        costs = np.array(costs)
        
        # Soft-min: -log(sum(exp(-costs/T))) * T approximates min(costs) for large T
        # Use temperature T=1.0 for a balance between smoothness and accuracy
        T = 1.0
        if len(costs) > 0:
            max_cost = np.max(costs)
            # Numerically stable soft-min
            exp_neg_costs = np.exp(-(costs - max_cost) / T)
            sum_exp = np.sum(exp_neg_costs)
            if sum_exp > 0:
                soft_min = max_cost + T * np.log(sum_exp)
            else:
                soft_min = max_cost
        else:
            soft_min = 0.0
        
        # Job urgency: jobs with more remaining work/ops are more critical
        # But we also want to prioritize jobs that are close to completion to free them up
        # Use a combination: high remaining_work means the job has a lot left, so starting
        # any op keeps it moving. High remaining_ops means many more ops to go.
        
        # Slack estimate: lower soft_min = better machine assignment available
        # Urgency: higher remaining_work = more urgent to keep the job moving
        
        # Final score: we want to maximize (urgency - cost)
        # Normalize remaining_work to be comparable to cost scale
        # Estimate typical cost scale from all operations
        score = -soft_min + remaining_work * 0.05 + remaining_ops * 0.3
        
        scores[i] = score
    
    return int(np.argmax(scores))


def dispatch(ready_operations, time):
    global _dispatch_call_count
    _dispatch_call_count += 1

    # Fast deadline check
    if _time.time() - _dispatch_start_wall > _MAX_WALL_SECONDS:
        # Fallback: pick the operation with the most remaining work
        if not ready_operations:
            return 0
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    if not ready_operations:
        return 0

    # Use the slack-based scoring with bipartite matching intuition
    return _compute_slack_score(ready_operations)
# EVOLVE-BLOCK-END
