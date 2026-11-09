"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Deadline guard to ensure we finish well before the 150s limit
_DEADLINE = 145.0
_START_TIME = _time.time()


def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        opts = op.get("machine_options", {})
        if opts:
            # If only one op is ready, pick the fastest machine
            best_machine = min(opts, key=opts.get)
            return (0, best_machine)
        return 0

    # Check deadline
    if _time.time() - _START_TIME > _DEADLINE:
        return 0

    # 1. Precompute metrics for all ready operations
    # We want to identify operations that are "critical" (high work, few options)
    
    op_metrics = []
    machine_load_estimate = {}  # machine -> estimated load (sum of work of ops that can use it)
    
    # First pass: calculate basic metrics and machine load contributions
    for op in ready_operations:
        opts = op.get("machine_options", {})
        work = op.get("remaining_work", 0.0)
        num_ops = op.get("remaining_ops", 0)
        flex = len(opts) if opts else 0
        
        # Add this operation's work to the "potential load" of each machine it can use
        # This is a rough estimate of congestion: if many high-work ops can use machine M,
        # machine M is likely to be busy.
        if opts:
            for m in opts:
                if m not in machine_load_estimate:
                    machine_load_estimate[m] = 0.0
                machine_load_estimate[m] += work
        
        op_metrics.append({
            'work': work,
            'ops': num_ops,
            'flex': flex,
            'opts': opts
        })

    # Normalize metrics to create a "Priority Score" for each operation
    # High work is generally good to start early (MWRF), but low flexibility is urgent.
    
    works = np.array([m['work'] for m in op_metrics], dtype=np.float64)
    flexes = np.array([m['flex'] for m in op_metrics], dtype=np.float64)
    ops_counts = np.array([m['ops'] for m in op_metrics], dtype=np.float64)
    
    def normalize(x):
        if np.max(x) - np.min(x) < 1e-9:
            return np.zeros_like(x)
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    
    norm_work = normalize(works)
    # Inverse normalize flexibility: lower flexibility (fewer options) should be higher urgency
    norm_flex_inv = 1.0 - normalize(flexes)
    norm_ops = normalize(ops_counts)
    
    # Combine into a single priority score for the operation itself
    # Weights: Work is most important, but constrained jobs (low flex) are also critical.
    # This is a structural change: we are not just picking the job with max work,
    # but the job with the best "urgency" profile.
    
    # Adaptive weighting: if all jobs have similar work, rely more on flexibility/urgency.
    work_spread = np.std(works) / (np.mean(works) + 1e-9)
    if work_spread < 0.2:
        w_work, w_flex = 0.4, 0.6
    else:
        w_work, w_flex = 0.7, 0.3
        
    # Small weight for remaining ops to break ties (more ops left = more complex job)
    w_ops = 0.1
    
    op_priority = w_work * norm_work + w_flex * norm_flex_inv + w_ops * norm_ops
    
    # 2. Score each (operation, machine) pair
    # For each ready operation, and each machine it can run on, calculate a pair score.
    # We want to maximize the pair score.
    
    best_pair_score = -float('inf')
    best_op_idx = 0
    best_machine = None
    
    # Total ready work for normalization of congestion
    total_ready_work = sum(works) if sum(works) > 0 else 1.0
    
    for i, m_data in enumerate(op_metrics):
        opts = m_data['opts']
        if not opts:
            continue
            
        op_prior = op_priority[i]
        
        for m, dur in opts.items():
            # Machine load estimate: how much work is "waiting" or can go to this machine?
            # We use the precomputed machine_load_estimate.
            m_load = machine_load_estimate.get(m, 0.0)
            
            # Congestion factor: 0 if no load, 1 if high load.
            # Normalize load by total ready work to make it scale-invariant.
            congestion = m_load / total_ready_work
            
            # Duration factor: shorter is better.
            # Normalize duration by the max duration for this specific operation?
            # Or global max? Let's use a relative duration score.
            # If dur is 1 and min is 1, score is 1. If dur is 10 and min is 1, score is 0.1.
            min_dur = min(opts.values())
            dur_score = min_dur / dur if dur > 0 else 0.0
            
            # Pair Score Calculation:
            # We want high op_priority, high dur_score (fast machine), and low congestion.
            # Congestion penalty: if a machine is heavily loaded by other ops, starting a job
            # on it might be fine if it's the only option, but generally we prefer less congested
            # paths to spread load.
            
            # Structural idea: "Slack" approximation.
            # If an op has high priority and a machine is fast AND not too congested, it's a great pair.
            
            # Weight for congestion: if the machine is used by many other high-work ops,
            # it's a bottleneck.
            congestion_penalty = 0.5 * congestion
            
            # Combine:
            # Score = OpPriority * (DurScore * (1 - CongestionPenalty))
            # This encourages assigning high-priority ops to fast, free machines.
            
            pair_score = op_prior * (dur_score * (1.0 - congestion_penalty))
            
            # Alternative: Additive mix to avoid zeroing out if congestion is high but op is critical.
            # If op is critical (high priority), we should pick it even if congestion is high.
            # Score = op_prior * 0.8 + dur_score * 0.2 - congestion_penalty * 0.5
            # Let's try a multiplicative with a floor.
            
            # Let's use a weighted sum for stability, but bias towards fast machines for high priority ops.
            # High priority op on slow machine: bad.
            # High priority op on fast machine: good.
            # Low priority op on fast machine: ok, but maybe let high priority use it?
            
            # Refined Score:
            # Base: Op Priority
            # Bonus: If machine is fast (high dur_score)
            # Penalty: If machine is congested
            
            # If Op Priority is high, we want the best machine for it.
            # If Op Priority is low, we can afford to use a slightly slower/less congested machine.
            
            # Let's go with:
            score = op_prior + 0.3 * dur_score - 0.4 * congestion_penalty
            
            if score > best_pair_score:
                best_pair_score = score
                best_op_idx = i
                best_machine = m

    # Fallback if no machine found (should not happen)
    if best_machine is None:
        return 0

    return (best_op_idx, best_machine)

# EVOLVE-BLOCK-END
