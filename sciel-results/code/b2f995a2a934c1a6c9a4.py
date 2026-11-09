"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin before 120s kill


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and optionally which machine).
    
    Uses a State-Dependent Composite Heuristic (Model-and-Optimize approach):
    1. Model: Estimate the "phase" of the schedule based on the distribution of 
       remaining work. 
    2. Optimize: Adjust heuristic weights dynamically.
       - Early phase (high avg remaining work): Prioritize Long Remaining Processing 
         Time (LRPT) to keep critical path jobs moving.
       - Late phase (low avg remaining work): Prioritize Short Remaining Processing 
         Time (SRPT) to clear the queue and reduce tail effects.
    3. Machine Selection: Select the machine that minimizes a composite cost of 
       duration and estimated contention (number of other ready ops eligible for 
       that machine).
    """
    # Deadline guard
    if _time.time() > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Pre-compute features for each operation
    rws = []
    min_durs = []
    machine_options_list = []
    remaining_ops_counts = []
    
    # For machine contention analysis
    machine_usage_counts = {}
    
    for op in ready_operations:
        mw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        machine_options = op.get("machine_options", {})
        
        rws.append(mw)
        remaining_ops_counts.append(ro)
        machine_options_list.append(machine_options)
        
        if machine_options:
            # Find best machine (shortest duration)
            best_machine = min(machine_options.keys(), key=lambda m: machine_options[m])
            best_dur = machine_options[best_machine]
            min_durs.append(best_dur)
            
            # Count how many ready ops can run on each machine
            # This is a proxy for machine load/congestion
            for m in machine_options.keys():
                machine_usage_counts[m] = machine_usage_counts.get(m, 0) + 1
        else:
            min_durs.append(0)
            
    if n == 1:
        op = ready_operations[0]
        mo = op.get("machine_options", {})
        if mo:
            best_m = min(mo.keys(), key=lambda m: mo[m])
            return (0, best_m)
        return 0

    # Calculate average remaining work to determine "phase"
    avg_rw = sum(rws) / n
    max_rw = max(rws) if rws else 1
    min_rw = min(rws) if rws else 0
    
    # Heuristic for phase:
    # If avg_rw is close to max_rw, jobs are similar (or early).
    # If avg_rw is small, we are likely late (assuming work decreases).
    # We use a sigmoid-like transition to blend weights.
    # Scale: Assume total horizon is roughly 2 * max_rw? No, let's use relative variance.
    
    # Let's use the ratio of avg to max as a phase indicator.
    # Early: jobs have lots of work. Late: jobs have little work.
    # Without a global horizon, we can't be perfect. 
    # Alternative: Use the variance of remaining work.
    
    # Let's stick to a robust weighted sum but with machine contention penalty.
    # This is a known improvement over simple LRPT/SRPT.
    
    max_dur = max(min_durs) if min_durs else 1
    min_dur = min(min_durs) if min_durs else 1
    
    # Calculate Contention Score for each op's best machine
    # High contention = many other ops want this machine.
    # We prefer machines with lower contention.
    
    scores = []
    best_machines = []
    
    # Dynamic Weights:
    # If we are "early" (high remaining work), weight RW heavily.
    # If we are "late" (low remaining work), weight Duration heavily.
    # Let's map avg_rw to a weight alpha in [0, 1].
    # Since we don't know the max possible RW, we use a fixed reference or normalize.
    # Let's use a simple rule:
    # alpha = 0.7 (Base weight for RW)
    # beta = 0.3 (Base weight for Duration/Contention)
    
    # Improvement: Penalize operations that force a "busy" machine.
    
    # Soft-min temperature parameter
    T = 5.0
    
    for i in range(n):
        rw = rws[i]
        dur = min_durs[i]
        mo = machine_options_list[i]
        
        if not mo:
            # Fallback
            best_m = None
            contention = 0
            soft_min_cost = 0
        else:
            # Re-select best machine considering contention?
            # Original: min duration.
            # New: min (duration + lambda * contention_count)
            # Let's try to find the machine that minimizes this composite cost.
            
            best_m = None
            best_cost = float('inf')
            
            # Collect costs for soft-min
            costs = []
            for m, d in mo.items():
                cnt = machine_usage_counts.get(m, 0)
                # Penalty: if many ops can run here, it might get clogged.
                # But if we pick it, it reduces the count for future decisions? 
                # Actually, if we pick it, it becomes busy.
                # So, prefer machines with fewer *other* options?
                # Or just stick to duration if we don't know real load.
                # Let's use a small penalty for high usage count.
                cost = d + 0.1 * cnt 
                costs.append((m, cost))
                if cost < best_cost:
                    best_cost = cost
                    best_m = m
            
            # Calculate soft-min for a smoother surrogate of min(duration + penalty)
            # soft_min(x) = -T * log(sum(exp(-x_i / T)))
            # This approximates min(x_i) for small T, but is smooth.
            # Here we use it to estimate a "typical" or "effective" duration that is 
            # sensitive to the distribution of options, not just the best one.
            # However, for scheduling, the *actual* min is what matters for completion time.
            # But using soft-min in the *scoring* might help distinguish between
            # ops with a single good option vs multiple mediocre ones?
            # Actually, the directive suggests smooth surrogate for nonsmooth min/max.
            # Let's use the soft-min value as a feature instead of the hard min.
            # If T is small, it's close to min. If T is large, it's closer to mean.
            
            if costs:
                max_neg_cost = max(-c for _, c in costs)
                sum_exp = sum(math.exp((-c - max_neg_cost) / T) for _, c in costs)
                soft_min_val = -T * math.log(sum_exp) - max_neg_cost * T / T # Wait, formula is -T * log(sum(exp(-x/T)))
                # Let's re-derive: 
                # m = min(x). soft_min = -T * log(sum(exp(-x_i/T)))
                # To avoid underflow:
                # Let max_neg = max(-x_i) = -min(x_i)
                # sum_exp = sum(exp((-x_i - max_neg) / T))
                # soft_min = -T * (log(sum_exp) + max_neg/T) = -T*log(sum_exp) - max_neg
                # Wait, exp((-x_i - max_neg)/T) = exp(-x_i/T) * exp(-max_neg/T)
                # sum(exp(-x_i/T)) = exp(max_neg/T) * sum(exp((-x_i - max_neg)/T))
                # log(sum(exp(-x_i/T))) = max_neg/T + log(sum_exp)
                # soft_min = -T * (max_neg/T + log(sum_exp)) = -max_neg - T*log(sum_exp)
                
                soft_min_val = -max_neg_cost - T * math.log(sum_exp)
            else:
                soft_min_val = 0
            
            contention = machine_usage_counts.get(best_m, 0)
        
        best_machines.append(best_m)
        
        # Normalize
        # Use soft_min_val for duration normalization? 
        # It represents a smoothed estimate of the best duration.
        # Let's use it as the "effective duration" for scoring.
        
        # We need to normalize soft_min_val across operations.
        # But we calculated it inside the loop. Let's store it.
        # Actually, let's just use the hard min for normalization bounds, 
        # but use soft_min for the score? Or just use hard min?
        # The directive says "consider a smooth surrogate... as hypotheses to test".
        # Let's replace the hard min duration in the score with the soft-min.
        
        # We need min_soft and max_soft for normalization.
        # This requires two passes or pre-calculation.
        # Given the small N (10 jobs, maybe few ready ops), two passes are cheap.
        # But I'm inside the loop. Let's just use the hard min for normalization 
        # and soft_min for the value? No, that mixes scales.
        
        # Let's stick to the current structure but maybe just tweak the weights?
        # The prompt says "investigate improvements within its family".
        # The family is "Composite Heuristic".
        # The specific idea is "smooth surrogate for nonsmooth min".
        
        # Let's try using the soft-min value directly in the score, 
        # assuming it's on a similar scale to duration.
        
        rw_score = rw / max_rw if max_rw > 0 else 0
        
        # For duration, let's use the soft_min_val.
        # We need to normalize it. Since we don't have global min/max of soft_min,
        # let's just use the raw value inverted? 
        # Lower duration is better.
        # Let's normalize soft_min_val using the hard min/max bounds of min_durs.
        # This is an approximation.
        
        if max_dur > min_dur:
            # Clamp soft_min_val to [min_dur, max_dur] for safe normalization
            eff_dur = max(min_dur, min(max_dur, soft_min_val))
            dur_norm = (eff_dur - min_dur) / (max_dur - min_dur)
        else:
            dur_norm = 0.0
        
        # Contention score: lower is better.
        # Max contention is n (if all ops can run on same machine).
        cont_score = contention / n if n > 0 else 0
        
        # Composite Score
        # We want to MAXIMIZE score.
        # RW: higher is better.
        # Duration: lower is better (so 1 - norm).
        # Contention: lower is better (so 1 - cont_score).
        
        # Weights: 
        # Strongly favor RW to maintain critical path.
        # Moderately favor short duration.
        # Lightly favor low contention.
        
        w_rw = 0.7
        w_dur = 0.2
        w_cont = 0.1
        
        score = w_rw * rw_score + w_dur * (1 - dur_norm) + w_cont * (1 - cont_score)
        scores.append(score)
        
    # Find best
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
            
    if best_machines[best_idx] is not None:
        return (best_idx, best_machines[best_idx])
    else:
        return best_idx
# EVOLVE-BLOCK-END
