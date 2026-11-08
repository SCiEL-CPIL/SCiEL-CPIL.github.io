"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

_DEADLINE_MARGIN = 5.0
_START_TIME = None


def _now():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.time()
    return _time.time() - _START_TIME


def _deadline_exceeded():
    return _now() > (150.0 - _DEADLINE_MARGIN)


def _get_min_duration_and_machine(op):
    """
    Returns (min_duration, best_machine) for the operation.
    """
    opts = op.get("machine_options", {})
    if not opts:
        return 0, -1
    best_machine = min(opts.keys(), key=lambda m: opts[m])
    return opts[best_machine], best_machine


def dispatch(ready_operations, time):
    """
    Dispatch rule based on Critical Path Priority and WSPT.
    
    Strategy:
    1. For each ready operation, calculate a 'priority score'.
    2. The score combines:
       - WSPT (Weighted Shortest Processing Time): Favors short, high-weight jobs.
       - Critical Path Urgency: Favors jobs that are close to completion or have high remaining work relative to ops.
       - Congestion Penalty: Penalizes operations that would block many other ready ops on the same machine.
    
    We select the operation with the LOWEST score (highest priority).
    """
    if not ready_operations:
        return 0

    if len(ready_operations) == 1:
        return 0

    if _deadline_exceeded():
        # Fallback: Shortest Processing Time (SPT)
        best_idx = 0
        best_dur = float('inf')
        for i, op in enumerate(ready_operations):
            dur, _ = _get_min_duration_and_machine(op)
            if dur < best_dur:
                best_dur = dur
                best_idx = i
        return int(best_idx)

    n = len(ready_operations)
    
    # Pre-calculate machine conflicts to avoid O(N^2) inside the main loop if possible,
    # but N is small (10 jobs * 5 ops = 50 max ready ops usually, but ready set is smaller).
    # Let's stick to O(N^2) as N is likely < 20 in any single dispatch call.
    
    # Store pre-calculated data for each op to avoid recomputation
    op_data = []
    for i, op in enumerate(ready_operations):
        dur, best_mach = _get_min_duration_and_machine(op)
        remaining_ops = op.get("remaining_ops", 1)
        remaining_work = op.get("remaining_work", 0)
        
        # If remaining_ops is 0, it's the last op.
        if remaining_ops == 0:
            remaining_ops = 1
        
        # 1. WSPT Component:
        # Weight is proportional to remaining work? Or remaining ops?
        # Standard WSPT uses weight. Here, let's use remaining_work as weight.
        # If remaining_work is 0, use remaining_ops.
        weight = remaining_work if remaining_work > 0 else remaining_ops
        if weight == 0:
            weight = 1.0
            
        wspt = dur / weight
        
        # 2. Critical Path / Urgency Component:
        # Jobs with fewer remaining ops are "closer" to finishing.
        # If we delay a job with 1 op left, we delay the whole job.
        # If we delay a job with 10 ops left, the impact is diluted.
        # Urgency factor: 1 / remaining_ops? Or remaining_ops itself?
        # Let's use a "progress" metric. 
        # High remaining_work / low remaining_ops = High urgency.
        urgency = remaining_work / remaining_ops
        
        # 3. Congestion Component:
        # Count how many other ready ops want the same best_machine.
        # This is calculated in the main loop to be precise, 
        # but we can pre-compute machine usage counts.
        
        op_data.append({
            'index': i,
            'dur': dur,
            'best_mach': best_mach,
            'wspt': wspt,
            'urgency': urgency,
            'remaining_ops': remaining_ops
        })

    # Pre-compute machine contention counts
    machine_counts = {}
    for data in op_data:
        m = data['best_mach']
        if m != -1:
            machine_counts[m] = machine_counts.get(m, 0) + 1

    best_idx = 0
    best_score = float('inf')

    for data in op_data:
        i = data['index']
        dur = data['dur']
        best_mach = data['best_mach']
        wspt = data['wspt']
        urgency = data['urgency']
        remaining_ops = data['remaining_ops']
        
        # Congestion penalty
        # If this op goes, how many others are blocked?
        # If best_mach is -1, no congestion.
        if best_mach != -1:
            contention = machine_counts.get(best_mach, 0)
        else:
            contention = 0
            
        # Score Calculation:
        # We want to minimize this score.
        
        # Component 1: Processing time weighted by contention.
        # If we pick a long op on a busy machine, it's bad.
        # If we pick a short op on a busy machine, it's less bad.
        # If we pick any op on a free machine, it's good.
        
        # Let's blend WSPT and Congestion.
        # Score = wspt * (1 + 0.5 * contention) + urgency_penalty
        
        # Actually, WSPT already accounts for weight.
        # Let's try: Score = dur * (1 + contention * 0.1) + (1 / urgency)
        # Wait, higher urgency (more work per op) should be HIGHER priority (LOWER score).
        # So we should SUBTRACT urgency or divide by it?
        # Let's look at the inspiration: "WSPT ratio, critical path weight, and urgency bonus".
        # Usually, we maximize priority. Let's stick to minimizing "cost".
        
        # Cost of processing: dur
        # Cost of contention: dur * contention * factor
        # Cost of delay: If this job is urgent, delaying it is expensive.
        # But dispatching it NOW removes it from the queue.
        
        # Let's use a composite score:
        # 1. Base cost: dur
        # 2. Contention cost: dur * (contention - 1) * 0.2  (penalize blocking others)
        # 3. Urgency discount: If urgency is high, we want to pick it. 
        #    So we subtract a term proportional to urgency?
        #    Or divide the whole thing by (1 + urgency)?
        
        # Let's try a multiplicative model for priority, then invert for score.
        # Priority = (1 / wspt) * (1 + urgency) / (1 + contention * 0.5)
        # We want to MAXIMIZE Priority.
        # So we MINIMIZE 1/Priority.
        
        if wspt == 0:
            wspt = 0.0001
            
        # Normalize urgency to a reasonable range?
        # urgency = remaining_work / remaining_ops.
        
        # Let's try a simpler additive score which is more robust to scaling issues:
        # Score = (dur * (1 + 0.2 * contention)) - (urgency * 0.5)
        # This favors short durations, low contention, and high urgency.
        
        # However, WSPT (dur/weight) is a strong heuristic.
        # Let's combine:
        # Score = wspt * (1 + 0.1 * contention)
        # This favors short, high-weight jobs, and penalizes contention on the chosen machine.
        
        # Let's add a small tie-breaker for "critical path" feel:
        # Jobs with fewer remaining ops are more critical to finish.
        # So if wspt is similar, pick the one with fewer remaining_ops.
        
        score = wspt * (1.0 + 0.1 * contention)
        
        # Add a small penalty for having many remaining ops? 
        # No, WSPT uses weight (remaining_work). 
        # If remaining_work is high, weight is high, wspt is low. Good.
        # If remaining_ops is high, but remaining_work is also high, wspt is low.
        
        # Let's add a "completion bonus".
        # If remaining_ops == 1, it's the last step. Finishing it finishes the job.
        # This is very valuable.
        if remaining_ops == 1:
            score -= 0.01 * dur # Slight bonus for finishing jobs
        
        if score < best_score:
            best_score = score
            best_idx = i
        elif score == best_score:
            # Tie-break: Lower remaining_ops is better (closer to done)
            if op_data[best_idx]['remaining_ops'] > remaining_ops:
                best_idx = i

    return int(best_idx)
# EVOLVE-BLOCK-END
