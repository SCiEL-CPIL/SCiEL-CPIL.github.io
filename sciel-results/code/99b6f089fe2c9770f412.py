"""Job-shop dispatch rule: Critical Path with Contention-Aware Penalty."""
# EVOLVE-BLOCK-START
import time as _time
import math

_dispatch_start_time = _time.time()
_DEADLINE_MARGIN = 135.0  # safety margin before 150s kill


def _deadline_exceeded():
    return (_time.time() - _dispatch_start_time) > _DEADLINE_MARGIN


def dispatch(ready_operations, time):
    """
    Pick the best ready operation to schedule next.
    
    Strategy: Critical Path (CP) prioritization with contention-aware penalty.
    1. Primary signal: Critical Path Length (approximated by remaining_work).
       Jobs with more remaining work are more critical for makespan.
    2. Contention Penalty: Penalize operations on heavily contended machines,
       but scale the penalty such that the most critical job on a machine is
       penalized least. This encourages scheduling the "right" job on bottlenecks.
    3. Tie-breaker: Shortest Processing Time (efficiency).
    """
    if not ready_operations:
        return 0
    
    if _deadline_exceeded():
        return 0
    
    n = len(ready_operations)
    
    # Precompute features
    min_durations = []
    best_machines = []
    remaining_works = []
    
    for op in ready_operations:
        rw = op.get("remaining_work", 0)
        mo = op.get("machine_options", {})
        
        if mo:
            # Find machine with shortest duration
            best_m, min_d = min(mo.items(), key=lambda item: item[1])
        else:
            best_m = None
            min_d = 0
            
        min_durations.append(min_d)
        best_machines.append(best_m)
        remaining_works.append(rw)
    
    # Calculate machine contention
    # Count how many ready operations have this machine as their *best* (shortest duration) option
    machine_counts = {}
    for m in best_machines:
        if m is not None:
            machine_counts[m] = machine_counts.get(m, 0) + 1
            
    # Also count how many operations *can* use each machine (broader contention)
    # This might be more indicative of actual load
    machine_available_counts = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m in mo.keys():
            machine_available_counts[m] = machine_available_counts.get(m, 0) + 1

    # Normalize remaining_work for comparison
    max_rw = max(remaining_works) if remaining_works else 1.0
    if max_rw <= 0:
        max_rw = 1.0
        
    # Calculate scores
    scores = []
    for i in range(n):
        rw = remaining_works[i]
        min_d = min_durations[i]
        best_m = best_machines[i]
        
        # Primary Score: Critical Path Urgency
        # Higher remaining_work means higher priority
        base_score = rw
        
        # Contention Penalty
        # We penalize being on a busy machine.
        # However, if this op has the highest remaining_work among those competing for the same machine,
        # the penalty should be reduced (it's the "right" one to schedule).
        
        penalty = 0.0
        if best_m is not None:
            # Use the count of ops that *prefer* this machine (min duration)
            count_pref = machine_counts.get(best_m, 1)
            # Use the count of ops that *can* use this machine for a broader view
            count_avail = machine_available_counts.get(best_m, 1)
            
            # Blend the two counts for a robust contention metric
            # weight 0.5 for preference, 0.5 for availability
            effective_contention = 0.5 * count_pref + 0.5 * count_avail
            
            if effective_contention > 1:
                # Determine the "criticality rank" of this op on this machine
                # Find the max remaining_work among all ops that prefer this machine
                max_rw_on_machine = 0
                for j in range(n):
                    if best_machines[j] == best_m:
                        if remaining_works[j] > max_rw_on_machine:
                            max_rw_on_machine = remaining_works[j]
                
                # If this op is the most critical on this machine, penalty is low.
                # If it's the least critical, penalty is high.
                if max_rw_on_machine > 0:
                    criticality_ratio = rw / max_rw_on_machine
                else:
                    criticality_ratio = 1.0
                
                # Penalty scales with contention and inverse criticality
                # Base penalty factor
                penalty_base = 0.1 * (effective_contention - 1)
                # Modulate by criticality: 1.0 - criticality_ratio
                # If criticality_ratio is 1.0 (most critical), penalty is 0.
                # If criticality_ratio is 0.0 (least critical), penalty is max.
                penalty = penalty_base * (1.0 - criticality_ratio)
        
        # Efficiency Tie-breaker
        # Add a small bonus for shorter operations to keep the system moving
        # Scale by 1/max_rw to keep it small relative to base_score
        efficiency_bonus = 0.01 * (1.0 / (min_d + 1.0)) * (max_rw / 10.0 if max_rw > 0 else 0.1)
        
        final_score = base_score - penalty + efficiency_bonus
        scores.append(final_score)
    
    # Select best index
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i
            
    return int(best_idx)

# EVOLVE-BLOCK-END
