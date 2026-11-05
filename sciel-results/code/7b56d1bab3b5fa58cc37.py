"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque


def priority(instance):
    """
    Compute a dynamic resource-weighted criticality score for each activity.
    
    Approach:
    1. Extract durations, demands, capacities, and predecessors.
    2. Compute static CPM criticality (longest path from activity to end).
    3. Simulate a greedy serial schedule using CPM criticality as priority.
    4. Measure resource-induced delay for each activity in this simulation.
    5. Combine CPM criticality, resource delay, and demand-to-capacity ratio.
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    n = len(durations)
    
    # Handle edge cases
    if demands is None or capacities is None:
        # Fallback to longest duration first
        return durations.copy()
    
    demands = np.asarray(demands)
    capacities = np.asarray(capacities)
    
    # Ensure demands is 2D (n_activities x n_resources)
    if demands.ndim == 1:
        demands = demands.reshape(-1, 1)
    
    # Ensure capacities is 1D (n_resources,)
    if capacities.ndim == 2:
        capacities = capacities.flatten()
    
    n_resources = len(capacities)
    
    # Build predecessor and successor sets
    if predecessors is None:
        pred_sets = [set() for _ in range(n)]
    else:
        pred_sets = [set() for _ in range(n)]
        for i, preds in enumerate(predecessors):
            if preds is not None:
                pred_sets[i] = set(preds)
    
    succ_sets = [set() for _ in range(n)]
    for i in range(n):
        for p in pred_sets[i]:
            succ_sets[p].add(i)
    
    # Topological Sort
    in_degree = [len(pred_sets[i]) for i in range(n)]
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    top_order = []
    while queue:
        node = queue.popleft()
        top_order.append(node)
        for s in succ_sets[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # Compute Longest Path to End (CPM Criticality)
    longest_to_end = [0.0] * n
    for node in reversed(top_order):
        if succ_sets[node]:
            longest_to_end[node] = max(longest_to_end[s] for s in succ_sets[node]) + durations[node]
        else:
            longest_to_end[node] = durations[node]
    
    # Normalize CPM Criticality
    max_lte = max(longest_to_end) if max(longest_to_end) > 0 else 1.0
    norm_lte = np.array([lte / max_lte for lte in longest_to_end])
    
    # Compute Demand-to-Capacity Ratio (DCR)
    # Average ratio per activity
    dcr = np.zeros(n)
    for i in range(n):
        ratios = []
        for r in range(n_resources):
            if capacities[r] > 0:
                ratios.append(demands[i][r] / capacities[r])
        if ratios:
            dcr[i] = np.mean(ratios)
    
    max_dcr = max(dcr) if max(dcr) > 0 else 1.0
    norm_dcr = dcr / max_dcr
    
    # Simulate a greedy schedule using CPM Criticality as priority
    # This helps identify which activities are truly bottlenecked by resources
    # even when they are high priority.
    
    sim_priority = longest_to_end.copy()
    scheduled = [False] * n
    start_times = [0.0] * n
    end_times = [0.0] * n
    remaining_preds = [len(pred_sets[i]) for i in range(n)]
    
    # Track resource delay: start_time - earliest_possible_start
    # Earliest possible start is max(end_time of predecessors)
    resource_delay = [0.0] * n
    
    # Event-based simulation
    time = 0.0
    scheduled_count = 0
    
    # Active activities: list of (end_time, activity_index)
    active_activities = []
    resource_usage = [0.0] * n_resources
    
    # To efficiently find eligible activities, we can just iterate all unscheduled
    # N is small (30), so O(N^2) is fine.
    
    while scheduled_count < n:
        # Find eligible activities
        eligible = [i for i in range(n) if not scheduled[i] and remaining_preds[i] == 0]
        
        if not eligible:
            # No eligible activities, advance time to next completion
            if active_activities:
                time = min(end for end, _ in active_activities)
                # Complete activities ending at this time
                new_active = []
                for end, idx in active_activities:
                    if end <= time:
                        for r in range(n_resources):
                            resource_usage[r] -= demands[idx][r]
                        for s in succ_sets[idx]:
                            remaining_preds[s] -= 1
                    else:
                        new_active.append((end, idx))
                active_activities = new_active
            else:
                # Should not happen if valid DAG
                break
            continue
        
        # Sort eligible by simulation priority (CPM Criticality)
        eligible_sorted = sorted(eligible, key=lambda x: -sim_priority[x])
        
        # Try to schedule the highest priority eligible activity
        scheduled_any = False
        for act in eligible_sorted:
            # Check if resources are available
            can_schedule = True
            for r in range(n_resources):
                if resource_usage[r] + demands[act][r] > capacities[r] + 1e-9:
                    can_schedule = False
                    break
            
            if can_schedule:
                # Calculate earliest possible start time (based on predecessors)
                # Since we are in a serial SGS, 'time' is the current simulation time.
                # The activity *could* have started earlier if resources were free,
                # but we are simulating a specific schedule.
                # For "resource delay", we compare actual start time to the
                # max end time of its predecessors.
                max_pred_end = 0.0
                for p in pred_sets[act]:
                    if end_times[p] > max_pred_end:
                        max_pred_end = end_times[p]
                
                start_times[act] = time
                end_times[act] = time + durations[act]
                
                # Delay is how much later it started than its predecessors finished
                if start_times[act] > max_pred_end:
                    resource_delay[act] = start_times[act] - max_pred_end
                else:
                    resource_delay[act] = 0.0
                
                scheduled[act] = True
                scheduled_count += 1
                for r in range(n_resources):
                    resource_usage[r] += demands[act][r]
                active_activities.append((end_times[act], act))
                scheduled_any = True
                break
            else:
                # Resource contention: this activity is blocked
                # We don't increment a counter here, we rely on the delay metric
                # which will be captured when it *does* eventually schedule.
                # However, if it's blocked now, it will be considered again later.
                pass
        
        if not scheduled_any:
            # All eligible activities are blocked by resources
            # Advance time to next completion
            if active_activities:
                time = min(end for end, _ in active_activities)
                # Complete activities ending at this time
                new_active = []
                for end, idx in active_activities:
                    if end <= time:
                        for r in range(n_resources):
                            resource_usage[r] -= demands[idx][r]
                        for s in succ_sets[idx]:
                            remaining_preds[s] -= 1
                    else:
                        new_active.append((end, idx))
                active_activities = new_active
            else:
                # Should not happen
                break
    
    # Normalize Resource Delay
    max_delay = max(resource_delay) if max(resource_delay) > 0 else 1.0
    norm_delay = np.array([d / max_delay for d in resource_delay])
    
    # Estimate Resource Tightness
    # If average DCR is high, resources are tight, so delay is more important
    avg_dcr = np.mean(dcr)
    # Scale weight of delay based on tightness
    # If avg_dcr is close to 1, weight is high.
    tightness = min(avg_dcr, 1.0)
    
    # Weights
    # Base weights
    w_cpm = 0.5
    w_delay = 0.35
    w_dcr = 0.15
    
    # Adjust: if tight, increase delay weight more aggressively
    w_delay *= (1.0 + 2.0 * tightness)
    
    # Normalize weights
    total_w = w_cpm + w_delay + w_dcr
    w_cpm /= total_w
    w_delay /= total_w
    w_dcr /= total_w
    
    # Final Score
    scores = w_cpm * norm_lte + w_delay * norm_delay + w_dcr * norm_dcr
    
    return scores
# EVOLVE-BLOCK-END
