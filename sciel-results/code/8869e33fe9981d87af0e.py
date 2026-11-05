"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(instance):
    """
    Compute a dynamic resource-weighted criticality score for each activity.
    
    Approach:
    1. Extract durations, demands, capacities, and predecessors.
    2. Simulate a greedy serial schedule to identify resource contention.
    3. Compute static CPM criticality (longest path from activity to end).
    4. Combine with empirical bottleneck detection (activities that caused
       resource contention during the greedy schedule) and demand-to-capacity ratios.
    5. Return priority scores (higher = should be scheduled first).
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
    
    # Build predecessor sets
    if predecessors is None:
        pred_sets = [set() for _ in range(n)]
    else:
        pred_sets = [set() for _ in range(n)]
        for i, preds in enumerate(predecessors):
            if preds is not None:
                pred_sets[i] = set(preds)
    
    # Build successor sets for CPM calculation
    succ_sets = [set() for _ in range(n)]
    for i in range(n):
        for p in pred_sets[i]:
            succ_sets[p].add(i)
    
    # Compute longest path from each activity to the end (reverse CPM)
    # Using topological order
    in_degree = [len(pred_sets[i]) for i in range(n)]
    from collections import deque
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    top_order = []
    while queue:
        node = queue.popleft()
        top_order.append(node)
        for s in succ_sets[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # Reverse topological order for computing longest path to end
    longest_to_end = [0.0] * n
    for node in reversed(top_order):
        if succ_sets[node]:
            longest_to_end[node] = max(longest_to_end[s] for s in succ_sets[node]) + durations[node]
        else:
            longest_to_end[node] = durations[node]
    
    # Normalize longest_to_end to [0, 1]
    max_lte = max(longest_to_end) if max(longest_to_end) > 0 else 1.0
    norm_lte = [lte / max_lte for lte in longest_to_end]
    
    # Simulate a greedy serial schedule to detect resource contention
    # Priority: use longest duration first as initial heuristic for the simulation
    sim_priority = durations.copy()
    
    scheduled = [False] * n
    start_times = [0.0] * n
    end_times = [0.0] * n
    remaining_preds = [len(pred_sets[i]) for i in range(n)]
    
    # Track resource usage over time to detect contention
    # We'll track which activities were delayed due to resource constraints
    resource_contention_count = [0] * n  # How many times each activity was blocked by resources
    contention_severity = [0.0] * n  # Severity of contention when blocked
    
    # Simulate the schedule
    time = 0.0
    scheduled_count = 0
    
    # We need to track active activities and resource usage
    # For simplicity, we'll simulate step by step
    
    # Event-based simulation
    # At each step, find eligible activities (all predecessors scheduled)
    # among eligible, pick highest priority, check if resources are available
    # If not, wait until resources free up
    
    active_activities = []  # List of (end_time, activity_index)
    resource_usage = [0.0] * n_resources  # Current resource usage
    
    while scheduled_count < n:
        # Find eligible activities
        eligible = [i for i in range(n) if not scheduled[i] and remaining_preds[i] == 0]
        
        if not eligible:
            # No eligible activities, advance time to next completion
            if active_activities:
                time = min(end for end, _ in active_activities)
                # Complete activities ending at this time
                completed = [idx for end, idx in active_activities if end <= time]
                for idx in completed:
                    active_activities.remove((end_times[idx], idx))
                    for r in range(n_resources):
                        resource_usage[r] -= demands[idx][r]
                    # Update remaining preds for successors
                    for s in succ_sets[idx]:
                        remaining_preds[s] -= 1
                # Remove all completed
                active_activities = [(end, idx) for end, idx in active_activities if end > time]
            else:
                # Should not happen if there's a valid DAG
                break
            continue
        
        # Sort eligible by simulation priority
        eligible_sorted = sorted(eligible, key=lambda x: -sim_priority[x])
        
        # Try to schedule the highest priority eligible activity
        scheduled_any = False
        for act in eligible_sorted:
            # Check if resources are available
            can_schedule = True
            for r in range(n_resources):
                if resource_usage[r] + demands[act][r] > capacities[r]:
                    can_schedule = False
                    break
            
            if can_schedule:
                # Schedule this activity
                start_times[act] = time
                end_times[act] = time + durations[act]
                scheduled[act] = True
                scheduled_count += 1
                for r in range(n_resources):
                    resource_usage[r] += demands[act][r]
                active_activities.append((end_times[act], act))
                scheduled_any = True
                break
            else:
                # Resource contention: this activity is blocked
                # Record contention
                resource_contention_count[act] += 1
                # Severity: how much over capacity we'd be
                max_over = 0.0
                for r in range(n_resources):
                    over = resource_usage[r] + demands[act][r] - capacities[r]
                    if over > 0:
                        max_over = max(max_over, over / max(capacities[r], 1.0))
                contention_severity[act] = max(contention_severity[act], max_over)
        
        if not scheduled_any:
            # All eligible activities are blocked by resources
            # Advance time to next completion
            if active_activities:
                time = min(end for end, _ in active_activities)
                # Complete activities ending at this time
                completed = [idx for end, idx in active_activities if end <= time]
                for idx in completed:
                    active_activities.remove((end_times[idx], idx))
                    for r in range(n_resources):
                        resource_usage[r] -= demands[idx][r]
                    # Update remaining preds for successors
                    for s in succ_sets[idx]:
                        remaining_preds[s] -= 1
                # Remove all completed
                active_activities = [(end, idx) for end, idx in active_activities if end > time]
            else:
                # Should not happen
                break
    
    # Compute demand-to-capacity ratio for each activity
    # Average over all resources
    dcr = [0.0] * n
    for i in range(n):
        ratios = []
        for r in range(n_resources):
            if capacities[r] > 0:
                ratios.append(demands[i][r] / capacities[r])
        if ratios:
            dcr[i] = sum(ratios) / len(ratios)
    
    # Normalize DCR to [0, 1]
    max_dcr = max(dcr) if max(dcr) > 0 else 1.0
    norm_dcr = [d / max_dcr for d in dcr]
    
    # Combine scores:
    # - Criticality (longest path to end): weight 0.5
    # - Resource contention: weight 0.3
    # - Demand-to-capacity ratio: weight 0.2
    max_contention = max(resource_contention_count) if max(resource_contention_count) > 0 else 1
    norm_contention = [c / max_contention for c in resource_contention_count]
    
    # Also consider contention severity
    max_sev = max(contention_severity) if max(contention_severity) > 0 else 1.0
    norm_sev = [s / max_sev for s in contention_severity]
    
    # Final score: higher means should be scheduled first
    scores = [0.0] * n
    for i in range(n):
        score = 0.5 * norm_lte[i] + 0.3 * norm_contention[i] + 0.2 * norm_dcr[i]
        # Add a small bonus for high severity contention
        score += 0.1 * norm_sev[i]
        # Slight boost for longer durations (tie-breaker)
        max_dur = max(durations) if max(durations) > 0 else 1.0
        score += 0.1 * (durations[i] / max_dur)
        scores[i] = score
    
    return np.asarray(scores, dtype=float)
# EVOLVE-BLOCK-END
