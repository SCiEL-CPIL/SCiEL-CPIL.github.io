"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def _simulate_sgs(durations, predecessors, successors, demands, capacities, priority_scores):
    """
    Simulate serial schedule generation scheme.
    Returns makespan.
    """
    n = len(durations)
    
    # Build pred_count for eligibility checking
    pred_count = [len(preds) for preds in predecessors]
    remaining_preds = pred_count[:]
    
    # Track resource usage over time
    # We'll use a simple approach: maintain a list of (end_time, resource_index, amount) events
    # and check availability when starting activities
    
    # For efficiency with small n, we can just track current resource usage
    # by keeping a list of active activities and their end times
    
    num_resources = 0
    if demands is not None:
        demands_arr = np.asarray(demands, dtype=float)
        if demands_arr.ndim == 2:
            num_resources = demands_arr.shape[1]
        elif demands_arr.ndim == 1:
            num_resources = 1
            demands_arr = demands_arr.reshape(-1, 1)
    else:
        demands_arr = np.zeros((n, 1))
        num_resources = 1
    
    if capacities is not None:
        caps_arr = np.asarray(capacities, dtype=float)
        if caps_arr.ndim == 0:
            caps_arr = np.full(num_resources, float(caps_arr))
        else:
            caps_arr = caps_arr[:num_resources] if len(caps_arr) >= num_resources else np.pad(caps_arr, (0, num_resources - len(caps_arr)))
    else:
        caps_arr = np.ones(num_resources)
    
    # Priority-based scheduling
    # Start time for each activity
    start_times = np.zeros(n)
    
    # Eligible set: activities with all predecessors completed
    completed = [False] * n
    
    current_time = 0.0
    scheduled_count = 0
    
    # Use a heap for efficiency: (priority, activity_index)
    # Higher priority should be scheduled first, so we negate for min-heap
    import heapq
    eligible_heap = []
    
    # Initially eligible: activities with no predecessors
    for i in range(n):
        if remaining_preds[i] == 0:
            heapq.heappush(eligible_heap, (-priority_scores[i], i))
    
    # Track resource usage: list of (end_time, resource_idx, amount)
    resource_events = []
    
    while scheduled_count < n:
        # Find the highest priority eligible activity that can be started
        best_activity = None
        best_priority = -np.inf
        
        while eligible_heap:
            neg_pri, act = heapq.heappop(eligible_heap)
            pri = -neg_pri
            
            # Check if this activity can start at current_time
            # Need to check resource availability
            can_start = True
            
            # Check resource constraints
            for r in range(num_resources):
                demand = demands_arr[act, r]
                if demand <= 0:
                    continue
                # Check if resource r is available at current_time
                # Sum up all ongoing usage of resource r
                usage = 0.0
                for end_t, r_idx, amt in resource_events:
                    if end_t > current_time and r_idx == r:
                        usage += amt
                
                if usage + demand > caps_arr[r] + 1e-9:
                    can_start = False
                    break
            
            if can_start:
                best_activity = act
                best_priority = pri
                break
            else:
                # This activity can't start now; we need to advance time
                # Find the earliest end time among active activities that use conflicting resources
                # For simplicity, let's just try to find when this activity could start
                # This is complex; for now, let's just skip and try next
                # Actually, we should put it back and try another approach
                # Let's find the minimum end time that would free up resources
                min_end = np.inf
                for end_t, r_idx, amt in resource_events:
                    if end_t > current_time:
                        for r2 in range(num_resources):
                            if demands_arr[act, r2] > 0:
                                # Check if this activity uses resource r2
                                # We need to see if freeing this would help
                                # This is getting complex; let's use a simpler approach
                                pass
                
                # Simpler: just advance to the next event time
                if resource_events:
                    next_time = min(end_t for end_t, _, _ in resource_events if end_t > current_time)
                    if next_time > current_time:
                        # Advance time and remove expired events
                        current_time = next_time
                        resource_events = [(et, ri, amt) for et, ri, amt in resource_events if et > current_time]
                        # Re-try this activity
                        heapq.heappush(eligible_heap, (neg_pri, act))
                        break
                else:
                    # No active resources, should be able to start
                    best_activity = act
                    best_priority = pri
                    break
        
        if best_activity is None:
            # No eligible activity can start; advance time to next event
            if resource_events:
                next_time = min(end_t for end_t, _, _ in resource_events if end_t > current_time)
                current_time = next_time
                resource_events = [(et, ri, amt) for et, ri, amt in resource_events if et > current_time]
                # Re-add all previously skipped activities
                # This is getting complex; let's use a simpler simulation approach
                # For now, break to avoid infinite loop
                break
            else:
                break
        
        # Start best_activity at current_time
        start_times[best_activity] = current_time
        end_time = current_time + durations[best_activity]
        
        # Add resource events
        for r in range(num_resources):
            demand = demands_arr[best_activity, r]
            if demand > 0:
                resource_events.append((end_time, r, demand))
        
        # Mark as completed
        completed[best_activity] = True
        scheduled_count += 1
        
        # Update successors
        for succ in successors[best_activity]:
            remaining_preds[succ] -= 1
            if remaining_preds[succ] == 0:
                heapq.heappush(eligible_heap, (-priority_scores[succ], succ))
    
    # Makespan is the maximum end time
    if scheduled_count < n:
        return float('inf')
    
    makespan = max(start_times[i] + durations[i] for i in range(n))
    return makespan

def _compute_features(instance):
    """Compute normalized feature vectors for each activity."""
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    predecessors = instance.get("predecessors", [])
    successors = [[] for _ in range(n)]
    pred_count = [0] * n
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        pred_count[i] = len(preds)
        for p in preds:
            successors[p].append(i)
    
    # Topological sort
    in_degree = pred_count[:]
    queue = deque()
    for i in range(n):
        if in_degree[i] == 0:
            queue.append(i)
    
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    
    if len(topo_order) < n:
        # Cycle; return simple features
        return durations.copy(), None, None
    
    # Backward pass
    dist_to_end = np.zeros(n)
    for i in reversed(range(len(topo_order))):
        node = topo_order[i]
        max_succ = 0.0
        for succ in successors[node]:
            if dist_to_end[succ] > max_succ:
                max_succ = dist_to_end[succ]
        dist_to_end[node] = durations[node] + max_succ
    
    # Resource scarcity
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    resource_score = np.zeros(n)
    if demands is not None and capacities is not None:
        demands_arr = np.asarray(demands, dtype=float)
        capacities_arr = np.asarray(capacities, dtype=float)
        
        if demands_arr.ndim == 1:
            if capacities_arr.ndim == 0:
                cap = float(capacities_arr)
                if cap > 0:
                    resource_score = demands_arr / cap
            else:
                cap = float(np.max(capacities_arr)) if len(capacities_arr) > 0 else 1.0
                if cap > 0:
                    resource_score = demands_arr / cap
        else:
            safe_caps = np.where(capacities_arr > 0, capacities_arr, 1.0)
            if len(safe_caps) < demands_arr.shape[1]:
                safe_caps = np.pad(safe_caps, (0, demands_arr.shape[1] - len(safe_caps)))
            ratio = demands_arr / safe_caps[np.newaxis, :]
            resource_score = np.sum(ratio, axis=1)
    
    # Downstream criticality
    descendant_crit = np.zeros(n)
    for i in reversed(range(len(topo_order))):
        node = topo_order[i]
        total = 0.0
        for succ in successors[node]:
            total += dist_to_end[succ] + descendant_crit[succ]
        descendant_crit[node] = total
    
    def normalize(arr):
        mn = np.min(arr)
        mx = np.max(arr)
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    features = np.column_stack([
        normalize(dist_to_end),
        normalize(resource_score),
        normalize(descendant_crit),
        normalize(durations)
    ])
    
    return features, predecessors, successors

def _nelder_mead(objective, x0, max_iter=50, tol=1e-4):
    """Simple Nelder-Mead optimization."""
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        point = np.array(x0, dtype=float)
        point[i] += 0.1
        simplex.append(point)
    
    fvals = [objective(x) for x in simplex]
    
    for iteration in range(max_iter):
        # Sort simplex by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        x_reflect = centroid + alpha * (centroid - simplex[-1])
        f_reflect = objective(x_reflect)
        
        if fvals[0] <= f_reflect < fvals[-2]:
            simplex[-1] = x_reflect
            fvals[-1] = f_reflect
        elif f_reflect < fvals[0]:
            # Expansion
            gamma = 2.0
            x_expand = centroid + gamma * (x_reflect - centroid)
            f_expand = objective(x_expand)
            if f_expand < f_reflect:
                simplex[-1] = x_expand
                fvals[-1] = f_expand
            else:
                simplex[-1] = x_reflect
                fvals[-1] = f_reflect
        else:
            # Contraction
            rho = 0.5
            x_contract = centroid + rho * (simplex[-1] - centroid)
            f_contract = objective(x_contract)
            if f_contract < fvals[-1]:
                simplex[-1] = x_contract
                fvals[-1] = f_contract
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
    
    # Return best point
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def priority(instance):
    """
    Compute priority scores using Nelder-Mead optimization over feature weights.
    The objective is the makespan from a simulated SGS.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    features, predecessors, successors = _compute_features(instance)
    
    if features is None or features.shape[1] == 0:
        return np.zeros(len(instance["durations"]))
    
    n_features = features.shape[1]
    
    # Initial weights: emphasize critical path
    x0 = np.array([0.4, 0.35, 0.25, 0.0])
    x0 = x0 / np.sum(x0)  # Normalize to sum to 1
    
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    def objective(weights):
        if time.time() > deadline:
            return float('inf')
        # Ensure weights are non-negative and normalized
        w = np.maximum(weights, 0)
        if np.sum(w) < 1e-12:
            w = np.ones(n_features) / n_features
        else:
            w = w / np.sum(w)
        
        priority_scores = features @ w
        makespan = _simulate_sgs(durations, predecessors, successors, demands, capacities, priority_scores)
        return makespan
    
    # Run Nelder-Mead
    try:
        best_weights, best_value = _nelder_mead(objective, x0, max_iter=30, tol=1e-3)
        
        if time.time() > deadline:
            # Timeout: use initial weights
            w = x0
            w = np.maximum(w, 0)
            if np.sum(w) < 1e-12:
                w = np.ones(n_features) / n_features
            else:
                w = w / np.sum(w)
            return features @ w
        
        # Ensure weights are valid
        w = np.maximum(best_weights, 0)
        if np.sum(w) < 1e-12:
            w = np.ones(n_features) / n_features
        else:
            w = w / np.sum(w)
        
        return features @ w
    
    except Exception:
        # Fallback to static weights
        w = np.array([0.4, 0.35, 0.25, 0.0])
        w = np.maximum(w, 0)
        if np.sum(w) < 1e-12:
            w = np.ones(n_features) / n_features
        else:
            w = w / np.sum(w)
        return features @ w
# EVOLVE-BLOCK-END
