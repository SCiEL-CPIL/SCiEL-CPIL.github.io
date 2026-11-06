import numpy as np
import time
from collections import deque
import random

_T0 = time.time()
_DEADLINE = 150.0
_MARGIN = 5.0

def _elapsed():
    return time.time() - _T0

def _topo_order(n, preds):
    """Return topological order and successor list."""
    indeg = [0] * n
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            indeg[i] += 1
            succ[p].append(i)
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) < n:  # cycle fallback
        order = list(range(n))
    return order, succ

def _simulate_schedule(durations, demands, capacities, preds, succ, eligible, rng):
    """
    Simulate a complete schedule starting from a given set of eligible activities.
    Uses a simple priority rule (e.g., shortest remaining processing time or random)
    to pick the next activity, respecting resource constraints.
    Returns the makespan.
    """
    n = len(durations)
    # Track state
    scheduled = [False] * n
    remaining_pred = [0] * n
    for i in range(n):
        remaining_pred[i] = len(preds[i])
    
    # Current resource usage
    current_usage = np.zeros(len(capacities), dtype=float)
    
    # Queue of activities that are ready (all predecessors done)
    ready = list(eligible)
    # Initialize remaining_pred for those already eligible? No, eligible means preds done.
    # But we need to track which ones are NOT yet scheduled.
    # Actually, let's just use a set of unscheduled activities and check eligibility on the fly
    # or maintain a ready list properly.
    
    # Let's restart state properly for simulation
    scheduled = [False] * n
    in_degree = [0] * n
    for i in range(n):
        in_degree[i] = len(preds[i])
        
    # Pre-compute successors for quick update
    # succ is already provided
    
    ready = []
    for i in range(n):
        if in_degree[i] == 0:
            ready.append(i)
            
    current_time = 0.0
    makespan = 0.0
    
    while ready:
        # Pick an activity from ready
        # Strategy: Random choice weighted by some heuristic? 
        # For MCTS, we usually want a fast rollout. 
        # Let's try: pick the one with shortest duration? Or random?
        # Random is unbiased but slow convergence. 
        # Let's try a simple heuristic: pick the one with longest tail? 
        # But we don't have tail precomputed here easily without overhead.
        # Let's stick to a simple rule: Random from ready.
        
        idx = rng.randint(0, len(ready) - 1)
        act = ready[idx]
        ready.pop(idx)
        
        # Check resource availability
        # In a serial SGS, we assume we can always start if resources are available.
        # If resources are NOT available, we would have to wait.
        # But in a serial SGS simulation, we typically advance time to when resources are free.
        # This is complex. 
        # Alternative: Assume we just check if it fits. If not, we might need to wait.
        # For simplicity in rollout, let's assume we can always start it at current_time 
        # IF resources are available. If not, we find the earliest time it can start.
        
        # Find earliest start time
        earliest_start = current_time
        for r in range(len(capacities)):
            if demands[act, r] > 0:
                # We need to find when resource r will have enough capacity.
                # This requires tracking future releases, which is complex in a simple rollout.
                # Approximation: Assume resources are only released when activities finish.
                # We don't track finish times of others here easily.
                
                # Simplified assumption for rollout: 
                # If current_usage[r] + demands[act, r] > capacities[r]:
                #   We cannot start now. 
                #   To keep it simple, let's just skip resource contention in the rollout 
                #   and only use it for the first step? No, that defeats the purpose.
                
                # Let's try a different approach for rollout:
                # Just use a greedy serial SGS that always picks the highest priority 
                # from the ready list based on a static rule (like tail).
                # But we are inside a simulation that was started by picking 'act' first.
                
                # Actually, the MCTS node value is determined by the rollouts.
                # If the rollout is too slow, we can't do many.
                
                # Let's simplify the rollout:
                # 1. Start with 'act' at t=0.
                # 2. For the rest, use a simple serial SGS with a fixed priority rule (e.g., min duration).
                # This makes the rollout O(n^2) or O(n*R).
                
                pass
        
        # This is getting complicated. Let's use a simpler rollout strategy:
        # Rollout = Serial SGS with a specific priority rule (e.g., Shortest Processing Time).
        # The "choice" in MCTS is just the first activity.
        # Then we complete the schedule greedily.
        
        # Let's restructure:
        # The function _simulate_schedule should take the first activity and then complete the rest greedily.
        
        break # Break out to use the simpler approach below

    return makespan

def _complete_schedule_greedy(durations, demands, capacities, preds, succ, first_act, rng=None):
    """
    Complete a schedule starting with 'first_act' at time 0.
    Uses a greedy serial SGS for the remaining activities.
    Priority rule: Shortest Remaining Processing Time (SRPT) or similar.
    """
    n = len(durations)
    if n == 0:
        return 0.0
        
    scheduled = [False] * n
    in_degree = [0] * n
    for i in range(n):
        in_degree[i] = len(preds[i])
        
    # Track resource usage over time? 
    # For serial SGS, we process one activity at a time.
    # We need to know the earliest start time for each activity based on:
    # 1. Predecessor finish times
    # 2. Resource availability
    
    finish_times = [0.0] * n
    current_usage = np.zeros(len(capacities), dtype=float)
    # To handle resource conflicts in serial SGS, we usually need to know when resources are free.
    # A simple approximation: maintain a list of (end_time, resource, amount) events.
    # Or, for each new activity, check all resources and find the max end time among 
    # activities that conflict? That's O(n^2).
    
    # Since n=30, O(n^2) is fine (900 ops).
    
    # List of activities to consider
    ready = []
    for i in range(n):
        if in_degree[i] == 0:
            ready.append(i)
            
    # Force first_act to be scheduled first?
    # The MCTS idea is: "What if we pick 'first_act' first?"
    # So we schedule 'first_act' at t=0.
    
    if first_act is not None:
        if in_degree[first_act] != 0:
            # Invalid start, shouldn't happen if we only pick eligible
            return float('inf')
            
        # Schedule first_act
        scheduled[first_act] = True
        start_t = 0.0
        finish_times[first_act] = start_t + durations[first_act]
        
        # Update resources
        for r in range(len(capacities)):
            current_usage[r] += demands[first_act, r]
            
        # Update successors
        for s in succ[first_act]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                ready.append(s)
                
        # Remove first_act from ready if it was there
        if first_act in ready:
            ready.remove(first_act)
    else:
        # No forced first activity, just start normally
        pass

    # Now schedule the rest using Serial SGS
    # At each step, pick the eligible activity with the best priority.
    # Priority: Min Duration? Max Tail?
    # Let's use Max Tail (critical path) as the greedy rule for the rollout.
    # We need to precompute tails.
    
    # Precompute tails for the original graph
    order, _ = _topo_order(n, preds)
    tail = np.zeros(n, dtype=float)
    for i in reversed(order):
        best = 0.0
        for s in succ[i]:
            if tail[s] > best:
                best = tail[s]
        tail[i] = durations[i] + best
        
    while ready:
        # Pick activity with max tail from ready
        # If tie, pick min duration?
        best_idx = -1
        best_key = (-1, 0) # (-tail, -duration) ? No, we want max tail.
        # Let's just pick max tail.
        max_tail = -1
        chosen = -1
        for act in ready:
            if tail[act] > max_tail:
                max_tail = tail[act]
                chosen = act
            elif tail[act] == max_tail:
                # Tie breaker: shorter duration
                if durations[act] < durations[chosen]:
                    chosen = act
                    
        if chosen == -1:
            break
            
        # Determine start time for 'chosen'
        # 1. Predecessor constraint: max(finish_times[p] for p in preds[chosen])
        pred_finish = 0.0
        for p in preds[chosen]:
            if finish_times[p] > pred_finish:
                pred_finish = finish_times[p]
                
        # 2. Resource constraint:
        # We need to find the earliest time t >= pred_finish such that
        # for all r, current_usage[r] + demands[chosen, r] <= capacities[r]
        # 
        # How to track current_usage?
        # In serial SGS, activities are started sequentially.
        # But they overlap in time.
        # So "current_usage" is not just a scalar, it's a function of time.
        #
        # Efficient way: Keep a list of active intervals for each resource.
        # Or, since n is small, we can just check against all previously scheduled activities.
        
        earliest_start = pred_finish
        for r in range(len(capacities)):
            if demands[chosen, r] == 0:
                continue
                
            # Find the max time until which resource r is "full" or "over capacity"
            # We need to find the earliest t >= earliest_start such that
            # sum(demands[j, r] for j in scheduled if start_j < t < finish_j) + demands[chosen, r] <= capacities[r]
            
            # This is a classic problem.
            # Since n is small, we can just iterate through potential start times.
            # Candidate start times: earliest_start, and finish times of all scheduled activities.
            
            candidates = [earliest_start]
            for j in range(n):
                if scheduled[j]:
                    candidates.append(finish_times[j])
                    
            # Sort candidates
            candidates = sorted(set([c for c in candidates if c >= earliest_start]))
            
            found = False
            for t in candidates:
                # Check if we can start at t
                usage_at_t = 0.0
                for j in range(n):
                    if scheduled[j]:
                        # If j overlaps with [t, t+duration)
                        # Actually, we just need to know usage at time t.
                        # An activity j is active at time t if start_j <= t < finish_j
                        # But we don't store start_j explicitly, we can derive it?
                        # No, we need to store start times.
                        
                        pass # Need start_times array
                # This is getting messy. Let's store start_times.
                break
        
        # Let's restart with a cleaner data structure for the greedy rollout.
        break
    
    # This manual implementation is error-prone. 
    # Let's write a helper function for the greedy rollout that is clean.
    
    return float('inf')

def _greedy_rollout(durations, demands, capacities, preds, succ, first_act):
    """
    Perform a greedy serial SGS rollout starting with 'first_act' (if not None).
    Returns the makespan.
    """
    n = len(durations)
    if n == 0:
        return 0.0

    # Precompute tails
    order, _ = _topo_order(n, preds)
    tail = np.zeros(n, dtype=float)
    for i in reversed(order):
        best = 0.0
        for s in succ[i]:
            if tail[s] > best:
                best = tail[s]
        tail[i] = durations[i] + best

    scheduled = [False] * n
    in_degree = [0] * n
    for i in range(n):
        in_degree[i] = len(preds[i])
        
    start_times = [0.0] * n
    finish_times = [0.0] * n
    
    # List of (end_time, activity_index) to track resource releases?
    # Or just check all scheduled activities for resource conflicts.
    
    ready = []
    for i in range(n):
        if in_degree[i] == 0:
            ready.append(i)
            
    # Step 1: Schedule first_act if provided
    if first_act is not None:
        if in_degree[first_act] != 0:
            return float('inf')
            
        scheduled[first_act] = True
        start_times[first_act] = 0.0
        finish_times[first_act] = durations[first_act]
        
        for s in succ[first_act]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                ready.append(s)
                
        if first_act in ready:
            ready.remove(first_act)
            
    # Step 2: Greedy schedule the rest
    # Priority: Max Tail
    while ready:
        # Select activity with max tail
        chosen = -1
        max_t = -1.0
        for act in ready:
            if tail[act] > max_t:
                max_t = tail[act]
                chosen = act
            elif tail[act] == max_t:
                if durations[act] < durations[chosen]:
                    chosen = act
                    
        if chosen == -1:
            break
            
        # Determine start time
        # 1. Predecessor constraint
        t_pred = 0.0
        for p in preds[chosen]:
            if finish_times[p] > t_pred:
                t_pred = finish_times[p]
                
        # 2. Resource constraint
        # We need the earliest t >= t_pred such that for all r:
        # sum(demands[j, r] for j in scheduled if start_times[j] <= t < finish_times[j]) + demands[chosen, r] <= capacities[r]
        
        t_res = t_pred
        # Candidate times: t_pred and all finish_times of scheduled activities
        candidates = [t_pred]
        for j in range(n):
            if scheduled[j]:
                candidates.append(finish_times[j])
                
        # Sort and unique
        candidates = sorted(set(candidates))
        
        valid_t = None
        for t in candidates:
            if t < t_pred:
                continue
            ok = True
            for r in range(len(capacities)):
                if demands[chosen, r] == 0:
                    continue
                usage = 0.0
                for j in range(n):
                    if scheduled[j]:
                        # Check if j is active at time t
                        # Active if start_times[j] <= t < finish_times[j]
                        if start_times[j] <= t < finish_times[j]:
                            usage += demands[j, r]
                if usage + demands[chosen, r] > capacities[r]:
                    ok = False
                    break
            if ok:
                valid_t = t
                break
                
        if valid_t is None:
            # Should not happen if capacities are sufficient, but just in case
            valid_t = t_pred
            
        start_times[chosen] = valid_t
        finish_times[chosen] = valid_t + durations[chosen]
        scheduled[chosen] = True
        
        # Update ready list
        for s in succ[chosen]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                ready.append(s)
                
        if chosen in ready:
            ready.remove(chosen)
            
    makespan = max(finish_times) if n > 0 else 0.0
    return makespan

def priority(instance):
    """
    Score each activity for a serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Uses MCTS-inspired scoring:
    1. Identify eligible activities (in-degree 0).
    2. For each eligible activity, simulate a greedy rollout where that activity is scheduled first.
    3. Score is based on the negative of the resulting makespan (lower makespan = higher score).
    4. For non-eligible activities, use a static heuristic (tail) to maintain a total ordering.
    """
    if _elapsed() > _DEADLINE - _MARGIN:
        return np.asarray(instance["durations"], dtype=float)

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    preds = instance["predecessors"]
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)

    if demands.ndim == 1:
        if n > 0:
            demands = demands.reshape(n, -1)
        else:
            demands = demands.reshape(0, 0)
    R = demands.shape[1] if demands.size else 0

    order, succ = _topo_order(n, preds)
    
    # Compute static tail for non-eligible or tie-breaking
    tail = np.zeros(n, dtype=float)
    for i in reversed(order):
        best = 0.0
        for s in succ[i]:
            if tail[s] > best:
                best = tail[s]
        tail[i] = durations[i] + best

    # Initialize scores with tail (static heuristic)
    scores = tail.copy()
    
    # Identify eligible activities
    eligible = [i for i in range(n) if len(preds[i]) == 0]
    
    if not eligible:
        # No eligible activities? Shouldn't happen in a valid DAG.
        return np.asarray(scores, dtype=float)
        
    # MCTS-like evaluation:
    # For each eligible activity, run a greedy rollout starting with that activity.
    # Use the makespan from the rollout to determine the score.
    
    makespans = {}
    for act in eligible:
        if _elapsed() > _DEADLINE - _MARGIN:
            break
        ms = _greedy_rollout(durations, demands, capacities, preds, succ, act)
        makespans[act] = ms
        
    if not makespans:
        return np.asarray(scores, dtype=float)
        
    # Normalize makespans to scores
    # Lower makespan is better -> Higher score
    min_ms = min(makespans.values())
    max_ms = max(makespans.values())
    
    if max_ms == min_ms:
        # All makespans are the same, use tail as tie-breaker
        for act in eligible:
            scores[act] = tail[act]
    else:
        for act, ms in makespans.items():
            # Map [min_ms, max_ms] to [1.0, 0.0] ?
            # Or just use -ms?
            # Let's use a normalized score where the best makespan gets the highest score.
            # score = 1.0 - (ms - min_ms) / (max_ms - min_ms)
            # But we need to combine with tail for non-eligible?
            # For eligible, the MCTS score should dominate.
            norm_score = 1.0 - (ms - min_ms) / (max_ms - min_ms)
            # Combine with tail to break ties or add stability?
            # Let's just use the MCTS score for eligible, scaled to be comparable to tail.
            # Tail is in time units. Makespan is in time units.
            # Let's just use -ms as the score? No, higher is better.
            # So score = -ms.
            # But we need to return a single array.
            # Let's set scores[act] = -ms
            scores[act] = -ms
            
    # For non-eligible activities, keep the tail score.
    # But we need to ensure eligible activities are prioritized over non-eligible?
    # The serial SGS only picks from eligible. So the relative order of non-eligible doesn't matter 
    # until they become eligible.
    # However, the priority function returns a score for ALL activities.
    # The SGS picks the highest priority among ELIGIBLE.
    # So we just need the eligible ones to have the correct relative order.
    # The non-eligible ones can have any score, as they won't be picked.
    
    # However, if we use -ms for eligible and tail for non-eligible, 
    # and tail is positive while -ms is negative, 
    # eligible will always have lower scores than non-eligible (if tail is positive).
    # This is fine because SGS only looks at eligible.
    
    # But wait, if we return negative scores for eligible and positive for non-eligible,
    # and the SGS picks "highest priority", it will correctly pick from eligible.
    
    # Let's verify the sign.
    # "Higher score = higher priority"
    # So we want eligible activities to have the highest scores among themselves.
    # And we don't care about their score relative to non-eligible, as long as 
    # the SGS implementation filters for eligibility first.
    # The problem statement says: "a serial schedule-generation scheme starts the highest-priority eligible activity".
    # So it filters by eligibility, then picks max priority.
    
    return np.asarray(scores, dtype=float)
