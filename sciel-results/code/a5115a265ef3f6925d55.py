import numpy as np
import heapq

def priority(instance):
    durations = np.asarray(instance["durations"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    predecessors = instance["predecessors"]

    n = len(durations)
    n_res = len(capacities)

    # Build successors
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)

    # CPM Forward Pass
    es = np.zeros(n)
    ef = np.zeros(n)
    # Topological order for forward pass
    in_degree = [len(predecessors[i]) for i in range(n)]
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
        
    for u in topo_order:
        if predecessors[u]:
            es[u] = max(ef[p] for p in predecessors[u])
        else:
            es[u] = 0
        ef[u] = es[u] + durations[u]
        
    makespan_cpm = max(ef) if n > 0 else 0.0

    # CPM Backward Pass
    lf = np.full(n, makespan_cpm)
    ls = np.full(n, makespan_cpm)
    for u in reversed(topo_order):
        if successors[u]:
            lf[u] = min(ls[s] for s in successors[u])
        else:
            lf[u] = makespan_cpm
        ls[u] = lf[u] - durations[u]
        
    total_float = ls - es

    # Internal SGS using Minimum Total Float as heuristic
    # We simulate the schedule to get a good ordering
    
    scheduled = [False] * n
    remaining = list(range(n))
    # We need to track resource availability over time.
    # Since N is small (30), we can just iterate through time events.
    
    # Events: times when activities finish.
    # State: current time, active activities, available resources.
    
    # To simplify, we can use a list of (start_time, end_time, activity_id)
    # and simulate step by step.
    
    # However, implementing a full SGS inside is risky if not careful.
    # Let's use a simpler proxy: 
    # The "order" of activities in a SGS using Min Slack is often correlated with 
    # their "Critical Path Position".
    
    # Let's stick to the "Schedule-Induced" idea but ensure it's robust.
    # If the internal SGS fails or takes too long, fallback to CPM slack.
    
    # Given N=30, a simple SGS is fast.
    
    active = [] # list of (end_time, activity_id)
    resource_usage = np.zeros(n_res)
    time = 0.0
    order = []
    
    # Pre-compute eligibility based on predecessors
    # We need to know when preds finish.
    # We can just check predecessors' finish times.
    
    finish_times = np.full(n, np.inf)
    
    while len(order) < n:
        # Find eligible activities
        eligible = []
        for i in range(n):
            if i in order or any(i == x[1] for x in active):
                continue
            # Check predecessors
            if all(p in order or (p in [x[1] for x in active] and finish_times[p] <= time) for p in predecessors[i]):
                # Actually, if pred is active, it hasn't finished yet.
                # So pred must be in order (finished) OR (active and finished at time? No, active means running).
                # In SGS, we only start activities when resources are free at time t.
                # Predecessors must be *finished*.
                if all(p in order for p in predecessors[i]):
                    # Check resources
                    if np.all(resource_usage + demands[i] <= capacities):
                        eligible.append(i)
        
        if not eligible:
            # Advance time to next finish
            if active:
                next_time = min(t for t, _ in active)
                # Remove finished activities
                new_active = []
                for t, aid in active:
                    if t == next_time:
                        finish_times[aid] = t
                        resource_usage -= demands[aid]
                    else:
                        new_active.append((t, aid))
                active = new_active
                time = next_time
            else:
                # No active, no eligible? Deadlock? Should not happen if DAG is valid.
                # Force schedule any remaining with no preds?
                break
        else:
            # Pick best from eligible based on Minimum Total Float (lower is better)
            # Tie-breaker: Higher resource pressure?
            # Let's use Min Total Float, then Min ES (from CPM)
            best = min(eligible, key=lambda i: (total_float[i], es[i]))
            
            # Schedule best
            order.append(best)
            finish_time = time + durations[best]
            active.append((finish_time, best))
            resource_usage += demands[best]
            
    # If we broke out early, fill remaining
    if len(order) < n:
        for i in range(n):
            if i not in order:
                order.append(i)
                
    # Generate priority scores based on this order
    # Earliest in order -> Highest priority
    rank = np.zeros(n)
    for idx, aid in enumerate(order):
        rank[aid] = n - 1 - idx
        
    return rank
