"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq

_DEADLINE_SECONDS = 145.0
_START_TIME = time.time()

def _remaining_budget():
    return _DEADLINE_SECONDS - (time.time() - _START_TIME)

def priority(instance):
    """
    Computes a static priority vector. 
    In this strategy, we use a hybrid of Longest Duration and Resource Intensity.
    However, the real power comes from the fact that the evaluator might just 
    use this static vector. But wait, the prompt says:
    "a serial schedule-generation scheme starts the highest-priority eligible activity"
    
    If the evaluator uses a FIXED priority vector returned by this function, 
    then dynamic re-evaluation inside the scheduling loop is NOT possible 
    unless the evaluator allows the priority function to be called repeatedly 
    or if we implement the scheduling logic ourselves and the evaluator 
    just runs our code?
    
    RE-READING CONTRACT:
    "Write a Python priority(instance) function that scores each activity; 
     a serial schedule-generation scheme starts the highest-priority eligible 
     activity under precedence and renewable-resource limits."
     
    This implies the EVALUATOR runs the serial SGS. It calls `priority(instance)` 
    ONCE to get a vector, and then uses that static vector to pick the next 
    eligible activity. It does NOT call priority again after scheduling one activity.
    
    THEREFORE: We cannot do dynamic re-prioritization inside the loop. 
    We must return a SINGLE static vector.
    
    However, we can make this static vector "smarter" by simulating the 
    schedule ourselves to see which activities are likely to be bottlenecks, 
    or by using a more sophisticated static metric like "Total Float" or 
    "Resource-Weighted Critical Path".
    
    Let's try a different static metric: 
    1. Compute Critical Path Length (CPL) for each node (backward pass).
    2. Compute Resource Pressure: Sum of demands / Capacity for each resource.
    3. Combine: Priority = CPL * (1 + avg_resource_pressure)
    
    Actually, a known good heuristic for static SGS is "Minimum Slack" or 
    "Latest Start Time" (LST). 
    LST = Project Duration (if no resources) - Duration - Forward Pass Time.
    Lower LST means more urgent.
    So Priority = -LST (higher score = more urgent).
    
    Let's compute Forward Pass (Earliest Start/Finish) and Backward Pass 
    (Latest Start/Finish) to get Slack.
    Priority = -Slack. If Slack is 0, it's on critical path, highest priority.
    
    To make it even better, we can weight by duration to break ties in favor 
    of longer tasks on the critical path (to finish them sooner).
    Priority = -Slack + 0.1 * Duration
    """
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    n = len(durations)
    if n == 0:
        return np.array([])
        
    # If predecessors is not provided, assume no precedence (all parallel)
    if predecessors is None:
        predecessors = [[] for _ in range(n)]
        
    # 1. Forward Pass (Earliest Start ES, Earliest Finish EF)
    # ES[i] = max(EF[p] for p in preds[i])
    # EF[i] = ES[i] + D[i]
    es = np.zeros(n, dtype=float)
    ef = np.zeros(n, dtype=float)
    
    # Topological sort order to ensure predecessors are processed first
    # Simple Kahn's algorithm
    in_degree = np.zeros(n, dtype=int)
    successors = [[] for _ in range(n)]
    for i in range(n):
        preds = predecessors[i]
        if preds:
            in_degree[i] = len(preds)
            for p in preds:
                successors[p].append(i)
                
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # If cycle exists, topo_order might be shorter than n. 
    # Assuming valid DAG as per task description.
    
    for u in topo_order:
        if predecessors[u]:
            es[u] = max(ef[p] for p in predecessors[u])
        else:
            es[u] = 0.0
        ef[u] = es[u] + durations[u]
        
    # Project makespan (ignoring resources)
    makespan_no_res = np.max(ef) if n > 0 else 0.0
    
    # 2. Backward Pass (Latest Start LS, Latest Finish LF)
    # LF[i] = min(LS[s] for s in succs[i])
    # LS[i] = LF[i] - D[i]
    ls = np.zeros(n, dtype=float)
    lf = np.zeros(n, dtype=float)
    
    for u in reversed(topo_order):
        if successors[u]:
            lf[u] = min(ls[s] for s in successors[u])
        else:
            lf[u] = makespan_no_res
        ls[u] = lf[u] - durations[u]
        
    # 3. Slack = LS - ES
    slack = ls - es
    
    # 4. Resource Pressure Metric
    # We want to prioritize activities that use scarce resources.
    # If demands and capacities are available, compute a "pressure" score.
    # Pressure = sum(demand[i][r] / capacity[r])
    # This is a static estimate.
    pressure = np.zeros(n, dtype=float)
    if demands is not None and capacities is not None:
        # demands is usually a matrix (n x R) or list of lists
        # capacities is a vector (R,)
        try:
            d_mat = np.asarray(demands, dtype=float)
            c_vec = np.asarray(capacities, dtype=float)
            # Normalize demands by capacities
            # d_mat / c_vec broadcasts
            normalized = d_mat / c_vec
            # Sum across resources to get total pressure per activity
            pressure = np.sum(normalized, axis=1)
        except Exception:
            pressure = np.zeros(n, dtype=float)
            
    # 5. Combine
    # High priority = Low Slack (urgent) + High Pressure (bottleneck)
    # Normalize slack and pressure to 0-1 range to combine reasonably?
    # Or just weight them.
    
    # Let's try: Priority = -slack + 0.5 * pressure
    # Note: slack is non-negative. Lower is better. So -slack is higher for critical.
    # Pressure is non-negative. Higher is better (more resource consumption).
    
    # To be safe against scaling issues, let's normalize.
    max_slack = np.max(slack) if np.max(slack) > 0 else 1.0
    max_pressure = np.max(pressure) if np.max(pressure) > 0 else 1.0
    
    norm_slack = slack / max_slack
    norm_pressure = pressure / max_pressure
    
    # We want to prioritize low slack (high urgency) and high pressure.
    # Score = (1 - norm_slack) + 0.5 * norm_pressure
    # This gives a score between 0 and 1.5.
    
    priority_vec = (1.0 - norm_slack) + 0.5 * norm_pressure
    
    # Add a tiny bit of duration to break ties in favor of longer tasks?
    # Actually, if two tasks have same slack and pressure, the longer one 
    # usually should go first to clear the path.
    # Let's add a small duration term.
    max_dur = np.max(durations) if np.max(durations) > 0 else 1.0
    priority_vec += 0.1 * (durations / max_dur)
    
    return priority_vec

# EVOLVE-BLOCK-END
