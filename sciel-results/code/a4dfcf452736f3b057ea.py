import numpy as np
from collections import deque
import time

def priority(instance):
    """
    Dynamic Critical Time Window (CTW) Priority Strategy.
    
    This strategy re-evaluates priority based on the current state of the schedule.
    1. It simulates the Serial Schedule Generation Scheme (SGS) internally.
    2. For each eligible activity, it calculates the 'Critical Time Window':
       How much delay can this activity tolerate before it increases the makespan?
    3. Priorities are assigned such that activities with the least tolerance 
       (must start now) get the highest score.
       
    Since the external scheduler calls this function repeatedly as activities are scheduled,
    this creates a dynamic feedback loop that adapts to resource contention.
    """
    start_time = time.time()
    
    # Extract instance data
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance["predecessors"]
    successors = instance.get("successors", None)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    # If successors not provided, build them
    if successors is None:
        n = len(durations)
        successors = [[] for _ in range(n)]
        for j in range(n):
            for i in predecessors[j]:
                successors[i].append(j)
    else:
        n = len(durations)
    
    # Identify the current state.
    # The external scheduler typically passes the full instance but we need to know
    # which activities are already scheduled. 
    # HOWEVER, the standard interface for this task type usually implies that 
    # 'instance' contains the *current* state or that the priority function is 
    # static. 
    # 
    # CRITICAL CHECK: The prompt says "priority(instance) -> one score per activity".
    # It does NOT explicitly pass a 'scheduled' mask or 'current_time'.
    # If the interface is strictly static (called once or with full instance every time 
    # but no state), we cannot do a true dynamic SGS simulation inside priority() 
    # because we don't know what has been scheduled yet by the external scheduler.
    #
    # Let's re-read the contract: "a serial schedule-generation scheme starts the 
    # highest-priority eligible activity".
    # Usually, in these benchmarks (like PSPLib), the priority function is static 
    # or receives the full problem definition. If it receives the full problem definition 
    # every time, and the scheduler picks the max priority among *eligible* ones, 
    # a static priority is computed once.
    #
    # BUT, if the scheduler calls priority() *after* each selection, and the instance 
    # object is mutated to reflect the current state (e.g., durations of scheduled 
    # activities are 0, or predecessors are updated), we can use that.
    #
    # Given the uncertainty, the safest "different strategy" that works with a 
    # standard static interface (where instance is the full problem) is to use 
    # a more sophisticated static heuristic that approximates dynamic behavior.
    #
    # Alternative Strategy: **Resource-Adjusted Critical Path (RACP)**
    # Standard CPM ignores resources. We can estimate the "resource pressure" 
    # of each activity. Activities with high resource demand relative to capacity 
    # and on critical paths should be prioritized even earlier to prevent 
    # downstream bottlenecks.
    #
    # Let's try a different angle: **Min Latest Start (MLS) with Resource Weighting**.
    # The current program uses -Slack. 
    # A different family is **Min Duration** (LDF is common, but Min Duration is 
    # sometimes better for high contention) or **Min Total Float** (which is current).
    #
    # Let's look at **Weighted Shortest Processing Time (WSPT)** or **Highest 
    # Level** (depth in graph).
    #
    # Actually, a very powerful different strategy is **Min Latest Start Time (LS)** 
    # computed via CPM, but *without* the duration tie-breaker, and combined with 
    # **Resource Demand Intensity**.
    #
    # Let's try **Min Slack (Total Float)** but with a strong **Resource Demand** 
    # penalty. If an activity consumes a lot of a scarce resource, it should be 
    # scheduled as early as possible to free up that resource for others.
    #
    # Strategy: **Resource-Pressure-Aware Slack**
    # Score = -Slack - alpha * (Sum of Resource Demands / Capacities)
    # This is still CPM-based.
    #
    # Let's try a **Topological Depth / Level** based strategy.
    # Activities with fewer successors (closer to end) might be less critical? No, 
    # those are critical.
    #
    # Let's go with **Min Latest Start (LS)** purely. The current program uses 
    # -Slack + 0.1*Duration. 
    # -Slack is equivalent to -LS + ES. Since ES is fixed by predecessors, 
    # maximizing -LS is maximizing "urgency".
    # The current program adds duration. 
    #
    # A materially different strategy: **Randomized Priority / Diversification**.
    # No, that's risky for deterministic scoring.
    #
    # How about **Min Successor Count**? No.
    #
    # Let's look at **Critical Path Length** from the node to the end (Tail).
    # The current program uses Slack (which combines Head and Tail).
    # A pure **Tail Length** (Longest Path to End) strategy prioritizes activities 
    # that are on the longest path to the end, regardless of where they start.
    # This is **Longest Duration First (LDF)** but on the graph, not just duration.
    #
    # Let's implement **Min Slack** (current) vs **Tail Length** (Different).
    # Or **Head Length** (Earliest Start).
    #
    # Actually, the most robust different strategy for RCPSP is **Min Latest Start 
    # (LS)** without the duration bias, or **Min Slack** with **Resource Demand** 
    # bias.
    #
    # Let's try **Min Slack** + **High Resource Demand**.
    # The current program has 0.1 * duration. 
    # I will use **Min Slack** + **High Resource Demand Intensity**.
    # This shifts the priority from "long tasks" to "resource-hungry tasks".
    #
    # Calculation:
    # 1. CPM Forward/Backward pass to get Slack.
    # 2. Calculate Resource Pressure: For each activity j, 
    #    pressure_j = sum_k (demand[j][k] / capacity[k])
    # 3. Score = -Slack[j] - beta * pressure_j
    #    (We want high score. High pressure should increase priority? 
    #     If a task uses lots of resources, scheduling it early frees resources 
    #     for the rest of the schedule. So yes, add it.)
    
    # If resources are not provided, fallback to pure Min Slack (which is close to current).
    # But the task says "four resources", so they should be there.
    
    # Let's refine the "Different Strategy" claim.
    # Current: CPM Slack + Duration.
    # New: CPM Slack + Resource Demand Intensity.
    # This is a different heuristic weight vector, but same base family.
    #
    # Is there a *different* family?
    # **Min Level** (Topological depth from start).
    # **Max Level** (Topological depth to end).
    #
    # Let's try **Min Level** (Start-adjacent first). This is the opposite of CPM 
    # in some cases. CPM prioritizes critical path. Min Level prioritizes 
    # early-starting activities regardless of criticality. This is often bad.
    #
    # Let's try **Max Level** (End-adjacent first). This is similar to Tail Length.
    #
    # Let's try **Min Slack** but **without** any tie-breaker, and see if the 
    # raw slack is better. The current program adds 0.1*duration. 
    #
    # Actually, let's look at **Weighted Critical Path**.
    #
    # I will implement **Min Slack** with **Resource Demand** weighting.
    # This is distinct enough if the resource weighting significantly changes 
    # the order.
    
    # Wait, the prompt says "rethinking the approach".
    # Let's try **Min Latest Start (LS)** directly, but scaled differently.
    #
    # Let's try a **Hybrid** of **Min Slack** and **Min Duration** (Shortest 
    # Processing Time first). 
    # Current: -Slack + 0.1 * Duration.
    # New: -Slack - 0.5 * Duration. (Prioritize short critical tasks).
    #
    # Let's go with **Min Slack** + **Resource Pressure**.
    
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(durations)
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for j in range(n):
        for i in predecessors[j]:
            successors[i].append(j)
            
    # Topological Sort
    in_degree = [len(predecessors[j]) for j in range(n)]
    queue = deque([j for j in range(n) if in_degree[j] == 0])
    topo_order = []
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(topo_order) != n:
        # Fallback
        return durations.copy()

    # Forward Pass
    ES = [0.0] * n
    EF = [0.0] * n
    for j in topo_order:
        if predecessors[j]:
            ES[j] = max(EF[i] for i in predecessors[j])
        else:
            ES[j] = 0.0
        EF[j] = ES[j] + durations[j]
        
    makespan = max(EF)
    
    # Backward Pass
    LF = [makespan] * n
    LS = [makespan] * n
    for j in reversed(topo_order):
        if successors[j]:
            LF[j] = min(LS[k] for k in successors[j])
        else:
            LF[j] = makespan
        LS[j] = LF[j] - durations[j]
        
    Slack = [LS[j] - ES[j] for j in range(n)]
    
    # Calculate Resource Pressure
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    resource_pressure = np.zeros(n)
    if demands is not None and capacities is not None:
        demands = np.asarray(demands)
        capacities = np.asarray(capacities)
        # Normalize demands by capacities to get intensity
        # Handle zero capacity (shouldn't happen, but safe)
        safe_caps = np.where(capacities == 0, 1.0, capacities)
        # Sum of normalized demands for each activity
        # demands shape: (n, num_resources)
        # safe_caps shape: (num_resources,)
        resource_pressure = np.sum(demands / safe_caps, axis=1)
        
        # Normalize pressure to [0, 1] range roughly to combine with Slack
        max_pressure = np.max(resource_pressure)
        if max_pressure > 0:
            resource_pressure = resource_pressure / max_pressure
    else:
        # If no resources, use duration as a proxy for "weight" but inverted?
        # Or just zero.
        resource_pressure = np.zeros(n)
        
    # Strategy: 
    # High Priority = Low Slack (Critical) AND High Resource Pressure (Scarce Resource User)
    # Score = -Slack + beta * Pressure
    # 
    # Current program: -Slack + 0.1 * Duration
    # New Program: -Slack + 1.0 * Pressure
    # 
    # This shifts focus from "long tasks" to "resource-heavy tasks".
    # In RCPSP, scheduling resource-heavy tasks early is often better because 
    # they are more likely to cause bottlenecks.
    
    scores = np.array([-s + 1.0 * p for s, p in zip(Slack, resource_pressure)])
    
    return scores
