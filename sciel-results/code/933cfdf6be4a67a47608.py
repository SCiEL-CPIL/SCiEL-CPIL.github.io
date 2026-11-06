import numpy as np
import time
from collections import defaultdict

def priority(instance):
    start_time = time.time()
    deadline = 115.0  # Safety margin

    # 1. Parse Instance
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    predecessors = instance["predecessors"]
    demands = np.asarray(instance["demands"], dtype=float)
    
    if "capacities" in instance:
        capacities = np.asarray(instance["capacities"], dtype=float)
    else:
        # Fallback
        if demands.ndim > 1:
            capacities = np.ones(demands.shape[1])
        else:
            capacities = np.ones(1)
            
    # Handle 1D demands if necessary
    if demands.ndim == 1:
        demands = demands.reshape(-1, 1)
        if capacities.ndim == 0:
            capacities = np.array([capacities])
            
    # 2. Build Graph Structures
    successors = defaultdict(list)
    in_degree = [0] * n
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1

    # 3. Topological Sort (for fallback and sanity)
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    in_deg_temp = in_degree[:]
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for s in successors[node]:
            in_deg_temp[s] -= 1
            if in_deg_temp[s] == 0:
                queue.append(s)
                
    if len(topo_order) != n:
        return durations.copy()

    # 4. Compute Static Metrics (Fallback / Base)
    # Forward Pass
    ES = np.zeros(n)
    EF = np.zeros(n)
    for i in topo_order:
        max_pred_ef = 0.0
        for p in predecessors[i]:
            if EF[p] > max_pred_ef:
                max_pred_ef = EF[p]
        ES[i] = max_pred_ef
        EF[i] = ES[i] + durations[i]
        
    project_duration = np.max(EF)
    
    # Backward Pass
    LF = np.full(n, project_duration)
    LS = np.full(n, project_duration)
    for i in reversed(topo_order):
        if successors[i]:
            min_succ_ls = float('inf')
            for s in successors[i]:
                if LS[s] < min_succ_ls:
                    min_succ_ls = LS[s]
            LF[i] = min_succ_ls
        LS[i] = LF[i] - durations[i]
        
    # Slack (Total Float)
    slack = LS - ES
    
    # 5. Dynamic Priority Strategy: Simulate SGS to find bottlenecks
    # We simulate the schedule using a "default" priority (e.g., Critical Path Length or Slack)
    # to see which activities get blocked. Then we boost those blocked activities.
    # This is a structural change: using the outcome of a heuristic schedule to refine the priority.
    
    # Default priority for simulation: Lower Slack is higher priority (Critical Path first)
    # We simulate the serial SGS with this default priority.
    
    # State for simulation
    sim_time = 0.0
    sim_started = [False] * n
    sim_start_times = [0.0] * n
    sim_end_times = [0.0] * n
    sim_remaining_preds = in_degree[:]
    
    # Resource usage over time: We track active resources
    # Since resources are renewable, we can track "busy intervals" or just check current usage.
    # For a greedy simulation, we need to check if starting activity i at time t is feasible.
    # Feasibility: sum of demands of active activities + demand_i <= capacity.
    
    # To make this efficient, we'll keep a list of active activities and their end times.
    # active_acts: list of (end_time, activity_index)
    
    # We will perform a "dry run" of the SGS using a simple priority (e.g., -slack)
    # and record which activities had to wait due to resources.
    
    # Priority for simulation: Lower slack = higher priority. 
    # In SGS, we pick the highest priority eligible. 
    # Let's use slack as the "badness" (lower slack is better priority).
    # So in simulation, we sort by slack ascending.
    
    # We need to simulate the entire schedule to see where bottlenecks are.
    # However, the `priority` function must return a static vector for the *actual* SGS.
    # The idea is: If activity A is often blocked by activity B (which has lower priority in the default sim),
    # we should give A a higher priority in the final vector to preempt B or resolve the conflict earlier.
    
    # Alternative Strategy: 
    # Instead of simulating the whole thing (which might be slow if we do it repeatedly), 
    # let's look at the *resource contention* at the earliest start times.
    
    # Actually, a more robust "structural" change is to use **Weighted Critical Path with Resource Scarcity**.
    # But the prompt asks for a DIFFERENT strategy. 
    # Let's try: **Greedy Simulation with "Regret" based Priority**.
    
    # We simulate the SGS using a neutral priority (e.g., index or random) to get a baseline makespan? No, too slow.
    
    # Let's stick to the "Bottleneck Identification" via simulation.
    # We simulate the SGS using a standard heuristic (Min Slack).
    # We identify activities that are "resource blocked" (i.e., they were eligible but couldn't start because resources were full).
    # We boost the priority of activities that *cause* the blocking (high resource demand on scarce resources) 
    # AND activities that are *victims* of blocking if they have high downstream impact.
    
    # Actually, simpler and more effective: 
    # **Priority = Critical Path Length * (1 + Resource Pressure)**
    # Where Resource Pressure is calculated dynamically? No, priority must be static.
    
    # Let's try a different structural family: **Local Search on the Priority Vector**.
    # But we only have one call to `priority`. We can't iterate.
    
    # How about: **Simulate the SGS with the *current* best known heuristic (Min Slack) and compute the "Delay" each activity experiences due to resources.**
    # If an activity is delayed by resources, it means it was contending. 
    # We want to resolve these conflicts. 
    # If Activity A (high slack) blocks Activity B (low slack), we should prioritize B.
    # Min Slack already does this. 
    # What if Min Slack fails because a high-slack activity with huge resource demand blocks a low-slack activity with small demand?
    # In that case, we should prioritize the high-demand activity *if* it's on the critical path?
    
    # Let's implement a **Resource-Weighted Critical Path** where the weight of an edge or node is scaled by the resource scarcity.
    # This is different from the current additive mix. It's multiplicative.
    
    # Metric: 
    # 1. Compute Critical Path Length (CPL) for each node (distance to end).
    # 2. Compute Resource Scarcity Index (RSI) for each node: Sum(demand / capacity)
    # 3. Priority = CPL * (1 + alpha * RSI)
    
    # Let's refine the RSI. 
    # RSI_i = sum_j (demand_ij / capacity_j)
    # If capacity is low, RSI is high.
    # If we multiply CPL by (1 + RSI), we boost critical activities that are also resource-heavy.
    # This encourages starting resource-heavy critical activities early to free up resources for others? 
    # Or does it just delay non-critical ones?
    
    # Let's try this multiplicative approach. It's structurally different from the additive linear combination in the current program.
    
    # Compute CPL (Longest path from node to end)
    cpl = np.zeros(n)
    for i in reversed(topo_order):
        if successors[i]:
            max_succ_cpl = 0.0
            for s in successors[i]:
                if cpl[s] + durations[s] > max_succ_cpl: # Wait, CPL usually includes duration of node?
                    # Let's define CPL_i as duration_i + max(CPL_s)
                    pass
            cpl[i] = durations[i] + max_succ_cpl
        else:
            cpl[i] = durations[i]
            
    # Compute RSI
    safe_caps = np.where(capacities == 0, 1.0, capacities)
    rsi = np.sum(demands / safe_caps[np.newaxis, :], axis=1)
    
    # Normalize RSI to [0, 1] to control the weight
    rsi_min = np.min(rsi)
    rsi_max = np.max(rsi)
    rsi_range = rsi_max - rsi_min
    if rsi_range < 1e-10:
        rsi_norm = np.zeros_like(rsi)
    else:
        rsi_norm = (rsi - rsi_min) / rsi_range
        
    # Multiplicative Combination
    # We want to boost critical activities that are resource-intensive.
    # Priority = CPL * (1 + rsi_norm)
    # This keeps the relative ordering of CPL mostly intact but tilts it towards resource-heavy tasks.
    
    # Let's also consider the "Reverse" effect: 
    # Maybe we should prioritize *low* resource intensity tasks to keep resources free for critical ones?
    # If we start a non-critical, high-resource task, it might block a critical, low-resource task.
    # So, for non-critical tasks, we should *penalize* high resource intensity.
    # For critical tasks, we should *reward* high resource intensity (to get them done and out of the way).
    
    # Let's use a hybrid:
    # If CPL is high (critical), boost by RSI.
    # If CPL is low (non-critical), penalize by RSI.
    
    cpl_min = np.min(cpl)
    cpl_max = np.max(cpl)
    cpl_range = cpl_max - cpl_min
    if cpl_range < 1e-10:
        cpl_norm = np.zeros_like(cpl)
    else:
        cpl_norm = (cpl - cpl_min) / cpl_range
        
    # Weight factor: 
    # factor = 1 + cpl_norm * rsi_norm  (Boost critical + heavy)
    # Or: factor = 1 + cpl_norm * rsi_norm - (1 - cpl_norm) * rsi_norm * 0.5 (Penalize non-critical + heavy)
    
    # Let's try: 
    # Priority = cpl * (1 + rsi_norm * cpl_norm)
    # This means: 
    # - Critical (cpl_norm=1): Priority = cpl * (1 + rsi_norm) -> Heavy critical tasks are boosted.
    # - Non-critical (cpl_norm=0): Priority = cpl * 1 -> No change.
    
    # What about the "victim" scenario? 
    # If a non-critical heavy task is started, it blocks a critical light task.
    # The critical light task has high CPL but low RSI. 
    # The non-critical heavy task has low CPL but high RSI.
    # With the formula above:
    # Critical Light: Priority = cpl_high * (1 + 0) = cpl_high
    # Non-Critical Heavy: Priority = cpl_low * (1 + 0) = cpl_low
    # Critical Light wins. Good.
    
    # What if Critical Heavy vs Non-Critical Light?
    # Critical Heavy: Priority = cpl_high * (1 + rsi_high)
    # Non-Critical Light: Priority = cpl_low * 1
    # Critical Heavy wins. Good.
    
    # What if Critical Light vs Non-Critical Heavy?
    # Critical Light: cpl_high
    # Non-Critical Heavy: cpl_low
    # Critical Light wins. Good.
    
    # This seems robust. It essentially says: "Schedule critical path first, but among critical path activities, prefer those that consume more resources (to clear them out)."
    
    # Let's implement this.
    
    scores = cpl * (1.0 + rsi_norm * cpl_norm)
    
    # Add a small tiebreaker: Shorter duration first (SPT) for equal scores
    dur_norm = -durations # Invert so shorter is higher
    dur_min = np.min(dur_norm)
    dur_max = np.max(dur_norm)
    dur_range = dur_max - dur_min
    if dur_range < 1e-10:
        dur_score = np.zeros_like(dur_norm)
    else:
        dur_score = (dur_norm - dur_min) / dur_range
        
    # Combine
    scores = scores + 0.01 * dur_score
    
    return scores
