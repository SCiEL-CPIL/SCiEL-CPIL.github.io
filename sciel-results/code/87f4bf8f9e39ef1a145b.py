"""
RCPSP Heuristic: Iterative Regret-Based Priority
This strategy uses an iterative local search approach. 
1. Initialize priorities using a strong static heuristic (Critical Path + Resource Pressure).
2. Simulate the serial schedule generation scheme (SGS) using these priorities.
3. Calculate the 'regret' for each activity: how much its start was delayed 
   compared to its earliest possible start (ES) due to resource constraints.
4. Update priorities to heavily weight activities with high regret, as they are 
   the bottlenecks causing the makespan.
5. Repeat for a limited number of iterations to converge on a better priority ordering.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

def priority(instance):
    """
    Returns a priority score for each activity.
    Higher score = higher priority.
    
    Strategy:
    1. Parse instance data.
    2. Compute CPM (Critical Path Method) for Earliest Start (ES) and Total Float.
    3. Initialize priority scores based on Criticality (low float) and Resource Pressure.
    4. Iteratively simulate SGS, measure 'regret' (delay caused by resources), 
       and update priorities to prioritize high-regret activities.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin
    
    # --- 1. Parse Instance ---
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    preds_raw = instance.get("predecessors", instance.get("preds", [[] for _ in range(n)]))
    demands_raw = instance.get("demands", instance.get("resource_demands", None))
    capacities_raw = instance.get("capacities", instance.get("resource_capacities", None))
    
    # Normalize predecessors
    preds = [list(p) if p else [] for p in preds_raw]
    if len(preds) < n:
        preds = [[] for _ in range(n)]
        for i in range(n):
            if i < len(preds_raw) and preds_raw[i]:
                preds[i] = list(preds_raw[i])
            
    # Build successors
    succs = [[] for _ in range(n)]
    for j in range(n):
        for i in preds[j]:
            if 0 <= i < n:
                succs[i].append(j)
                
    # Handle resources
    if demands_raw is not None:
        try:
            demands = np.asarray(demands_raw, dtype=float)
            if demands.ndim == 1:
                demands = demands.reshape(-1, 1)
        except:
            demands = np.zeros((n, 1))
    else:
        demands = np.zeros((n, 1))
        
    if capacities_raw is not None:
        try:
            capacities = np.asarray(capacities_raw, dtype=float)
            if capacities.ndim == 0:
                capacities = np.array([capacities])
        except:
            capacities = np.ones(demands.shape[1])
    else:
        capacities = np.ones(demands.shape[1])
        
    m_resources = demands.shape[1]
    if len(capacities) < m_resources:
        capacities = np.pad(capacities, (0, m_resources - len(capacities)), mode='constant')
        
    # Topological Sort
    in_degree = np.array([len(preds[i]) for i in range(n)], dtype=int)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    indeg_work = in_degree.copy()
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            indeg_work[v] -= 1
            if indeg_work[v] == 0:
                queue.append(v)
                
    if len(topo_order) < n:
        # Cycle or invalid, fallback
        return durations.copy()
        
    # --- 2. CPM Analysis ---
    es = np.zeros(n)
    ef = np.zeros(n)
    for u in topo_order:
        es[u] = 0
        for p in preds[u]:
            if ef[p] > es[u]:
                es[u] = ef[p]
        ef[u] = es[u] + durations[u]
        
    makespan_cpm = np.max(ef)
    
    lf = np.zeros(n)
    ls = np.zeros(n)
    for u in reversed(topo_order):
        if len(succs[u]) > 0:
            lf[u] = np.min([ls[v] for v in succs[u]])
        else:
            lf[u] = makespan_cpm
        ls[u] = lf[u] - durations[u]
        
    tf = ls - es
    
    # Resource Pressure (static component)
    if m_resources > 0 and np.all(capacities > 0):
        safe_cap = np.where(capacities > 0, capacities, 1.0)
        pressure = np.sum(demands / safe_cap, axis=1)
        max_pressure = np.max(pressure)
        if max_pressure > 0:
            pressure = pressure / max_pressure
        else:
            pressure = np.zeros(n)
    else:
        pressure = np.zeros(n)
        
    # Normalize static components
    max_tf = np.max(tf) if np.max(tf) > 0 else 1.0
    crit_score = 1.0 - (tf / max_tf)
    max_dur = np.max(durations) if np.max(durations) > 0 else 1.0
    dur_score = durations / max_dur
    
    # Initial Priority: Blend of Criticality, Pressure, Duration
    # We start with a strong static heuristic
    p_init = 0.6 * crit_score + 0.3 * pressure + 0.1 * dur_score
    
    # --- 3. Iterative Local Search ---
    # We will run a few iterations of:
    # 1. Simulate SGS with current priority
    # 2. Calculate regret (delay from ES due to resources)
    # 3. Update priority: new_p = alpha * old_p + beta * regret_score
    
    # Precompute for speed: 
    # We need to simulate SGS efficiently.
    # SGS: 
    # - Maintain set of eligible activities (preds done)
    # - Pick highest priority eligible
    # - Find earliest start time where resources are available
    # - Schedule it, update resource usage over time
    
    # To make this fast, we'll use a simple event-based simulation or 
    # step-by-step time check. Given n=30, step-by-step is fine if optimized.
    
    # Let's define a helper to simulate SGS and return start times
    def simulate_sgs(priorities):
        # Returns start times array
        start_times = np.zeros(n)
        scheduled = [False] * n
        # Resource usage over time: dict time -> array of usage
        # Or simpler: list of (start, end, demands) for scheduled jobs
        scheduled_jobs = []
        
        # Precompute eligible lists? No, dynamic.
        # We need to know when resources are free.
        # We'll keep track of resource usage in a list of intervals or just check against scheduled jobs.
        # For n=30, checking all scheduled jobs for each candidate is O(N^2 * M) which is ~30*30*4 = 3600 ops per step.
        # Total steps N=30. Total ops ~100k. Very fast.
        
        remaining = set(range(n))
        
        # To speed up finding earliest start, we can track the next available time for each resource?
        # No, because resources are renewable but shared. 
        # Simplest robust method: 
        # For a candidate job j, earliest start = max(ES_j, max_end_time_of_conflicts)
        # Conflict: another job i that uses same resources and overlaps.
        # Actually, we just need to find the first time t >= ES_j such that 
        # for all resources k: usage_k(t) + demand_jk <= capacity_k.
        
        # We'll maintain a list of scheduled jobs.
        # For each candidate, we scan time forward from ES_j.
        
        while remaining:
            # Find eligible activities
            eligible = []
            for i in remaining:
                if all(scheduled[p] for p in preds[i]):
                    eligible.append(i)
            
            if not eligible:
                # Should not happen if DAG is valid and no cycles, but safe break
                break
                
            # Pick highest priority
            # We want to pick max priority. 
            # Note: priority function returns scores. Higher is better.
            best_idx = -1
            best_score = -np.inf
            for i in eligible:
                if priorities[i] > best_score:
                    best_score = priorities[i]
                    best_idx = i
                    
            if best_idx == -1:
                break
                
            # Determine start time for best_idx
            es_t = es[best_idx]
            # Find earliest t >= es_t where resources fit
            t = es_t
            # We need to check resource availability at time t.
            # Resource usage at time t is sum of demands of jobs j where start_j <= t < end_j.
            
            # Optimization: 
            # Since we schedule in order, we can keep a list of active jobs.
            # But for simplicity and n=30, we just check all scheduled jobs.
            
            while True:
                feasible = True
                for k in range(m_resources):
                    # Check capacity k
                    current_usage = 0.0
                    for job in scheduled_jobs:
                        if job['start'] <= t < job['end']:
                            current_usage += job['demands'][k]
                    if current_usage + demands[best_idx, k] > capacities[k] + 1e-9:
                        feasible = False
                        # Find the earliest end time among conflicting jobs
                        # This allows us to jump forward
                        next_free = np.inf
                        for job in scheduled_jobs:
                            if job['start'] <= t < job['end']:
                                if job['demands'][k] > 0: # Only if it uses this resource
                                    if job['end'] < next_free:
                                        next_free = job['end']
                        if next_free != np.inf:
                            t = next_free
                        else:
                            t += 1 # Fallback increment, shouldn't happen if logic is right
                        break # Re-check from start of while loop for this t
                if feasible:
                    break
                # If we broke due to infeasibility, t has been updated. Loop continues.
                
            # Schedule best_idx
            start_times[best_idx] = t
            end_t = t + durations[best_idx]
            scheduled_jobs.append({
                'id': best_idx,
                'start': t,
                'end': end_t,
                'demands': demands[best_idx]
            })
            scheduled[best_idx] = True
            remaining.remove(best_idx)
            
        return start_times

    # Initial Priority Vector
    current_priorities = p_init.copy()
    best_makespan = np.inf
    best_priorities = current_priorities.copy()
    
    # Iterative Search
    # We want to see if we can improve the makespan by adjusting priorities.
    # The "regret" of an activity i is (actual_start[i] - es[i]).
    # If an activity was delayed a lot, it's a bottleneck. 
    # We should increase its priority so it gets scheduled earlier in the next iteration.
    
    max_iterations = 20
    for _ in range(max_iterations):
        if time.time() - start_time > DEADLINE:
            break
            
        start_times = simulate_sgs(current_priorities)
        makespan = np.max(start_times + durations)
        
        if makespan < best_makespan:
            best_makespan = makespan
            best_priorities = current_priorities.copy()
            
        # Calculate Regret
        regret = start_times - es
        # Normalize regret
        max_regret = np.max(regret) if np.max(regret) > 0 else 1.0
        regret_score = regret / max_regret
        
        # Update Priorities
        # We want to blend the static heuristic (which ensures good baseline) 
        # with the dynamic regret (which targets bottlenecks).
        # If we use only regret, we might lose the global structure.
        # Blend: new_p = (1 - lambda) * p_static + lambda * regret_score
        # Or: new_p = p_static + alpha * regret_score
        
        # Let's try: new_p = 0.5 * p_init + 0.5 * regret_score
        # This effectively says: "Keep the critical path stuff, but boost the ones that got delayed."
        
        current_priorities = 0.5 * p_init + 0.5 * regret_score
        
        # Optional: Add a small duration component to break ties or maintain LPT flavor
        # current_priorities += 0.1 * dur_score
        
        # If regret is all zeros, we've reached CPM makespan (no resource contention delays)
        if np.allclose(regret, 0):
            break
            
    # Return the best priorities found
    # The evaluator uses these priorities to run ITS OWN SGS.
    # So we return the priority vector that *we* found to produce the best makespan in our simulation.
    
    return best_priorities

# EVOLVE-BLOCK-END
