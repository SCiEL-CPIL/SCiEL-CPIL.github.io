"""
RCPSP Heuristic: Local Search over Priority Rules via SGS Evaluation
This strategy abandons static proxy features in favor of an explicit search.
It defines a family of priority rules parameterized by weights for Criticality 
(Total Float), Resource Pressure, and Duration. It evaluates these rules by 
actually running the Serial Schedule Generation Scheme (SGS) on the instance 
and measuring the resulting makespan. It uses a grid search followed by 
random perturbations to find the best-performing weight vector within the 
time budget.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import deque

def _parse_instance(instance, n):
    """Parse instance data into standard arrays."""
    durations = np.asarray(instance["durations"], dtype=float)
    
    preds_raw = instance.get("predecessors", instance.get("preds", [[] for _ in range(n)]))
    preds = [list(p) if p else [] for p in preds_raw]
    if len(preds) < n:
        preds = [[] for _ in range(n)]
        for i in range(n):
            if i < len(preds_raw) and preds_raw[i]:
                preds[i] = list(preds_raw[i])
                
    succs = [[] for _ in range(n)]
    for j in range(n):
        for i in preds[j]:
            if 0 <= i < n:
                succs[i].append(j)
                
    demands_raw = instance.get("demands", instance.get("resource_demands", None))
    capacities_raw = instance.get("capacities", instance.get("resource_capacities", None))
    
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
        
    return durations, preds, succs, demands, capacities

def _topo_sort(n, preds, succs):
    in_degree = np.array([len(preds[i]) for i in range(n)], dtype=int)
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(topo_order) < n:
        return None
    return topo_order

def _compute_cpm(n, durations, preds, succs, topo_order):
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
    return es, ef, tf, makespan_cpm

def _compute_pressure(n, demands, capacities):
    m_resources = demands.shape[1]
    if m_resources == 0 or np.any(capacities <= 0):
        return np.zeros(n)
    safe_cap = np.where(capacities > 0, capacities, 1.0)
    pressure = np.sum(demands / safe_cap, axis=1)
    max_p = np.max(pressure)
    if max_p > 0:
        pressure = pressure / max_p
    return pressure

def _run_sgs(n, durations, preds, demands, capacities, topo_order, priority_scores):
    """
    Execute Serial Schedule Generation Scheme.
    Returns makespan.
    """
    scheduled = np.zeros(n, dtype=bool)
    current_time = 0.0
    remaining_resources = capacities.copy()
    
    # We need to iterate through time or eligible list.
    # Efficient SGS:
    # 1. Find eligible activities (all preds scheduled)
    # 2. Among eligible, pick highest priority
    # 3. Start it at current_time if resources allow, else wait.
    
    # To make it efficient, we can simulate the timeline.
    # However, a simpler, slightly slower but robust approach for n=30:
    # Maintain a set of eligible activities.
    # At each step, find the earliest time an eligible activity can start.
    
    # Actually, standard SGS:
    # While not all scheduled:
    #   1. Identify eligible activities (preds done)
    #   2. Select highest priority eligible activity 'a'
    #   3. Determine start time 't' for 'a':
    #      - t must be >= completion time of all preds
    #      - t must be such that resources are available for [t, t+dur_a]
    #      - Find the earliest such t.
    #   4. Schedule 'a' at t.
    
    # Optimization: Since n is small (30), we can afford a simple loop.
    # But we need to be careful with resource availability checks.
    
    # Let's use a list of events or just step through time?
    # Stepping through time is risky if gaps are large.
    # Better: Maintain a list of "resource intervals" or just check feasibility.
    
    # Given n=30, let's do a direct simulation.
    # We track completion times of scheduled activities.
    # For a new activity 'a' with preds P, earliest start is max(ef[p] for p in P).
    # We then need to find the first time t >= earliest_start where resources are free.
    
    # To check resource availability, we can maintain a list of scheduled intervals.
    # scheduled_intervals: list of (start, end, demands_vector)
    
    scheduled_intervals = []
    completion_times = np.full(n, -1.0)
    remaining_count = n
    scheduled_set = set()
    
    # Pre-compute predecessors for fast lookup
    # preds is already a list of lists.
    
    while remaining_count > 0:
        # 1. Find eligible activities
        eligible = []
        for i in range(n):
            if i not in scheduled_set:
                is_eligible = True
                for p in preds[i]:
                    if p not in scheduled_set:
                        is_eligible = False
                        break
                if is_eligible:
                    eligible.append(i)
                    
        if not eligible:
            # Should not happen if graph is valid and acyclic
            break
            
        # 2. Select highest priority eligible activity
        # priority_scores: higher is better
        best_act = max(eligible, key=lambda i: priority_scores[i])
        
        # 3. Determine start time
        earliest_start = 0.0
        for p in preds[best_act]:
            if completion_times[p] > earliest_start:
                earliest_start = completion_times[p]
                
        # Check resource availability starting from earliest_start
        t = earliest_start
        dur = durations[best_act]
        dem = demands[best_act]
        
        # We need to find the first t such that for all resources r:
        # sum(demands of activities active at t) + dem[r] <= capacities[r]
        # Active at t means start <= t < end
        
        # Since n is small, we can just scan forward in time steps defined by existing interval boundaries.
        # Or, simpler: just check t, then t+epsilon, etc? No, that's inefficient.
        
        # Let's collect all relevant time points:
        # - earliest_start
        # - End times of all currently scheduled activities that overlap or start after earliest_start
        
        # Actually, a simpler robust method for small N:
        # Try t = earliest_start.
        # If conflict, jump to the end time of one of the conflicting activities.
        
        while True:
            # Check conflict at time t
            conflict = False
            for r in range(dem.shape[0]):
                total_load = dem[r]
                for (s, e, d) in scheduled_intervals:
                    if s <= t < e:
                        total_load += d[r]
                if total_load > capacities[r]:
                    conflict = True
                    break
            
            if not conflict:
                break
            
            # Conflict found. We must wait until one of the conflicting activities finishes.
            # Find the earliest finish time among activities active at t.
            next_time = float('inf')
            for (s, e, d) in scheduled_intervals:
                if s <= t < e:
                    if e < next_time:
                        next_time = e
            
            if next_time == float('inf'):
                # Should not happen
                break
                
            t = next_time
            
        # 4. Schedule
        completion_times[best_act] = t + dur
        scheduled_intervals.append((t, t + dur, dem))
        scheduled_set.add(best_act)
        remaining_count -= 1
        
    makespan = np.max(completion_times) if n > 0 else 0.0
    return makespan

def priority(instance):
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin for 150s limit
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if n == 0:
        return np.array([])
        
    durations, preds, succs, demands, capacities = _parse_instance(instance, n)
    
    topo_order = _topo_sort(n, preds, succs)
    if topo_order is None:
        return durations.copy()
        
    es, ef, tf, makespan_cpm = _compute_cpm(n, durations, preds, succs, topo_order)
    pressure = _compute_pressure(n, demands, capacities)
    
    # Normalize features to [0, 1]
    max_tf = np.max(tf) if np.max(tf) > 0 else 1.0
    crit_score = 1.0 - (tf / max_tf)
    
    max_dur = np.max(durations) if np.max(durations) > 0 else 1.0
    dur_score = durations / max_dur
    
    # pressure is already normalized to [0, 1]
    
    # Define candidate weight vectors (w_crit, w_press, w_dur)
    # We want to find the best combination.
    candidates = []
    
    # Grid search over a coarse grid
    # Weights don't need to sum to 1, but relative scale matters.
    # Let's fix w_crit = 1.0 and vary others? Or just sample a 3D grid.
    # Given time constraints, let's use a smaller grid.
    
    # Base heuristics
    candidates.append([1.0, 0.0, 0.0]) # Pure Criticality
    candidates.append([0.0, 1.0, 0.0]) # Pure Pressure
    candidates.append([0.0, 0.0, 1.0]) # Pure Duration (LPT)
    candidates.append([1.0, 1.0, 1.0]) # Balanced
    
    # Add some grid points
    for w_c in [0.5, 1.0, 2.0]:
        for w_p in [0.0, 0.5, 1.0]:
            for w_d in [0.0, 0.5, 1.0]:
                candidates.append([w_c, w_p, w_d])
                
    # Remove duplicates (optional, but good for speed)
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
            
    best_score = float('inf')
    best_weights = [1.0, 0.0, 0.0] # Default to Criticality
    
    # Evaluate candidates
    for w in unique_candidates:
        if time.time() - start_time > DEADLINE - 5: # Leave 5s for final computation
            break
            
        scores = w[0] * crit_score + w[1] * pressure + w[2] * dur_score
        makespan = _run_sgs(n, durations, preds, demands, capacities, topo_order, scores)
        
        if makespan < best_score:
            best_score = makespan
            best_weights = w[:]
            
    # Local Search / Perturbation around best weights
    # We have used some time. Let's see how much is left.
    time_left = DEADLINE - (time.time() - start_time)
    
    if time_left > 2.0:
        # Perform random perturbations around best_weights
        # Number of perturbations depends on time left.
        # Assume each evaluation takes ~T seconds. We don't know T exactly, but we can estimate.
        # Let's just do a fixed number of high-quality perturbations if time allows.
        
        # Estimate cost of one evaluation:
        # We can't easily measure it without running, but for n=30, it's fast.
        # Let's try 50 perturbations if we have > 10s left, 20 if > 5s, etc.
        
        num_perturbations = 0
        if time_left > 10:
            num_perturbations = 50
        elif time_left > 5:
            num_perturbations = 20
        elif time_left > 2:
            num_perturbations = 5
            
        for _ in range(num_perturbations):
            if time.time() - start_time > DEADLINE - 1:
                break
                
            # Perturb best_weights
            new_w = best_weights[:]
            # Gaussian noise
            for i in range(3):
                new_w[i] += random.gauss(0, 0.2)
                if new_w[i] < 0:
                    new_w[i] = 0
                    
            # Optional: Normalize? No, relative scale is what matters for ranking, 
            # but absolute scale doesn't matter for max(). So we can just use it.
            
            scores = new_w[0] * crit_score + new_w[1] * pressure + new_w[2] * dur_score
            makespan = _run_sgs(n, durations, preds, demands, capacities, topo_order, scores)
            
            if makespan < best_score:
                best_score = makespan
                best_weights = new_w[:]
                
    # Final scores based on best_weights
    final_scores = best_weights[0] * crit_score + best_weights[1] * pressure + best_weights[2] * dur_score
    
    # Safety check
    if time.time() - start_time > DEADLINE:
        # Fallback to simple criticality if we are dangerously close to timeout
        return crit_score
        
    return final_scores

# EVOLVE-BLOCK-END
