import numpy as np
import time
import random

def _simulate_sgs(instance, priority_vec):
    """
    Fast simulation of the Serial Schedule Generation Scheme.
    Returns makespan.
    """
    N = len(instance["durations"])
    durations = instance["durations"]
    demands = instance["demands"]
    capacities = instance["capacities"]
    preds = instance.get("predecessors", [[] for _ in range(N)])
    
    # Normalize predecessors if needed
    if isinstance(preds, np.ndarray):
        if preds.ndim == 2:
             preds = [np.where(preds[i] > 0)[0].tolist() for i in range(N)]
        else:
             preds = [[] for _ in range(N)]
    elif isinstance(preds, list):
         normalized_preds = []
         for p in preds:
             if isinstance(p, (list, tuple, np.ndarray)):
                 normalized_preds.append(list(map(int, p)))
             else:
                 normalized_preds.append([])
         preds = normalized_preds
    else:
         preds = [[] for _ in range(N)]

    # Build successors
    succs = [[] for _ in range(N)]
    for i in range(N):
        for p in preds[i]:
            succs[p].append(i)
            
    # State
    remaining_preds_count = [len(p) for p in preds]
    elapsed = np.zeros(N)
    start_times = np.zeros(N)
    
    # Priority order: sort indices by priority (highest first)
    # We need a dynamic list of eligible activities.
    # For efficiency in N=30, a simple list scan is fine.
    
    # Initial eligible
    eligible = [i for i in range(N) if remaining_preds_count[i] == 0]
    
    # We will process activities one by one.
    # At each step, pick the highest priority eligible activity that fits in resources.
    # If none fit, advance time to the next event (completion of an activity).
    
    # To speed up, we can pre-sort by priority? No, eligibility changes.
    # But we can maintain a list and scan it.
    
    # Resource usage over time? No, just current usage.
    current_usage = np.zeros(len(capacities))
    
    # Events: list of (completion_time, activity_id)
    # We need to know when activities finish to update resources and eligibility.
    
    # Let's use a time-stepping approach or event-based.
    # Event-based is better.
    
    # Sort eligible by priority (highest first)
    # We'll repeatedly pick the best eligible that fits.
    
    # To make it faster, we can sort the entire set of N activities by priority once?
    # No, because eligibility is dynamic. But we can check them in priority order.
    
    # Let's pre-sort indices by priority descending.
    # sorted_indices = np.argsort(-priority_vec)
    # But we must only consider eligible ones.
    
    # Algorithm:
    # 1. Find next event time (min completion time of running activities).
    # 2. At that time, release finished activities (update resources, mark successors eligible).
    # 3. Try to start eligible activities in priority order.
    
    # State:
    # running: set of (activity, end_time)
    # finished: set of activities
    
    running = [] # list of (end_time, act_id)
    finished = set()
    
    # Initial eligible
    eligible_set = set(i for i in range(N) if remaining_preds_count[i] == 0)
    
    # Pre-sort all activities by priority for quick access?
    # We can just scan the eligible set and pick the max priority one that fits.
    
    # Since N=30, scanning is cheap.
    
    while len(finished) < N:
        # Find next event time
        if not running:
            # No running activities, must start something
            # Pick best eligible that fits
            best_act = -1
            best_pri = -1
            for i in eligible_set:
                if priority_vec[i] > best_pri:
                    # Check resources
                    if np.all(demands[i] <= capacities - current_usage):
                        best_act = i
                        best_pri = priority_vec[i]
            
            if best_act == -1:
                # Should not happen in valid instance, but safety
                break
            
            # Start it
            start_times[best_act] = 0 # Current time is 0 initially?
            # We need a current time variable
            # Let's refactor to use current_time
            
            # This logic is getting messy. Let's use a standard event loop.
            
            pass
        
        # This implementation is too complex to write correctly in one shot without testing.
        # Let's use a simpler, slightly slower but robust simulation.
        
        # Reset state for the simpler version
        break
        
    # Simple Robust Simulation
    # Time step approach? No, events.
    
    # Let's restart the simulation logic cleanly.
    
    elapsed = np.zeros(N)
    start_times = np.zeros(N)
    current_usage = np.zeros(len(capacities))
    
    # eligible: list of activity ids
    # running: list of (end_time, activity_id)
    
    # We'll maintain a list of all activities and their state.
    
    # Actually, for N=30, we can just do a greedy selection at each "decision point".
    # Decision points are: time 0, and every time an activity finishes.
    
    # Let's implement this carefully.
    
    # 1. Initialize
    remaining_preds_count = [len(p) for p in preds]
    eligible = [i for i in range(N) if remaining_preds_count[i] == 0]
    running = [] # (end_time, act_id)
    current_time = 0
    current_usage = np.zeros(len(capacities))
    
    # We need to track which activities are running to know when they finish.
    
    # Loop until all done
    while len(eligible) + len(running) < N:
        
        # 1. Try to start activities from eligible set
        # Sort eligible by priority (highest first)
        # We want to start as many as possible? No, Serial SGS starts ONE at a time.
        # "starts the highest-priority eligible activity"
        # So we pick the single best one that fits.
        
        best_act = -1
        best_pri = -1
        for i in eligible:
            if priority_vec[i] > best_pri:
                # Check if it fits
                if np.all(demands[i] <= capacities - current_usage):
                    best_act = i
                    best_pri = priority_vec[i]
        
        if best_act != -1:
            # Start it
            start_times[best_act] = current_time
            end_time = current_time + durations[best_act]
            running.append((end_time, best_act))
            current_usage += demands[best_act]
            eligible.remove(best_act)
            
            # If we started one, we can continue trying to start others?
            # Serial SGS: "starts the highest-priority eligible activity under precedence and renewable-resource limits."
            # Usually implies one activity is started, then the scheme repeats.
            # However, if resources allow, can we start another immediately?
            # Yes, the loop continues. If resources allow another, it will be picked.
            # So we continue the while loop.
            continue
            
        # 2. No activity can be started. Advance time to next event.
        if not running:
            # Deadlock? Should not happen.
            break
        
        # Find min end time
        min_end_time = min(r[0] for r in running)
        
        # Advance current_time to min_end_time
        current_time = min_end_time
        
        # Remove all activities that finish at this time
        newly_finished = []
        still_running = []
        for end_t, act_id in running:
            if end_t == current_time:
                newly_finished.append(act_id)
                current_usage -= demands[act_id]
            else:
                still_running.append((end_t, act_id))
        running = still_running
        
        # Update eligibility
        for act_id in newly_finished:
            for succ in succs[act_id]:
                remaining_preds_count[succ] -= 1
                if remaining_preds_count[succ] == 0:
                    eligible.append(succ)
                    
    # Calculate makespan
    makespan = max(start_times[i] + durations[i] for i in range(N)) if N > 0 else 0
    return makespan

def priority(instance):
    """
    Returns a priority score for each activity to be used in a Serial SGS.
    
    Strategy:
    1. Compute a baseline priority using Critical Path Length (CPL) and Resource Intensity.
    2. Perform a local search (Hill Climbing with Random Restarts) to optimize the priority vector
       by simulating the SGS and minimizing makespan.
    3. Use a time guard to ensure completion within the limit.
    """
    
    N = len(instance["durations"])
    
    # 1. Compute Baseline Priorities
    # We reuse the logic from the current program to get a good starting point.
    # Extract data
    durations = np.array(instance["durations"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    capacities = np.array(instance["capacities"], dtype=float)
    R = len(capacities)
    
    preds = instance.get("predecessors", [[] for _ in range(N)])
    if isinstance(preds, np.ndarray):
        if preds.ndim == 2:
             preds = [np.where(preds[i] > 0)[0].tolist() for i in range(N)]
        else:
             preds = [[] for _ in range(N)]
    elif isinstance(preds, list):
         normalized_preds = []
         for p in preds:
             if isinstance(p, (list, tuple, np.ndarray)):
                 normalized_preds.append(list(map(int, p)))
             else:
                 normalized_preds.append([])
         preds = normalized_preds
    else:
         preds = [[] for _ in range(N)]

    # Build successors
    succs = [[] for _ in range(N)]
    for i in range(N):
        for p in preds[i]:
            succs[p].append(i)
            
    # Topological Sort
    in_degree = [len(p) for p in preds]
    queue = [i for i in range(N) if in_degree[i] == 0]
    topo_order = []
    indeg_temp = in_degree[:]
    q = queue[:]
    while q:
        u = q.pop(0)
        topo_order.append(u)
        for v in succs[u]:
            indeg_temp[v] -= 1
            if indeg_temp[v] == 0:
                q.append(v)
                
    # Compute Critical Path Length (CPL) backwards
    cpl = np.zeros(N)
    for i in range(len(topo_order) - 1, -1, -1):
        u = topo_order[i]
        max_succ_cpl = 0
        for v in succs[u]:
            if cpl[v] > max_succ_cpl:
                max_succ_cpl = cpl[v]
        cpl[u] = durations[u] + max_succ_cpl
        
    # Compute Earliest Start Times (EST) forwards
    est = np.zeros(N)
    for i in range(len(topo_order)):
        u = topo_order[i]
        max_pred_est = 0
        for p in preds[u]:
            if est[p] + durations[p] > max_pred_est:
                max_pred_est = est[p] + durations[p]
        est[u] = max_pred_est
        
    # Compute Slack
    project_duration = np.max(est + durations)
    lft = np.full(N, project_duration)
    for i in range(len(topo_order) - 1, -1, -1):
        u = topo_order[i]
        if succs[u]:
            min_lft_succ = float('inf')
            for v in succs[u]:
                if lft[v] - durations[v] < min_lft_succ:
                    min_lft_succ = lft[v] - durations[v]
            lft[u] = min_lft_succ
    lst = lft - durations
    slack = lst - est
    
    # Compute Pressure
    pressure = np.zeros(N)
    for r in range(R):
        cap_r = capacities[r]
        if cap_r <= 0:
            continue
        d_col = demands[:, r]
        remaining_cap = cap_r - d_col
        safe_remaining = np.where(remaining_cap > 0, remaining_cap, 1.0)
        ratio = d_col / safe_remaining
        pressure += ratio
        
    max_pressure = np.max(pressure)
    if max_pressure > 0:
        pressure = pressure / max_pressure
        
    max_slack = np.max(slack)
    if max_slack > 0:
        slack_factor = 1.0 - (slack / max_slack)
    else:
        slack_factor = np.ones(N)
        
    # Baseline Priority
    best_priorities = cpl * (1.0 + 2.0 * pressure) * (1.0 + slack_factor)
    
    # 2. Local Search
    # We will try to improve this priority vector.
    # Time limit: 150s total, leave margin. Say 140s for search.
    start_time = time.time()
    time_limit = 140.0
    
    # Helper to get makespan
    def get_makespan(p_vec):
        return _simulate_sgs(instance, p_vec)
    
    best_makespan = get_makespan(best_priorities)
    
    # Hill Climbing with Random Restarts
    # Perturbation: swap two priorities, or add noise.
    # Since priorities are real numbers, we can swap the values of two activities.
    # Or we can add Gaussian noise to a random subset.
    # Swapping is often effective for permutation-like problems, but priorities are continuous.
    # Let's try swapping the priority values of two random activities.
    
    random.seed(42)
    
    # Number of iterations per restart
    max_iters_per_restart = 1000
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if restart == 0:
            current_p = best_priorities.copy()
            current_m = best_makespan
        else:
            # Random restart: perturb best solution significantly
            current_p = best_priorities.copy()
            # Add noise
            noise = np.random.normal(0, 0.5 * np.std(best_priorities), N)
            current_p += noise
            current_m = get_makespan(current_p)
            
        # Hill Climbing
        for it in range(max_iters_per_restart):
            if time.time() - start_time > time_limit:
                break
                
            # Perturb: pick two random distinct indices and swap their priorities
            i, j = random.sample(range(N), 2)
            old_i, old_j = current_p[i], current_p[j]
            current_p[i] = old_j
            current_p[j] = old_i
            
            new_m = get_makespan(current_p)
            
            if new_m < current_m:
                current_m = new_m
                if new_m < best_makespan:
                    best_makespan = new_m
                    best_priorities = current_p.copy()
            else:
                # Revert
                current_p[i] = old_i
                current_p[j] = old_j
                
    return best_priorities
