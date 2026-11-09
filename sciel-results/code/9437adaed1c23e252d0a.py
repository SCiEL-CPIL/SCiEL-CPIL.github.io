"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _simulate_greedy(instance, scores):
    """
    Simulates the greedy assignment using the provided scores.
    Returns the number of stations used.
    """
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Convert predecessors to a set of sets for fast lookup
    # predecessors[i] is a list/set of indices that must precede i
    pred_sets = [set(p) for p in predecessors]
    
    # Track remaining predecessors for each task
    remaining_preds = [len(p) for p in pred_sets]
    # Track which tasks depend on the current task (successors)
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in pred_sets[i]:
            successors[p].append(i)
    
    # Initial eligible tasks: those with 0 remaining predecessors
    eligible = [i for i in range(n) if remaining_preds[i] == 0]
    
    # Sort eligible by score descending initially
    # We will maintain a max-heap or just sort list each time for simplicity if n is small (30)
    # n=30 is very small, sorting a list of size <=30 is cheap.
    
    stations = 0
    current_station_load = 0.0
    assigned = [False] * n
    
    while any(not a for a in assigned):
        if not eligible:
            # Should not happen if DAG is valid, but break to avoid infinite loop
            break
        
        # Sort eligible tasks by score (highest first)
        # We want the highest priority (score) to be picked first
        # So we sort by score descending
        eligible_sorted = sorted(eligible, key=lambda x: scores[x], reverse=True)
        
        placed_any = False
        new_eligible = []
        
        # Try to pick the best fitting task
        # The rule: "stations are filled with the highest-priority eligible task that fits the cycle time"
        # This implies we iterate through eligible tasks in priority order and pick the first one that fits.
        
        for task_id in eligible_sorted:
            if assigned[task_id]:
                continue
            if current_station_load + times[task_id] <= cycle_time:
                # Place this task
                assigned[task_id] = True
                current_station_load += times[task_id]
                placed_any = True
                
                # Update successors
                for succ in successors[task_id]:
                    remaining_preds[succ] -= 1
                    if remaining_preds[succ] == 0:
                        new_eligible.append(succ)
                break
        
        if placed_any:
            # Remove the placed task from eligible list
            # Note: eligible_sorted is a copy. We need to update the main eligible list.
            # Since we picked the first one that fit, we remove it from eligible.
            # It's easier to rebuild eligible or mark it.
            # Let's just filter it out.
            eligible = [t for t in eligible if not assigned[t]]
            eligible.extend(new_eligible)
        else:
            # No task fits in the current station.
            # Move to the next station.
            stations += 1
            current_station_load = 0.0
            # Eligible list remains the same, but we start a new station.
            # The loop continues, and we will try to place a task in the new station.
            # Note: The problem statement says "stations are filled with the highest-priority eligible task that fits".
            # If no task fits, we presumably open a new station.
            # However, if NO task fits, it means the smallest eligible task is > cycle_time?
            # Or just the current load is full?
            # "highest-priority eligible task that fits the cycle time"
            # Usually, this means: Look at all eligible tasks. Find the one with highest priority.
            # If it fits (load + time <= CT), assign it.
            # If it doesn't fit, try the next highest priority?
            # "highest-priority eligible task that fits" -> This phrasing is ambiguous.
            # Interpretation A: Sort by priority. Iterate. Pick the FIRST one that fits.
            # Interpretation B: Pick the highest priority task. If it fits, assign. If not, open new station?
            # Standard GLB (Greedy Line Balancing) usually uses a rule like "Assign the eligible task with the largest priority that fits in the remaining time of the current station."
            # If no eligible task fits, start a new station.
            # My implementation above (Interpretation A) is standard for "First Fit Decreasing" style heuristics where priority dictates order.
            
            # If we opened a new station, we retry with the same eligible list.
            # To prevent infinite loop if a single task > cycle_time, we should check that.
            # But assuming valid instances, tasks <= cycle_time.
            # If current_station_load is 0 and no task fits, then min(times[eligible]) > cycle_time.
            # This would cause an infinite loop of opening stations.
            # We should break or handle this.
            if current_station_load == 0:
                # A single task doesn't fit?
                # Just assign the smallest one to avoid infinite loop? Or break?
                # Let's assume valid data. If this happens, it's an error in instance generation.
                # To be safe, force assign the smallest eligible task.
                min_task = min(eligible, key=lambda x: times[x])
                assigned[min_task] = True
                current_station_load = times[min_task]
                for succ in successors[min_task]:
                    remaining_preds[succ] -= 1
                    if remaining_preds[succ] == 0:
                        new_eligible.append(succ)
                eligible = [t for t in eligible if not assigned[t]]
                eligible.extend(new_eligible)
                # Don't increment stations again here, we just forced it.
                # Actually, if it doesn't fit, it's an invalid instance for this CT.
                # But let's just count it as a station.
                pass

    # Count the last station if any tasks were placed in it
    # The loop increments `stations` when a new station is STARTED.
    # Wait, my logic:
    # 1. Check if eligible empty.
    # 2. Try to place.
    # 3. If placed, continue.
    # 4. If not placed, stations += 1, reset load.
    # This counts the number of times we had to open a NEW station because the current one was full.
    # The FIRST station is not counted by `stations += 1` initially.
    # So total stations = stations + 1 (for the first one) IF any tasks were assigned.
    # Let's trace:
    # Start: stations=0, load=0.
    # Place task A. load > 0.
    # Place task B. load full.
    # Next task C doesn't fit. stations becomes 1. load=0.
    # Place task C.
    # ...
    # If we finish with load > 0, we used `stations` + 1 stations.
    # If we finish with load == 0 (exactly filled), we used `stations` stations?
    # No, if we exactly fill a station, the next task won't fit (unless time 0).
    # So we open a new one.
    # Actually, if the last task exactly fills the station, the loop will try to place the next one, fail, and increment stations.
    # So `stations` is the count of "new stations opened after the first".
    # Total = stations + 1.
    
    return stations + 1

def priority(instance):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    # Pre-compute some features
    # 1. Time
    # 2. Number of successors (out-degree)
    # 3. Number of predecessors (in-degree)
    # 4. Rank Position Weight (RPW): Sum of times of all successors + own time
    # 5. Slack: (Cycle Time - Own Time) / (Sum of Predecessor Times + Own Time) ? Or similar.
    
    # Build successor graph
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
    
    # Calculate RPW (Rank Position Weight)
    # RPW[i] = time[i] + sum(time[j] for j in all successors of i)
    # We can compute this via topological sort or just DFS memoization.
    # With n=30, simple DFS is fine.
    rpw = [0.0] * n
    visited = [False] * n
    
    def calc_rpw(i):
        if visited[i]:
            return rpw[i]
        visited[i] = True
        rpw[i] = times[i]
        for s in successors[i]:
            rpw[i] += calc_rpw(s)
        return rpw[i]
    
    for i in range(n):
        calc_rpw(i)
    
    # Features:
    # F1: times
    # F2: rpw
    # F3: len(successors[i]) (Successor count)
    # F4: len(predecessors[i]) (Predecessor count) - usually lower is better? Or higher?
    # F5: (times[i] / rpw[i]) if rpw[i] > 0 else 0
    
    f1 = times
    f2 = np.array(rpw)
    f3 = np.array([len(succ) for succ in successors], dtype=float)
    f4 = np.array([len(preds) for preds in predecessors], dtype=float)
    f5 = np.where(f2 > 0, f1 / f2, 0.0)
    
    # Candidate scoring vectors
    # We want to find weights w such that score = w1*f1 + w2*f2 + ... minimizes stations.
    # Since n is small, we can try a few random combinations or specific known heuristics.
    
    # Known heuristics:
    # 1. Longest Task (f1)
    # 2. RPW (f2)
    # 3. Successor Count (f3)
    # 4. Combination: f1 + f2
    
    candidates = [
        f1,
        f2,
        f3,
        f1 + f2,
        f1 * 0.5 + f2 * 0.5,
        f5, # Ratio
    ]
    
    best_score_vec = f1
    best_stations = float('inf')
    
    # Evaluate base candidates
    for cand in candidates:
        sim_stations = _simulate_greedy(instance, cand)
        if sim_stations < best_stations:
            best_stations = sim_stations
            best_score_vec = cand.copy()
    
    # Local Search / Random Perturbation
    # We can try to improve the best candidate by adding small random noise or mixing with other features.
    # Since the evaluator uses a fixed seed for instances but our function is called per instance,
    # we can use a deterministic random seed based on the instance hash to ensure reproducibility if needed,
    # or just use a fixed random state.
    
    random.seed(42) # Fixed seed for stability
    
    # Number of iterations
    # Each simulation is O(n^2 log n) roughly. n=30. Very fast.
    # We can do many iterations.
    max_iters = 500
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
        
        # Mix the best found so far with a random candidate or random weights
        # Let's try linear combinations of the base features with random weights
        w1 = random.uniform(0, 1)
        w2 = random.uniform(0, 1)
        w3 = random.uniform(0, 1)
        w4 = random.uniform(0, 1)
        w5 = random.uniform(0, 1)
        
        # Normalize weights? Or just use them.
        # We want higher score to mean higher priority.
        # All features are "higher is better" in terms of criticality usually.
        # f4 (pred count) is tricky. More preds means it's later in the sequence?
        # Let's stick to positive combinations of f1, f2, f3, f5.
        
        candidate = w1 * f1 + w2 * f2 + w3 * f3 + w5 * f5
        
        # Occasionally try a pure random vector?
        if random.random() < 0.1:
             candidate = np.random.rand(n)
        
        sim_stations = _simulate_greedy(instance, candidate)
        if sim_stations < best_stations:
            best_stations = sim_stations
            best_score_vec = candidate.copy()
            
            # If we found a perfect solution (lower bound), we can stop.
            # Lower bound = ceil(sum(times) / cycle_time)
            lower_bound = int(np.ceil(np.sum(times) / instance["cycle_time"]))
            if best_stations <= lower_bound:
                break
    
    return best_score_vec
# EVOLVE-BLOCK-END
