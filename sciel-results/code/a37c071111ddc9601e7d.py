import numpy as np
import time
import random
from collections import deque

def priority(instance):
    """
    Strategy: Stochastic Multi-Start Greedy SGS Simulation.
    
    Instead of calculating static analytical scores (like Slack or Resource Pressure),
    this function runs a large number of fast, randomized Serial Schedule Generation 
    Scheme (SGS) simulations. 
    
    In each simulation, at every step, we select an eligible activity based on a 
    simple, fast heuristic (Min CPM Slack, with random tie-breaking to explore 
    different schedules). 
    
    We record the start time of each activity in every simulation. 
    The final priority score for an activity is the negative of its average start 
    time across all simulations. 
    
    Rationale:
    - Activities that consistently start early in good schedules are likely to be 
      critical for makespan minimization.
    - Randomization allows the heuristic to escape local minima of a single 
      deterministic greedy path, effectively "searching" the schedule space 
      within the time budget.
    - This is structurally different from static CPM/RCCP as it uses actual 
      simulated schedule outcomes rather than graph-theoretic properties alone.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Extract instance data
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance["predecessors"]
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)
    
    n = len(durations)
    num_resources = len(capacities)
    
    # Build adjacency and topological order for CPM
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for j in range(n):
        for i in predecessors[j]:
            successors[i].append(j)
            in_degree[j] += 1
            
    # Topological sort (Kahn's algorithm)
    queue = deque([j for j in range(n) if in_degree[j] == 0])
    topo_order = []
    temp_in_degree = in_degree[:]
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            temp_in_degree[v] -= 1
            if temp_in_degree[v] == 0:
                queue.append(v)
                
    if len(topo_order) != n:
        # Cycle detected or invalid graph, fallback to random
        return np.random.rand(n)

    # Compute CPM Slack for the base greedy heuristic
    ES = [0.0] * n
    EF = [0.0] * n
    for j in topo_order:
        if predecessors[j]:
            ES[j] = max(EF[i] for i in predecessors[j])
        EF[j] = ES[j] + durations[j]
        
    makespan_cpm = max(EF)
    
    LF = [makespan_cpm] * n
    LS = [makespan_cpm] * n
    for j in reversed(topo_order):
        if successors[j]:
            LF[j] = min(LS[k] for k in successors[j])
        LS[j] = LF[j] - durations[j]
        
    Slack = [LS[j] - ES[j] for j in range(n)]
    
    # Pre-compute resource demand vectors for faster lookup
    # demands is shape (n, num_resources)
    
    # Accumulator for average start times
    total_start_times = np.zeros(n, dtype=float)
    num_simulations = 0
    
    # Fast Greedy SGS Simulation
    # We use a simple priority rule: Min Slack, with random tie-breaking.
    # To speed up, we limit the number of simulations based on time.
    
    while time.time() < deadline:
        # State
        # active: list of currently running activities
        # finished: set of finished activities
        # current_time: float
        
        # We will track start times for this simulation
        sim_start_times = np.zeros(n)
        
        # Initialize
        # eligible: activities with no predecessors
        eligible = [j for j in range(n) if len(predecessors[j]) == 0]
        
        # To make it fast, we don't use a full event-driven simulation with 
        # fine-grained time steps. We use a "step-by-step" SGS where at each 
        # "decision point" (when resources allow), we start one activity.
        
        # However, standard Serial SGS starts one activity at a time at the 
        # earliest possible time.
        
        # Let's implement a fast Serial SGS:
        # 1. Start with time t=0.
        # 2. Find all eligible activities.
        # 3. Among eligible, find those that can start at t (resource check).
        # 4. If none can start at t, jump t to the next time an activity finishes.
        # 5. If some can start, pick one (using Min Slack + random noise), 
        #    start it, update resources, remove from eligible, add to active.
        # 6. Repeat until all activities are scheduled.
        
        # Data structures for speed
        remaining_preds = [len(predecessors[j]) for j in range(n)]
        active = []  # List of (finish_time, activity_id)
        finished = [False] * n
        current_time = 0.0
        remaining_count = n
        
        # Copy resource usage
        current_usage = np.zeros(num_resources, dtype=float)
        
        # We need to efficiently find the next eligible activity.
        # Since n=30, a linear scan is fine.
        
        while remaining_count > 0:
            # Find eligible activities (preds done)
            # And check if they can start at current_time
            
            best_candidate = -1
            best_score = float('inf')
            
            # Optimization: If no activity can start at current_time, 
            # we should jump time. But in Serial SGS, we usually just pick 
            # the one with highest priority that fits. If none fit, we wait.
            
            # Let's first check if any eligible activity fits at current_time.
            any_fits = False
            for j in range(n):
                if remaining_preds[j] == 0 and not finished[j]:
                    # Check resources
                    if np.all(current_usage + demands[j] <= capacities):
                        any_fits = True
                        # Score: Min Slack, with random tie-break
                        # We add a small random component to explore
                        score = Slack[j] + random.random() * 0.1
                        if score < best_score:
                            best_score = score
                            best_candidate = j
            
            if best_candidate != -1:
                # Start best_candidate at current_time
                sim_start_times[best_candidate] = current_time
                current_usage += demands[best_candidate]
                finish_time = current_time + durations[best_candidate]
                active.append((finish_time, best_candidate))
                remaining_count -= 1
                
                # Update successors
                for k in successors[best_candidate]:
                    remaining_preds[k] -= 1
                    if remaining_preds[k] == 0:
                        # It's now eligible (resource check happens in next loop)
                        pass
            else:
                # No activity can start at current_time.
                # Jump time to the next finish time of an active activity.
                if not active:
                    # Deadlock? Should not happen in valid RCPSP with no cycles.
                    # But if it does, break to avoid infinite loop.
                    break
                
                # Find min finish time
                min_finish = min(t for t, _ in active)
                current_time = min_finish
                
                # Remove all activities finishing at this time
                still_active = []
                for t, j in active:
                    if t == min_finish:
                        # Activity j finishes
                        current_usage -= demands[j]
                        finished[j] = True
                        # Note: In Serial SGS, we don't "finish" predecessors 
                        # in a way that blocks others except via resource release.
                        # The remaining_preds check handles eligibility.
                    else:
                        still_active.append((t, j))
                active = still_active
        
        if remaining_count > 0:
            # Simulation failed or timed out internally, skip accumulation
            continue
            
        total_start_times += sim_start_times
        num_simulations += 1
        
        # If we've done enough simulations, we can stop to ensure we return.
        # 30 activities, 4 resources. Each sim is very fast.
        # Let's cap at 5000 sims or until time limit.
        if num_simulations > 5000:
            break

    if num_simulations == 0:
        # Fallback to CPM Slack if no simulations completed
        return np.array([-s for s in Slack])

    # The priority should favor activities that start EARLY.
    # So we want the score to be HIGH for early start times.
    # Score = - Average Start Time
    avg_start_times = total_start_times / num_simulations
    scores = -avg_start_times
    
    # Add a tiny bit of CPM slack to break ties in the average start time
    # if two activities have identical average start times (unlikely with random noise)
    # But the random noise in the sim already handles this.
    
    return scores
