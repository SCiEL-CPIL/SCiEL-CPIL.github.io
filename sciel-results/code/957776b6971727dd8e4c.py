import numpy as np
import time

def _greedy_balance(times, predecessors, cycle_time, priorities):
    n = len(times)
    # indegree: number of unassigned predecessors
    indegree = [0] * n
    successors = [[] for _ in range(n)]
    
    for i in range(n):
        for p in predecessors[i]:
            indegree[i] += 1
            successors[p].append(i)
            
    assigned = [False] * n
    num_stations = 0
    
    while assigned.count(False) > 0:
        # Start a new station
        num_stations += 1
        remaining_time = cycle_time
        
        while True:
            # Find eligible tasks (indegree 0, not assigned)
            eligible_tasks = [i for i in range(n) if indegree[i] == 0 and not assigned[i]]
            if not eligible_tasks:
                break # Should not happen in valid DAG if tasks remain, but safety check
            
            # Find best task among eligible that fits
            candidate = -1
            max_prio = -np.inf
            for i in eligible_tasks:
                if times[i] <= remaining_time:
                    if priorities[i] > max_prio:
                        max_prio = priorities[i]
                        candidate = i
            
            if candidate == -1:
                # No eligible task fits in remaining time.
                # Start new station.
                break
            
            # Assign candidate
            assigned[candidate] = True
            remaining_time -= times[candidate]
            
            # Update successors
            for s in successors[candidate]:
                indegree[s] -= 1
                
    return num_stations

def _calculate_initial_priorities(times, predecessors, cycle_time):
    n = len(times)
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in predecessors[i]:
            successors[p].append(i)
            
    # Longest Path (Critical Path) from each node to the end
    lp = [-1] * n
    def calc_lp(i):
        if lp[i] != -1:
            return lp[i]
        max_succ = 0
        for s in successors[i]:
            max_succ = max(max_succ, calc_lp(s))
        lp[i] = times[i] + max_succ
        return lp[i]
        
    for i in range(n):
        calc_lp(i)
        
    # Weighted Precedence: sum of times of all successors (transitive)
    wp = [-1] * n
    def calc_wp(i):
        if wp[i] != -1:
            return wp[i]
        total = 0
        for s in successors[i]:
            total += times[s] + calc_wp(s)
        wp[i] = total
        return wp[i]
        
    for i in range(n):
        calc_wp(i)
        
    # Combined priority: lp + wp
    priorities = np.array(lp) + np.array(wp)
    
    return priorities

def _fast_local_search(times, predecessors, cycle_time, initial_priority, deadline):
    n = len(times)
    current_priority = initial_priority.copy()
    current_score = _greedy_balance(times, predecessors, cycle_time, current_priority)
    best_priority = current_priority.copy()
    best_score = current_score
    
    rng = np.random.RandomState(42)
    max_evals = 5000 # Increased slightly, but still fast
    evals = 0
    no_improve_count = 0
    max_no_improve = 100
    
    while evals < max_evals and time.time() < deadline:
        if no_improve_count > max_no_improve:
            break
            
        i, j = rng.choice(n, 2, replace=False)
        new_priority = current_priority.copy()
        new_priority[i], new_priority[j] = current_priority[j], new_priority[i]
        
        new_score = _greedy_balance(times, predecessors, cycle_time, new_priority)
        evals += 1
        
        if new_score < current_score:
            current_priority = new_priority
            current_score = new_score
            no_improve_count = 0
            
            if current_score < best_score:
                best_score = current_score
                best_priority = current_priority.copy()
        else:
            no_improve_count += 1
            
    return best_priority

def priority(instance):
    times = instance['times']
    predecessors = instance['predecessors']
    cycle_time = instance['cycle_time']
    
    start_time = time.time()
    # Allow 140 seconds to be safe against the 150s kill
    deadline = start_time + 140.0
    
    initial_prio = _calculate_initial_priorities(times, predecessors, cycle_time)
    
    best_prio = _fast_local_search(times, predecessors, cycle_time, initial_prio, deadline)
    
    return best_prio
