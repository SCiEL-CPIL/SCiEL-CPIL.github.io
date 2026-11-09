"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
import random
import math
from collections import deque

def priority(instance):
    """
    Score each assembly task for the priority rule.
    
    Strategy:
    1. Compute graph properties (rank, successors, etc.).
    2. Initialize a strong heuristic priority vector.
    3. Use Simulated Annealing to optimize the priority vector by 
       minimizing the number of stations in the greedy simulation.
    4. Return the optimized priority scores.
    """
    start_time = _time.time()
    # We have 150s total for 5 instances. Let's aim for ~25s per instance.
    # However, priority is called per instance. We don't know how many calls total 
    # but the contract says "five generated 30-task instances". 
    # So we can assume this function is called 5 times.
    # To be safe, we use a per-call deadline of 28s.
    DEADLINE = 28.0 

    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor lists and in-degree
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        preds = predecessors[i] if i < len(predecessors) else []
        in_degree[i] = len(preds)
        for p in preds:
            successors[p].append(i)
    
    # Topological order
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo = []
    indeg_copy = in_degree[:]
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in successors[u]:
            indeg_copy[v] -= 1
            if indeg_copy[v] == 0:
                queue.append(v)
    
    if len(topo) != n:
        # Cycle detected, fallback
        return times.copy()

    # Compute Longest Path to End (Rank)
    rank = np.zeros(n, dtype=float)
    for u in reversed(topo):
        max_succ = 0.0
        for v in successors[u]:
            if rank[v] > max_succ:
                max_succ = rank[v]
        rank[u] = times[u] + max_succ
        
    # Compute Number of Successors
    num_successors = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Compute Longest Path from Start
    lp_start = np.zeros(n, dtype=float)
    for node in topo:
        preds = predecessors[node] if node < len(predecessors) else []
        if preds:
            max_pred = 0.0
            for p in preds:
                val = times[p] + lp_start[p]
                if val > max_pred:
                    max_pred = val
            lp_start[node] = max_pred
        lp_start[node] += times[node]

    # Initial Heuristic: Combine Rank, Successors, and Time
    # Weights are tunable, but Rank and Successors are usually dominant.
    base_scores = rank * 1.0 + num_successors * 0.5 + times * 0.1
    
    def simulate_stations(priority_scores):
        """
        Fast simulation of greedy station filling.
        Returns number of stations.
        """
        # remaining_preds[i] is the number of uncompleted predecessors for task i
        remaining_preds = in_degree[:]
        completed = [False] * n
        stations = 0
        current_load = 0.0
        remaining_tasks = n
        
        # Pre-compute successors list for fast access
        # We already have 'successors' list of lists.
        
        while remaining_tasks > 0:
            # Find eligible tasks
            eligible = []
            # Iterate through all tasks. 
            # Optimization: We could maintain a list of available tasks, 
            # but for N=30, scanning is fast enough and simpler.
            for task in range(n):
                if not completed[task]:
                    if remaining_preds[task] == 0:
                        if times[task] <= cycle_time - current_load:
                            eligible.append(task)
            
            if not eligible:
                # Start new station
                stations += 1
                current_load = 0.0
                # Re-evaluate eligible with full cycle time
                for task in range(n):
                    if not completed[task]:
                        if remaining_preds[task] == 0:
                            if times[task] <= cycle_time:
                                eligible.append(task)
                
                if not eligible:
                    # Infeasible (should not happen if instance is valid)
                    return float('inf')
            
            # Pick highest priority task from eligible
            # Using max with key is efficient for small lists
            best_task = max(eligible, key=lambda t: priority_scores[t])
            
            # Mark as completed
            completed[best_task] = True
            remaining_tasks -= 1
            current_load += times[best_task]
            
            # Update predecessors count for successors
            for succ in successors[best_task]:
                remaining_preds[succ] -= 1
                
        return stations

    # Initial evaluation
    current_scores = base_scores.copy()
    current_K = simulate_stations(current_scores)
    best_scores = current_scores.copy()
    best_K = current_K
    
    # Simulated Annealing
    # K is an integer. Delta K is at least 1.
    # T should be comparable to Delta K.
    T = 5.0 
    cooling_rate = 0.995
    min_T = 0.5
    
    # Random seed for reproducibility within the run, but different per instance?
    # The task says "five generated instances". We should be deterministic if possible, 
    # or use a fixed seed. Let's use a fixed seed for stability.
    random.seed(42)
    
    while _time.time() - start_time < DEADLINE:
        # Generate neighbor
        # Strategy: Swap scores of two random tasks, or add noise to one task.
        # Swapping is good for exploring permutations.
        # Noise is good for fine-tuning.
        
        new_scores = current_scores.copy()
        
        # Choose mutation type
        if random.random() < 0.5:
            # Swap two tasks
            i, j = random.sample(range(n), 2)
            new_scores[i], new_scores[j] = new_scores[j], new_scores[i]
        else:
            # Add noise to one task
            i = random.randint(0, n-1)
            # Noise magnitude relative to score range
            score_range = np.max(base_scores) - np.min(base_scores)
            if score_range == 0:
                score_range = 1.0
            noise = random.uniform(-0.2 * score_range, 0.2 * score_range)
            new_scores[i] += noise
            
        new_K = simulate_stations(new_scores)
        
        # Acceptance criterion
        if new_K < current_K:
            current_scores = new_scores
            current_K = new_K
            if new_K < best_K:
                best_K = new_K
                best_scores = new_scores.copy()
        else:
            # Accept with probability exp(-(current_K - new_K) / T)
            # Note: new_K >= current_K, so current_K - new_K <= 0
            # We want to accept worse solutions.
            # Prob = exp((current_K - new_K) / T)
            # If new_K == current_K, prob = 1.
            # If new_K > current_K, prob < 1.
            prob = math.exp((current_K - new_K) / T)
            if random.random() < prob:
                current_scores = new_scores
                current_K = new_K
        
        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T

    return np.asarray(best_scores, dtype=float)


# EVOLVE-BLOCK-END
