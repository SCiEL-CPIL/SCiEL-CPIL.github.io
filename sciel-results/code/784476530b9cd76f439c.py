"""
Strategy: Dynamic Priority Heuristic based on Critical Path Density and Fan-out.

Instead of searching for an optimal assignment (which is NP-hard and slow) or
reverse-engineering static scores from a found solution, we compute a robust
heuristic score for each task that reflects its structural importance in the
assembly line.

The score for a task t is a combination of:
1. Critical Path Density: times[t] / (times[t] + max_downstream_height[t])
   This prioritizes tasks that are significant portions of the critical path
   leading through them. It helps identify "bottlenecks" that constrain the
   cycle time more than others.
2. Fan-out Pressure: len(successors[t])
   Tasks with many immediate successors are branching points. Placing them
   early (high priority) ensures their successors are eligible sooner,
   reducing the "waiting time" for downstream tasks and potentially allowing
   better packing of subsequent stations.

We combine these into a single score. Since the evaluator uses a greedy
"highest priority eligible task" rule, higher scores mean the task is picked
earlier. We want to pick critical, high-fanout tasks early to establish the
backbone of the line.

This approach is O(N^2) or O(N log N) depending on implementation details,
which is extremely fast (microseconds), allowing us to run it safely within
the time limit without risk of timeout. It avoids the local minima traps of
static simple heuristics (like just "longest first") by considering the
global structure (downstream paths).
"""

# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time

def _soft_max(values, temperature=1.0):
    """
    Compute a smooth approximation of max(values) using log-sum-exp.
    As temperature -> 0, this approaches the hard max.
    """
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    max_val = np.max(arr)
    if max_val == 0:
        return 0.0
    # Avoid overflow by subtracting max
    shifted = arr - max_val
    log_sum = np.log(np.sum(np.exp(shifted / temperature)))
    return max_val + temperature * log_sum

def priority(instance):
    """
    Computes priority scores for each task in the assembly line.
    
    Args:
        instance: dict with keys 'times', 'predecessors', 'cycle_time'
        
    Returns:
        numpy array of scores, one per task.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    times = np.array(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n_tasks = len(times)
    
    # 1. Build successor lists and in-degrees for topological sort
    succs = [[] for _ in range(n_tasks)]
    in_deg = [0] * n_tasks
    
    for i in range(n_tasks):
        for p in preds[i]:
            if p >= 0 and p < n_tasks:
                succs[p].append(i)
                in_deg[i] += 1
                
    # 2. Topological Sort (Kahn's algorithm)
    q = deque([i for i in range(n_tasks) if in_deg[i] == 0])
    topo_order = []
    indeg_copy = in_deg[:]
    
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in succs[u]:
            indeg_copy[v] -= 1
            if indeg_copy[v] == 0:
                q.append(v)
                
    # If not all tasks are in topo_order, there's a cycle or invalid input.
    if len(topo_order) < n_tasks:
        topo_order = list(range(n_tasks))

    # 3. Compute "Height" (Longest Path to End) and "Soft Height"
    # height[t] = times[t] + max(height[s] for s in successors[t])
    # soft_height[t] = times[t] + soft_max(soft_height[s] for s in successors[t])
    # The soft height captures the aggregate workload pressure, not just the single worst path.
    
    height = [0.0] * n_tasks
    soft_height = [0.0] * n_tasks
    
    # Temperature for soft-max: smaller is closer to hard max.
    # We want to capture "bottleneck" nature, so a moderate temperature.
    temp = 2.0
    
    for u in reversed(topo_order):
        if not succs[u]:
            height[u] = times[u]
            soft_height[u] = times[u]
        else:
            succ_heights = [height[v] for v in succs[u]]
            succ_soft_heights = [soft_height[v] for v in succs[u]]
            
            max_h = max(succ_heights)
            height[u] = times[u] + max_h
            
            soft_max_h = _soft_max(succ_soft_heights, temperature=temp)
            soft_height[u] = times[u] + soft_max_h
            
            # Check deadline
            if time.time() > deadline:
                # Fallback to simple scores if we're running out of time
                scores = np.zeros(n_tasks, dtype=float)
                for i in range(n_tasks):
                    scores[i] = times[i]
                return scores

    # 4. Compute Scores
    # We want to prioritize tasks that are "critical" and "high fanout".
    
    # Component 1: Critical Path Density (based on hard max height)
    # density[u] = times[u] / height[u]
    
    # Component 2: Soft Bottleneck Pressure
    # pressure[u] = times[u] / soft_height[u]
    # This is higher if the task is a large fraction of the *aggregate* downstream workload.
    
    # Component 3: Fan-out
    # fanout[u] = len(succs[u])
    
    max_height = max(height) if height else 1.0
    if max_height == 0:
        max_height = 1.0
        
    max_soft_height = max(soft_height) if soft_height else 1.0
    if max_soft_height == 0:
        max_soft_height = 1.0
        
    max_fanout = max([len(s) for s in succs]) if succs else 1
    if max_fanout == 0:
        max_fanout = 1
        
    scores = np.zeros(n_tasks, dtype=float)
    
    for u in range(n_tasks):
        # Density: hard critical path
        density = times[u] / height[u] if height[u] > 0 else 0
        
        # Pressure: soft aggregate workload
        pressure = times[u] / soft_height[u] if soft_height[u] > 0 else 0
        
        # Fanout: normalized
        fanout = len(succs[u])
        norm_fanout = fanout / max_fanout
        
        # Combine:
        # We weight the soft pressure slightly higher because it captures more global structure.
        # The hard density is a specific case of soft pressure when temperature -> 0.
        # Using both might provide robustness.
        
        scores[u] = 0.5 * density + 0.4 * pressure + 0.1 * norm_fanout
        
        # Check deadline
        if time.time() > deadline:
            break
            
    return scores

# EVOLVE-BLOCK-END
