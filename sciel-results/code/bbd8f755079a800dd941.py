"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(instance):
    """
    Score each assembly task for the greedy line-balancing filler.
    
    Strategy: 'bottleneck density' priority.
    1. Compute the 'downstream depth' (longest path in time units) from each task
       to the end of the line (successors). This represents the critical path
       length that depends on this task being finished early.
    2. Compute the 'upstream depth' (longest path from start to this task).
    3. The priority is a combination of:
       - Task duration (longer tasks are harder to fit, so prioritize them to avoid fragmentation? 
         Actually, in assembly line balancing, we usually want to place tasks that have few successors 
         or are on critical paths first? 
         Let's rethink: The greedy algorithm picks the highest priority ELIGIBLE task that FITS.
         Eligible means all predecessors are already assigned to previous stations.
         We want to minimize the number of stations.
         
         Common heuristics:
         - Largest Processing Time (LPT): Good for makespan minimization in parallel machines.
         - Most Successors: Ensures that tasks with many dependents are placed early, leaving room for them.
         - Critical Path Rank: Tasks on the critical path should be prioritized.
         
         The prompt suggests: "ratio of a task's processing time to the minimum number of remaining stations required to complete its downstream subtree".
         This is a bit complex to define "minimum number of remaining stations" statically.
         
         Let's try a simpler, robust heuristic that often outperforms raw time:
         Priority = Time + (Sum of times of all successors) * weight
         Or simply: Longest path from the task to the end (inclusive).
         This is often called the "Critical Path" priority or "Longest Processing Time to End".
         
         Let's compute:
         - `succ_depth[i]`: The longest path time starting from task i, including i's time.
           succ_depth[i] = time[i] + max(succ_depth[j] for j in successors(i), default=0)
         
         Tasks with higher `succ_depth` are on longer critical paths. Placing them early ensures the downstream work can be started as soon as possible?
         Actually, if we place a task early, its successors become eligible earlier.
         If we prioritize tasks with long downstream chains, we ensure that the "critical path" tasks are placed in earlier stations, which minimizes the total length of the critical path in the final schedule.
         
         Let's test this hypothesis.
         Seed: `times` (LPT).
         Proposed: `succ_depth` (Longest path to end).
         
         Another option: `succ_depth` + `time` (which is just succ_depth).
         
         Let's also consider "Most Immediate Successors" (MIS) or "Most Successors" (MS).
         But `succ_depth` captures both time and depth.
         
         Let's implement `succ_depth`.
    """
    start_time = time.time()
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    cycle_time = instance["cycle_time"]
    
    # Get predecessors to build successors graph
    # instance["predecessors"] is likely a list of lists, where instance["predecessors"][i] is the list of predecessor indices for task i.
    preds = instance["predecessors"]
    
    # Build successors list
    # successors[i] is a list of task indices that depend on i
    successors = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            successors[p].append(i)
    
    # Compute longest path from each node to the end (succ_depth)
    # We need topological order. Since predecessors are given, we can do a DFS or Kahn's algorithm.
    # Given n=30, we can just use memoized DFS or simple iterative relaxation if we assume DAG.
    # Assembly line graphs are DAGs.
    
    # Topological sort (Kahn's algorithm)
    in_degree = [0] * n
    for i in range(n):
        for p in preds[i]:
            in_degree[i] += 1 # Wait, preds[i] are predecessors of i. So edges p -> i.
            # So in_degree[i] is the number of predecessors.
            
    # Actually, let's just compute in_degree correctly.
    in_degree = [0] * n
    for i in range(n):
        for p in preds[i]:
            pass # p is a predecessor of i. So p -> i.
            # We need to count edges coming INTO i.
            # The structure is: preds[i] = [p1, p2, ...]
            # So for each i, for each p in preds[i], there is an edge p -> i.
            # Thus, in_degree[i] is len(preds[i]).
            # And for each p, out_degree[p] increases.
            
    in_degree = [len(preds[i]) for i in range(n)]
    
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    # Copy in_degree for processing
    temp_in_degree = in_degree[:]
    
    while queue:
        # Sort queue to ensure deterministic order if needed, though for longest path it doesn't matter as long as we process in topo order
        u = queue.pop(0) 
        topo_order.append(u)
        for v in successors[u]:
            temp_in_degree[v] -= 1
            if temp_in_degree[v] == 0:
                queue.append(v)
                
    # If topo_order length < n, there's a cycle, but assume DAG.
    
    # Compute succ_depth (longest path from node to end)
    # We process in reverse topological order.
    succ_depth = [0.0] * n
    # Reverse topo order
    for u in reversed(topo_order):
        max_succ = 0.0
        for v in successors[u]:
            if succ_depth[v] > max_succ:
                max_succ = succ_depth[v]
        succ_depth[u] = times[u] + max_succ
        
    # The priority is succ_depth.
    # Tasks with longer downstream critical paths are prioritized.
    
    return np.array(succ_depth, dtype=float)

# EVOLVE-BLOCK-END
