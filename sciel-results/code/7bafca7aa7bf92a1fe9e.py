"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def priority(instance):
    """
    Scores tasks using a lookahead heuristic.
    
    Strategy:
    1. Parse instance data.
    2. Compute a base score for each task based on:
       - Duration (longer is better)
       - Number of successors (more successors = more critical, higher priority)
       - "Slack" or "Critical Path" position (tasks on the critical path are prioritized)
    3. To make it "structurally different" and more robust than simple LPT, 
       we incorporate a dynamic simulation estimate:
       We estimate how many stations a task *forces* to be created if it's placed late.
       
    However, the interface requires a static score per task returned by `priority(instance)`.
    The evaluator uses these scores to fill stations.
    
    A more effective static heuristic for ALB (Assembly Line Balancing) is often:
    Priority = (Duration + Sum of Successor Durations) / (1 + Number of Successors)
    Or simply:
    Priority = Duration + Sum of Successor Durations (Critical Path Length from that node)
    
    Let's use the "Longest Processing Time" (LPT) combined with "Successor Count".
    Specifically, we calculate the "Critical Path Length" (CPL) from each task to the end.
    Tasks with higher CPL are more critical.
    
    Formula:
    score[i] = times[i] + sum(times[j] for j in successors[i])
    
    This is essentially the "Rank" or "Position" in the precedence graph, often used in
    heuristic ALB algorithms (like the one by Jackson or Schrage).
    """
    start_time = time.time()
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    
    # Predecessors are given. We need successors to compute downstream critical path.
    # instance["predecessors"] is a list of lists, where predecessors[i] is a list of indices
    # that must be completed before i.
    
    predecessors = instance["predecessors"]
    
    # Build successor list
    successors = [[] for _ in range(n)]
    for i in range(n):
        if isinstance(predecessors, list):
            preds = predecessors[i]
        else:
            preds = list(predecessors[i])
            
        for p in preds:
            successors[p].append(i)
    
    # Calculate Critical Path Length (CPL) for each node.
    # CPL[i] = times[i] + max(CPL[j] for j in successors[i])
    # We need to process nodes in reverse topological order.
    
    # Since it's a DAG, we can use Kahn's algorithm or DFS.
    # Given n=30, DFS with memoization is very fast and safe.
    
    cpl = [0.0] * n
    visited = [False] * n
    
    def calc_cpl(i):
        if visited[i]:
            return cpl[i]
        visited[i] = True
        max_succ_cpl = 0.0
        for s in successors[i]:
            max_succ_cpl = max(max_succ_cpl, calc_cpl(s))
        cpl[i] = times[i] + max_succ_cpl
        return cpl[i]
    
    for i in range(n):
        if not visited[i]:
            calc_cpl(i)
    
    # The score is the CPL.
    # This prioritizes tasks that are on the "bottleneck" path of the assembly line.
    # This is structurally different from just using `times` because it accounts for
    # the dependency structure and downstream impact.
    
    scores = np.array(cpl, dtype=float)
    
    # Safety check: if any score is NaN or Inf, fall back to times
    if not np.all(np.isfinite(scores)):
        scores = times.copy()
        
    return scores

# EVOLVE-BLOCK-END
