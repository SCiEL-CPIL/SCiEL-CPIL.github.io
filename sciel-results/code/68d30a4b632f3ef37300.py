import numpy as np
import time
from collections import deque

def priority(instance):
    start_time = time.time()
    deadline = 115  # Safety margin before 120s limit
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n = len(times)
    
    if n == 0:
        return np.array([])
        
    # 1. Build successor graph
    successors = [[] for _ in range(n)]
    
    for j in range(n):
        preds = predecessors[j]
        for i in preds:
            successors[i].append(j)
            
    # Correct In-Degree Calculation for Kahn's Topological Sort
    # in_degree[i] = number of predecessors of i
    in_degree = [0] * n
    for j in range(n):
        for i in predecessors[j]:
            in_degree[j] += 1
            
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # Handle potential cycles gracefully by appending remaining nodes
    if len(topo_order) < n:
        remaining = [i for i in range(n) if i not in topo_order]
        topo_order.extend(remaining)
        
    # 2. Calculate Positional Weights
    # w(i) = t(i) + sum(w(j) for j in successors(i))
    # Process in reverse topological order (sinks to sources)
    pos_weights = times.copy()
    
    for u in reversed(topo_order):
        for v in successors[u]:
            pos_weights[u] += pos_weights[v]
            
    # 3. Calculate Immediate Successor Count (ISC)
    # Tasks with fewer immediate successors are often "tighter" in the end
    # and should be prioritized to prevent bottlenecks at the end of the line.
    isc = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # 4. Combine Heuristics
    # Primary: Positional Weight (Longest Path)
    # Secondary: Immediate Successor Count (Fewer is better, so we negate or subtract)
    # We want to maximize the score.
    # High positional weight is good.
    # Low immediate successor count is good.
    # So: Score = pos_weights - alpha * isc
    # We choose a small alpha so that positional weight dominates,
    # but ties are broken by fewer successors.
    
    # Normalize isc to be in a similar range to pos_weights to avoid scaling issues?
    # Or just use a small constant.
    # pos_weights can be large (sum of times).
    # isc is small (integer count).
    # Let's try subtracting a small fraction of the max time or just a small constant.
    # Actually, a common heuristic is "Rank" = (pos_weight, -isc).
    # Since we return a single float, we can do:
    # priority = pos_weights - 0.01 * isc
    # This ensures that if two tasks have very close positional weights,
    # the one with fewer successors is preferred.
    
    # Let's refine the combination.
    # If we just return pos_weights, we get the "Longest Path" heuristic.
    # The "Ranked Positional Weight" (RPW) is standard.
    # Another common heuristic is "Minimum Slack" or "Maximum Precedence".
    # Let's stick to RPW but break ties with ISC.
    
    # To make the tie-breaking robust, we can scale isc by a very small epsilon.
    # However, if times are integers, pos_weights are integers.
    # If we subtract 0.1 * isc, it breaks integer ties.
    
    # Let's try a weighted sum.
    # Maximize: pos_weights
    # Minimize: isc
    
    # A robust way:
    # priority = pos_weights * 1000.0 - isc
    # This effectively sorts by pos_weights first, then by -isc.
    
    max_pw = np.max(pos_weights) if n > 0 else 1.0
    if max_pw == 0:
        max_pw = 1.0
        
    # Scale isc to be negligible compared to differences in pos_weights,
    # but significant enough to break ties.
    # If pos_weights are integers, a difference of 1 is the smallest unit.
    # So we can subtract isc * 0.1.
    
    final_priority = pos_weights - 0.1 * isc
    
    # Check time
    if time.time() - start_time > deadline - 5:
         # Fallback to simple LPT if somehow slow (unlikely)
         return times
         
    return final_priority
