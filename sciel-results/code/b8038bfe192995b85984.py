import numpy as np
from collections import deque
import time

def priority(instance):
    """
    Spectral Resource-Contention Priority Strategy.
    
    This strategy models the problem as a maximum-weight independent set on a 
    conflict graph. Nodes are activities. Edges connect pairs that cannot 
    overlap due to resource exhaustion. 
    Priorities are assigned based on the eigenvector centrality of this graph, 
    identifying structurally dominant bottlenecks that propagate delays through 
    the network. This spectral approach captures global resource contention 
    patterns that local CPM heuristics might miss.
    """
    start_time = time.time()
    
    # Extract instance data
    durations = np.asarray(instance["durations"], dtype=float)
    predecessors = instance["predecessors"]
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    
    n = len(durations)
    
    # If no resource data, fallback to simple duration-based (or 1.0)
    if demands is None or capacities is None:
        return np.ones(n)
    
    demands = np.asarray(demands, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    
    # Ensure shapes are correct
    # demands: (n, R), capacities: (R,)
    if demands.ndim == 1:
        # If 1D, maybe it's (n,) for a single resource? Or (R,) for all same?
        # Standard RCPSP is (n, R). If 1D, assume single resource column.
        demands = demands.reshape(-1, 1)
    if capacities.ndim == 0:
        capacities = np.array([capacities])
        
    # Safety check for time
    if time.time() - start_time > 10:
        return durations.copy()
        
    # 1. Build Conflict Graph
    # Two activities i and j conflict if they cannot be executed simultaneously.
    # Condition: For any resource k, demand[i][k] + demand[j][k] > capacity[k]
    
    # Vectorized check for efficiency
    # We need to check for all pairs (i, j) if sum(demands[i] + demands[j], axis=1) > capacities
    
    # To avoid O(n^2 * R) explicit loops in Python, we can use matrix operations
    # or a clever broadcasting trick, but n=30 is small enough that even a loop 
    # or simple numpy broadcasting is fast.
    
    # Let's construct the adjacency matrix A where A[i,j] = 1 if they conflict.
    # We can compute the "resource load" of pair (i,j) as demands[i] + demands[j].
    # Conflict if max_k (demands[i,k] + demands[j,k] - capacities[k]) > 0.
    
    # Efficient way:
    # diff_matrix[i, j, k] = demands[i, k] + demands[j, k] - capacities[k]
    # This is (n, n, R). For n=30, R=4, this is 30*30*4 = 3600 floats. Very small.
    
    # demands: (n, R)
    # We want to add demands[i] and demands[j] for all i, j.
    # demands_i = demands[:, np.newaxis, :]  # (n, 1, R)
    # demands_j = demands[np.newaxis, :, :]  # (1, n, R)
    # pair_demands = demands_i + demands_j   # (n, n, R)
    # excess = pair_demands - capacities     # (n, n, R)
    # conflicts = np.any(excess > 0, axis=2) # (n, n)
    
    # Let's do it safely with broadcasting
    d_i = demands[:, np.newaxis, :]  # (n, 1, R)
    d_j = demands[np.newaxis, :, :]  # (1, n, R)
    
    # Check time again
    if time.time() - start_time > 20:
        return durations.copy()

    pair_demands = d_i + d_j
    excess = pair_demands - capacities
    
    # A conflict exists if ANY resource is over capacity
    # Note: i == j is always a self-loop, but we ignore self-loops in eigenvector centrality usually.
    # Also, if an activity alone exceeds capacity (which shouldn't happen in valid instances), 
    # it conflicts with everyone.
    
    conflicts = np.any(excess > 0, axis=2)
    
    # Zero out self-loops
    np.fill_diagonal(conflicts, 0)
    
    # The conflict graph is undirected. 
    # We want to find the "most central" nodes in this conflict graph.
    # Eigenvector centrality: x = A x / lambda.
    # Power iteration: x_{t+1} = A x_t / ||A x_t||
    
    # Initialize x with 1s
    x = np.ones(n)
    x /= np.linalg.norm(x)
    
    # Power iteration parameters
    max_iter = 100
    tol = 1e-6
    
    # Check time
    if time.time() - start_time > 30:
        return x
    
    for _ in range(max_iter):
        x_new = conflicts.dot(x)
        norm = np.linalg.norm(x_new)
        if norm == 0:
            break
        x_new /= norm
        
        # Check convergence
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
        
        # Time check inside loop
        if time.time() - start_time > 40:
            break
            
    # x now contains the eigenvector centrality scores.
    # Higher centrality means the node is connected to many other high-centrality nodes.
    # In the context of "conflicts", a high centrality node is involved in many conflicts
    # with other heavily conflicting nodes. These are the "bottlenecks".
    
    # We want to schedule bottlenecks FIRST? Or LAST?
    # If a node is a bottleneck (high centrality in conflict graph), it causes the most
    # "friction" in the schedule. 
    # Usually, in RCPSP, we want to schedule "critical" or "resource-heavy" tasks early 
    # to secure resources. 
    # However, if we schedule a highly conflicting task early, it might block others.
    # But if we schedule it late, it might be delayed by others.
    # 
    # Let's look at the "Max Weight Independent Set" analogy.
    # We want to pick a set of non-conflicting tasks.
    # The "dominant" tasks are those that, if scheduled, force the most other tasks to wait.
    # Scheduling them early might be good to "clear the deck".
    # Alternatively, scheduling them early secures the scarce resources.
    # 
    # Let's assume higher centrality = higher priority (schedule earlier).
    
    # We should also consider the "weight" of the node. 
    # A node with high duration and high centrality is very important.
    # Let's combine Centrality with Duration to ensure long tasks aren't ignored.
    # But the directive says "eigenvector centrality ... to identify structurally dominant bottlenecks".
    # Let's stick to the spectral score primarily, but maybe normalize.
    
    # Normalize x to [0, 1] for stability
    min_x = np.min(x)
    max_x = np.max(x)
    if max_x > min_x:
        x_norm = (x - min_x) / (max_x - min_x)
    else:
        x_norm = np.ones(n)
        
    # Final Score:
    # We want to return a score such that higher is scheduled first.
    # Score = Centrality.
    
    # One potential issue: If the graph is disconnected or sparse, power iteration might converge to a trivial solution or be slow.
    # With n=30, it's fast.
    
    return x_norm
