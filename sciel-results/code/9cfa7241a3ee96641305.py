import numpy as np
from collections import defaultdict, deque

def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    # Build successor lists
    successors = defaultdict(list)
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            successors[p].append(i)
            
    # Calculate in-degrees for topological sort (Kahn's algorithm)
    # We need to process nodes such that all successors are processed before the node
    # Wait, to compute total successors, we need to process in REVERSE topological order.
    # i.e., if A -> B, B must be processed before A.
    # So we need a topological order, then iterate backwards.
    
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        in_degree[i] = len(predecessors[i])
        
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # If topo_order length is not n, there's a cycle (should not happen in ALB)
    # Compute total successors in reverse topological order
    total_successors = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        # Total successors of u = count of immediate successors + sum of total successors of immediate successors?
        # No, that would double count if there are diamond dependencies.
        # A -> B, A -> C, B -> D, C -> D.
        # Immediate succ of A: B, C.
        # Total succ of B: D.
        # Total succ of C: D.
        # Total succ of A should be {B, C, D} = 3.
        # Sum of immediate (2) + sum of total succ of immediate (1+1=2) = 4. Overcounting.
        
        # To count unique nodes, we need a set or a bitset, which is slow for large N.
        # However, N=30 is small. We can use a set for each node during the reverse pass?
        # Or simpler: For N=30, we can just do a BFS/DFS from each node to count successors.
        # O(N*(N+E)) is 30 * 30 = 900 ops. Very fast.
        
        pass
        
    # Given N=30, simple BFS/DFS for each node is sufficient and avoids topological complexity/errors.
    total_succ_counts = np.zeros(n, dtype=float)
    for i in range(n):
        visited = set([i])
        q = deque([i])
        count = 0
        while q:
            curr = q.popleft()
            for next_node in successors[curr]:
                if next_node not in visited:
                    visited.add(next_node)
                    count += 1
                    q.append(next_node)
        total_succ_counts[i] = count

    # Compute slack
    slack = np.maximum(cycle_time - times, 1e-6)
    
    # Compute ratio: slack / (total_successors + 1)
    # Using total_successors + 1 to handle sinks (0 successors) and avoid div by zero
    # Sinks have 0 total successors. Ratio = slack / 1.
    # This maintains the behavior for sinks similar to immediate successors (which was also 1).
    
    denom = total_succ_counts + 1.0
    ratio = slack / denom
    
    # Score: 1 / ratio
    scores = 1.0 / (ratio + 1e-6)
    
    return scores
