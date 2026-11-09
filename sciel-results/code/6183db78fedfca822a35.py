import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    N = 256
    
    # 1. Filter nodes: 2*v must not be a perfect square
    valid_nodes = []
    for v in range(1, N + 1):
        s = 2 * v
        r = int(math.isqrt(s))
        if r * r != s:
            valid_nodes.append(v)
    
    if not valid_nodes:
        return [1]
        
    M = len(valid_nodes)
    
    # 2. Build adjacency list
    # adj[i] is a set of indices j such that valid_nodes[i] + valid_nodes[j] is a square
    adj = [set() for _ in range(M)]
    
    # Precompute squares for fast lookup
    squares = set()
    max_sum = 2 * N
    k = 1
    while k * k <= max_sum:
        squares.add(k * k)
        k += 1
        
    for i in range(M):
        vi = valid_nodes[i]
        for j in range(i + 1, M):
            vj = valid_nodes[j]
            s = vi + vj
            if s in squares:
                adj[i].add(j)
                adj[j].add(i)
                
    # 3. Solve for Maximum Independent Set using Local Search with Restarts
    
    def greedy_init(adj, M):
        # Greedy: pick node with min degree, add to set, remove neighbors
        remaining = set(range(M))
        in_set = [False] * M
        while remaining:
            if time.time() > deadline:
                break
            # Find node with min degree in remaining subgraph
            min_deg_node = None
            min_deg = float('inf')
            for u in remaining:
                deg = len(adj[u] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_deg_node = u
            if min_deg_node is None:
                break
            in_set[min_deg_node] = True
            remaining.discard(min_deg_node)
            for v in adj[min_deg_node]:
                if v in remaining:
                    remaining.discard(v)
        return in_set

    def local_improve(in_set, adj, M):
        # Try to add any node that is not in set but has no neighbors in set
        # We can do this iteratively until no more can be added
        changed = True
        while changed:
            changed = False
            # We need to know which nodes are "available" (not in set)
            # and check if their neighbors are all not in set.
            # To be efficient, we can just iterate over all nodes not in set.
            for u in range(M):
                if in_set[u]:
                    continue
                # Check if u can be added
                can_add = True
                for v in adj[u]:
                    if in_set[v]:
                        can_add = False
                        break
                if can_add:
                    in_set[u] = True
                    changed = True
        return in_set

    best_set = [False] * M
    best_size = 0
    
    # Run multiple restarts with random perturbations
    num_restarts = 50 # Adjust based on time
    random.seed(42) # Deterministic for reproducibility if needed, or just use system time
    
    for r in range(num_restarts):
        if time.time() > deadline - 5: # Leave 5s for final processing
            break
            
        # 1. Greedy Init
        current_set = greedy_init(adj, M)
        
        # 2. Perturbation (if not first restart)
        if r > 0:
            # Remove k random nodes from current_set to allow different configurations
            # k can be small, e.g., 5-10
            k_remove = random.randint(3, 10)
            set_indices = [i for i in range(M) if current_set[i]]
            if len(set_indices) > k_remove:
                to_remove = random.sample(set_indices, k_remove)
                for idx in to_remove:
                    current_set[idx] = False
        
        # 3. Local Improvement
        current_set = local_improve(current_set, adj, M)
        
        # 4. Update Best
        current_size = sum(current_set)
        if current_size > best_size:
            best_size = current_size
            best_set = current_set[:]
            
    # Extract result
    result = [valid_nodes[i] for i in range(M) if best_set[i]]
    
    if not result:
        # Fallback if something went wrong
        return [valid_nodes[0]]
        
    return result
