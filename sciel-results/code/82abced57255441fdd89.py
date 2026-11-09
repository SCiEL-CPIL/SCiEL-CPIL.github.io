import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin for safety
    
    N = 256
    
    # Precompute squares up to 2*N
    max_sum = 2 * N
    squares = set()
    i = 2
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    
    # 1. Filter nodes that violate the diagonal condition (a=a => 2a is square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
            
    M = len(valid_nodes)
    
    # 2. Build conflict graph using adjacency lists for faster greedy iteration
    # neighbors[i] is a set of indices j such that valid_nodes[i] + valid_nodes[j] is a square
    neighbors = [set() for _ in range(M)]
    
    for i in range(M):
        vi = valid_nodes[i]
        for j in range(i + 1, M):
            vj = valid_nodes[j]
            if vi + vj in squares:
                neighbors[i].add(j)
                neighbors[j].add(i)
                
    # Calculate degrees
    degrees = [len(neighbors[i]) for i in range(M)]
    
    best_set = []
    best_size = 0
    
    def greedy_mis(order):
        """
        Runs a greedy Maximum Independent Set algorithm given a specific order of nodes.
        Returns the set of node indices.
        """
        selected = []
        blocked = set()
        
        for idx in order:
            if idx in blocked:
                continue
            # Select idx
            selected.append(idx)
            # Block idx and all its neighbors
            blocked.add(idx)
            blocked.update(neighbors[idx])
            
        return selected
    
    # Strategy 1: Order by degree descending (common heuristic for MIS)
    order_desc = sorted(range(M), key=lambda x: degrees[x], reverse=True)
    set_indices = greedy_mis(order_desc)
    best_set = [valid_nodes[i] for i in set_indices]
    best_size = len(best_set)
    
    # Strategy 2: Order by degree ascending
    order_asc = sorted(range(M), key=lambda x: degrees[x])
    set_indices = greedy_mis(order_asc)
    if len(set_indices) > best_size:
        best_set = [valid_nodes[i] for i in set_indices]
        best_size = len(set_indices)
        
    # Strategy 3: Random restarts to escape local optima of the greedy choice
    # We use a fixed seed for reproducibility if needed, but random is fine here.
    # We limit the number of restarts to ensure speed.
    num_restarts = 50
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        # Random order, but maybe weighted by degree? 
        # Simple random shuffle is a good baseline for diversity.
        random_order = list(range(M))
        random.shuffle(random_order)
        set_indices = greedy_mis(random_order)
        if len(set_indices) > best_size:
            best_set = [valid_nodes[i] for i in set_indices]
            best_size = len(set_indices)
            
    # Strategy 4: Greedy with "least constraining value" style tie-breaking?
    # Or simply: sort by (degree, value) or similar.
    # Let's try sorting by degree descending, then by value descending.
    order_desc_val = sorted(range(M), key=lambda x: (degrees[x], valid_nodes[x]), reverse=True)
    set_indices = greedy_mis(order_desc_val)
    if len(set_indices) > best_size:
        best_set = [valid_nodes[i] for i in set_indices]
        best_size = len(set_indices)

    # If we still have time, we can try a simple local search (swap) to improve.
    # But greedy with multiple orders is usually very strong for this size.
    # Let's ensure we have a valid non-empty set.
    if not best_set:
        # Fallback: pick the first valid node
        if valid_nodes:
            best_set = [valid_nodes[0]]
        else:
            # This should not happen as 1 is valid (2*1=2 not square)
            best_set = [1]
            
    return best_set

# EVOLVE-BLOCK-START
# The function construct is defined above.
# EVOLVE-BLOCK-END
