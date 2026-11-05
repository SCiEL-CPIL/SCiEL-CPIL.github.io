"""Search for a Hamilton cycle with minimal maximum edge arrival index."""
import time
import sys
import random

# The edge list from the pinned shuffle
EDGE_LIST = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

N = 24
TOTAL_EDGES = len(EDGE_LIST)

# Build edge index: edge_index[(u,v)] = 0-based index in EDGE_LIST
def _build_edge_index():
    idx = {}
    for i, (u, v) in enumerate(EDGE_LIST):
        if u > v:
            u, v = v, u
        idx[(u, v)] = i
    return idx

EDGE_INDEX = _build_edge_index()

def _get_edge_idx(u, v):
    """Get 0-based index of edge (u,v) in the shuffled list."""
    if u > v:
        u, v = v, u
    return EDGE_INDEX[(u, v)]

def _cycle_max_index(cycle):
    """Compute the max 0-based edge index in the Hamilton cycle.
    cycle is a list of N vertices in order; closing edge is (cycle[-1], cycle[0])."""
    mx = 0
    for i in range(N):
        u = cycle[i]
        v = cycle[(i + 1) % N]
        e = _get_edge_idx(u, v)
        if e > mx:
            mx = e
    return mx

def _local_search(cycle, time_limit):
    """Perform local search to minimize max edge index."""
    start_time = time.time()
    best_cycle = list(cycle)
    best_max = _cycle_max_index(best_cycle)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # 2-opt moves
        for i in range(N - 1):
            for j in range(i + 2, N):
                a = best_cycle[i]
                b = best_cycle[i + 1]
                c = best_cycle[j]
                d = best_cycle[(j + 1) % N]
                
                # Check if new edges exist (they always do in K_N)
                e1 = _get_edge_idx(a, c)
                e2 = _get_edge_idx(b, d)
                
                # Calculate new max without full cycle scan
                # Old max is best_max.
                # New edges are e1, e2. Removed edges are (a,b) and (c,d).
                # If best_max was one of the removed edges, the new max might be lower.
                # We need to check if max(e1, e2, other_edges_max) < best_max
                
                # Optimization: Only compute full max if potential improvement
                # If e1 < best_max and e2 < best_max, then new_max <= max(e1, e2, second_worst_old)
                # But simply computing new cycle max is O(N). N=24 is small.
                
                new_cycle = list(best_cycle)
                new_cycle[i+1:j+1] = reversed(new_cycle[i+1:j+1])
                new_max = _cycle_max_index(new_cycle)
                
                if new_max < best_max:
                    best_cycle = new_cycle
                    best_max = new_max
                    improved = True
                    break
            if improved:
                break
        
        # 3-opt moves (optional, but 2-opt is often sufficient for small N)
        # For N=24, 2-opt is very fast.
        
    return best_cycle, best_max

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline to be safe
    
    best_cycle = None
    best_max = float('inf')
    
    rng = random.Random(42)
    
    # Multiple random restarts
    # Each restart: generate random permutation, then local search
    # We can run many restarts in 85s for N=24
    
    max_restarts = 1000
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        # Generate random permutation
        cycle = list(range(N))
        rng.shuffle(cycle)
        
        # Local search
        current_cycle, current_max = _local_search(cycle, 0.5) # 0.5s per restart limit
        
        if current_max < best_max:
            best_max = current_max
            best_cycle = current_cycle
            
        # If we found a very good solution, we can stop or tighten time
        # The minimum possible max index is at least N-1 (since we need N edges, 
        # and the earliest N edges might not form a cycle, but theoretically 
        # the max index must be >= N-1 if the first N edges formed a cycle, 
        # but usually it's higher).
        
        # Heuristic: if best_max is close to N, stop.
        if best_max < N + 5:
            break

    if best_cycle is None:
        return list(range(N))
        
    return best_cycle
