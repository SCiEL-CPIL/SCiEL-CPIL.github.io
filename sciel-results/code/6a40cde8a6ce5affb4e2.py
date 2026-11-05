"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']

# Build edge index lookup: (u, v) -> 1-based index
edge_idx = {}
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    if u > v:
        u, v = v, u
    edge_idx[(u, v)] = i

def edge_weight(u, v):
    """Return the 1-based arrival index of edge (u,v)."""
    if u > v:
        u, v = v, u
    return edge_idx[(u, v)]

def cycle_score(cycle):
    """Score = max edge index over all edges in the cycle (including closing edge)."""
    n = len(cycle)
    max_idx = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        w = edge_weight(u, v)
        if w > max_idx:
            max_idx = w
    return max_idx

def two_opt(cycle, max_iters=10000):
    """Improve cycle via 2-opt moves, minimizing max edge index."""
    n = len(cycle)
    best = cycle[:]
    best_score = cycle_score(best)
    
    for _ in range(max_iters):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n - 1):
                # Try reversing segment i+1..j
                new_cycle = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                # Check if this creates a valid cycle (no self-loops)
                # The 2-opt move on a cycle: edges (i, i+1) and (j, j+1) are removed
                # and (i, j) and (i+1, j+1) are added
                # In the new cycle, we need to check the changed edges
                # Actually, let's just compute the score
                new_score = cycle_score(new_cycle)
                if new_score < best_score:
                    best = new_cycle
                    best_score = new_score
                    improved = True
        if not improved:
            break
    return best, best_score

def optimize_cycle(cycle):
    """Run 2-opt optimization."""
    cycle, score = two_opt(cycle)
    return cycle, score

def find_hamilton_cycle_greedy(start=0, max_restarts=500):
    """
    Greedy construction of Hamilton cycle: at each step, pick the unused vertex
    with the minimum edge index to the current vertex.
    """
    n = N
    best_cycle = None
    best_score = float('inf')
    
    for restart in range(max_restarts):
        # Randomly shuffle the order of candidates for tie-breaking
        visited = [False] * n
        cycle = [start]
        visited[start] = True
        current = start
        
        for step in range(1, n):
            best_next = -1
            best_w = float('inf')
            # Collect candidates
            candidates = []
            for v in range(n):
                if not visited[v]:
                    w = edge_weight(current, v)
                    candidates.append((w, v))
            
            # Sort by weight, then by vertex (for determinism with randomness)
            candidates.sort()
            
            # With some probability, pick the second-best to add diversity
            if restart > 0 and len(candidates) > 1 and random.random() < 0.1:
                w, v = candidates[1]
            else:
                w, v = candidates[0]
            
            cycle.append(v)
            visited[v] = True
            current = v
        
        # Check if closing edge is valid (always is in complete graph)
        # Compute score
        score = cycle_score(cycle)
        if score < best_score:
            best_score = score
            best_cycle = cycle[:]
        
        # Early stop if we found a very good cycle
        if best_score <= 30:
            break
    
    return best_cycle, best_score

def hill_climbing(cycle, max_iters=5000):
    """
    Local search: try swapping two non-adjacent vertices in the cycle,
    or rotating segments, to reduce max edge index.
    """
    n = len(cycle)
    best = cycle[:]
    best_score = cycle_score(best)
    
    for it in range(max_iters):
        improved = False
        
        # Try 2-opt moves
        for i in range(n - 1):
            for j in range(i + 1, n - 1):
                new_cycle = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                new_score = cycle_score(new_cycle)
                if new_score < best_score:
                    best = new_cycle
                    best_score = new_score
                    improved = True
        
        # Try 3-opt moves (limited)
        if not improved:
            for i in range(n - 2):
                for j in range(i + 1, n - 1):
                    for k in range(j + 1, n - 1):
                        # Reverse segment i+1..k
                        new_cycle = best[:i+1] + best[i+1:k+1][::-1] + best[k+1:]
                        new_score = cycle_score(new_cycle)
                        if new_score < best_score:
                            best = new_cycle
                            best_score = new_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Phase 1: Greedy construction with multiple restarts
    best_cycle, best_score = find_hamilton_cycle_greedy(max_restarts=200)
    
    # Phase 2: Hill climbing / 2-opt on the best cycle
    best_cycle, best_score = hill_climbing(best_cycle, max_iters=3000)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        return best_cycle
    
    # Phase 3: More aggressive search if time remains
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Try different starting points and more restarts
        for start in range(N):
            if time.time() > deadline - 2:
                break
            cycle, score = find_hamilton_cycle_greedy(start=start, max_restarts=50)
            if score < best_score:
                best_cycle, best_score = cycle, score
                best_cycle, best_score = hill_climbing(best_cycle, max_iters=2000)
    
    # Phase 4: Final polish
    if time.time() < deadline - 2:
        best_cycle, best_score = hill_climbing(best_cycle, max_iters=5000)
    
    return best_cycle
# EVOLVE-BLOCK-END
