"""Find minimum-threshold Hamilton cycle using binary search + backtracking/SA."""
import time
import random
import numpy as np

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
TOTAL_EDGES = len(P['edge_order'])

# Build edge position lookup: (min(u,v), max(u,v)) -> 1-based index
edge_pos = {}
for i, (u, v) in enumerate(P['edge_order']):
    key = (min(u, v), max(u, v))
    edge_pos[key] = i + 1

def get_edge_idx(u, v):
    return edge_pos[(min(u, v), max(u, v))]

def cycle_score(cycle):
    """Return the max 1-based edge index in the cycle."""
    n = len(cycle)
    max_idx = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        idx = get_edge_idx(u, v)
        if idx > max_idx:
            max_idx = idx
    return max_idx

def has_hamilton_cycle_threshold(K, deadline):
    """Check if graph with edges having index <= K contains a Hamilton cycle.
    Uses backtracking with pruning."""
    # Build adjacency: for each vertex, list of neighbors connected by edges with idx <= K
    adj = [set() for _ in range(N)]
    for (u, v), idx in edge_pos.items():
        if idx <= K:
            adj[u].add(v)
            adj[v].add(u)
    
    # Quick check: every vertex must have degree >= 2
    for i in range(N):
        if len(adj[i]) < 2:
            return None
    
    # Backtracking DFS
    start = 0
    visited = [False] * N
    visited[0] = True
    path = [0]
    
    def dfs(node, depth):
        if time.time() > deadline:
            return None
        if depth == N - 1:
            # Check if we can close the cycle
            if start in adj[node]:
                return list(path)
            return None
        
        # Get unvisited neighbors, sorted by degree (fewest options first)
        candidates = [n for n in adj[node] if not visited[n]]
        # Sort by degree for better pruning
        candidates.sort(key=lambda x: len(adj[x]))
        
        for nxt in candidates:
            # Pruning: check if nxt can still reach all remaining vertices
            # Simple check: if remaining unvisited count > 0, nxt must have at least 2 connections to unvisited+start
            remaining = N - depth - 1
            if remaining > 0:
                # nxt needs to connect to at least one more unvisited (unless it's the last)
                unvisited_neighbors = [n for n in adj[nxt] if not visited[n] and n != start]
                if depth == N - 2:
                    # Next is the last vertex before closing
                    if start not in adj[nxt]:
                        continue
                else:
                    if len(unvisited_neighbors) < 1:
                        continue
            
            visited[nxt] = True
            path.append(nxt)
            result = dfs(nxt, depth + 1)
            if result is not None:
                return result
            path.pop()
            visited[nxt] = False
            
        return None
    
    return dfs(start, 1)

def two_opt_improve(cycle, K, deadline, rng):
    """Try to improve a cycle using 2-opt moves, keeping all edges <= K."""
    n = len(cycle)
    best = cycle
    best_score = cycle_score(best)
    
    for _ in range(10000):
        if time.time() > deadline:
            break
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if abs(i - j) < 2:
            continue
        # Try 2-opt reversal
        new_cycle = cycle[:i+1] + cycle[i+1:j+1][::-1] + cycle[j+1:]
        # Check if all edges in new cycle have index <= K
        valid = True
        for idx in range(n):
            u = new_cycle[idx]
            v = new_cycle[(idx + 1) % n]
            if get_edge_idx(u, v) > K:
                valid = False
                break
        if valid:
            new_score = cycle_score(new_cycle)
            if new_score < best_score:
                best = new_cycle
                best_score = new_score
    
    return best, best_score

def simulated_annealing_find_cycle(K, deadline, rng, max_iters=50000):
    """Use simulated annealing to find a Hamilton cycle using only edges <= K."""
    # Build adjacency for threshold K
    adj = [set() for _ in range(N)]
    for (u, v), idx in edge_pos.items():
        if idx <= K:
            adj[u].add(v)
            adj[v].add(u)
    
    for i in range(N):
        if len(adj[i]) < 2:
            return None
    
    # Start with a random valid cycle if possible, otherwise greedy
    # Greedy construction
    cycle = [0]
    visited = {0}
    current = 0
    for _ in range(N - 1):
        candidates = [v for v in adj[current] if v not in visited and v != 0]
        if not candidates:
            return None
        # Pick neighbor with fewest options (Warnsdorff-like)
        candidates.sort(key=lambda x: len([y for y in adj[x] if y not in visited and y != 0]))
        current = candidates[0]
        cycle.append(current)
        visited.add(current)
    
    # Check if cycle closes
    if 0 not in adj[cycle[-1]]:
        return None
    
    # Now use SA to optimize
    best_cycle = list(cycle)
    best_score = cycle_score(best_cycle)
    T = 10.0
    T_min = 0.01
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        # Random 2-opt move
        i = rng.randint(0, N - 1)
        j = rng.randint(0, N - 1)
        if abs(i - j) < 2:
            continue
        new_cycle = cycle[:i+1] + cycle[i+1:j+1][::-1] + cycle[j+1:]
        
        # Validate
        valid = True
        for idx in range(N):
            u = new_cycle[idx]
            v = new_cycle[(idx + 1) % N]
            if get_edge_idx(u, v) > K:
                valid = False
                break
        
        if not valid:
            continue
        
        new_score = cycle_score(new_cycle)
        delta = new_score - best_score
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 0.001)):
            cycle = new_cycle
            if new_score < best_score:
                best_cycle = list(new_cycle)
                best_score = new_score
        
        T = max(T * 0.9995, T_min)
    
    return best_cycle

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Binary search for minimum K
    # Lower bound: need at least N edges, so K >= N (but really we need a cycle)
    # Upper bound: TOTAL_EDGES
    lo = N  # minimum possible
    hi = TOTAL_EDGES
    
    # First, find a feasible solution using the full edge set
    rng = random.Random(42)
    
    # Get an initial cycle using SA on full graph
    initial_cycle = simulated_annealing_find_cycle(TOTAL_EDGES, min(deadline, start_time + 5.0), rng, max_iters=20000)
    
    if initial_cycle is None:
        # Fallback: identity permutation
        return list(range(N))
    
    best_cycle = initial_cycle
    best_score = cycle_score(best_cycle)
    
    # Binary search for minimum K
    while lo < hi and time.time() < deadline - 5.0:
        mid = (lo + hi) // 2
        # Try to find a cycle with threshold mid
        found = simulated_annealing_find_cycle(mid, min(deadline, time.time() + 15.0), rng, max_iters=30000)
        
        if found is not None:
            score = cycle_score(found)
            if score <= mid:
                if score < best_score:
                    best_score = score
                    best_cycle = found
                hi = mid
            else:
                lo = mid + 1
        else:
            lo = mid + 1
    
    # Try to further improve with more SA iterations
    if time.time() < deadline - 3.0:
        # Try SA with the best known threshold
        improved = simulated_annealing_find_cycle(best_score, deadline - 1.0, rng, max_iters=100000)
        if improved is not None:
            s = cycle_score(improved)
            if s < best_score:
                best_score = s
                best_cycle = improved
    
    # Final 2-opt polish
    if time.time() < deadline - 1.0:
        polished, _ = two_opt_improve(best_cycle, best_score, deadline - 0.5, rng)
        s = cycle_score(polished)
        if s < best_score:
            best_score = s
            best_cycle = polished
    
    return best_cycle

# Pre-compute edge positions at module load for speed
edge_pos = {}
for i, (u, v) in enumerate(P['edge_order']):
    key = (min(u, v), max(u, v))
    edge_pos[key] = i + 1
