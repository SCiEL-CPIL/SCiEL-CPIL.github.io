import time
import random
import heapq
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = P['n']
edge_order = P['edge_order']
m = len(edge_order)

# Arrival index (0-based) for each unordered edge
arrival = {}
for idx, (u, v) in enumerate(edge_order):
    arrival[(min(u, v), max(u, v))] = idx

def edge_arrival(u, v):
    if u > v:
        u, v = v, u
    return arrival[(u, v)]

def cycle_score(cycle):
    s = 0
    L = len(cycle)
    for i in range(L):
        a = cycle[i]
        b = cycle[(i + 1) % L]
        ar = edge_arrival(a, b) + 1
        if ar > s:
            s = ar
    return s

def dijkstra_bottleneck_hamiltonian(deadline):
    """
    Dijkstra best-first search over state space (mask, last) with bottleneck cost.
    State: (mask of visited vertices, last vertex)
    Cost: maximum edge arrival index (0-based) used so far
    Goal: mask == full_mask and last connects back to start (vertex 0)
    """
    full_mask = (1 << n) - 1
    start = 0
    
    # Precompute adjacency lists with arrival indices
    adj = [[] for _ in range(n)]
    for v in range(n):
        for w in range(n):
            if v != w:
                a = edge_arrival(v, w)
                adj[v].append((a, w))
        # Sort by arrival index for efficiency
        adj[v].sort()
    
    # Dijkstra: priority queue of (bottleneck_cost, mask, last)
    # dist[mask][last] = minimum bottleneck cost to reach state (mask, last)
    INF = 2**31
    dist = {}  # (mask, last) -> bottleneck cost
    prev = {}  # (mask, last) -> (prev_mask, prev_last) for path reconstruction
    
    # Start state: only vertex 0 visited, at vertex 0, cost = 0
    start_state = (1 << start, start)
    dist[start_state] = 0
    pq = [(0, 1 << start, start)]  # (cost, mask, last)
    
    while pq and time.time() < deadline:
        cost, mask, last = heapq.heappop(pq)
        
        # Check if this state was already processed with better cost
        if dist.get((mask, last), INF) < cost:
            continue
        
        # Check goal: all vertices visited and can return to start
        if mask == full_mask:
            close_cost = edge_arrival(last, start)
            total_bottleneck = max(cost, close_cost)
            # Reconstruct path
            path = []
            state = (mask, last)
            while state in prev:
                pm, pl = prev[state]
                path.append(state[1])
                state = (pm, pl)
            path.append(start)
            path.reverse()
            return path, total_bottleneck + 1  # 1-based score
        
        # Expand: try all neighbors not yet visited
        for (arr, nb) in adj[last]:
            if mask & (1 << nb):
                continue
            new_mask = mask | (1 << nb)
            new_cost = max(cost, arr)
            
            state = (new_mask, nb)
            if dist.get(state, INF) > new_cost:
                dist[state] = new_cost
                prev[state] = (mask, last)
                heapq.heappush(pq, (new_cost, new_mask, nb))
    
    return None, None

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # safety margin under 90s
    
    default_cycle = list(range(n))
    best_cycle = default_cycle
    best_score = cycle_score(default_cycle)
    
    # Try Dijkstra bottleneck search - this should find the exact minimum bottleneck
    path, score = dijkstra_bottleneck_hamiltonian(deadline)
    if path is not None and score < best_score:
        best_score = score
        best_cycle = path
    
    return best_cycle
# EVOLVE-BLOCK-END
