"""Find a Hamilton cycle minimizing the maximum edge arrival index."""
import time
import random
import math

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']

# Build position lookup: edge (a,b) -> 1-based index
EDGE_POS = {}
for i, (a, b) in enumerate(EDGE_ORDER):
    if a > b:
        a, b = b, a
    EDGE_POS[(a, b)] = i + 1

def get_edge_pos(u, v):
    if u > v:
        u, v = v, u
    return EDGE_POS[(u, v)]

def cycle_edges(cycle):
    """Return list of edges in the cycle (including closing edge)."""
    n = len(cycle)
    edges = []
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        edges.append((u, v))
    return edges

def max_edge_index(cycle):
    """Return the maximum 1-based arrival index of edges in the cycle."""
    mx = 0
    n = len(cycle)
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        if u > v:
            u, v = v, u
        pos = EDGE_POS[(u, v)]
        if pos > mx:
            mx = pos
    return mx

def greedy_cycle():
    """Build a cycle using a greedy approach: always pick the neighbor with smallest edge index."""
    n = N
    # Start from vertex 0
    visited = [False] * n
    cycle = [0]
    visited[0] = True
    
    for _ in range(n - 1):
        current = cycle[-1]
        best_next = -1
        best_pos = float('inf')
        for v in range(n):
            if not visited[v]:
                u, v2 = current, v
                if u > v2:
                    u, v2 = v2, u
                pos = EDGE_POS[(u, v2)]
                if pos < best_pos:
                    best_pos = pos
                    best_next = v
        cycle.append(best_next)
        visited[best_next] = True
    
    # Check if closing edge is valid (it always is in complete graph)
    return cycle

def two_opt(cycle, max_iter=100000):
    """Apply 2-opt moves to minimize max edge index."""
    n = len(cycle)
    best_cycle = list(cycle)
    best_score = max_edge_index(best_cycle)
    
    for it in range(max_iter):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Try reversing segment [i+1, j]
                # New cycle: [0..i] + reversed([i+1..j]) + [j+1..n-1]
                new_cycle = list(cycle)
                new_cycle[i+1:j+1] = reversed(cycle[i+1:j+1])
                new_score = max_edge_index(new_cycle)
                if new_score < best_score:
                    best_score = new_score
                    best_cycle = new_cycle
                    cycle = new_cycle
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_cycle, best_score

def simulated_annealing(cycle, max_iter=200000, t_start=10.0, t_end=0.01):
    """Simulated annealing to minimize max edge index."""
    n = len(cycle)
    current = list(cycle)
    current_score = max_edge_index(current)
    best = list(current)
    best_score = current_score
    
    rng = random.Random(42)
    
    for it in range(max_iter):
        # Temperature schedule
        t = t_start * (t_end / t_start) ** (it / max_iter)
        
        # Random 2-opt move
        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, n - 1)
        new_cycle = list(current)
        new_cycle[i+1:j+1] = reversed(current[i+1:j+1])
        new_score = max_edge_index(new_cycle)
        
        delta = new_score - current_score
        if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-10)):
            current = new_cycle
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best = list(current)
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # Initial greedy solution
    cycle = greedy_cycle()
    best_cycle, best_score = two_opt(cycle, max_iter=50000)
    
    # Simulated annealing with multiple restarts
    rng = random.Random(123)
    
    while time.time() < deadline:
        # Start from best solution found so far
        sa_cycle, sa_score = simulated_annealing(
            best_cycle, 
            max_iter=50000,
            t_start=5.0,
            t_end=0.005
        )
        if sa_score < best_score:
            best_score = sa_score
            best_cycle = sa_cycle
            # Follow up with 2-opt
            best_cycle, best_score = two_opt(best_cycle, max_iter=20000)
        
        # Try a random restart
        if time.time() < deadline - 5:
            random_cycle = list(range(N))
            rng.shuffle(random_cycle)
            rc, rs = two_opt(random_cycle, max_iter=10000)
            if rs < best_score:
                best_score = rs
                best_cycle = rc
                sa_cycle, sa_score = simulated_annealing(
                    best_cycle,
                    max_iter=30000,
                    t_start=5.0,
                    t_end=0.005
                )
                if sa_score < best_score:
                    best_score = sa_score
                    best_cycle = sa_cycle
                    best_cycle, best_score = two_opt(best_cycle, max_iter=20000)
    
    return best_cycle
# EVOLVE-BLOCK-END
