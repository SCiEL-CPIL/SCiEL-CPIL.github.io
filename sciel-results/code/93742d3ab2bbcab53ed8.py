"""Search for a Hamilton cycle minimizing the maximum edge arrival index."""
import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']

# Build cost matrix
cost = [[0] * N for _ in range(N)]
for idx, (a, b) in enumerate(P['edge_order']):
    cost[a][b] = idx + 1
    cost[b][a] = idx + 1

# Sorted next vertices by cost
sorted_next = []
for u in range(N):
    lst = [(cost[u][v], v) for v in range(N) if v != u]
    lst.sort()
    sorted_next.append(lst)

# Vertices sorted by cost to 0
verts_sorted_by_cost_to_0 = sorted(range(N), key=lambda v: cost[v][0])

def cycle_score(cyc):
    m = 0
    for i in range(N):
        c = cost[cyc[i]][cyc[(i + 1) % N]]
        if c > m:
            m = c
    return m

# Initial upper bound: identity
best_score = cycle_score(list(range(N)))
best_cycle = list(range(N))

# Nearest-neighbor heuristic from each start
for start in range(N):
    path = [start]
    visited = 1 << start
    cur = start
    for _ in range(N - 1):
        best_v = -1
        for c, v in sorted_next[cur]:
            if not (visited >> v) & 1:
                best_v = v
                break
        if best_v == -1:
            for v in range(N):
                if not (visited >> v) & 1:
                    best_v = v
                    break
        path.append(best_v)
        visited |= (1 << best_v)
        cur = best_v
    s = cycle_score(path)
    if s < best_score:
        best_score = s
        best_cycle = path

sys.setrecursionlimit(10000)

def min_cost_to_unvisited(u, visited_mask):
    for c, v in sorted_next[u]:
        if not (visited_mask >> v) & 1:
            return c
    return 10**9

def min_cost_to_0_from_unvisited_fast(visited_mask):
    for v in verts_sorted_by_cost_to_0:
        if not (visited_mask >> v) & 1:
            return cost[v][0]
    return 10**9

start_time = time.time()
DEADLINE = 80.0
nodes_visited = 0

def _dfs(cur, path, visited, cur_max):
    global nodes_visited
    if time.time() - start_time > DEADLINE:
        return False
    
    nodes_visited += 1
    if nodes_visited % 100000 == 0 and time.time() - start_time > DEADLINE:
        return False
    
    if len(path) == N:
        closing_cost = cost[cur][0]
        total_max = cur_max if cur_max > closing_cost else closing_cost
        if total_max < best_score:
            best_score = total_max
            best_cycle = list(path)
        return True
    
    min_from_cur = min_cost_to_unvisited(cur, visited)
    min_to_0 = min_cost_to_0_from_unvisited_fast(visited)
    lb = cur_max
    if min_from_cur > lb:
        lb = min_from_cur
    if min_to_0 > lb:
        lb = min_to_0
    if lb >= best_score:
        return True
    
    for c, v in sorted_next[cur]:
        if (visited >> v) & 1:
            continue
        new_max = cur_max if cur_max > c else c
        if new_max >= best_score:
            if c >= best_score:
                break
            continue
        
        path.append(v)
        visited |= (1 << v)
        
        if not _dfs(v, path, visited, new_max):
            path.pop()
            return False
        
        path.pop()
        visited ^= (1 << v)
    
    return True

path = [0]
visited = 1 << 0
_dfs(0, path, visited, 0)

final_cycle = best_cycle

def construct():
    return final_cycle
# EVOLVE-BLOCK-END
