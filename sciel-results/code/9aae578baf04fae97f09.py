import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

_n = P['n']
_edge_order = P['edge_order']

# Precompute edge position map (1-based)
_edge_pos = {}
for i, (u, v) in enumerate(_edge_order, 1):
    _edge_pos[(u, v)] = i
    _edge_pos[(v, u)] = i

# For each vertex, list of (neighbor, edge_position) sorted by position
_adj = [None] * _n
for v in range(_n):
    lst = []
    for u in range(_n):
        if u != v:
            pos = _edge_pos[(v, u)]
            lst.append((pos, u))
    lst.sort()
    _adj[v] = lst

def _find_cycle(deadline):
    """Find Hamilton cycle minimizing max edge position. Returns (best_cycle, best_score)."""
    best_cycle = None
    best_score = float('inf')
    
    # Try each starting vertex
    for start in range(_n):
        if time.time() > deadline:
            break
        # DFS: build path from start
        path = [start]
        used = [False] * _n
        used[start] = True
        current_max = 0
        
        def dfs():
            nonlocal best_cycle, best_score, current_max
            if time.time() > deadline:
                return
            if len(path) == _n:
                # Check closing edge
                last = path[-1]
                first = path[0]
                close_pos = _edge_pos[(last, first)]
                total_max = max(current_max, close_pos)
                if total_max < best_score:
                    best_score = total_max
                    best_cycle = path[:]
                return
            
            last = path[-1]
            # Try neighbors in order of edge position (prefer early edges)
            for pos, nxt in _adj[last]:
                if used[nxt]:
                    continue
                new_max = max(current_max, pos)
                if new_max >= best_score:
                    # Prune: can't improve
                    continue
                # Additional pruning: if we're near the end, check if closing edge can be small
                if len(path) == _n - 1:
                    # Only one vertex left unused
                    for u in range(_n):
                        if not used[u]:
                            # Closing edge would be from nxt to start
                            close_pos = _edge_pos[(nxt, start)]
                            if max(new_max, close_pos) >= best_score:
                                break
                            else:
                                path.append(nxt)
                                used[nxt] = True
                                dfs()
                                used[nxt] = False
                                path.pop()
                            break
                else:
                    path.append(nxt)
                    used[nxt] = True
                    current_max = new_max
                    dfs()
                    used[nxt] = False
                    path.pop()
                    current_max = new_max  # restore
        
        dfs()
    
    return best_cycle, best_score

def construct():
    deadline = time.time() + 50  # 50s safety margin within 60s limit
    
    # First, get a baseline: the identity cycle 0,1,2,...,23,0
    # Compute its score
    base_score = float('inf')
    for i in range(_n):
        u = i
        v = (i + 1) % _n
        base_score = max(base_score, _edge_pos[(u, v)])
    
    best_cycle = list(range(_n))
    best_score = base_score
    
    # Search for better cycles
    result_cycle, result_score = _find_cycle(deadline)
    if result_cycle is not None and result_score < best_score:
        best_cycle = result_cycle
        best_score = result_score
    
    return best_cycle
# EVOLVE-BLOCK-END
