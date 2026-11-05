"""Hamilton cycle finder using binary search + optimized backtracking with connectivity pruning."""
import sys
sys.setrecursionlimit(10000)
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
edge_order = P['edge_order']

# Build edge index: edge -> position (1-based)
edge_pos = {}
for i, (u, v) in enumerate(edge_order):
    if u > v:
        u, v = v, u
    edge_pos[(u, v)] = i + 1

def get_edge_pos(u, v):
    if u > v:
        u, v = v, u
    return edge_pos[(u, v)]

# Precompute adjacency lists sorted by edge position
adj_by_pos = [[] for _ in range(N)]
for i, (u, v) in enumerate(edge_order):
    pos = i + 1
    adj_by_pos[u].append((pos, v))
    adj_by_pos[v].append((pos, u))
for i in range(N):
    adj_by_pos[i].sort(key=lambda x: x[0])

def find_ham_cycle(threshold):
    """Find a Hamiltonian cycle using only edges with position <= threshold."""
    # Build allowed adjacency
    allowed = [set() for _ in range(N)]
    for i in range(N):
        for pos, v in adj_by_pos[i]:
            if pos <= threshold:
                allowed[i].add(v)
            else:
                break
    
    # Check degree >= 2 for all vertices
    for i in range(N):
        if len(allowed[i]) < 2:
            return None
    
    # Backtracking: build path from vertex 0
    path = [0]
    used = [False] * N
    used[0] = True
    
    def backtrack():
        if len(path) == N:
            last = path[-1]
            return 0 in allowed[last]
        
        current = path[-1]
        candidates = [v for v in allowed[current] if not used[v]]
        
        if not candidates:
            return False
        
        # MRV: try candidates with fewest remaining options first
        candidates.sort(key=lambda v: len([u for u in allowed[v] if not used[u] and u != current]))
        
        for v in candidates:
            # Check if v can still connect to complete the cycle
            if len(path) == N - 1:
                # v is the last vertex, must connect to 0
                if 0 not in allowed[v]:
                    continue
            else:
                # v is internal, needs at least one other unused neighbor
                other_unused = sum(1 for u in allowed[v] if not used[u] and u != current)
                if other_unused < 1:
                    continue
            
            # Connectivity check: ensure all unused vertices can still reach the path
            feasible = True
            for w in range(N):
                if not used[w] and w != v:
                    # w needs to have a connection to either the path or other unused vertices
                    has_conn = False
                    for u in allowed[w]:
                        if used[u] and u != v:
                            has_conn = True
                            break
                        if not used[u] and u != v:
                            has_conn = True
                            break
                    if not has_conn:
                        feasible = False
                        break
            
            if not feasible:
                continue
            
            path.append(v)
            used[v] = True
            
            if backtrack():
                return True
            
            path.pop()
            used[v] = False
        
        return False
    
    if backtrack():
        return path
    return None

# Binary search for minimum threshold
lo, hi = 1, len(edge_order)
best_cycle = None

while lo <= hi:
    mid = (lo + hi) // 2
    cycle = find_ham_cycle(mid)
    if cycle is not None:
        best_cycle = cycle
        hi = mid - 1
    else:
        lo = mid + 1

def construct():
    if best_cycle is not None:
        return best_cycle
    # Fallback: identity permutation
    return list(range(N))
# EVOLVE-BLOCK-END
