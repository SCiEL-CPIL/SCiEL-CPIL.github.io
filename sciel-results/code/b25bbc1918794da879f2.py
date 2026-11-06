import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
# Fixed edge order from the task description
EDGE_ORDER = [
    [3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23],
    [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19],
    [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19],
    [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8],
    [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17],
    [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23],
    [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4],
    [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12],
    [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20],
    [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16],
    [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18],
    [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21],
    [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14],
    [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6],
    [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8],
    [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18],
    [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12],
    [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23],
    [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20],
    [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5],
    [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11],
    [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7],
    [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9],
    [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4],
    [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14],
    [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23],
    [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16],
    [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]
]

N = 24
TIME_LIMIT = 85.0  # Safety margin before 90s limit

# Precompute edge index lookup: (u,v) -> 1-based arrival index
EDGE_INDEX = {}
for _i, (_u, _v) in enumerate(EDGE_ORDER, 1):
    EDGE_INDEX[(_u, _v)] = _i
    EDGE_INDEX[(_v, _u)] = _i

def cycle_score(cycle):
    """Compute the max edge arrival index for a Hamilton cycle."""
    max_idx = 0
    for i in range(N):
        u = cycle[i]
        v = cycle[(i + 1) % N]
        idx = EDGE_INDEX[(u, v)]
        if idx > max_idx:
            max_idx = idx
    return max_idx

def local_search_polish(cycle, time_budget=5.0):
    """
    Try to reduce the max edge index by 2-opt style reversals.
    Returns improved cycle or original.
    """
    start = time.time()
    best_cycle = list(cycle)
    best_score = cycle_score(best_cycle)
    
    for _ in range(50):
        if time.time() - start > time_budget:
            break
        improved = False
        for i in range(N):
            if time.time() - start > time_budget:
                break
            for j in range(i+1, N):
                if time.time() - start > time_budget:
                    break
                new_cycle = best_cycle[:i] + best_cycle[i:j+1][::-1] + best_cycle[j+1:]
                new_score = cycle_score(new_cycle)
                if new_score < best_score:
                    best_cycle = new_cycle
                    best_score = new_score
                    improved = True
        if not improved:
            break
    
    return best_cycle

def construct():
    start_time = time.time()
    
    def find_cycle(K, start_node=0):
        """
        Finds a Hamilton cycle using only edges from the first K in EDGE_ORDER.
        Returns the list of vertices in the cycle (length N) or None if not found.
        Uses iterative DFS with bitmasks and Warnsdorff ordering.
        """
        # Build adjacency bitmasks for the first K edges
        adj = [0] * N
        for i in range(K):
            u, v = EDGE_ORDER[i]
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Check min degree
        for i in range(N):
            if adj[i].bit_count() < 2:
                return None

        # Iterative DFS
        stack = [(start_node, 1, 1 << start_node, 0)]
        path = [start_node]
        
        while stack:
            current, depth, v_mask, next_idx = stack.pop()
            
            if depth == N:
                if adj[current] & (1 << start_node):
                    return path
                continue
            
            m = adj[current] & ~v_mask
            if not m:
                continue
            
            # Build candidate list with Warnsdorff ordering
            candidates = []
            mm = m
            while mm:
                bit = mm & (-mm)
                v = bit.bit_length() - 1
                unvisited_count = (adj[v] & ~v_mask).bit_count()
                candidates.append((unvisited_count, v))
                mm ^= bit
            
            candidates.sort()
            
            # Push candidates in reverse order so the best is processed first
            for idx in range(len(candidates) - 1, -1, -1):
                _, v = candidates[idx]
                stack.append((v, depth + 1, v_mask | (1 << v), 0))
                path.append(v)
            
            # We need to handle backtracking properly
            # Let me use a different approach - explicit stack with state
            break
        
        # Better iterative DFS with proper backtracking
        path = [start_node]
        stack = [(start_node, 1, 1 << start_node, -1)]  # (node, depth, v_mask, next_candidate_idx)
        
        while stack:
            current, depth, v_mask, next_idx = stack[-1]
            
            if depth == N:
                if adj[current] & (1 << start_node):
                    return path
                stack.pop()
                if path:
                    path.pop()
                continue
            
            m = adj[current] & ~v_mask
            if not m:
                stack.pop()
                if path:
                    path.pop()
                continue
            
            # Build candidate list
            candidates = []
            mm = m
            while mm:
                bit = mm & (-mm)
                v = bit.bit_length() - 1
                unvisited_count = (adj[v] & ~v_mask).bit_count()
                candidates.append((unvisited_count, v))
                mm ^= bit
            
            candidates.sort()
            
            if next_idx < 0:
                # First visit, set next_idx to 0
                stack[-1] = (current, depth, v_mask, 0)
                continue
            
            if next_idx >= len(candidates):
                # No more candidates, backtrack
                stack.pop()
                if path:
                    path.pop()
                continue
            
            _, v = candidates[next_idx]
            stack[-1] = (current, depth, v_mask, next_idx + 1)
            path.append(v)
            stack.append((v, depth + 1, v_mask | (1 << v), -1))
        
        return None

    # Binary search for minimal K
    lo = N
    hi = N * (N - 1) // 2
    best_cycle = None
    best_K = hi
    
    while lo <= hi:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        mid = (lo + hi) // 2
        cycle = find_cycle(mid)
        
        if cycle:
            best_cycle = cycle
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1

    if best_cycle:
        # Polish with 2-opt
        remaining = TIME_LIMIT - 10 - (time.time() - start_time)
        if remaining > 0.5:
            best_cycle = local_search_polish(best_cycle, time_budget=min(remaining, 10.0))
        return best_cycle
    else:
        return list(range(N))

# EVOLVE-BLOCK-END
