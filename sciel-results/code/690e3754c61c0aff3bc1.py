import time
import sys
import random
from collections import defaultdict
import numpy as np

N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Precompute edge positions
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    edge_pos[(u, v)] = i
    edge_pos[(v, u)] = i

# Precompute masks for all K values (0 to len(EDGE_ORDER))
MAX_K = len(EDGE_ORDER)
all_masks = [[0] * N for _ in range(MAX_K + 1)]
for K in range(1, MAX_K + 1):
    for v in range(N):
        all_masks[K][v] = all_masks[K-1][v]
    u, v = EDGE_ORDER[K-1]
    all_masks[K][u] |= (1 << v)
    all_masks[K][v] |= (1 << u)

# Precompute degree counts for each K and vertex
all_degrees = [[0] * N for _ in range(MAX_K + 1)]
for K in range(MAX_K + 1):
    for v in range(N):
        all_degrees[K][v] = bin(all_masks[K][v]).count('1')

# Precompute connectivity for each K
all_connected = [False] * (MAX_K + 1)
for K in range(MAX_K + 1):
    masks = all_masks[K]
    visited = [False] * N
    stack = [0]
    visited[0] = True
    count = 1
    while stack:
        u = stack.pop()
        neighbors = masks[u]
        while neighbors:
            lsb = neighbors & (-neighbors)
            nb = lsb.bit_length() - 1
            neighbors ^= lsb
            if not visited[nb]:
                visited[nb] = True
                count += 1
                stack.append(nb)
    all_connected[K] = (count == N)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    best_cycle = None
    best_K = None
    
    def check_feasibility(K, max_restarts=10):
        # 1. Degree check (precomputed)
        for v in range(N):
            if all_degrees[K][v] < 2:
                return None
        
        # 2. Connectivity check (precomputed)
        if not all_connected[K]:
            return None
        
        masks = all_masks[K]
        
        # 3. DFS with Warnsdorff and Restart
        sys.setrecursionlimit(10000)
        
        def dfs(v, visited_mask, path, rng):
            if len(path) == N:
                if masks[v] & (1 << 0):
                    return path[:]
                return None
            
            available = masks[v] & ~visited_mask
            if available == 0:
                return None
                
            neighbors = []
            temp = available
            while temp:
                lsb = temp & (-temp)
                nb = lsb.bit_length() - 1
                temp ^= lsb
                onward = masks[nb] & ~visited_mask & ~(1 << v)
                if len(path) == N - 1:
                    onward = masks[nb] & (1 << 0)
                neighbors.append((bin(onward).count('1'), nb))
            
            # Sort by onward degree
            neighbors.sort(key=lambda x: x[0])
            
            # Group by degree to randomize within groups
            deg_map = defaultdict(list)
            for deg, nb in neighbors:
                deg_map[deg].append(nb)
                
            # Process degrees in increasing order
            for deg in sorted(deg_map.keys()):
                nbs = deg_map[deg]
                rng.shuffle(nbs)
                for nb in nbs:
                    result = dfs(nb, visited_mask | (1 << nb), path + [nb], rng)
                    if result is not None:
                        return result
            return None
            
        # Try multiple restarts with different seeds to escape local minima
        for attempt in range(max_restarts):
            if time.time() > deadline - 5:
                break
            rng = random.Random(attempt * 1000 + 42)
            cycle = dfs(0, 1 << 0, [0], rng)
            if cycle is not None:
                return cycle
                
        return None

    # First, find the minimum feasible K using binary search
    lo, hi = 24, len(EDGE_ORDER)
    
    while lo <= hi and time.time() < deadline - 15:
        mid = (lo + hi) // 2
        
        # Quick check: if degree or connectivity fails, skip DFS
        feasible = True
        for v in range(N):
            if all_degrees[mid][v] < 2:
                feasible = False
                break
        if feasible and not all_connected[mid]:
            feasible = False
        
        if not feasible:
            lo = mid + 1
            continue
        
        cycle = check_feasibility(mid, max_restarts=10)
        
        if cycle is not None:
            best_cycle = cycle
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_cycle is not None:
        # Polish: try to find a cycle for smaller K values near best_K
        # This handles cases where the heuristic failed for a feasible K during binary search
        for K in range(best_K, max(23, best_K - 10), -1):
            if time.time() > deadline - 5:
                break
            # Use more restarts for the polish phase
            cycle = check_feasibility(K, max_restarts=50)
            if cycle is not None:
                best_cycle = cycle
                best_K = K
            else:
                # If we failed to find a cycle for K, it's likely infeasible
                # But we can't be sure, so we stop polishing if we fail for a few consecutive K
                # However, since we found one at best_K+1, and we are going down,
                # if we fail at K, we can break if we are confident.
                # Given the monotonicity, if K is infeasible, K-1 is also.
                # So if we fail at K, we can break.
                break
                
    if best_cycle is not None:
        return best_cycle
        
    # Fallback: try higher K values if binary search failed to find anything
    for K in range(24, len(EDGE_ORDER) + 1, 5):
        if time.time() > deadline - 5:
            break
        cycle = check_feasibility(K, max_restarts=5)
        if cycle is not None:
            best_cycle = cycle
            break
            
    if best_cycle is not None:
        return best_cycle
        
    return list(range(N))
