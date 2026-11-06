import time
import sys

N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    edge_pos[frozenset((u, v))] = i

# Precompute edge list sorted by index (it is already)
# We will use bitmasks for adjacency

def find_hamilton_cycle_optimized(K, deadline):
    """
    Finds a Hamilton cycle using edges with index <= K.
    Uses bitmask DFS with pruning.
    """
    # Build adjacency bitmasks
    adj_mask = [0] * N
    for i, (u, v) in enumerate(EDGE_ORDER, 1):
        if i > K:
            break
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Check basic feasibility: all nodes must have degree >= 2
    for v in range(N):
        if bin(adj_mask[v]).count('1') < 2:
            return None
            
    # Check connectivity
    # BFS/DFS from 0
    visited = [False] * N
    stack = [0]
    visited[0] = True
    count = 0
    while stack:
        v = stack.pop()
        count += 1
        neighbors = adj_mask[v]
        while neighbors:
            nb_bit = neighbors & -neighbors
            nb = nb_bit.bit_length() - 1
            neighbors ^= nb_bit
            if not visited[nb]:
                visited[nb] = True
                stack.append(nb)
    if count < N:
        return None

    full_mask = (1 << N) - 1
    sys.setrecursionlimit(10000)
    
    # Try starting from vertex 0 (or any vertex)
    # Symmetry breaking: fix start to 0? No, the graph is not necessarily symmetric in edge indices.
    # But we can try multiple starts.
    
    def dfs(v, visited_mask, path):
        if time.time() > deadline:
            return None
        
        if len(path) == N:
            # Check if v connects to start (path[0])
            if (adj_mask[v] & (1 << path[0])):
                return path[:]
            return None
        
        # Get unvisited neighbors of v
        unvisited = full_mask ^ visited_mask
        candidates = adj_mask[v] & unvisited
        
        if not candidates:
            return None
            
        # Warnsdorff heuristic: sort candidates by number of unvisited neighbors
        # We want to visit nodes with fewer options first to avoid getting stuck
        cand_list = []
        c = candidates
        while c:
            nb_bit = c & -c
            nb = nb_bit.bit_length() - 1
            c ^= nb_bit
            # Count unvisited neighbors of nb
            # Note: nb is not visited yet. 
            # Its unvisited neighbors are adj_mask[nb] & unvisited
            # But we are about to visit nb, so for the NEXT step, nb will be visited.
            # The pruning check will be done after moving to nb.
            # For sorting, we just use the current count.
            cnt = bin(adj_mask[nb] & unvisited).count('1')
            cand_list.append((cnt, nb))
            
        cand_list.sort()
        
        for cnt, nb in cand_list:
            # Pruning: Check if remaining unvisited nodes are "safe"
            # After visiting nb, the new unvisited set is unvisited ^ (1 << nb)
            new_unvisited = unvisited ^ (1 << nb)
            
            # If new_unvisited is empty, we handle it in the base case (len(path)==N)
            if new_unvisited:
                # Check if any node in new_unvisited is isolated from new_unvisited
                # It must have at least one neighbor in new_unvisited, 
                # UNLESS it is the last node to be visited (which requires connection to start? No, 
                # the last node connects to start. But intermediate nodes must connect to other unvisited nodes).
                # Actually, every node in new_unvisited must have a path to the rest.
                # Simple prune: if any node has 0 neighbors in new_unvisited, 
                # it can only be connected to nb (visited) or start (visited).
                # If it's not the last node, it's a dead end.
                # If it IS the last node, it must connect to start.
                
                # Let's check degrees in new_unvisited
                dead_end = False
                nu = new_unvisited
                while nu:
                    v_bit = nu & -nu
                    v_idx = v_bit.bit_length() - 1
                    nu ^= v_bit
                    
                    # Neighbors of v_idx in new_unvisited
                    deg = bin(adj_mask[v_idx] & new_unvisited).count('1')
                    
                    if deg == 0:
                        # v_idx has no unvisited neighbors.
                        # It can only be the last node in the path.
                        # If it is the last node, it must connect to start.
                        # But we don't know if it will be the last node yet.
                        # However, if there is ANY node with deg 0, it MUST be the last one.
                        # If there are MULTIPLE nodes with deg 0, it's impossible.
                        # If there is ONE node with deg 0, it must be the last one.
                        # This is a weak prune. Let's just check if deg < 1 and it's not the only one?
                        # Actually, if deg == 0, v_idx is isolated from the rest of the unvisited nodes.
                        # The path must enter v_idx from nb (or some visited node? No, only from current path).
                        # And exit to... nowhere, unless it's the end.
                        # So v_idx MUST be the last node.
                        # If there is another node w in new_unvisited with deg == 0, then both must be last, impossible.
                        # So: count how many nodes have deg 0.
                        pass # We'll do a more efficient check later or just rely on Warnsdorff
                    
                # More efficient prune:
                # If any node in new_unvisited has deg 0 in new_unvisited, 
                # and there is more than 1 such node, return None.
                # If there is exactly 1 such node, it must be the last one.
                # This check is O(N) per step. N=24, so 24*24=576 ops. Acceptable.
                
                zero_deg_count = 0
                zero_deg_node = -1
                nu = new_unvisited
                while nu:
                    v_bit = nu & -nu
                    v_idx = v_bit.bit_length() - 1
                    nu ^= v_bit
                    if bin(adj_mask[v_idx] & new_unvisited).count('1') == 0:
                        zero_deg_count += 1
                        zero_deg_node = v_idx
                        if zero_deg_count > 1:
                            return None # Two isolated nodes, impossible
                            
                # If zero_deg_count == 1, the path MUST end at zero_deg_node.
                # This doesn't prune the current choice, but restricts future choices.
                # We can't easily enforce this in the recursion without passing state.
                # So we just use the >1 prune.

            result = dfs(nb, visited_mask | (1 << nb), path + [nb])
            if result:
                return result
        return None

    # Try start 0
    result = dfs(0, 1 << 0, [0])
    if result:
        return result
        
    # If start 0 failed, try others? 
    # Given the time limit, maybe just return None if 0 fails?
    # Or try a few others.
    for start in range(1, N):
        if time.time() > deadline:
            return None
        result = dfs(start, 1 << start, [start])
        if result:
            return result
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    lo, hi = 24, len(EDGE_ORDER)
    best_cycle = None
    best_K = None
    
    while lo <= hi:
        if time.time() > deadline - 5:
            break
        mid = (lo + hi) // 2
        cycle = find_hamilton_cycle_optimized(mid, deadline)
        if cycle is not None:
            best_cycle = cycle
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_cycle is None:
        # Fallback
        return list(range(N))
        
    # Try to improve: search lower K
    # The binary search might have skipped some K if the first success was high.
    # But binary search finds the lowest K that works IF the property is monotonic.
    # "If a Hamilton cycle exists with max edge index K, it also exists with K+1".
    # Yes, this is true. So binary search finds the minimum K.
    
    return best_cycle
