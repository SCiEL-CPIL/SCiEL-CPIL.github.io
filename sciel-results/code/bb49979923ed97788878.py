import time
import sys
import random

N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Precompute edge positions
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    edge_pos[(u, v)] = i
    edge_pos[(v, u)] = i

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # adj_masks[v] is a bitmask of neighbors reachable in the current graph
    adj_masks = [0] * N
    
    # We will iterate K from 1 to len(EDGE_ORDER)
    # But we can start later. Min degree 2 requires K >= 24 roughly.
    # Let's just iterate from 1, but skip checks until K >= 20.
    
    best_cycle = None
    best_K = None
    
    for K in range(1, len(EDGE_ORDER) + 1):
        if time.time() > deadline - 2:
            break
            
        u, v = EDGE_ORDER[K-1]
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
        # Quick checks
        if K < 24:
            continue
            
        # Check min degree
        feasible = True
        for i in range(N):
            if bin(adj_masks[i]).count('1') < 2:
                feasible = False
                break
                
        if not feasible:
            continue
            
        # Check connectivity
        visited = [False] * N
        queue = [0]
        visited[0] = True
        count = 1
        while queue:
            curr = queue.pop(0)
            neighbors = adj_masks[curr]
            while neighbors:
                lsb = neighbors & (-neighbors)
                nb = lsb.bit_length() - 1
                neighbors ^= lsb
                if not visited[nb]:
                    visited[nb] = True
                    count += 1
                    queue.append(nb)
        if count != N:
            continue
            
        # Try to find cycle
        # Time budget for this K: 1 second
        time_budget = 1.0
        if K > 50:
            time_budget = 0.5 # Less time for higher K, assume easier
        
        cycle = find_cycle(adj_masks, time_budget)
        if cycle is not None:
            best_cycle = cycle
            best_K = K
            break # First found is the minimum K
            
    if best_cycle is not None:
        return best_cycle
        
    # Fallback: Use a high K and a simple heuristic
    # Reset adj_masks to full graph
    for i in range(N):
        adj_masks[i] = 0
    for i, (u, v) in enumerate(EDGE_ORDER, 1):
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    # Greedy construction
    path = [0]
    visited = 1 << 0
    curr = 0
    while len(path) < N:
        next_node = -1
        min_deg = N + 1
        candidates = adj_masks[curr] & ~visited
        while candidates:
            lsb = candidates & (-candidates)
            nb = lsb.bit_length() - 1
            candidates ^= lsb
            deg = bin(adj_masks[nb] & ~visited).count('1')
            if deg < min_deg:
                min_deg = deg
                next_node = nb
        if next_node == -1:
            break
        path.append(next_node)
        visited |= (1 << next_node)
        curr = next_node
        
    if len(path) == N and (adj_masks[curr] & (1 << 0)):
        return path
    else:
        # Last resort: return a valid cycle from a known dense subgraph
        # Just return 0..23 if it forms a cycle? No, edges might not exist.
        # But in K24, all edges exist. The only constraint is the *order*.
        # The score is the max index. If we return 0..23, the max index is max(edge_pos[(i, i+1)]).
        # This is a valid cycle.
        return list(range(N))

def find_cycle(adj_masks, time_budget):
    start_time = time.time()
    deadline = start_time + time_budget
    N = 24
    all_mask = (1 << N) - 1
    
    def dfs(v, visited, path):
        if time.time() > deadline:
            return None
            
        if len(path) == N:
            if adj_masks[v] & (1 << 0):
                return path[:]
            return None
            
        available = adj_masks[v] & ~visited
        if available == 0:
            return None
            
        neighbors = []
        temp = available
        while temp:
            lsb = temp & (-temp)
            nb = lsb.bit_length() - 1
            temp ^= lsb
            
            # Pruning: Check if nb can be part of a valid cycle
            # 1. If nb is not 0, and we are not at the last step, nb must have enough onward connections
            new_visited = visited | (1 << nb)
            unvisited = all_mask & ~new_visited
            
            if len(path) == N - 1:
                # Last step
                if adj_masks[nb] & (1 << 0):
                    neighbors.append((0, nb)) # Priority: valid
            else:
                # Check degree constraint for unvisited nodes
                valid = True
                # The node 'nb' is now current. It needs to connect to unvisited nodes eventually.
                # Actually, the constraint is on the unvisited nodes.
                # For any u in unvisited, max degree available is deg_in_unvisited(u) + (1 if u~nb) + (1 if u~0)
                # If this is < 2, invalid.
                
                # Optimization: Only check if unvisited is not empty
                if unvisited:
                    temp_u = unvisited
                    while temp_u:
                        lsb_u = temp_u & (-temp_u)
                        u = lsb_u.bit_length() - 1
                        temp_u ^= lsb_u
                        
                        deg_unvisited = bin(adj_masks[u] & unvisited).count('1')
                        conn_nb = 1 if (adj_masks[u] & (1 << nb)) else 0
                        conn_0 = 1 if (adj_masks[u] & (1 << 0)) else 0
                        
                        if deg_unvisited + conn_nb + conn_0 < 2:
                            valid = False
                            break
                            
                if valid:
                    # Heuristic: Warnsdorff
                    onward = adj_masks[nb] & ~new_visited
                    deg_onward = bin(onward).count('1')
                    neighbors.append((deg_onward, nb))
                    
        if not neighbors:
            return None
            
        neighbors.sort(key=lambda x: x[0])
        
        for deg, nb in neighbors:
            result = dfs(nb, visited | (1 << nb), path + [nb])
            if result is not None:
                return result
                
        return None
        
    return dfs(0, 1 << 0, [0])
