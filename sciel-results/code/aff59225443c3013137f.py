import time
import sys
from collections import defaultdict

N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Precompute edge positions
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    a, b = min(u, v), max(u, v)
    edge_pos[(a, b)] = i + 1

def find_hamilton_cycle_bitmask(adj_masks, n, deadline):
    """
    Finds a Hamilton cycle using bitmask backtracking.
    adj_masks[v] is a bitmask of neighbors of v.
    """
    # Quick checks
    for v in range(n):
        if bin(adj_masks[v]).count('1') < 2:
            return None
    
    # Connectivity check using BFS on bitmasks
    visited_mask = 1
    stack = [0]
    while stack:
        u = stack.pop()
        # Neighbors of u that are not visited
        neighbors = adj_masks[u] & ~visited_mask
        while neighbors:
            lsb = neighbors & -neighbors
            v = lsb.bit_length() - 1
            visited_mask |= lsb
            stack.append(v)
            neighbors ^= lsb
    if visited_mask != (1 << n) - 1:
        return None

    start = 0
    start_mask = 1 << start
    
    def backtrack(path, used_mask, current):
        if time.time() > deadline:
            return None
        
        if len(path) == n:
            # Check closing edge
            if (adj_masks[current] & start_mask) != 0:
                return path[:]
            return None
        
        last = current
        # Candidates are unvisited neighbors of last
        candidates_mask = adj_masks[last] & ~used_mask
        
        # If no candidates, fail
        if candidates_mask == 0:
            return None
            
        # Heuristic: Warnsdorff
        # Sort candidates by number of unvisited neighbors
        # To do this fast, we can iterate and count bits of (adj_masks[nxt] & ~used_mask)
        # Note: used_mask includes 'last' but not 'nxt' yet.
        # When we pick 'nxt', the new used_mask will be used_mask | (1<<nxt).
        # The "future" unvisited set is ~used_mask & ~(1<<nxt).
        # We want to pick 'nxt' with fewest options in the future.
        
        candidates = []
        cm = candidates_mask
        while cm:
            lsb = cm & -cm
            nxt = lsb.bit_length() - 1
            # Count unvisited neighbors of nxt
            # Future unvisited = ~used_mask & ~lsb
            # Neighbors of nxt in future = adj_masks[nxt] & (~used_mask & ~lsb)
            # Note: ~used_mask includes 'last' and 'start' if not used?
            # 'last' is in used_mask. 'start' is in used_mask (since path starts with start).
            # So ~used_mask is exactly the unvisited set.
            future_unvisited = ~used_mask & ~lsb
            # We also need to ensure 'nxt' can connect to 'start' if it's the last step?
            # Warnsdorff usually just counts degree.
            deg = bin(adj_masks[nxt] & future_unvisited).count('1')
            candidates.append((deg, nxt, lsb))
            cm ^= lsb
            
        candidates.sort(key=lambda x: x[0])
        
        for deg, nxt, lsb in candidates:
            new_used = used_mask | lsb
            path.append(nxt)
            
            # Connectivity check on remaining
            # Remaining vertices: ~new_used
            # Endpoints: nxt and start
            # We need the subgraph induced by (~new_used) U {nxt, start} to be connected.
            # Let's check if there is a path from nxt to start in the graph induced by (~new_used) U {nxt, start}.
            # Actually, we just need the component of 'nxt' in this induced graph to contain 'start'.
            
            # BFS/DFS from 'nxt' in the induced graph
            # Induced graph nodes: mask = ~new_used | (1<<nxt) | (1<<start)
            # Note: 'start' is already in used_mask (it's the first element of path).
            # So ~new_used does NOT contain start.
            # We must explicitly include start in the search space.
            
            search_mask = (~new_used) | (1 << nxt) | (1 << start)
            
            visited_bfs = 1 << nxt
            stack_bfs = [nxt]
            found_start = False
            if nxt == start:
                found_start = True # Should not happen unless n=1
            
            while stack_bfs and not found_start:
                u = stack_bfs.pop()
                # Neighbors of u in the induced graph
                # Neighbors in original graph: adj_masks[u]
                # Intersect with search_mask
                # Exclude already visited in BFS
                neighbors_bfs = adj_masks[u] & search_mask & ~visited_bfs
                while neighbors_bfs:
                    lsb_bfs = neighbors_bfs & -neighbors_bfs
                    v_bfs = lsb_bfs.bit_length() - 1
                    visited_bfs |= lsb_bfs
                    if v_bfs == start:
                        found_start = True
                        break
                    stack_bfs.append(v_bfs)
                    neighbors_bfs ^= lsb_bfs
            
            if not found_start:
                path.pop()
                continue

            result = backtrack(path, new_used, nxt)
            if result is not None:
                return result
            
            path.pop()
            
        return None

    return backtrack([start], start_mask, start)

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    n = N
    total_edges = n * (n - 1) // 2
    
    # Binary search for the minimum threshold T
    lo, hi = 24, total_edges
    best_cycle = None
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        
        # Build adj masks
        adj_masks = [0] * n
        for (u, v), pos in edge_pos.items():
            if pos <= mid:
                adj_masks[u] |= 1 << v
                adj_masks[v] |= 1 << u
        
        # Check for Hamilton cycle
        # Give a fraction of remaining time to this check
        # If we spend too long, we might not finish.
        # Simple heuristic: give 10s per check? No, binary search has ~8 steps.
        # Total 80s. 10s per step is safe.
        check_deadline = time.time() + 10
        
        cycle = find_hamilton_cycle_bitmask(adj_masks, n, check_deadline)
        
        if cycle is not None:
            best_cycle = cycle
            hi = mid - 1
        else:
            # If timed out, we don't know. 
            # Assuming "No" is safer for finding *a* valid cycle with lower bound?
            # If we assume "Yes", we lower hi. If it was "No", we skip valid low T.
            # If we assume "No", we raise lo. If it was "Yes", we miss the minimum.
            # However, if we miss the minimum, we might still have a valid cycle from a higher T?
            # No, we only update best_cycle on success.
            # If we timeout on a "Yes", we treat as "No" -> lo increases -> we might end up with a higher T than necessary.
            # If we timeout on a "No", we treat as "No" -> lo increases -> correct.
            # If we timeout on a "Yes" and treat as "Yes" -> hi decreases -> we might find a lower T that is actually "No" (timeout again) or "Yes".
            # Given the risk of returning an invalid cycle (if we assume Yes and it's No, we might return a cycle from a previous higher T? No, best_cycle is only updated on success).
            # So if we assume Yes, we just search lower. If lower ones are No, we will eventually get a "No" (fast) or timeout.
            # If we get a timeout on a "No", we treat as "No" (if we change logic) or "Yes"?
            # Let's stick to "No" on timeout to be conservative about the existence of a cycle at that T.
            # This might result in a suboptimal T (higher than true minimum), but a valid cycle.
            lo = mid + 1
    
    if best_cycle is not None:
        return best_cycle
    
    # Fallback
    if time.time() < deadline:
        adj_masks = [0] * n
        for (u, v), pos in edge_pos.items():
            adj_masks[u] |= 1 << v
            adj_masks[v] |= 1 << u
        cycle = find_hamilton_cycle_bitmask(adj_masks, n, start_time + 5)
        if cycle:
            return cycle
            
    return list(range(n))

if __name__ == "__main__":
    c = construct()
    max_pos = 0
    for i in range(N):
        u = c[i]
        v = c[(i+1)%N]
        a, b = min(u, v), max(u, v)
        p = edge_pos[(a, b)]
        if p > max_pos:
            max_pos = p
    print(f"Cycle: {c}")
    print(f"Score: {max_pos}")
