import time
import sys
from collections import defaultdict

N = 24
ALL_MASK = (1 << N) - 1
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Precompute edge positions
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    a, b = min(u, v), max(u, v)
    edge_pos[(a, b)] = i + 1

# Precompute popcount table for 24-bit integers (use byte-level lookup)
_POPCOUNT_8 = [bin(i).count('1') for i in range(256)]
def popcount(x):
    c = 0
    while x:
        c += _POPCOUNT_8[x & 0xFF]
        x >>= 8
    return c

def find_hamilton_cycle_bitmask(adj_masks, n, deadline):
    """
    Finds a Hamilton cycle using bitmask backtracking with optimized heuristics.
    adj_masks[v] is a bitmask of neighbors of v.
    """
    # Quick degree check
    for v in range(n):
        if popcount(adj_masks[v]) < 2:
            return None
    
    # Connectivity check
    visited_mask = 1
    stack = [0]
    while stack:
        u = stack.pop()
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
    full_mask = (1 << n) - 1
    
    # Precompute: for each vertex, its adjacency mask
    # We'll use a class-like structure for state to avoid passing many args
    
    def backtrack(path, used_mask, current):
        if time.time() > deadline:
            return None
        
        if len(path) == n:
            if (adj_masks[current] & start_mask) != 0:
                return path[:]
            return None
        
        # Candidates are unvisited neighbors of current
        candidates_mask = adj_masks[current] & ~used_mask
        
        if candidates_mask == 0:
            return None
            
        # Check for forced moves: if any unvisited vertex (other than current's candidates)
        # has only 1 available neighbor among unvisited + {current}, it's forced
        # More precisely: for each unvisited vertex v != current, count neighbors in (unvisited \ {v}) + {current}
        # If count < 2, problem. If count == 1 and that neighbor is not current, then v must connect to that neighbor.
        
        unvisited = full_mask & ~used_mask
        # unvisited includes current? No, current is in used_mask.
        # unvisited = all vertices not yet placed in path
        
        # For each unvisited vertex, check its available degree
        # available neighbors of v = adj_masks[v] & (unvisited | (1 << current))
        # But we need at least 2 such neighbors for v to be in a cycle (one in, one out)
        # Actually in the path, v needs exactly 2 path-neighbors from (unvisited \ {v}) ∪ {current, start}
        # But start is special - it's already placed.
        
        # Let's check: for each unvisited v, deg_avail = popcount(adj_masks[v] & (unvisited | start_mask))
        # If deg_avail < 2: dead end (unless v is adjacent to current, which is handled by the path)
        # Actually: v's path-neighbors must come from (unvisited \ {v}) ∪ {current} ∪ {start}
        # But start is already in the path. So v can connect to start only at the very end.
        # For now: if v has < 2 neighbors in (unvisited \ {v}) ∪ {current}, it's problematic.
        
        # Simpler check: for each unvisited v, count neighbors in (unvisited | (1<<current) | start_mask)
        # If < 2, prune. But this is expensive. Let's do a lighter version.
        
        # Check: is there any unvisited vertex with only 1 available neighbor (excluding itself)?
        # available for v = adj_masks[v] & (unvisited | (1 << current) | start_mask) & ~(1 << v)
        # If popcount < 2, dead end.
        
        cm_check = unvisited
        while cm_check:
            lsb_v = cm_check & -cm_check
            v = lsb_v.bit_length() - 1
            avail = adj_masks[v] & (unvisited | (1 << current) | start_mask) & ~lsb_v
            if popcount(avail) < 2:
                # This vertex can't be in a valid cycle
                return None
            cm_check ^= lsb_v
        
        # Warnsdorff: sort by number of future unvisited neighbors
        candidates = []
        cm = candidates_mask
        while cm:
            lsb = cm & -cm
            nxt = lsb.bit_length() - 1
            # Future unvisited after picking nxt
            future = unvisited & ~lsb
            deg = popcount(adj_masks[nxt] & future)
            # Bonus: prefer vertices that can still reach start
            can_reach_start = 1 if (adj_masks[nxt] & start_mask) else 0
            candidates.append((deg, -can_reach_start, nxt, lsb))
            cm ^= lsb
            
        candidates.sort(key=lambda x: (x[0], x[1]))
        
        for deg, _, nxt, lsb in candidates:
            new_used = used_mask | lsb
            path.append(nxt)
            
            # Connectivity check: can we reach start from nxt through unvisited + {nxt, start}?
            search_mask = (full_mask & ~new_used) | lsb | start_mask
            
            # Quick check: is start reachable from nxt in this induced subgraph?
            if nxt != start:
                visited_bfs = lsb
                stack_bfs = [nxt]
                found_start = False
                while stack_bfs and not found_start:
                    u = stack_bfs.pop()
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
    deadline = start_time + 85
    
    n = N
    total_edges = n * (n - 1) // 2
    
    # Binary search for the minimum threshold T
    lo, hi = 24, total_edges
    best_cycle = None
    best_score = float('inf')
    
    # Track results for potential refinement
    results = {}
    
    while lo <= hi and time.time() < deadline - 5:
        mid = (lo + hi) // 2
        
        # Build adj masks
        adj_masks = [0] * n
        for (u, v), pos in edge_pos.items():
            if pos <= mid:
                adj_masks[u] |= 1 << v
                adj_masks[v] |= 1 << u
        
        # Check for Hamilton cycle
        # Dynamic time allocation based on remaining time
        remaining = deadline - 5 - time.time()
        check_deadline = time.time() + min(10, max(2, remaining * 0.35))
        
        cycle = find_hamilton_cycle_bitmask(adj_masks, n, check_deadline)
        results[mid] = cycle is not None
        
        if cycle is not None:
            best_cycle = cycle
            best_score = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Polish: try to find a cycle with lower max edge position
    if best_cycle is not None and time.time() < deadline - 3:
        # Try slightly lower thresholds around best_score
        for offset in range(1, 8):
            test_t = best_score - offset
            if test_t < 24:
                break
            if test_t in results and results[test_t]:
                continue
            adj_masks = [0] * n
            for (u, v), pos in edge_pos.items():
                if pos <= test_t:
                    adj_masks[u] |= 1 << v
                    adj_masks[v] |= 1 << u
            check_deadline = time.time() + min(4, deadline - 3 - time.time())
            if check_deadline <= time.time():
                break
            cycle = find_hamilton_cycle_bitmask(adj_masks, n, check_deadline)
            if cycle is not None:
                best_cycle = cycle
                best_score = test_t
                results[test_t] = True
            else:
                results[test_t] = False
    
    if best_cycle is not None:
        return best_cycle
    
    # Fallback
    if time.time() < deadline - 1:
        adj_masks = [0] * n
        for (u, v), pos in edge_pos.items():
            adj_masks[u] |= 1 << v
            adj_masks[v] |= 1 << u
        cycle = find_hamilton_cycle_bitmask(adj_masks, n, deadline - 1)
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
