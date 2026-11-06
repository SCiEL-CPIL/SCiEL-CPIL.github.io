import time
import sys
from collections import deque
import heapq

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = P['n']
edge_order = P['edge_order']
m = len(edge_order)

# Precompute arrival index for each edge
arrival_matrix = [[-1] * n for _ in range(n)]
for idx, (u, v) in enumerate(edge_order):
    arrival_matrix[u][v] = idx
    arrival_matrix[v][u] = idx

def get_edge_arrival(u, v):
    return arrival_matrix[u][v]

def cycle_score(cycle):
    s = 0
    L = len(cycle)
    for i in range(L):
        a = cycle[i]
        b = cycle[(i + 1) % L]
        ar = get_edge_arrival(a, b) + 1
        if ar > s:
            s = ar
    return s

def greedy_initial_bound():
    """
    Finds a reasonably good Hamiltonian cycle using a greedy 2-factor approach 
    and simple repair, to provide an upper bound for Dijkstra.
    """
    # Greedy 2-regular subgraph
    degree = [0] * n
    adj = [set() for _ in range(n)]
    parent = list(range(n))
    size = [1] * n
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry: return False
        if size[rx] < size[ry]: rx, ry = ry, rx
        parent[ry] = rx
        size[rx] += size[ry]
        return True

    for idx in range(m):
        u, v = edge_order[idx]
        if degree[u] >= 2 or degree[v] >= 2: continue
        ru, rv = find(u), find(v)
        if ru == rv:
            if size[ru] < n: continue
        degree[u] += 1
        degree[v] += 1
        adj[u].add(v)
        adj[v].add(u)
        union(u, v)
        if all(d == 2 for d in degree):
            # Check if single cycle
            root = find(0)
            ok = all(find(i) == root for i in range(n))
            if ok:
                # Extract cycle
                cycle = [0]
                prev = -1
                cur = 0
                for _ in range(n - 1):
                    nxts = list(adj[cur])
                    nxt = nxts[0] if nxts[0] != prev else nxts[1]
                    prev, cur = cur, nxt
                    cycle.append(cur)
                return cycle_score(cycle)
            else:
                # Multiple cycles, try to merge? For now, just return a high bound
                # We can compute the max edge in this 2-factor
                max_ar = 0
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            ar = get_edge_arrival(u, v) + 1
                            if ar > max_ar:
                                max_ar = ar
                return max_ar
    return m + 1

def dijkstra_bottleneck(upper_bound):
    """
    Finds the minimum bottleneck Hamiltonian cycle using Dial's Algorithm.
    Returns the cycle and its score.
    """
    start = 0
    full_mask = (1 << n) - 1
    
    # dist[mask][last] = min bottleneck cost
    # Using a dictionary of dictionaries to save memory
    dist = [{} for _ in range(1 << n)]
    
    # Parent for path reconstruction
    parent = [{} for _ in range(1 << n)]
    
    # Dial's Algorithm Buckets
    # Max cost is upper_bound
    buckets = [[] for _ in range(upper_bound + 1)]
    
    # Initial state
    init_mask = 1 << start
    dist[init_mask][start] = 0
    buckets[0].append((start, init_mask))
    
    best_score = upper_bound
    best_last = -1
    best_mask = -1
    
    for c in range(upper_bound + 1):
        if not buckets[c]:
            continue
        
        # Process all states in this bucket
        # We need to be careful: adding to buckets[c'] where c' >= c
        # If we add to buckets[c], we might process it in the same iteration?
        # Dial's algorithm usually processes nodes in non-decreasing order of distance.
        # If we add a node with distance c to bucket c, and we are currently at c,
        # it will be processed later in the same loop iteration if we iterate the list correctly,
        # or in the next if we use a pointer. 
        # Standard Dial's uses a circular array of size C+1.
        # Here, since c is increasing, any new node added will have cost >= c.
        # If cost == c, it goes to bucket[c]. We should process it.
        
        # To handle additions to the current bucket, we can use a while loop on the bucket list
        # but we must ensure we don't miss items added during the loop.
        # Simple approach: pop from the end, or use a separate list for new items?
        # Actually, if we pop from the front, and add to the back, it's a queue.
        # But we are iterating `c` from 0 to UB.
        # If we add to bucket[c] while processing bucket[c], we need to process those too.
        
        # Let's use a while loop for the current bucket
        while buckets[c]:
            last, mask = buckets[c].pop(0) # Pop from front? O(N). 
            # Better to use a deque or pop from end if order doesn't matter within same cost
            # Order within same cost doesn't matter for correctness.
            # Let's pop from end for O(1).
            
            if dist[mask].get(last, float('inf')) != c:
                continue # Outdated entry
                
            # Check if we can close the cycle
            if mask == full_mask:
                if get_edge_arrival(last, start) != -1:
                    final_cost = max(c, get_edge_arrival(last, start) + 1)
                    if final_cost < best_score:
                        best_score = final_cost
                        best_last = last
                        best_mask = mask
                        # We can update upper_bound to prune further?
                        # If we found a solution, we can stop if c > best_score?
                        # No, we are at c. best_score >= c.
                        # If best_score == c, we can break?
                        # If we found a solution with cost C, any state with cost > C is useless.
                        # So we can break the outer loop if c > best_score.
            
            # Expand
            for next_v in range(n):
                if mask & (1 << next_v):
                    continue
                ar = get_edge_arrival(last, next_v)
                if ar == -1:
                    continue
                new_cost = max(c, ar + 1)
                if new_cost > best_score:
                    continue
                new_mask = mask | (1 << next_v)
                if new_cost < dist[new_mask].get(next_v, float('inf')):
                    dist[new_mask][next_v] = new_cost
                    parent[new_mask][next_v] = last
                    if new_cost <= upper_bound:
                        buckets[new_cost].append((next_v, new_mask))
                    # If new_cost > upper_bound, we ignore it because we already have a better solution
                    # (or equal, but we want min, so strictly better is needed to update)
                    # Actually, if new_cost == best_score, we don't need to explore further 
                    # unless we want to find another cycle? No, we just need one.
                    # But wait, if new_cost == best_score, we might find a path that leads to 
                    # a final cost < best_score? No, final cost >= new_cost.
                    # So if new_cost >= best_score, we can prune.
            
            # Optimization: If c > best_score, break outer loop
            if c > best_score:
                return best_score, None # Signal to stop
        
        if c > best_score:
            break
            
    if best_mask != -1:
        # Reconstruct path
        cycle = []
        cur_mask = best_mask
        cur_last = best_last
        while cur_last != -1:
            cycle.append(cur_last)
            prev_last = parent[cur_mask].get(cur_last, -1)
            if prev_last == -1:
                break
            # Need to remove cur_last from mask to get prev mask
            cur_mask = cur_mask ^ (1 << cur_last)
            cur_last = prev_last
        cycle.reverse()
        # The start is 0, which should be at the end of the reversed list (or beginning?)
        # Path: 0 -> ... -> best_last
        # Reconstruction: 
        # cur_last = best_last, cur_mask = full
        # append best_last
        # prev = parent[full][best_last]
        # cur_mask = full ^ (1<<best_last)
        # ...
        # Last step: cur_last = 0, cur_mask = 1<<0
        # append 0
        # prev = -1
        # So cycle is [best_last, ..., 0]
        # Reversed: [0, ..., best_last]
        # This is a valid path from 0 to best_last.
        # The cycle is closed by edge (best_last, 0).
        return best_score, cycle
    else:
        return upper_bound, None

def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    
    # Get initial upper bound
    ub = greedy_initial_bound()
    if ub > m:
        ub = m + 1
        
    # Run Dijkstra
    score, cycle = dijkstra_bottleneck(ub)
    
    if cycle is None:
        # Fallback to greedy if Dijkstra failed or timed out (though it shouldn't timeout easily with UB)
        # Or if no cycle found (should not happen in complete graph)
        # Try a simple backtracking with the UB as limit
        # For now, return a default or greedy cycle
        # Let's try to extract the greedy cycle again
        # (Re-implementing greedy extraction is redundant, just return a valid cycle)
        # Identity permutation is a valid cycle (0-1-2...-23-0)
        # Score might be high, but it's valid.
        return list(range(n))
        
    return cycle

# EVOLVE-BLOCK-END
