import random
import time
from itertools import combinations

def _luw_construct(k, g):
    # Seed logic
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _count_short_cycles(adj, max_len=8):
    n = len(adj)
    count = 0
    # 4-cycles
    # For each u, for each pair of neighbors v, w, if v and w share a neighbor x != u, that's a 4-cycle.
    # To avoid double counting, count each 4-cycle once.
    # A 4-cycle u-v-w-x-u.
    # Iterate u < v < w < x? No, complex.
    # Standard way: Count triangles in the line graph? Or just:
    # Number of 4-cycles = 1/4 * sum_{u} (number of common neighbors of pairs of neighbors of u)
    # Actually, for each u, let N(u) be neighbors.
    # For each pair v, w in N(u), let c_{vw} be number of common neighbors of v and w excluding u.
    # Each 4-cycle is counted 4 times (once for each vertex).
    # So count_4 = sum_{u} sum_{v<w in N(u)} c_{vw} / 4
    
    # We just need the count to be 0.
    # If any c_{vw} > 0, we have a 4-cycle.
    
    # 6-cycles
    # This is harder to count incrementally.
    # For the purpose of SA, we can just compute the girth via BFS from each vertex up to depth 4 (for 4-cycles) and depth 5 (for 6-cycles, 8-cycles).
    # Actually, if we ensure no 4-cycles, we still need to check 6 and 8.
    # BFS from each vertex to depth 4 takes O(n * k^4) = 320 * 256 ~ 80k ops.
    # 320 BFS * 80k = 25 million ops. ~ 0.1s in Python? Maybe too slow for 100k iterations.
    
    # Better: 
    # Check 4-cycles fast.
    # If no 4-cycles, check 6-cycles.
    # If no 6-cycles, check 8-cycles.
    
    # Let's implement a fast 4-cycle check.
    # If 4-cycles exist, return 4.
    # Else check 6-cycles.
    # Else check 8-cycles.
    # Else return 10 (assuming no shorter).
    
    # 4-cycle check:
    # Build set of edges for O(1) lookup.
    edge_set = set()
    for i in range(n):
        for j in adj[i]:
            edge_set.add((i, j) if i < j else (j, i))
            
    # Count 4-cycles
    # For each vertex u
    four_count = 0
    for u in range(n):
        neigh = adj[u]
        for i in range(len(neigh)):
            v = neigh[i]
            for j in range(i+1, len(neigh)):
                w = neigh[j]
                # Check if v and w have a common neighbor x != u
                # Neighbors of v: adj[v]
                # Neighbors of w: adj[w]
                # Intersection minus u
                common = set(adj[v]) & set(adj[w])
                if u in common: common.remove(u)
                if common:
                    four_count += len(common)
    if four_count > 0:
        return 4
        
    # If no 4-cycles, check 6-cycles
    # A 6-cycle is u-v-w-x-y-z-u
    # We can find all paths of length 2 from u, and see if they connect to u via another path of length 2.
    # For each u, for each v in N(u), for each w in N(v)\{u},
    # We are at w. We need to reach u in 3 steps? No, 6-cycle is length 6.
    # Path u-v-w-x-y-z-u.
    # From u, go 3 steps to x. From u, go 3 steps to x via different path?
    # No, simpler:
    # Count paths of length 3 between pairs of vertices?
    # If there is a 6-cycle, there is a path of length 3 from u to v (where v is neighbor of u) that doesn't use u?
    # Actually, a 6-cycle implies for some u, there is a path of length 5 from u back to u.
    
    # Let's just do BFS from each vertex up to depth 5, but only if 4-cycle check passed.
    # Since 4-cycle check is fast, we only do BFS if 4-cycles are gone.
    # This is a good strategy.
    
    # Check 6-cycles and 8-cycles via BFS
    # We want to know if any cycle < 10 exists.
    # So we check for cycles of length 6 and 8.
    # BFS from u:
    # dist[u][u] = 0
    # If we reach u again at dist d < 10, we found a cycle of length d.
    # Note: Non-backtracking walks.
    
    # To speed up BFS, we stop if we find a cycle.
    # But we need to know if ANY cycle exists.
    
    # Optimization: If no 4-cycles, min cycle is >= 6.
    # Check for 6-cycles.
    # If found, return 6.
    # Check for 8-cycles.
    # If found, return 8.
    # Else return 10.
    
    # BFS implementation
    def has_cycle_len(target_len):
        # Check if there is a cycle of length target_len
        # Actually, we can just check for any cycle < 10.
        pass

    # Simpler: Just run BFS from each vertex, track parent to reconstruct cycle?
    # No, just track distance.
    # If we visit a vertex already visited in the current BFS tree (not the immediate parent), 
    # we found a cycle. Its length is dist[current] + 1 - dist[visited]? No.
    # Standard BFS for girth:
    # dist[v] = shortest distance from u to v.
    # parent[v] = previous node.
    # If we encounter an edge (v, w) such that dist[v] + 1 == dist[w] and v != parent[w], 
    # that's not a cycle.
    # A cycle is formed if we see an edge (v, w) where w is already visited and w != parent[v] and parent[w] != v.
    # The cycle length is dist[v] + dist[w] + 1.
    # We want this to be >= 10.
    
    # We iterate u from 0 to n-1.
    # BFS from u.
    # If we find a cycle of length < 10, return that length.
    # If BFS completes without finding such a cycle, continue.
    # If all BFS complete, return 10 (or more).
    
    for u in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[u] = 0
        queue = [u]
        found_cycle_len = None
        while queue:
            v = queue.pop(0)
            for w in adj[v]:
                if dist[w] == -1:
                    dist[w] = dist[v] + 1
                    parent[w] = v
                    queue.append(w)
                else:
                    # w is already visited
                    # Avoid counting the edge back to parent
                    if parent[v] == w or parent[w] == v:
                        continue
                    # Cycle found
                    # Length = dist[v] + dist[w] + 1
                    cyc_len = dist[v] + dist[w] + 1
                    if cyc_len < 10:
                        return cyc_len
        # If no cycle < 10 found from u, next u
    return 10

def _local_search(n, k, deadline):
    # Initialize random k-regular graph
    # Use configuration model
    stubs = []
    for i in range(n):
        stubs.extend([i] * k)
    random.shuffle(stubs)
    adj = [[] for _ in range(n)]
    i = 0
    while i < len(stubs):
        u = stubs[i]
        v = stubs[i+1]
        if u != v and v not in adj[u]:
            adj[u].append(v)
            adj[v].append(u)
            i += 2
        else:
            # Retry pairing
            # Simple method: shuffle again or swap with later element
            # For simplicity, just reshuffle if invalid pair found in first few steps?
            # Better: find a valid j > i
            found = False
            for j in range(i+2, len(stubs)):
                if stubs[j] != u and stubs[j] not in adj[u]:
                    # Swap stubs[i+1] and stubs[j]
                    stubs[i+1], stubs[j] = stubs[j], stubs[i+1]
                    v = stubs[i+1]
                    adj[u].append(v)
                    adj[v].append(u)
                    i += 2
                    found = True
                    break
            if not found:
                # Reshuffle all and retry?
                # This might loop.
                # For robustness, just reshuffle and restart initialization
                return _local_search(n, k, deadline) # Recursive retry, risky?
                # Better: break and use a different init
                pass

    # Check if valid
    if any(len(adj[i]) != k for i in range(n)):
        return None

    current_cost = _count_short_cycles(adj)
    best_cost = current_cost
    best_adj = [lst[:] for lst in adj]
    
    start_time = time.time()
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() - start_time < deadline:
        # Pick random edges to swap
        # Pick u, v in adj[u] and x, y in adj[x]
        u = random.randint(0, n-1)
        x = random.randint(0, n-1)
        if u == x: continue
        v_idx = random.randint(0, k-1)
        y_idx = random.randint(0, k-1)
        v = adj[u][v_idx]
        y = adj[x][y_idx]
        
        # Swap (u,v) and (x,y) with (u,y) and (x,v)
        # Check validity: no loops, no multi-edges
        if y == x or v == u: continue # Loops
        if y in adj[x] or v in adj[u]: continue # Multi-edges (already are, so check new ones)
        if y == x: continue
        if v == u: continue
        
        # New edges: (u,y) and (x,v)
        # Check if (u,y) is a loop: u==y
        if u == y: continue
        # Check if (x,v) is a loop: x==v
        if x == v: continue
        # Check if (u,y) already exists
        if y in adj[u]: continue
        # Check if (x,v) already exists
        if v in adj[x]: continue
        
        # Perform swap
        adj[u].remove(v)
        adj[v].remove(u)
        adj[x].remove(y)
        adj[y].remove(x)
        
        adj[u].append(y)
        adj[y].append(u)
        adj[x].append(v)
        adj[v].append(x)
        
        new_cost = _count_short_cycles(adj)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_adj = [lst[:] for lst in adj]
                if best_cost == 10: # Girth >= 10
                    # Verify
                    return best_adj
        else:
            # Accept with probability
            delta = new_cost - current_cost
            if delta > 0:
                prob = pow(2.71828, -delta / T)
                if random.random() < prob:
                    current_cost = new_cost
                else:
                    # Revert
                    adj[u].remove(y)
                    adj[y].remove(u)
                    adj[x].remove(v)
                    adj[v].remove(x)
                    
                    adj[u].append(v)
                    adj[v].append(u)
                    adj[x].append(y)
                    adj[y].append(x)
        
        T = max(T_min, T * cooling)
        
    return best_adj if best_cost < 10 else None

def construct(k, g):
    if k == 4 and g == 10:
        # Try to find 320-vertex cage
        deadline = 100
        adj = _local_search(320, 4, deadline)
        if adj is not None:
            # Convert to edge list
            edges = []
            for i in range(320):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return 320, edges
        
    # Fallback to LUW
    return _luw_construct(k, g)
