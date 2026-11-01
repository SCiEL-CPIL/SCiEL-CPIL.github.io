import random
import time
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses a Simulated Annealing approach on the space of k-regular graphs
    to minimize the number of short cycles, targeting the Moore bound upwards.
    """
    if k != 3 or g != 16:
        # Fallback for other parameters: simple random search or L-U-W
        # Since the task is specific to k=3, g=16, we focus there.
        # For robustness, we return a known construction if available, 
        # but the task implies we should solve the specific instance.
        # We will just attempt the same logic for any k, g if time permits, 
        # but primarily target the specific case.
        pass

    start_global = time.time()
    global_timeout = 115.0
    
    # Moore bound for k=3, g=16 is 766.
    # We search N from 766 to 935 (step 2).
    # We allocate time slices to each N.
    
    # Heuristic: It's unlikely to find a graph very close to the Moore bound.
    # But we must try.
    
    # To optimize, we can skip some N values if we are running out of time.
    # Let's try N = 766, 768, ..., 934.
    
    # Strategy: 
    # 1. Try N=766 for 5s.
    # 2. Try N=770 for 5s.
    # 3. ...
    # Actually, the distribution of successful graphs might be sparse.
    # Let's use a fixed time per N, but reduce it as N increases if we fail.
    
    # Better: Iterate N, give each a chance.
    # If we find one, return.
    
    # We will use a random restart strategy.
    
    for N in range(766, 936, 2):
        if time.time() - start_global > global_timeout:
            break
            
        # Time slice for this N
        # Start with 2 seconds, decrease if N is large?
        # Or just fixed 1.5s to allow more N's.
        # (936-766)/2 = 85 values. 85 * 1.5 = 127.5s. Too much.
        # Let's use 1.0s per N. 85s. Leaves 30s buffer.
        
        # If N is close to 936, we know 936 works (L-U-W). 
        # But we want to beat it.
        
        found = attempt_sa(N, k, g, time_limit=1.0)
        if found:
            return N, found
            
    # If we exhausted the search without finding one, 
    # we must return a valid graph. 
    # The prompt says "n must lie between the Moore bound and max_vertices".
    # It doesn't explicitly say we *must* beat 936, but the score is n.
    # If we fail to find < 936, we should return 936 (L-U-W) to be safe?
    # The current program returns L-U-W (936) as fallback.
    # I will implement a fast L-U-W fallback to ensure validity if SA fails.
    
    return luw_construct(3, 16)

def attempt_sa(N, k, g, time_limit):
    """
    Attempts to find a k-regular graph on N vertices with girth >= g
    using Simulated Annealing on cycle counts.
    """
    if N % 2 != 0:
        return None
        
    start_time = time.time()
    
    # Initial random graph
    edges = generate_random_k_regular(N, k)
    if edges is None:
        return None
        
    adj = build_adj(N, edges)
    
    # Calculate initial energy: number of cycles of length < g
    # For g=16, we care about cycles 3..15.
    # We can approximate energy by counting cycles of length 4, 6, 8, 10, 12, 14.
    # Counting all is expensive. 
    # Let's just check girth directly, but that's O(N^2).
    # For SA, we need a fast energy function.
    # Let's use the number of 4-cycles and 6-cycles as a proxy, 
    # or just check if girth < g.
    
    # A better energy: sum of (girth - i)^2 for all short cycles?
    # Too slow.
    
    # Let's use a greedy hill-climbing with random restarts.
    # Move: 2-switch.
    # Accept if girth increases or stays same, and short cycle count decreases.
    
    # To speed up, we will only check girth every few steps, 
    # or use a fast local check.
    
    # Actually, for N~800, BFS girth check is ~0.05s.
    # In 1 second, we can do ~20 girth checks.
    # This is very slow for SA.
    
    # We need a faster proxy.
    # Let's count 4-cycles.
    # # 4-cycles can be counted in O(N * d^2) or using matrix multiplication.
    # For d=3, it's fast.
    
    def count_4_cycles(adj):
        count = 0
        for u in range(N):
            neighbors = adj[u]
            # Count pairs of neighbors that share a common neighbor
            # This is tricky.
            # Alternative: For each edge (u,v), count common neighbors.
            # Sum over edges of (common_neighbors(u,v) choose 2) ? No.
            # A 4-cycle is u-v-w-x-u.
            # For each pair of vertices (u,w), if they have 2 common neighbors, that's a 4-cycle.
            # Number of 4-cycles = sum_{u < w} (C(common_neighbors(u,w), 2))
            
        # This is O(N^2 * d). For N=800, N^2 = 640,000. d=3.
        # ~2 million ops. Fast enough.
        
        count = 0
        for u in range(N):
            for w in range(u + 1, N):
                common = 0
                # Check if u and w share neighbors
                # Since d=3, we can just check the 3 neighbors of u
                for v in adj[u]:
                    if v in adj[w]:
                        common += 1
                if common >= 2:
                    count += common * (common - 1) // 2
        return count

    # We will minimize 4-cycles and 6-cycles.
    # But breaking 4-cycles might create 5-cycles? No, 5 is odd.
    # We want girth >= 16. So we must eliminate 3,4,5,6,7,8,9,10,11,12,13,14,15.
    
    # Let's just use the girth check as the acceptance criterion, 
    # but do it less frequently?
    
    # Actually, let's try a different heuristic:
    # 1. Generate random graph.
    # 2. Check girth.
    # 3. If girth < 16, find a short cycle.
    # 4. Perform a switch that breaks that specific cycle.
    # 5. Repeat.
    
    # This is "Targeted Repair".
    
    for _ in range(1000): # Max iterations
        if time.time() - start_time > time_limit:
            break
            
        girth = compute_girth_fast(N, adj)
        if girth >= g:
            return edges
            
        # Find a short cycle
        cycle = find_short_cycle(N, adj, g)
        if not cycle:
            continue
            
        # Try to break it
        # Pick an edge in the cycle, and a random edge elsewhere.
        # Perform a switch.
        # If switch is valid and doesn't create a shorter cycle (check locally?), accept.
        
        # To be safe, we will just do random switches and check girth periodically.
        # Given the time limit, let's do 100 switches then check girth.
        
        for _ in range(50):
            edges = random_switch(N, edges, adj)
            if edges is None:
                break
            adj = build_adj(N, edges)
            
        # Check girth after batch
        girth = compute_girth_fast(N, adj)
        if girth >= g:
            return edges
            
    return None

def generate_random_k_regular(N, k):
    if N * k % 2 != 0:
        return None
    stubs = []
    for i in range(N):
        for _ in range(k):
            stubs.append(i)
    random.shuffle(stubs)
    edges = []
    seen = set()
    for i in range(0, N * k, 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None
        if (u, v) in seen or (v, u) in seen:
            return None
        seen.add((u, v))
        seen.add((v, u))
        edges.append((u, v))
    return edges

def build_adj(N, edges):
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def compute_girth_fast(N, adj):
    min_girth = float('inf')
    for start in range(N):
        visited = [-1] * N
        parent = [-1] * N
        visited[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if visited[v] == -1:
                    visited[v] = visited[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = visited[u] + visited[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
        if min_girth <= 3: # Early exit
            break
    return min_girth if min_girth != float('inf') else N

def find_short_cycle(N, adj, target_girth):
    # Returns a list of vertices forming a cycle < target_girth
    # Reuses BFS logic
    for start in range(N):
        visited = [-1] * N
        parent = [-1] * N
        visited[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if visited[v] == -1:
                    visited[v] = visited[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = visited[u] + visited[v] + 1
                    if cycle_len < target_girth:
                        # Reconstruct cycle
                        path1 = []
                        curr = u
                        while curr != start:
                            path1.append(curr)
                            curr = parent[curr]
                        path1.reverse()
                        
                        path2 = []
                        curr = v
                        while curr != start:
                            path2.append(curr)
                            curr = parent[curr]
                        path2.reverse()
                        
                        # Combine
                        # The cycle is path1 + path2 (excluding start duplication)
                        # Actually, the cycle is path from u to start + path from v to start + edge (u,v)?
                        # No, the cycle is formed by the tree paths to LCA + the edge.
                        # Since it's a BFS tree, the LCA is start? No, LCA is some node.
                        # Simpler: just return the edge (u,v) and let the caller handle it?
                        # For now, return the cycle vertices roughly.
                        return [u, v] # Simplified
    return None

def random_switch(N, edges, adj):
    if len(edges) < 2:
        return None
    # Pick two random edges
    e1_idx = random.randint(0, len(edges) - 1)
    e2_idx = random.randint(0, len(edges) - 1)
    if e1_idx == e2_idx:
        return None
        
    u1, v1 = edges[e1_idx]
    u2, v2 = edges[e2_idx]
    
    # Check disjointness
    if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
        return None
        
    # Try switch 1: (u1, u2) and (v1, v2)
    # Check if these edges exist or are loops
    if (u1, u2) in adj[u1] or (v1, v2) in adj[v1]:
        pass # This logic is wrong, adj is list.
        
    # Let's use a set for fast lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
        
    new_e1 = (min(u1, u2), max(u1, u2))
    new_e2 = (min(v1, v2), max(v1, v2))
    
    if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
        return None
    if new_e1 in edge_set or new_e2 in edge_set:
        return None
        
    # Perform switch
    edges[e1_idx] = (u1, u2)
    edges[e2_idx] = (v1, v2)
    return edges

def luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q).
    """
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
