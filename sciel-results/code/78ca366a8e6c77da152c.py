import random
import time
from collections import deque

def construct(k, g):
    # Target: 4-regular graph, girth 10.
    # Known best: 320 vertices.
    # Moore bound: 242.
    
    # 1. Try to find a circulant graph C_N({a, b}) with girth >= 10
    # This is fast and might yield a small N.
    def check_circulant_girth(N, a, b, target_girth):
        # Check for cycles of length < target_girth
        # We only need to check up to target_girth - 1
        # In a circulant graph, a cycle of length L corresponds to a sequence of steps
        # from {a, -a, b, -b} that sums to 0 mod N.
        # We can do a BFS in the Cayley graph to find the shortest cycle.
        # Since it's vertex-transitive, we just need to find the shortest cycle containing vertex 0.
        # A cycle containing 0 is a path from 0 to 0 of length L with no repeated vertices (except 0).
        # Actually, girth is the shortest cycle.
        
        # BFS from 0
        # State: (current_node, path_length)
        # We need to avoid immediate backtracking? No, BFS handles that.
        # But we need to ensure simple cycle.
        # For small girth (10), we can just enumerate all walks of length < 10 from 0
        # and check if they return to 0.
        
        steps = [a, -a % N, b, -b % N]
        
        # Check for cycles of length 4, 6, 8
        # Length 4: 0 -> s1 -> s2 -> s3 -> 0
        # s1+s2+s3+s4 = 0
        # We can just do a BFS up to depth target_girth - 1 and see if we hit 0.
        # But BFS from 0 will hit 0 at depth 0.
        # We need to find the shortest cycle.
        
        # Better: Check all combinations of steps of length L < target_girth
        # L=4: 4 steps
        # L=6: 6 steps
        # L=8: 8 steps
        
        # For L=4:
        # s1, s2, s3, s4 in steps
        # s1+s2+s3+s4 = 0
        # And the intermediate nodes are distinct?
        # 0, s1, s1+s2, s1+s2+s3, 0
        # Distinctness: s1 != 0, s1+s2 != 0, s1+s2+s3 != 0, s1+s2+s3+s4=0
        # Also s1 != s1+s2 (s2!=0), etc.
        
        # This combinatorial check is feasible for L=8.
        # 4^8 = 65536 checks. Very fast.
        
        for L in range(4, target_girth, 2): # Only even cycles in bipartite? No, circulant can be non-bipartite.
            # Actually, if N is even and steps are odd, it might be bipartite.
            # Let's just check all L from 3 to target_girth - 1.
            pass
        
        # Simpler: BFS on the state space (node, visited_mask) is too big.
        # Just check if any sequence of steps of length L sums to 0 mod N
        # and the partial sums are distinct.
        
        def has_short_cycle(L):
            # Generate all sequences of L steps
            # This is 4^L. For L=8, 65536. For L=9, 262144.
            # We can prune.
            
            # We just need to know if a cycle of length L exists.
            # We can use a set of partial sums.
            # Start with {0}
            # Level 1: {s for s in steps}
            # Level 2: {s1+s2}
            # ...
            # If 0 appears at level L, and all intermediate sums are distinct and non-zero (except start),
            # then we have a cycle.
            
            current_sums = {0}
            # We need to track the path to ensure distinctness?
            # If we just track sums, we might merge paths.
            # But if a sum repeats, it's a shorter cycle or a non-simple path.
            # We want to know if there is a SIMPLE cycle.
            
            # For L <= 10, we can just brute force the sequences.
            # 4^8 = 65536.
            # For each sequence, check if sum == 0 and partial sums are unique.
            
            # Optimization: Only check even L if the graph is bipartite?
            # We don't know if it's bipartite.
            
            # Let's just check L=4, 6, 8.
            
            # L=4
            for s1 in steps:
                for s2 in steps:
                    p1 = s1
                    if p1 == 0: continue
                    for s3 in steps:
                        p2 = (p1 + s3) % N
                        if p2 == 0 or p2 == p1: continue
                        for s4 in steps:
                            p3 = (p2 + s4) % N
                            if p3 == 0:
                                # Check if p1, p2 are distinct and non-zero
                                # p1 != 0, p2 != 0, p1 != p2
                                # s1, s3, s4 are non-zero by construction (a,b != 0)
                                # p1 = s1
                                # p2 = s1+s3
                                # p3 = s1+s3+s4 = 0 => s4 = -s1-s3
                                # Cycle: 0 -> s1 -> s1+s3 -> 0
                                # Length 3? No, 4 steps.
                                # Vertices: 0, s1, s1+s3, 0.
                                # Wait, 3 vertices? 0, v1, v2, 0.
                                # This is a triangle if s1+s3+s4=0.
                                # But we are checking L=4.
                                # The sequence is s1, s2, s3, s4.
                                # My loop above used s1, s3, s4. I skipped s2?
                                # Let's rewrite properly.
                                pass
                                
            # This is getting complicated to code inline without errors.
            # Let's use a BFS that tracks the path for small depths.
            
            # BFS from 0
            # Queue: (node, path)
            # Max depth 9.
            # If we reach 0 at depth L > 0, we have a cycle of length L.
            # We must ensure the path is simple (no repeated nodes).
            
            visited = set() # (node, last_node) to avoid backtracking?
            # Actually, for girth, we just need the shortest cycle.
            # BFS from 0.
            # dist[0] = 0
            # When we visit a neighbor, if it is 0, we found a cycle.
            # But we must not backtrack immediately.
            
            # Standard BFS for girth in a graph:
            # For each edge (u, v), if dist[u] + 1 + dist[v] forms a cycle?
            # No, that's for finding *a* cycle.
            
            # For circulant, we can just do BFS from 0 up to depth 9.
            # If we encounter 0 again, record the length.
            # We need to avoid 0 -> s -> 0 (length 2, backtracking).
            
            q = deque()
            q.append((0, 0, [])) # node, depth, path
            # Actually, just depth is enough if we don't store path?
            # We need to ensure we don't visit a node twice in the same path.
            # For depth 9, we can store the tuple of nodes.
            
            # This might be slow if branching factor is 4.
            # 4^9 = 262144 paths.
            # For each N, we do this for each (a,b) pair.
            # Number of pairs ~ 150 * 150 = 22500.
            # 22500 * 262144 is too slow.
            
            return True # Assume valid for now to test structure
            
    # Given the complexity, let's stick to the Greedy Heuristic which is more likely to find a 320-vertex graph.
    
    def greedy_construct(N, k, g):
        # Returns edges if successful, None otherwise
        edges = []
        adj = [set() for _ in range(N)]
        
        # Precompute all possible edges
        # We will add them randomly
        
        # To ensure we can check distance quickly, we will run BFS from every node
        # after each addition? No, too slow.
        
        # Instead, we maintain a "forbidden" set?
        # If we add (u, v), we forbid any (x, y) that would create a cycle < g.
        # This is hard to update dynamically.
        
        # Alternative: Build the graph, then check girth.
        # If girth < g, try to fix?
        
        # Let's try a simple random regular graph generation and check.
        # 1. Generate random 4-regular graph on N vertices.
        # 2. Check girth.
        # 3. If girth >= 10, return.
        # 4. Repeat.
        
        # Probability of girth 10 is low.
        
        # Better: Greedy edge addition.
        # Start with empty graph.
        # While edges < N*k/2:
        #   Pick random (u, v)
        #   If dist(u, v) >= g - 1 in current graph, add edge.
        #   Else, skip.
        
        # If we get stuck, restart.
        
        def get_distance(u, v, adj):
            if u == v: return 0
            visited = {u}
            queue = deque([(u, 0)])
            while queue:
                node, d = queue.popleft()
                for nei in adj[node]:
                    if nei == v:
                        return d + 1
                    if nei not in visited:
                        visited.add(nei)
                        queue.append((nei, d + 1))
            return float('inf')
            
        random.seed(42)
        for attempt in range(50): # 50 attempts
            edges = []
            adj = [set() for _ in range(N)]
            target_edges = N * k // 2
            
            # List of all possible edges
            all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                if len(adj[u]) == k and len(adj[v]) == k:
                    continue
                if len(adj[u]) >= k or len(adj[v]) >= k:
                    continue
                
                # Check distance
                d = get_distance(u, v, adj)
                if d >= g - 1: # Adding edge creates cycle of length d+1 >= g
                    # Also check if u and v are already connected?
                    # If d is inf, it's fine.
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append((u, v))
                    if len(edges) == target_edges:
                        # Check if all vertices have degree k
                        if all(len(a) == k for a in adj):
                            # Final girth check
                            # We assume it's correct due to greedy constraint
                            # But let's verify one vertex's BFS to be safe?
                            # The greedy check ensures no cycle < g is formed.
                            return edges
            
        return None

    # Try N = 320
    start_time = time.time()
    edges = greedy_construct(320, k, g)
    
    if edges and time.time() - start_time < 110:
        return 320, edges
        
    # If that fails, try N = 324, 328 ...
    for N in range(324, 400, 4):
        if time.time() - start_time > 110:
            break
        edges = greedy_construct(N, k, g)
        if edges:
            return N, edges
            
    # Fallback: LUW (from seed)
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
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
    return len(order), edges_luw
