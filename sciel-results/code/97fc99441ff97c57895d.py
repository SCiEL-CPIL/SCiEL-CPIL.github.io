import random
import time
from collections import deque

def construct(k, g):
    # Fallback to seed logic if local search fails
    def seed_construct():
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

    # Local search for n=320
    target_n = 320
    # We are looking for a bipartite graph
    half_n = target_n // 2
    if target_n % 2 != 0:
        # If k=4, n must be even for bipartite?
        # Sum of degrees = 4n. 4n must be even. n can be odd?
        # But bipartite graphs have even number of vertices? No, K1 is bipartite.
        # But 4-regular requires n >= 4.
        # If n is odd, 4n is even, so possible.
        # But we construct bipartite, so n must be even.
        # The (4, 10) cage is 320 (even).
        pass

    deadline = time.time() + 110.0 # Safety margin
    
    def generate_random_bipartite_4reg(n):
        # n vertices, n/2 left, n/2 right
        # 4 perfect matchings
        half = n // 2
        matchings = []
        for _ in range(4):
            # Random permutation of range(half)
            perm = list(range(half))
            random.shuffle(perm)
            matchings.append(perm)
        
        # Build adjacency
        # left i connected to right perm[j][i]
        adj = [[0]*n for _ in range(n)] # or list of sets
        # Use sets for easy checking
        adj_sets = [set() for _ in range(n)]
        
        for p in matchings:
            for i in range(half):
                l = i
                r = half + p[i]
                adj_sets[l].add(r)
                adj_sets[r].add(l)
        
        return adj_sets

    def has_short_cycle(adj_sets, n, max_len=8):
        # Check 4-cycles
        # For each pair of left nodes (i, j), if they share > 1 neighbor, 4-cycle exists.
        left_nodes = range(n//2)
        for i in left_nodes:
            for j in range(i+1, n//2):
                common = adj_sets[i] & adj_sets[j]
                if len(common) > 1:
                    return True, "4-cycle"
        
        # Check 6-cycles
        # Path of length 3: u-v-w-x. If x connects to u, 6-cycle.
        # u (left) - v (right) - w (left) - x (right).
        # If x connects to u, we have u-v-w-x-u (length 4 edges, 4-cycle? No, u-v, v-w, w-x, x-u. 4 edges. 4-cycle.)
        # Wait.
        # 4-cycle: L1-R1-L2-R2-L1.
        # 6-cycle: L1-R1-L2-R2-L3-R3-L1.
        
        # Check 6-cycles:
        # For each edge (u, v) where u is left, v is right.
        # For each neighbor w of v (w is left, w != u).
        # For each neighbor x of w (x is right, x != v).
        # If x is connected to u, we have a 6-cycle?
        # u-v-w-x-u. Length 4 edges. That's a 4-cycle.
        # We already checked 4-cycles.
        # So we need length 6.
        # u-v-w-x-y-z-u?
        
        # Actually, if we check all cycles up to length g-1.
        # BFS from each node up to depth g/2?
        # n=320 is small.
        # BFS from each node to depth 4 (for 8-cycles) is O(n * 4^4) = 320 * 256 = 81920. Very fast.
        
        for start in range(n):
            visited = {start: 0}
            queue = deque([(start, 0, -1)])
            while queue:
                node, depth, parent = queue.popleft()
                if depth >= max_len:
                    continue
                for nei in adj_sets[node]:
                    if nei == parent:
                        continue
                    if nei in visited:
                        # Found a cycle
                        # Length = depth + 1 + visited[nei]
                        # If visited[nei] + depth + 1 < g, we have a short cycle
                        # Note: BFS finds shortest path.
                        # If we reach a visited node, the cycle length is depth + 1 + dist[nei]?
                        # Actually, if we use simple BFS, `visited` stores distance from start.
                        # If `nei` is visited, and `depth + 1 + visited[nei] < g`?
                        # No, `visited[nei]` is dist from start to nei.
                        # Path: start -> ... -> node -> nei.
                        # Cycle: start -> ... -> node -> nei -> ... -> start.
                        # This logic is tricky with simple BFS for *any* cycle.
                        # Better: check specific cycle lengths.
                        pass
                    else:
                        visited[nei] = depth + 1
                        queue.append((nei, depth + 1, node))
                        
        # The above BFS doesn't correctly identify all short cycles easily without backtracking or specific checks.
        # Given the time, checking 4-cycles is cheap.
        # 6-cycles:
        # For each pair of edges (u,v) and (v,w) forming a path u-v-w.
        # Check if w connects to u (4-cycle, already handled).
        # Check if w connects to a neighbor of u (6-cycle).
        # Let's just say we check 4-cycles and 6-cycles explicitly.
        
        # 6-cycle check:
        # For each right node v:
        #   For each pair of neighbors u1, u2 of v (u1 < u2):
        #     For each neighbor w of u1 (w != v):
        #       For each neighbor x of w (x != u1):
        #         If x == u2: 6-cycle?
        #         Path: u1-v-u2-w-x-u1? No.
        #         Path: u1-v-u2 ...
        #         Let's trace: u1-v (1), v-u2 (2).
        #         We need path from u2 to u1 of length 4.
        #         u2 - w - x - y - u1?
        #         This is getting complex.
        
        # Given the 120s limit, and that 4-cycles are the most frequent in random graphs,
        # removing 4-cycles might be enough to get a valid graph for g=10?
        # No, g=10 requires no 6 or 8 cycles either.
        
        return False, "none"

    # Given the complexity, I will stick to the seed but try to run it faster?
    # Or just return the seed result.
    # The seed is guaranteed to work.
    # If I can't guarantee a better result, I should return the seed result.
    # But the seed result is ~8000.
    # Is there a known small graph I can hardcode?
    # The (4, 10) cage is 320 vertices.
    # I can try to find it online? No.
    
    # Let's try a simple local search that only fixes 4-cycles.
    # If it fails, return seed.
    
    best_n, best_edges = None, None
    start_time = time.time()
    
    # Try a few random instances
    for attempt in range(20):
        if time.time() > deadline:
            break
            
        adj_sets = generate_random_bipartite_4reg(target_n)
        
        # Simple local search to remove 4-cycles
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Find a 4-cycle
            cycle_found = False
            for i in range(target_n // 2):
                if time.time() > deadline: break
                for j in range(i+1, target_n // 2):
                    common = adj_sets[i] & adj_sets[j]
                    if len(common) > 1:
                        # We have a 4-cycle
                        # Pick two common neighbors w1, w2
                        w_list = list(common)
                        w1, w2 = w_list[0], w_list[1]
                        
                        # Rewire
                        # Current: i-w1, i-w2, j-w1, j-w2
                        # Try to change one of these edges.
                        # Pick a random edge in the cycle, say i-w1.
                        # Pick a random "compatible" edge to swap with.
                        # Find a right node w3 not connected to i, and a left node i3 not connected to w3.
                        # This is heuristic.
                        
                        # Simpler: Just restart?
                        # No, let's just break and restart.
                        # Local search for 4-cycles is tricky to make efficient in Python.
                        
                        cycle_found = True
                        break
                if cycle_found:
                    break
                    
            if not cycle_found:
                # No 4-cycles. Check 6-cycles?
                # If we assume no 6-cycles (optimistic), return this.
                # But we need to be sure.
                # For g=10, we must ensure no 6, 8 cycles.
                # Since we can't easily check 6/8 in this quick loop,
                # we might return an invalid graph.
                # So we MUST use a reliable checker or fallback.
                
                # Let's use a reliable checker for this candidate.
                # Simple BFS for shortest cycle from each node up to depth 4.
                valid = True
                for s in range(target_n):
                    dist = [-1] * target_n
                    dist[s] = 0
                    q = deque([s])
                    while q:
                        u = q.popleft()
                        if dist[u] >= 4: continue
                        for v in adj_sets[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                q.append(v)
                            elif dist[v] == dist[u] + 1:
                                # Same level, edge between them?
                                # Bipartite graph has no edges within same level.
                                pass
                            # If dist[v] + dist[u] + 1 < g, cycle found?
                            # This checks paths, not cycles directly.
                            
                # This BFS doesn't prove girth >= 10.
                # Given the risk, I will NOT return this candidate unless I'm sure.
                # So I will fall back to seed.
                
                break
                
        if cycle_found == False and time.time() < deadline:
             # We found a graph with no 4-cycles.
             # But we don't know about 6/8.
             # Risky.
             pass
             
    # Fallback
    return seed_construct()
