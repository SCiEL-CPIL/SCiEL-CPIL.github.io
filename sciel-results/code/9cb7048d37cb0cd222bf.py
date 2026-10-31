import time
import sys
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    
    Strategy:
    1. The current program uses LUW algebraic construction which yields large graphs (4374 for d=7).
    2. We switch to a Greedy Tree-Embedding Search (GTE).
    3. We start with the unique k-regular tree of depth d = g/2 (Moore tree).
    4. For k=3, g=16, d=8? No, girth 16 means no cycles < 16.
       The Moore bound for n(3,16) is 1 + 3 * sum_{i=0}^{7} 2^i = 1 + 3 * (2^8 - 1) = 1 + 3*255 = 766.
       Wait, the Moore bound calculation:
       n >= 1 + k * sum_{i=0}^{d-1} (k-1)^i for even girth g=2d.
       Here g=16, so d=8.
       n >= 1 + 3 * (1 + 2 + 4 + ... + 128) = 1 + 3 * 255 = 766.
       The prompt says lower bound 512. Let's re-verify Moore bound for cubic graphs.
       Actually, for cubic graphs, the Moore bound is often cited differently or I am misremembering the formula.
       Let's check:
       Depth 0: 1 node.
       Depth 1: 3 nodes.
       Depth 2: 3*2 = 6 nodes.
       Depth 3: 3*2^2 = 12 nodes.
       Depth 4: 24 nodes.
       Depth 5: 48 nodes.
       Depth 6: 96 nodes.
       Depth 7: 192 nodes.
       Total nodes in tree of depth 7 (distance 7 from center) is 1 + 3 + 6 + 12 + 24 + 48 + 96 + 192 = 382.
       If girth is 16, the radius is 7 (cycles of length 16 go 8 steps from center to center? No).
       A cycle of length 2d in a graph with center c implies two paths of length d from c to a vertex v.
       So if girth is 16, we can have paths of length 7 that meet.
       The Moore bound for cubic graphs of girth g is:
       If g is even, g=2d, n >= 1 + k * sum_{i=0}^{d-1} (k-1)^i.
       For k=3, g=16 (d=8):
       n >= 1 + 3 * (2^8 - 1) = 1 + 3 * 255 = 766.
       
       However, the prompt states "best lower bound is 512". 
       512 is 2 * 256. 
       This suggests the prompt might be referring to a bipartite Moore bound or a different specific bound.
       Actually, for bipartite cubic graphs of girth 16:
       n >= 2 * sum_{i=0}^{d-1} 2^i = 2 * (2^8 - 1) = 510.
       So 512 is close to the bipartite Moore bound.
       
       The best known graph has 936 vertices.
       
       My GTE strategy:
       1. Generate the full Moore tree of depth 7 (radius 7) centered at node 0.
          This tree has 382 nodes.
          The leaves are at depth 7. There are 192 leaves.
          Each leaf has degree 1 in the tree. It needs 2 more edges to become degree 3.
          The internal nodes (depth 1..6) have degree 3 in the tree?
          Depth 1: connected to center and 2 children. Degree 3. OK.
          Depth 2..6: connected to parent and 2 children. Degree 3. OK.
          So only the 192 leaves need 2 more edges each.
          Total "stubs" to connect: 192 * 2 = 384 stubs.
          We need to pair these 384 stubs to form 192 edges.
          Constraint: No cycle shorter than 16.
          Since the tree has diameter 14 (depth 7 to depth 7 via center), any edge connecting two leaves at depth 7 creates a cycle of length:
          dist(leaf1, leaf2) + 1.
          dist(leaf1, leaf2) <= 14.
          If we connect leaf1 and leaf2, the cycle length is dist(leaf1, leaf2) + 1.
          We need cycle length >= 16.
          So dist(leaf1, leaf2) + 1 >= 16 => dist(leaf1, leaf2) >= 15.
          But max distance in the tree is 14.
          Therefore, we CANNOT connect any two leaves of the depth-7 tree without creating a cycle of length <= 15.
          
          This means the simple "close the tree" approach for girth 16 with depth 7 fails immediately.
          We need a larger structure or a different embedding.
          
          Alternative: Start with a smaller tree? No, we need girth 16.
          The issue is that the Moore tree of radius 7 is "too small" to close without short cycles.
          To get girth 16, we need to ensure that any two paths from the center that meet do so at distance >= 16.
          
          Let's rethink the GTE.
          Instead of starting with a full tree, we can build the graph layer by layer, or use a "hash" based approach.
          
          A more robust strategy for small cages (n ~ 1000) is **Randomized Local Search** with a very strong initialization.
          However, the directive asks for a "different design or modelling family".
          
          Let's try **Cayley Graphs on Abelian Groups**.
          The best known 936-vertex graph might be a Cayley graph.
          936 = 8 * 117 = 8 * 9 * 13 = 2^3 * 3^2 * 13.
          Possible groups: Z_936, Z_13 x Z_72, Z_9 x Z_104, etc.
          Generating Cayley graphs is fast.
          We can iterate through small subsets S of a group G of size 936 (or nearby sizes like 512, 766, 936) such that |S|=3, S=S^{-1}, 0 not in S.
          Then check girth.
          
          Let's implement a search over Cayley graphs of cyclic groups Z_n for n in [512, 936] (step 2 to ensure evenness for bipartite? No, cubic graphs don't have to be bipartite, but high girth cubic graphs are often bipartite or close to it).
          Actually, if we find a bipartite graph, n must be even.
          
          Let's try Z_n with generators {1, a, b}.
          Girth of Cayley graph Cay(Z_n, S) can be computed by BFS.
          
          This is a "Design Family" change: From LUW/Tree to Cayley Graph Search.
          
          Plan:
          1. Iterate n from 512 to 936 (step 2).
          2. For each n, iterate possible generators (a, b) with 1 < a < b < n/2.
          3. Construct Cayley graph.
          4. Check girth.
          5. Return first valid graph.
          
          Optimization:
          Checking girth for n=936 is O(n).
          Number of pairs (a,b) for n=936 is ~ (468^2)/2 ~ 100,000.
          100,000 * 936 ~ 100 million ops. Might be too slow in Python.
          
          We can prune:
          If a+b < g, we have a short cycle? Not necessarily.
          But if a == b, invalid.
          
          Let's restrict the search space.
          We know the answer is 936. We can just search around 936.
          Or, we can search smaller n first.
          
          However, if we can't find it in Cayley graphs, we fall back to LUW(3,7) which is known to work (score 3.0).
          
          Let's try a smarter Cayley search:
          Use Z_n.
          To be fast, we can check girth using a bitset or optimized BFS.
          
          Actually, there is a known family: **Generalized Petersen Graphs** were tried and failed in the current code (or not tried correctly).
          Let's try **Circulant Graphs** which are Cayley graphs on Z_n.
          
          Implementation details for speed:
          - Use numpy for adjacency? No, list of lists is fine for n=1000.
          - BFS for girth.
          
          Let's add a timeout guard.
          
          If Cayley search fails, return LUW(3,7).
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Fallback: LUW(3, 7)
    def luw_construct(k, g, d_override=None):
        if k != 3:
            return None, None
        q = 3
        d = d_override
        if d is None:
            d = 7
        
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
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

    def verify_girth(n, edges, k, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        max_depth = g // 2
        for start in range(n):
            visited_depth = {}
            queue = deque()
            for v in adj[start]:
                queue.append((v, start, 1))
                visited_depth[(v, start)] = 1
            
            while queue:
                u, prev, depth = queue.popleft()
                if depth == max_depth:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    if v == start:
                        if depth + 1 < g:
                            return False
                        continue
                    state = (v, u)
                    if state in visited_depth:
                        if depth - visited_depth[state] < g:
                            return False
                    else:
                        visited_depth[state] = depth
                        queue.append((v, u, depth + 1))
        return True

    # New Strategy: Search Circulant Graphs C(n, {1, a, b})
    # n must be even for bipartite? Not strictly, but high girth cubic graphs are often bipartite.
    # Let's search n from 512 to 936.
    # We prioritize n close to the known best (936) or the lower bound (512).
    # Let's try n in [512, 936] step 2.
    
    found_n = None
    found_edges = None
    
    # To speed up, we can test a subset of (a,b) first.
    # For each n, we test pairs (a,b).
    # Optimization: If time is low, break.
    
    # We'll iterate n in increasing order to find the smallest valid graph.
    # But 512 might not have a solution. 936 is known to have one.
    # Let's try n=936 first, as it is the target.
    # If we find a valid graph at 936, we can then try smaller n if time permits.
    # However, the score is n. Smaller is better.
    # So we should iterate n from 512 upwards.
    
    # But checking all n from 512 to 936 might be too slow if we check all pairs.
    # Let's limit the pairs checked per n.
    # Or, let's just try n=936 first to ensure we get a valid graph (score >= 3.0),
    # then try to improve by searching smaller n.
    
    # Actually, if we find 936, score is 14.0 (assuming seed 13122).
    # If we find 800, score is 16.4.
    # So finding ANY valid graph < 4374 is a huge win.
    
    # Let's try n=936 with a limited search for (a,b).
    # If found, return immediately? No, we might find a smaller one.
    # But 120s is tight.
    # Let's try n=936 first. If found, save it. Then try n=934, 932... down to 512.
    # If we find a smaller one, return it.
    # If time runs out, return the best found (likely 936 or LUW).
    
    best_valid_n = float('inf')
    best_valid_edges = []
    
    # Helper to check a specific circulant graph
    def check_circulant(n, a, b):
        # Construct edges
        # Neighbors of i: (i+1)%n, (i-1)%n, (i+a)%n, (i-a)%n, (i+b)%n, (i-b)%n
        # Wait, degree must be 3.
        # Generators S = {1, a, b}.
        # If 1, a, b are distinct and not inverses of each other (mod n), degree is 6?
        # No, in a Cayley graph, if S contains x and -x, they form one undirected edge.
        # For degree 3, we need |S|=3 where S is a set of elements such that S = -S?
        # No, if S = {1, a, b}, the neighbors are i+1, i-1, i+a, i-a, i+b, i-b.
        # This gives degree 6 unless some generators are inverses.
        # For a 3-regular graph, we need 3 "directions".
        # If we use S = {1, a, b}, we need 1=-a? No.
        # Standard 3-regular circulant: C(n, {1, a, b}) where the edges are (i, i+1), (i, i+a), (i, i+b).
        # This is directed? No, undirected.
        # So edges are {i, i+1}, {i, i+a}, {i, i+b}.
        # For this to be regular degree 3, we must ensure that 1, a, b are distinct and none is the negative of another (mod n) in a way that collapses edges?
        # Actually, if we define edges as (i, i+s) for s in S, and the graph is undirected,
        # then (i, i+s) is the same as (i+s, i).
        # So if s is in S, -s is effectively in S.
        # To have degree exactly 3, the set of undirected edges must be 3.
        # This happens if S = {1, a, b} and 1 != -a, 1 != -b, a != -b, and 1, a, b are distinct.
        # Also 1, a, b must not be 0.
        # And n must be large enough.
        
        # Let's just build the adjacency list and check degree.
        adj = [[] for _ in range(n)]
        edges = []
        for i in range(n):
            n1 = (i + 1) % n
            n2 = (i + a) % n
            n3 = (i + b) % n
            
            # Add edges if not already added (to avoid duplicates in list, though for BFS it's fine to have duplicates, but for degree check we need unique neighbors)
            # We'll add to adj and check uniqueness later or use sets.
            adj[i].append(n1)
            adj[i].append(n2)
            adj[i].append(n3)
            
            # For undirected, we add reverse
            adj[n1].append(i)
            adj[n2].append(i)
            adj[n3].append(i)
            
            # Record edges for return (avoid duplicates)
            if i < n1: edges.append((i, n1))
            if i < n2: edges.append((i, n2))
            if i < n3: edges.append((i, n3))
            
        # Check degree
        for i in range(n):
            if len(adj[i]) != 3:
                return False, None
            # Check for self-loops or multi-edges?
            # If a == 1, then n2 == n1. Degree would be 2 (if we count unique) or 3 (if multi).
            # We want simple graphs.
            # So we must ensure 1, a, b are distinct and not negatives of each other.
            # 1 != a, 1 != b, a != b.
            # 1 != -a => a != n-1.
            # 1 != -b => b != n-1.
            # a != -b => a+b != n.
            
        # Check simple:
        # If a == 1, edge (i, i+1) added twice.
        # We should filter S first.
        
        return True, edges

    def is_valid_generators(n, a, b):
        if a == b or a == 1 or b == 1:
            return False
        if (a + b) % n == 0: # a == -b
            return False
        if a == n - 1: # a == -1
            return False
        if b == n - 1:
            return False
        return True

    # Search n=936 first
    n_target = 936
    # We need to find a, b such that girth >= 16.
    # To be fast, we can't check all ~100k pairs for n=936 if BFS is slow.
    # BFS on 936 nodes is ~1000 ops.
    # 100k * 1000 = 100M ops. ~10-30s in Python. Feasible.
    
    # Let's iterate a, b for n=936.
    # To save time, we can restrict a, b to a subset?
    # No, just iterate.
    
    for n_try in [936]: # Just 936 for now to ensure a win.
        if time.time() > deadline - 5:
            break
        for a in range(2, n_try // 2):
            if time.time() > deadline - 5:
                break
            for b in range(a + 1, n_try // 2):
                if not is_valid_generators(n_try, a, b):
                    continue
                
                # Construct and check
                # Optimization: We can do a quick girth check.
                # Build adj
                adj = [[] for _ in range(n_try)]
                for i in range(n_try):
                    adj[i].append((i + 1) % n_try)
                    adj[i].append((i + a) % n_try)
                    adj[i].append((i + b) % n_try)
                
                # Check degree (should be 3 if valid generators)
                # Quick check for short cycles (girth < 16)
                # We can use the verify_girth function, but it's a bit heavy.
                # Let's use a faster inline check for girth < 16.
                
                # Actually, just use verify_girth.
                # But we need the edge list for verify_girth.
                # We can modify verify_girth to take adj directly?
                # Or just build edges.
                
                edges = []
                for i in range(n_try):
                    n1 = (i + 1) % n_try
                    n2 = (i + a) % n_try
                    n3 = (i + b) % n_try
                    if i < n1: edges.append((i, n1))
                    if i < n2: edges.append((i, n2))
                    if i < n3: edges.append((i, n3))
                    
                if verify_girth(n_try, edges, 3, g):
                    best_valid_n = n_try
                    best_valid_edges = edges
                    print(f"Found valid circulant graph n={n_try}, a={a}, b={b}")
                    break
            if best_valid_n != float('inf'):
                break
    
    # If found 936, try to find smaller?
    # If time permits, try n=934, 932...
    if best_valid_n != float('inf') and time.time() < deadline - 20:
        for n_try in range(934, 511, -2):
            if time.time() > deadline - 5:
                break
            if n_try >= best_valid_n:
                continue
            # Limited search for smaller n
            # Just check a few random pairs or a small range?
            # To be safe, we'll just check if we can find ANY valid graph for this n.
            # We'll iterate a, b but break early if found.
            found = False
            for a in range(2, n_try // 2):
                if time.time() > deadline - 5:
                    break
                for b in range(a + 1, n_try // 2):
                    if not is_valid_generators(n_try, a, b):
                        continue
                    edges = []
                    for i in range(n_try):
                        n1 = (i + 1) % n_try
                        n2 = (i + a) % n_try
                        n3 = (i + b) % n_try
                        if i < n1: edges.append((i, n1))
                        if i < n2: edges.append((i, n2))
                        if i < n3: edges.append((i, n3))
                    
                    if verify_girth(n_try, edges, 3, g):
                        best_valid_n = n_try
                        best_valid_edges = edges
                        print(f"Found smaller valid circulant graph n={n_try}")
                        found = True
                        break
                if found:
                    break
            if found:
                # Continue to smaller n?
                # Yes, keep going if time permits.
                pass

    # If no circulant found, fall back to LUW
    if best_valid_n == float('inf'):
        if time.time() < deadline - 10:
            n_luw, edges_luw = luw_construct(3, g, 7)
            if n_luw and verify_girth(n_luw, edges_luw, 3, g):
                best_valid_n = n_luw
                best_valid_edges = edges_luw
    
    if best_valid_n == float('inf'):
        # Last resort
        n_luw, edges_luw = luw_construct(3, g, 8)
        return n_luw, edges_luw
        
    return best_valid_n, best_valid_edges

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"Constructed graph with {n} vertices")

