import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 110
    
    # For k=4, g=10, try Cayley graph on Z_m with generators {1, -1, a, -a}
    # This gives a 4-regular graph on m vertices.
    # We need girth >= 10.
    
    # The (4,10)-cage is known to have 320 vertices.
    # It is a Cayley graph on Z_320? Or Z_160 x Z_2?
    
    # Let's search for m and a.
    # m must be even (for -1 to be distinct from 1 unless m=2).
    # a must be such that {1, -1, a, -a} are distinct.
    
    best_n = None
    best_edges = None
    
    # Try m from 243 to 1280 (step 1 or 2)
    # Actually, m must be even for the graph to be bipartite? No.
    # But if m is odd, -1 != 1.
    
    # Let's try m up to 1280.
    # For each m, try a from 2 to m//2.
    
    # To speed up, we can check girth incrementally.
    
    def check_girth_cayley(m, a):
        # Graph on Z_m, edges (x, x+1), (x, x-1), (x, x+a), (x, x-a)
        # We need to check if there is a cycle of length < 10.
        # A cycle is a sequence of steps s_1, ..., s_k such that sum s_i = 0 mod m.
        # Steps are from S = {1, -1, a, -a}.
        # We need to check if there is a non-trivial relation of length < 10.
        # This is equivalent to checking if the product of generators is 1.
        
        # We can do BFS from 0 in the Cayley graph.
        # Since m is small, BFS is fast.
        
        visited = [-1] * m
        visited[0] = 0
        queue = deque([(0, 0)]) # (node, dist)
        
        while queue:
            node, dist = queue.popleft()
            if dist >= 9: # We only care about cycles < 10
                continue
            for step in [1, -1, a, -a]:
                next_node = (node + step) % m
                if visited[next_node] == -1:
                    visited[next_node] = dist + 1
                    queue.append((next_node, dist + 1))
                else:
                    # If we hit a visited node, we might have a cycle.
                    # But we need to be careful with the "parent" check.
                    # Standard BFS girth check:
                    # If next_node is visited and next_node != parent, 
                    # then cycle length = dist + 1 + visited[next_node].
                    # But we don't track parent here.
                    
                    # Let's use a different check:
                    # If we reach a node that was visited at distance d', 
                    # and the current path length is d, 
                    # then we have a cycle of length d + d' + 1?
                    # No, this is for undirected graphs.
                    
                    # Actually, for Cayley graphs, we can just check if 
                    # any product of < 10 generators is 1.
                    # This is a subset sum problem mod m.
                    
                    pass
        return True # Placeholder
        
    # Let's use a more direct girth check.
    
    def girth_cayley(m, a):
        # BFS from 0
        # dist[v] = shortest distance from 0 to v
        # But we need the shortest cycle.
        # The shortest cycle through 0 is the shortest non-trivial relation.
        # The shortest cycle not through 0 is more complex.
        
        # However, for Cayley graphs, the girth is the length of the shortest 
        # non-trivial relation in the generators.
        
        # We can check all words of length < 10.
        # Number of words: 4^9 = 262144.
        # For each word, check if product is 1 mod m.
        
        # This is fast enough for m < 1000.
        
        # We generate all words of length 1 to 9.
        # We can do this iteratively.
        
        # S = {1, -1, a, -a}
        # We want to know if there is a word s_1 ... s_k = 1 mod m for k < 10.
        
        # We can use a set of reachable sums.
        # Reachable sums after k steps.
        
        # Let's do BFS on the sums.
        # State: (sum, last_step) ? No, sum is enough.
        # But we need to avoid backtracking? 
        # In a Cayley graph, a cycle is a closed walk.
        # A closed walk of length k is a sequence s_1 ... s_k such that sum s_i = 0.
        # We want the minimum k >= 3 (simple cycle).
        # Actually, girth is the length of the shortest simple cycle.
        # In a Cayley graph, the shortest cycle is the shortest non-trivial relation.
        
        # We can use BFS from 0, but we must not go back to the previous node.
        # This is the standard "non-backtracking" BFS.
        
        # Let's implement non-backtracking BFS on the Cayley graph.
        
        # State: (node, last_step)
        # last_step is the step taken to arrive at node.
        # We start from (0, None).
        
        # But we can just use the standard BFS on the directed graph of "arcs".
        # Arcs are (u, v).
        # We want the shortest cycle.
        
        # For m < 1000, we can just use the word check.
        
        # Let's check all words of length k from 3 to 9.
        # If any word sums to 0 mod m, and the word is "reduced" (no s_i = -s_{i+1}), 
        # then we have a cycle of length k.
        # Actually, if a word sums to 0, it contains a cycle.
        # The shortest such word gives the girth.
        
        # We can generate all words of length k and check sum.
        # For k=3: 4^3 = 64 words.
        # For k=9: 4^9 = 262144 words.
        # Total ~ 350k checks. Very fast.
        
        S = [1, -1, a, -a]
        
        # We need to check if there is a word of length k < 10 that sums to 0.
        # We also need to ensure it's a simple cycle?
        # If a word sums to 0, it represents a closed walk.
        # The closed walk contains a simple cycle of length <= k.
        # So if we find a word of length k that sums to 0, the girth is <= k.
        # We want the minimum k.
        
        # We can iterate k from 3 to 9.
        # For each k, check if any word of length k sums to 0.
        
        # To avoid checking all 4^k words, we can use dynamic programming.
        # Let R_k be the set of sums of words of length k.
        # R_0 = {0}
        # R_k = { (r + s) % m for r in R_{k-1} for s in S }
        
        # We want to know if 0 is in R_k for k >= 3.
        # But we must exclude the trivial word (all 0s? No, S has no 0).
        # We must exclude backtracking? 
        # If we allow backtracking, we might find a "cycle" of length 2 (s, -s).
        # But girth is >= 3.
        # So we just check for k >= 3.
        
        # However, if we find a word of length 4 that sums to 0, it might be (1, 1, -1, -1).
        # This is not a simple cycle.
        # But it implies a cycle of length 2? No, (1, -1) is length 2.
        # If we check k=2, we will find (1, -1) sums to 0.
        # So we must start from k=3.
        
        # But if a word of length 4 sums to 0, it might contain a shorter cycle.
        # So if we find any word of length k that sums to 0, the girth is <= k.
        # We want the minimum k.
        
        # So we check k=3, 4, ..., 9.
        # If we find a sum of 0 at k, return k.
        
        # But we must ensure the word is not "trivial" in a way that it's not a cycle?
        # Any non-empty word that sums to 0 is a closed walk.
        # The shortest closed walk is the girth? 
        # No, the shortest closed walk might be (1, -1) which is length 2.
        # But girth is the shortest *simple* cycle.
        # In a simple graph, there are no 2-cycles.
        # So the shortest closed walk of length >= 3 is the girth.
        
        # So we just need to find the minimum k >= 3 such that 0 is in R_k.
        
        R = {0}
        for k in range(1, 10):
            new_R = set()
            for r in R:
                for s in S:
                    new_R.add((r + s) % m)
            R = new_R
            if k >= 3 and 0 in R:
                return k
        
        return 10 # No cycle < 10
        
    # Now search for m, a
    # We want to minimize m.
    # Start from m = 243 (lower bound) up to 1280.
    
    # m must be even? No.
    # But if m is odd, -1 != 1.
    
    # Let's try m from 243 to 1280.
    # To speed up, we can skip m where gcd(m, a) is not 1?
    # No, the graph is always 4-regular.
    
    # Let's try.
    
    for m in range(243, 1281):
        if time.time() > deadline - 5:
            break
        
        # Try a from 2 to m//2
        # We can limit a to avoid symmetry?
        # a and m-a are equivalent.
        # 1 and m-1 are equivalent.
        
        for a in range(2, m // 2 + 1):
            if a == 1 or a == m - 1:
                continue
            if a == m // 2 and m % 2 == 0:
                continue # a = -a mod m, so only 2 distinct generators
            
            # Check if generators are distinct
            gens = {1, m-1, a, m-a}
            if len(gens) < 4:
                continue
            
            girth = girth_cayley(m, a)
            if girth >= 10:
                # Found a valid graph
                # Construct edges
                edges = []
                for u in range(m):
                    for v in [ (u+1)%m, (u-1)%m, (u+a)%m, (u-a)%m ]:
                        if u < v:
                            edges.append((u, v))
                
                return m, edges
        
    # Fallback to LUW if not found
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
