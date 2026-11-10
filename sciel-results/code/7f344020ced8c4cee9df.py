import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Fallback: LUW construction
    def luw_construct(k, g):
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

    # Try Cayley graph on SL(2, 16)
    if k == 3 and g == 16:
        try:
            # GF(16) setup
            # x^4 + x + 1 is primitive
            def gf16_add(a, b): return a ^ b
            def gf16_mul(a, b):
                # Multiply in GF(16)
                # a, b are 4-bit integers
                res = 0
                for _ in range(4):
                    if b & 1:
                        res ^= a
                    b >>= 1
                    a <<= 1
                    if a & 16:
                        a ^= 0b10001 # x^4 + x + 1 -> x^4 = x + 1
                return res
            
            # Generate SL(2, 16)
            # Matrix: (a, b, c, d) with ad - bc = 1
            # In char 2, - is +. So ad + bc = 1.
            group = []
            group_map = {}
            for a in range(16):
                for b in range(16):
                    for c in range(16):
                        # d is determined by a*d + b*c = 1 => a*d = 1 + b*c
                        # If a != 0, d = (1 + b*c) * a^(-1)
                        # If a == 0, then b*c = 1, so b, c non-zero. d can be anything?
                        # No, if a=0, then bc=1. d is free?
                        # Wait, det = ad - bc = 1.
                        # If a=0, -bc = 1 => bc = 1.
                        # Then d is not constrained by determinant?
                        # No, the matrix is [0 b; c d]. Det = 0*d - b*c = -bc.
                        # In char 2, -bc = bc. So bc = 1.
                        # d is free?
                        # Yes, if a=0 and bc=1, d can be any value.
                        # But wait, is the group SL(2, q) defined by det=1?
                        # Yes.
                        # So if a=0, we need bc=1.
                        # If a!=0, d = (1 + bc) * a^(-1).
                        
                        # Let's just iterate all 16^4 and check det=1.
                        pass
            
            # Better: iterate a,b,c, compute d if possible.
            inv_map = {}
            for i in range(1, 16):
                # Find inverse of i in GF(16)
                for j in range(1, 16):
                    if gf16_mul(i, j) == 1:
                        inv_map[i] = j
                        break
            
            for a in range(16):
                for b in range(16):
                    for c in range(16):
                        if a == 0:
                            if gf16_mul(b, c) == 1:
                                for d in range(16):
                                    m = (a, b, c, d)
                                    if m not in group_map:
                                        group_map[m] = len(group)
                                        group.append(m)
                        else:
                            # a*d = 1 + b*c
                            rhs = gf16_add(1, gf16_mul(b, c))
                            d = gf16_mul(rhs, inv_map[a])
                            m = (a, b, c, d)
                            if m not in group_map:
                                group_map[m] = len(group)
                                group.append(m)
            
            n_group = len(group)
            
            # Multiplication table
            def mat_mul(m1, m2):
                a1, b1, c1, d1 = m1
                a2, b2, c2, d2 = m2
                a = gf16_add(gf16_mul(a1, a2), gf16_mul(b1, c2))
                b = gf16_add(gf16_mul(a1, b2), gf16_mul(b1, d2))
                c = gf16_add(gf16_mul(c1, a2), gf16_mul(d1, c2))
                d = gf16_add(gf16_mul(c1, b2), gf16_mul(d1, d2))
                return (a, b, c, d)
            
            # Precompute multiplication
            mul_table = {}
            for i, m1 in enumerate(group):
                for j, m2 in enumerate(group):
                    res = mat_mul(m1, m2)
                    mul_table[(i, j)] = group_map[res]
            
            # Find inverses
            inv_idx = [0] * n_group
            for i, m in enumerate(group):
                # Find j such that m * group[j] = identity
                # Identity is (1, 0, 0, 1)
                # We can just compute m^(-1)
                # Or search
                pass
            
            # Simpler: compute inverse matrix
            def mat_inv(m):
                a, b, c, d = m
                # det = 1
                # inv = [d -b; -c a] = [d b; c a] in char 2
                return (d, b, c, a)
            
            for i, m in enumerate(group):
                inv_m = mat_inv(m)
                inv_idx[i] = group_map[inv_m]
            
            # Search for generating set
            # S = {s1, s2, s3}
            # Case 1: s1^2=1, s2^2=1, s3^2=1
            # Case 2: s1^2=1, s2^(-1)=s3
            
            self_inverse = [i for i in range(n_group) if group[i][0] == group[i][3] and group[i][1] == group[i][2] and group[i][0] != 1 or (group[i][0]==1 and group[i][1]==0)] # Identity check
            # Identity is (1,0,0,1). Self inverse means M^2 = I.
            # In char 2, M^2 = I.
            
            # Let's just pick random elements and check girth.
            
            def get_girth(s1, s2, s3):
                # Build adjacency
                adj = [[] for _ in range(n_group)]
                for i in range(n_group):
                    n1 = mul_table[(i, s1)]
                    n2 = mul_table[(i, s2)]
                    n3 = mul_table[(i, s3)]
                    adj[i].append(n1)
                    adj[i].append(n2)
                    adj[i].append(n3)
                
                # Check for cycles < 16
                # BFS from each node
                # To speed up, we can just check if any cycle of length < 16 exists.
                # Since it's a Cayley graph, we can just check words in generators of length < 16 that equal identity.
                # Generators: s1, s2, s3.
                # We need to check all words of length 1 to 15.
                # 3^1 + 3^2 + ... + 3^15 is too large?
                # 3^15 = 14 million.
                # We can use BFS in the Cayley graph of the group generated by S.
                # But the group is SL(2,16).
                # We just need to check if the identity is reached by a non-trivial word of length < 16.
                
                # BFS from identity (index 0)
                # dist[0] = 0
                # visited = {0: 0}
                # queue = deque([0])
                # while queue:
                #   u = queue.popleft()
                #   d = dist[u]
                #   if d >= 7: continue # depth 7 covers cycles up to 14?
                #   # Cycle length L corresponds to path of length L-1 returning to start?
                #   # No, cycle of length L is a path of length L from v to v.
                #   # In BFS, if we reach a node at distance d, and it has a neighbor at distance d-1, that's a back edge.
                #   # Shortest cycle containing v is related to distances.
                #   # Simpler: just BFS and check if we hit a node already visited at a distance that implies a short cycle.
                #   # Actually, for girth check, we can just run BFS from identity up to depth 7.
                #   # If we find a word of length < 16 equal to identity, girth < 16.
                #   # Words of length L correspond to nodes in the Cayley graph.
                #   # If a node is reached by two different words of length < 8, we might have a short cycle?
                #   # No, we need to check if identity is reached by a word of length < 16.
                
                # Let's just enumerate words up to length 7?
                # If there is a cycle of length L < 16, there is a word of length L equal to identity.
                # Since the graph is undirected, we can assume the word doesn't immediately backtrack?
                # Non-backtracking walks.
                
                # Let's use the standard BFS girth check on the graph adj.
                # n=4080.
                # For each start node:
                #   BFS up to depth 7.
                #   If we find a neighbor that is already visited in the current BFS tree at a depth that creates a cycle < 16, return False.
                #   Specifically, if we are at node u with depth d, and we look at neighbor v:
                #   if v is visited at depth d_v:
                #      if d_v < d:
                #         cycle_len = d - d_v + 1
                #         if cycle_len < 16: return False
                #   Note: This detects cycles.
                
                for start in range(n_group):
                    visited = [-1] * n_group
                    visited[start] = 0
                    q = deque([(start, 0)])
                    while q:
                        u, d = q.popleft()
                        if d >= 7:
                            continue
                        for v in adj[u]:
                            if visited[v] == -1:
                                visited[v] = d + 1
                                q.append((v, d + 1))
                            else:
                                # If v is visited, check cycle length
                                # Only consider if v is not the parent?
                                # In BFS tree, if v is visited, the cycle length is d - visited[v] + 1
                                # But this might count the same cycle multiple times or long ones.
                                # We only care about < 16.
                                # Since d <= 7, visited[v] <= 7.
                                # Max cycle detected: 7 - 0 + 1 = 8?
                                # No, if start is 0, and we go 0->1->2->3->4->5->6->7->0 (cycle 8).
                                # d=7, visited[0]=0. cycle = 7-0+1=8.
                                # What about cycle 16?
                                # 0->...->7->...->15->0.
                                # BFS depth 7 won't see the connection from depth 7 to depth 0 if the path is long?
                                # Wait.
                                # If there is a cycle of length 16, there is a path of length 16 from v to v.
                                # This implies a path of length 8 from v to some midpoint?
                                # No.
                                # To detect girth 16, we need to ensure no cycle < 16.
                                # BFS up to depth 7 from each node detects all cycles of length <= 2*7+1 = 15.
                                # Because any cycle of length L <= 15 has a node on it such that the rest of the cycle is within distance 7?
                                # Yes. If L <= 15, pick any node on the cycle. The farthest node on the cycle is at distance floor(L/2) <= 7.
                                # So BFS depth 7 is sufficient to detect all cycles < 16.
                                
                                if visited[v] < d:
                                    cycle_len = d - visited[v] + 1
                                    if cycle_len < 16:
                                        return False
                return True
            
            # Random search
            random.seed(42)
            # Pick random self-inverse elements
            # M^2 = I
            self_inv_candidates = []
            for i in range(n_group):
                m = group[i]
                # Check M^2 = I
                mm = mat_mul(m, m)
                if mm == (1, 0, 0, 1):
                    if m != (1, 0, 0, 1):
                        self_inv_candidates.append(i)
            
            # Pick random elements
            all_elements = list(range(n_group))
            
            found = False
            for _ in range(1000): # Try 1000 times
                if time.time() > deadline:
                    break
                
                # Strategy 1: 3 self-inverse
                if len(self_inv_candidates) >= 3:
                    s1, s2, s3 = random.sample(self_inv_candidates, 3)
                    if get_girth(s1, s2, s3):
                        # Build edges
                        edges = []
                        for i in range(n_group):
                            n1 = mul_table[(i, s1)]
                            n2 = mul_table[(i, s2)]
                            n3 = mul_table[(i, s3)]
                            edges.append((i, n1))
                            edges.append((i, n2))
                            edges.append((i, n3))
                        return n_group, edges
                
                # Strategy 2: 1 self-inverse, 1 pair
                if len(self_inv_candidates) >= 1:
                    s1 = random.choice(self_inv_candidates)
                    s2 = random.choice(all_elements)
                    s3 = inv_idx[s2]
                    if s2 == s3:
                        continue
                    if get_girth(s1, s2, s3):
                        edges = []
                        for i in range(n_group):
                            n1 = mul_table[(i, s1)]
                            n2 = mul_table[(i, s2)]
                            n3 = mul_table[(i, s3)]
                            edges.append((i, n1))
                            edges.append((i, n2))
                            edges.append((i, n3))
                        return n_group, edges
        
        except Exception as e:
            pass
    
    return luw_construct(k, g)
