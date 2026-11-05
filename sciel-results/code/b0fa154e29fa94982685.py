import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Fallback: L-U-W
    def luw_fallback():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        # ... (copy of original L-U-W logic) ...
        return len(order), edges

    # Try Benson-style on UT(3, q)
    # q must be power of 2 for even characteristic to have self-inverse elements easily
    # Try q=8 (N=512)
    # Try q=16 (N=4096)
    
    # Implement GF(2^m)
    def make_gf(m):
        # Irreducible polynomial x^m + x + 1 for m=3,4?
        # m=3: x^3+x+1
        # m=4: x^4+x+1
        # We'll use a simple table
        if m == 3:
            # x^3 = x + 1
            mul_table = [[0]*8 for _ in range(8)]
            for a in range(8):
                for b in range(8):
                    res = 0
                    x = b
                    for i in range(m):
                        if a & (1 << i):
                            res ^= x
                        # x = x * x (shift left, reduce)
                        # shift left by 1
                        x <<= 1
                        if x & (1 << m):
                            x ^= (1 << m) + 1 # x^3+x+1 -> x^3 = x+1
                            x &= (1 << m) - 1
                    mul_table[a][b] = res
            return mul_table
        elif m == 4:
            # x^4 = x + 1
            mul_table = [[0]*16 for _ in range(16)]
            for a in range(16):
                for b in range(16):
                    res = 0
                    x = b
                    for i in range(m):
                        if a & (1 << i):
                            res ^= x
                        x <<= 1
                        if x & (1 << m):
                            x ^= (1 << m) + 1
                            x &= (1 << m) - 1
                    mul_table[a][b] = res
            return mul_table
        else:
            return None

    def try_ut3(m):
        if time.time() > deadline: return None
        mul = make_gf(m)
        if not mul: return None
        q = 1 << m
        # Group elements: (a, b, c)
        # Index: a * q*q + b * q + c
        n = q**3
        # Precompute inverses and products?
        # Product: (a,b,c) * (a',b',c') = (a^a', b^b', c^c'^ (a*b'))
        # Inverse: (a,b,c)^-1 = (a,b,c^(a*b))
        
        def prod(p, q_):
            a, b, c = p
            a2, b2, c2 = q_
            return (a ^ a2, b ^ b2, c ^ c2 ^ mul[a][b2])
        
        def inv(p):
            a, b, c = p
            return (a, b, c ^ mul[a][b])
        
        def to_idx(p):
            a, b, c = p
            return a * q * q + b * q + c
        
        def from_idx(idx):
            c = idx % q
            b = (idx // q) % q
            a = idx // (q * q)
            return (a, b, c)

        # Random search for generators
        # 1 self-inverse, 2 pairs
        for _ in range(5000):
            if time.time() > deadline: return None
            # Pick random self-inverse
            # ab=0
            a = random.randint(0, q-1)
            b = random.randint(0, q-1)
            if mul[a][b] != 0: continue
            c = random.randint(0, q-1)
            s0 = (a, b, c)
            
            s1 = (random.randint(0, q-1), random.randint(0, q-1), random.randint(0, q-1))
            s2 = (random.randint(0, q-1), random.randint(0, q-1), random.randint(0, q-1))
            
            S = [s0, s1, inv(s1), s2, inv(s2)]
            # Ensure distinct and non-identity
            if len(set(S)) < 5: continue
            if (0,0,0) in S: continue
            
            # Check girth
            # BFS from 0
            visited = [False] * n
            visited[0] = True
            queue = [(0, 0)] # (node, dist)
            # Non-backtracking: store (node, prev)
            # For Cayley graph, state is (node, prev_gen_index)
            # Simpler: BFS on (node, prev_node)
            # But n is small (512 or 4096)
            
            min_girth = float('inf')
            # Use queue of (node, dist, prev)
            q_bfs = [(0, 0, -1)]
            visited_state = set()
            visited_state.add((0, -1))
            
            while q_bfs:
                node, dist, prev = q_bfs.pop(0)
                if dist >= g:
                    continue
                for i, s in enumerate(S):
                    next_node = prod(from_idx(node), s)
                    ni = to_idx(next_node)
                    if ni == prev:
                        continue
                    if (ni, i) in visited_state:
                        # Cycle found?
                        # If ni was visited at dist d2, and we are at dist+1
                        # We need to track dist for each (node, prev)
                        pass
                    else:
                        visited_state.add((ni, i))
                        q_bfs.append((ni, dist + 1, node))
            
            # This BFS is too slow/complex for 5000 trials.
            # Simplify: Just check if girth >= g using standard BFS on nodes, 
            # but Cayley graphs are vertex-transitive.
            # Girth is the length of the shortest cycle.
            # We can do BFS from 0, and check if we revisit a node.
            # But we must avoid backtracking.
            
            # Fast check:
            # dist[node] = min dist
            # If we find an edge (u, v) such that dist[u] + 1 + dist[v] < g?
            # No, that's for bipartite.
            
            # Let's just do a limited BFS depth g-1.
            # If no cycle is found in depth g-1, girth >= g.
            # State: (node, prev_node)
            visited = set()
            visited.add((0, -1))
            current = [(0, -1)]
            for d in range(g):
                next_current = []
                for node, prev in current:
                    for s in S:
                        next_node = prod(from_idx(node), s)
                        ni = to_idx(next_node)
                        if ni == prev:
                            continue
                        if (ni, node) in visited:
                            # Cycle of length d+1
                            # If d+1 < g, fail
                            return None
                        visited.add((ni, node))
                        next_current.append((ni, node))
                current = next_current
            # If we reach here, no cycle < g found
            # Construct graph
            edges = []
            for u in range(n):
                for s in S:
                    v = to_idx(prod(from_idx(u), s))
                    if u < v:
                        edges.append((u, v))
            return n, edges

    # Try q=8
    res = try_ut3(3)
    if res:
        return res
        
    # Try q=16
    if time.time() < deadline:
        res = try_ut3(4)
        if res:
            return res

    # Fallback
    return luw_fallback()
