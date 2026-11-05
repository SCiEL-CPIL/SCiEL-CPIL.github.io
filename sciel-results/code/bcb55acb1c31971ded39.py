import random
import itertools
from collections import deque
import time

def construct(k, g):
    # Moore bound check
    def moore_bound(k, g):
        if g % 2 == 0:
            return 2 * sum((k-1)**i for i in range(g//2))
        else:
            return 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    mb = moore_bound(k, g)
    
    # Fallback: LUW Construction
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
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

    # If k, g are small enough for S_n search
    # For k=5, g=9, try S_6
    if k == 5 and g == 9:
        # Try to find a Cayley graph on S_6
        n_group = 6
        # Generate involutions in S_n_group
        def get_involutions(n):
            invs = []
            # Permutations as tuples
            # Identity
            id_perm = tuple(range(n))
            for perm in itertools.permutations(range(n)):
                # Check if involution: perm(perm(i)) == i for all i
                # And not identity
                if perm == id_perm:
                    continue
                is_inv = True
                for i in range(n):
                    if perm[perm[i]] != i:
                        is_inv = False
                        break
                if is_inv:
                    invs.append(perm)
            return invs

        def compose(p, q):
            # p * q means apply q then p
            return tuple(p[q[i]] for i in range(len(p)))

        def check_girth(generators, n, girth_target):
            # BFS from identity
            id_perm = tuple(range(n))
            dist = {id_perm: 0}
            parent = {id_perm: None}
            queue = deque([id_perm])
            min_cycle = float('inf')
            
            while queue:
                v = queue.popleft()
                dv = dist[v]
                if dv >= girth_target // 2: 
                    # We can stop early if we've explored enough depth?
                    # Actually, we need to check all edges for cycles.
                    # But if dv + dist[u] + 1 >= girth_target for all neighbors, we are good?
                    # No, we need to ensure NO cycle < girth_target.
                    # So we must check all.
                    pass
                
                for s in generators:
                    u = compose(s, v) # s * v
                    if u not in dist:
                        dist[u] = dv + 1
                        parent[u] = v
                        queue.append(u)
                    else:
                        # If u is visited, we might have a cycle
                        # Cycle length = dist[v] + dist[u] + 1
                        # But only if u is not the parent of v
                        if u != parent[v]:
                            cycle_len = dv + dist[u] + 1
                            if cycle_len < min_cycle:
                                min_cycle = cycle_len
                            if min_cycle < girth_target:
                                return False
            return True

        invs = get_involutions(n_group)
        # We need k=5 generators
        # Random search
        found = False
        start_time = time.time()
        attempts = 0
        while time.time() - start_time < 100: # 100 seconds limit
            attempts += 1
            if attempts % 1000 == 0:
                # Optional: print status? No, just continue
                pass
            # Pick 5 random distinct involutions
            # To ensure they generate the group, we might need to check generation
            # But first check girth, it's faster?
            # Girth check is O(N*k). Generation check is O(N) or less.
            # Let's check generation first to avoid wasting time on small subgroups?
            # Actually, if they don't generate S_6, the graph is disconnected.
            # The task requires a simple graph, usually implying connected?
            # "Construct a k-regular graph...". Usually cages are connected.
            # If disconnected, the girth is the min girth of components.
            # But n is the number of vertices in the component?
            # "n must lie between the Moore bound and max_vertices".
            # Moore bound applies to connected graphs.
            # So we should ensure connectivity (generation of whole group).
            
            gen_set = random.sample(invs, k)
            
            # Check if they generate S_6
            # BFS on group elements using generators
            visited = {id_perm}
            queue = deque([id_perm])
            while queue:
                v = queue.popleft()
                for s in gen_set:
                    u = compose(s, v)
                    if u not in visited:
                        visited.add(u)
                        queue.append(u)
                    u2 = compose(v, s) # v * s? 
                    # Wait, Cayley graph is usually right or left regular.
                    # If we use left multiplication by generators:
                    # Neighbors of v are s*v.
                    # So we only need s*v.
                    # My BFS above used s*v.
                    # So visited size should be n_group! if it generates.
            
            if len(visited) < n_group:
                continue
                
            # Check girth
            if check_girth(gen_set, n_group, g):
                # Found!
                # Build edges
                # Map permutations to indices 0..n-1
                perm_to_idx = {p: i for i, p in enumerate(visited)}
                edges = []
                for v in visited:
                    idx_v = perm_to_idx[v]
                    for s in gen_set:
                        u = compose(s, v)
                        idx_u = perm_to_idx[u]
                        if idx_v < idx_u:
                            edges.append((idx_v, idx_u))
                return n_group, edges
                
        # If not found in S_6, try A_7? Or S_7?
        # Or just fall back to LUW
        pass

    # Fallback to LUW
    return luw_construct(k, g)
