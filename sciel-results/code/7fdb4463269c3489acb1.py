import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    def remaining():
        return deadline - time.time()

    # ---------------- girth verification (fast, early exit) ----------------
    def check_girth(n, edges, g, time_limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        min_cycle = float('inf')
        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g:
                return min_cycle
        return min_cycle

    # ---------------- LUW construction ----------------
    def _luw_ops(q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        return ((lambda x, y: (x + y) % q),
                (lambda x, y: (x - y) % q),
                (lambda x, y: x * y % q))

    def generate_rules(d, variant=0, offset=0):
        base = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        if variant == 0:
            for c in range(5, d + 1):
                r, c0 = ((c - 5 + offset) % 4), c - ((c - 5 + offset) % 4)
                base[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        elif variant == 1:
            for c in range(5, d + 1):
                base[c] = (1, c - 2) if (c + offset) % 2 == 0 else (c - 1, 1)
        elif variant == 2:
            for c in range(5, d + 1):
                if (c + offset) % 3 == 0:
                    base[c] = (1, c - 3)
                elif (c + offset) % 3 == 1:
                    base[c] = (c - 1, 1)
                else:
                    base[c] = (c - 2, 1)
        elif variant == 3:
            for c in range(5, d + 1):
                base[c] = (1, c - (1 + (c + offset) % 2))
        elif variant == 4:
            for c in range(5, d + 1):
                base[c] = [(1, c - 1), (c - 1, 1), (1, c - 2), (c - 2, 1)][(c + offset) % 4]
        elif variant == 5:
            for c in range(5, d + 1):
                base[c] = (c - 2, 1) if (c + offset) % 2 == 0 else (1, c - 1)
        elif variant == 6:
            for c in range(5, d + 1):
                base[c] = (1, c - 1) if (c + offset) % 3 == 0 else (c - 2, 1)
        elif variant == 7:
            for c in range(5, d + 1):
                base[c] = (c - 1, 1) if (c + offset) % 4 == 0 else (1, c - 2)
        else:
            for c in range(5, d + 1):
                r, c0 = ((c - 5 + offset) % 4), c - ((c - 5 + offset) % 4)
                base[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        return base

    def build_luw(q, d, rule, time_limit):
        add, sub, mul = _luw_ops(q)

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
            if time.time() > time_limit:
                return None
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def luw_construct(k, g):
        q = k
        d = max(3, (g - 2) // 2)
        best = None
        best_n = float('inf')

        for variant in range(8):
            for offset in range(6):
                if remaining() < 12:
                    break
                rule = generate_rules(d, variant, offset)
                res = build_luw(q, d, rule, deadline - 12)
                if res is None:
                    continue
                n, edges = res
                if n >= best_n:
                    continue
                gc = check_girth(n, edges, g, deadline - 8)
                if gc == -1:
                    continue
                if gc >= g:
                    best, best_n = (n, edges), n
                    break

        if best is None:
            for d_try in range(d + 1, d + 10):
                if remaining() < 10:
                    break
                for variant in range(3):
                    for offset in range(3):
                        if remaining() < 5:
                            break
                        rule = generate_rules(d_try, variant, offset)
                        res = build_luw(q, d_try, rule, deadline - 10)
                        if res is None:
                            continue
                        n, edges = res
                        gc = check_girth(n, edges, g, deadline - 5)
                        if gc == -1:
                            continue
                        if gc >= g:
                            best = (n, edges)
                            break
                    if best is not None:
                        break
                if best is not None:
                    break

        if best is None:
            rule = generate_rules(d, 0, 0)
            res = build_luw(q, d, rule, deadline - 2)
            if res is not None:
                best = res
        return best

    # ---------------- Fixed Ramanujan cubic graph construction ----------------
    def build_ramanujan_cubic(q):
        """Build cubic Ramanujan graph using PSL(2,q) Cayley graph with 3 involutions."""
        if q < 5 or (q % 4 != 1 and q % 4 != 3):
            return None
        # Elements of PSL(2,q): pairs (a,b) with a^2 - b = 1 (for q odd)
        # Actually use matrices mod q with det=1, identifying M and -M
        # Simpler: use the standard representation
        # For q ≡ 1 mod 4: involutions are matrices of form [[a,b],[c,-a]] with a^2+bc = -1
        # Let's use a direct approach: vertices are (u,v) in F_q^2 / ~ where (u,v)~(-u,-v)
        # and the three generators are specific involutions.
        
        # Standard construction: vertices = elements of PSL(2,q)
        # Represent element as (a,b,c) with a^2 + bc = 1 (unipotent parametrization won't work for all)
        # Better: use 2x2 matrices mod q with det=1, quotient by ±I
        # Number of elements: q(q-1) if q odd
        
        # Involutions in PSL(2,q): matrices M with M^2 = ±I and M ≠ ±I
        # For q odd: M = [[a,b],[c,-a]] with a^2+bc = -1 gives M^2 = -(a^2+bc)I = I in PSL
        # So involutions: a^2 + bc = -1 mod q, not all zero, mod ±I
        
        # Three standard involutions:
        # s1: [[0,1],[1,0]] -> a=0,b=1,c=1: 0+1=1≠-1... 
        # Let me use: s1=[[0,1],[q-1,0]], s2=[[1,1],[q-1,-1]], s3=[[1,0],[q-1,1]]
        # Check s1^2: [[0,1],[q-1,0]]^2 = [[q-1,0],[0,q-1]] = -I, so in PSL it's identity. Good.
        # s2^2: [[1,1],[q-1,-1]]^2 = [[1+1-q+1, 1-1],[q-1-q+1, q-1+1]] = [[q+1,0],[0,q]] hmm not right.
        
        # Let me use a cleaner approach: 
        # Vertices: (x,y) in F_q^2, identify (x,y)~(-x,-y). Total: q(q-1)/2... no, q^2/2.
        # Actually PSL(2,q) has order q(q^2-1)/2 for q odd.
        
        # Let me just do BFS on the group directly.
        def mat_mul(a, b):
            return ((a[0]*b[0] + a[1]*b[2]) % q, (a[0]*b[1] + a[1]*b[3]) % q,
                    (a[2]*b[0] + a[3]*b[2]) % q, (a[2]*b[1] + a[3]*b[3]) % q)
        
        def mat_det(a):
            return (a[0]*a[3] - a[1]*a[2]) % q
        
        def mat_inv(a):
            d = mat_det(a)
            if d == 0:
                return None
            invd = pow(d, q - 2, q)
            return ((a[3]*invd) % q, (-a[1]*invd) % q, (-a[2]*invd) % q, (a[0]*invd) % q)
        
        def canon(a):
            # Canonical representative: make top-left positive, or if 0, make top-right positive
            if a[0] > q//2 or (a[0] == q//2 and a[1] > q//2):
                return ((q-a[0])%q, (q-a[1])%q, (q-a[2])%q, (q-a[3])%q)
            return a
        
        I = (1, 0, 0, 1)
        
        # Find 3 involutions (M^2 = -I mod q, i.e., in PSL they square to I)
        # M = [[a,b],[c,-a]]: M^2 = [[a^2-bc, 0],[0, a^2-bc]]... wait
        # [[a,b],[c,-a]] * [[a,b],[c,-a]] = [[a^2+bc, ab-ab],[ca-ac, cb+a^2]] = [[a^2+bc,0],[0,a^2+bc]]
        # So M^2 = (a^2+bc)I. For M^2 = -I: a^2+bc = -1 mod q.
        # Also M ≠ ±I, so not (a=±1, b=c=0) and not (a=0, b=0, c=0)
        
        involutions = []
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    if (a*a + b*c) % q != (q-1) % q:
                        continue
                    M = (a, b, c, (q-a)%q)
                    if M == I or M == (q-1, 0, 0, q-1):
                        continue
                    # Check it's truly an involution in PSL: canon(M^2) == canon(I)
                    M2 = mat_mul(M, M)
                    if canon(M2) != canon(I):
                        continue
                    # Avoid duplicates under ±
                    if any(canon(M) == canon(inv) for inv in involutions):
                        continue
                    involutions.append(M)
                    if len(involutions) >= 20:
                        break
                if len(involutions) >= 20:
                    break
            if len(involutions) >= 20:
                break
        
        if len(involutions) < 3:
            return None
        
        # Pick 3 involutions that generate PSL(2,q)
        # Try combinations
        best_gens = None
        for i in range(min(5, len(involutions))):
            for j in range(i+1, min(10, len(involutions))):
                for l in range(j+1, min(15, len(involutions))):
                    gens = [involutions[i], involutions[j], involutions[l]]
                    # BFS to check generation
                    visited = {canon(I): 0}
                    queue = deque([canon(I)])
                    order = [canon(I)]
                    while queue:
                        v = queue.popleft()
                        for gen in gens:
                            # Right multiply: v * gen
                            prod = mat_mul(v, gen)
                            cp = canon(prod)
                            if cp not in visited:
                                visited[cp] = len(order)
                                order.append(cp)
                                queue.append(cp)
                    if len(order) == q*(q*q-1)//2:
                        best_gens = gens
                        break
                if best_gens:
                    break
            if best_gens:
                break
        
        if best_gens is None:
            # Just use first 3
            best_gens = involutions[:3]
        
        # Build Cayley graph
        gens = best_gens
        visited = {canon(I): 0}
        queue = deque([canon(I)])
        order = [canon(I)]
        edges = []
        while queue:
            v = queue.popleft()
            vi = visited[v]
            for gen in gens:
                prod = mat_mul(v, gen)
                cp = canon(prod)
                if cp not in visited:
                    visited[cp] = len(order)
                    order.append(cp)
                    queue.append(cp)
                if vi < visited[cp]:
                    edges.append((vi, visited[cp]))
        
        n = len(order)
        # Verify 3-regular
        deg = [0] * n
        for i, j in edges:
            deg[i] += 1
            deg[j] += 1
        if any(d != 3 for d in deg):
            return None
        
        return n, edges

    # ---------------- Greedy random construction with cycle elimination ----------------
    def greedy_construct(k, g, time_limit):
        """Build a random k-regular graph and try to eliminate short cycles via edge swaps."""
        if k != 3:
            return None
        
        # Moore bound for (3,16): 2 + 3*sum_{i=0}^{6} 2^i = 2+3*(127) = 383... 
        # Actually Moore bound: 1 + k*sum_{i=0}^{(g-2)/2 - 1} (k-1)^i for even g
        # For k=3, g=16: 1 + 3*(1 + 2 + 4 + 8 + 16 + 32 + 64) = 1 + 3*127 = 382
        # Try sizes from Moore bound upward
        moore = 1 + k * sum((k-1)**i for i in range((g-2)//2))
        
        for n in range(moore, moore + 200):
            if n % 2 != 0:  # 3-regular needs even number of vertices
                continue
            if time.time() > time_limit:
                break
            
            # Build random 3-regular graph
            for attempt in range(3):
                if time.time() > time_limit:
                    break
                # Configuration model
                half_edges = list(range(n)) + list(range(n)) + list(range(n))
                random.shuffle(half_edges)
                edges = set()
                valid = True
                for i in range(0, len(half_edges), 2):
                    u, v = half_edges[i], half_edges[i+1]
                    if u == v:
                        valid = False
                        break
                    if (min(u,v), max(u,v)) in edges:
                        valid = False
                        break
                    edges.add((min(u,v), max(u,v)))
                if not valid:
                    continue
                
                edge_list = list(edges)
                gc = check_girth(n, edge_list, g, time_limit)
                if gc == -1:
                    break
                if gc >= g:
                    return n, edge_list
                
                # Try edge swaps to eliminate short cycles
                # Find a short cycle and try to break it
                adj = [[] for _ in range(n)]
                for u, v in edge_list:
                    adj[u].append(v)
                    adj[v].append(u)
                
                improved = True
                max_iters = 5000
                iters = 0
                while improved and iters < max_iters and time.time() < time_limit:
                    improved = False
                    iters += 1
                    # Find shortest cycle
                    min_cyc = float('inf')
                    min_cyc_edges = None
                    for s in range(n):
                        if min_cyc <= g - 1:
                            break
                        dist = [-1] * n
                        parent = [-1] * n
                        dist[s] = 0
                        q2 = [s]
                        qi = 0
                        while qi < len(q2):
                            node = q2[qi]
                            qi += 1
                            for nb in adj[node]:
                                if nb == parent[node]:
                                    continue
                                if dist[nb] != -1:
                                    cl = dist[node] + dist[nb] + 1
                                    if cl < min_cyc:
                                        min_cyc = cl
                                        # Reconstruct cycle
                                        # This is expensive; skip for now
                                else:
                                    dist[nb] = dist[node] + 1
                                    parent[nb] = node
                                    q2.append(nb)
                        if min_cyc <= g - 1:
                            break
                    
                    if min_cyc >= g:
                        break
                    
                    # Try random edge swaps
                    for _ in range(100):
                        u1, v1 = random.choice(edge_list)
                        u2, v2 = random.choice(edge_list)
                        # Try swapping: remove (u1,v1) and (u2,v2), add (u1,u2) and (v1,v2)
                        # or (u1,v2) and (v1,u2)
                        for new1, new2 in [((u1,u2),(v1,v2)), ((u1,v2),(v1,u2))]:
                            if new1[0] == new1[1] or new2[0] == new2[1]:
                                continue
                            if (min(new1[0],new1[1]), max(new1[0],new1[1])) in edges:
                                continue
                            if (min(new2[0],new2[1]), max(new2[0],new2[1])) in edges:
                                continue
                            # Check degree preservation (always true for swap)
                            # Apply swap
                            edges.discard((min(u1,v1), max(u1,v1)))
                            edges.discard((min(u2,v2), max(u2,v2)))
                            edges.add((min(new1[0],new1[1]), max(new1[0],new1[1])))
                            edges.add((min(new2[0],new2[1]), max(new2[0],new2[1])))
                            edge_list = list(edges)
                            # Rebuild adj
                            adj = [[] for _ in range(n)]
                            for a, b in edge_list:
                                adj[a].append(b)
                                adj[b].append(a)
                            gc2 = check_girth(n, edge_list, g, time_limit)
                            if gc2 == -1:
                                break
                            if gc2 >= g:
                                return n, edge_list
                            improved = True
                            break
                        if improved:
                            break
                    if time.time() > time_limit:
                        break
                if time.time() > time_limit:
                    break
        
        return None

    # ---------------- main ----------------
    if k != 3:
        res = luw_construct(k, g)
        return res if res is not None else (0, [])

    # k == 3: try multiple strategies
    best = None
    best_n = float('inf')

    # Strategy 1: Ramanujan graphs (properly constructed)
    for q in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]:
        if remaining() < 8:
            break
        res = build_ramanujan_cubic(q)
        if res is None:
            continue
        n, edges = res
        if n >= best_n:
            continue
        gc = check_girth(n, edges, g, deadline - 5)
        if gc == -1:
            continue
        if gc >= g:
            best, best_n = (n, edges), n

    # Strategy 2: LUW construction
    if remaining() > 15:
        res = luw_construct(k, g)
        if res is not None:
            n, edges = res
            if n < best_n:
                best, best_n = res, n

    # Strategy 3: Greedy random (only if time remains)
    if remaining() > 20 and (best is None or best_n > 500):
        res = greedy_construct(k, g, deadline - 5)
        if res is not None:
            n, edges = res
            if n < best_n:
                best, best_n = res, n

    if best is not None:
        return best

    # Guaranteed fallback: LUW
    res = luw_construct(k, g)
    return res if res is not None else (0, [])
