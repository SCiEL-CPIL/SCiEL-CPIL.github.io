import sys
import time

def construct():
    from random import Random
    
    # Increase recursion limit to be safe
    sys.setrecursionlimit(10000)

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_instance(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Pre-compute degrees
        degrees = [len(adj[i]) for i in range(n)]
        
        # Domains as bitmasks. Bit i is 1 if colour i is available.
        # Colours are 0..4, so bits 0..4.
        domains = [0] * n
        for i in range(n):
            mask = 0
            for c in lists[i]:
                mask |= (1 << c)
            domains[i] = mask
        
        assigned = [False] * n
        assignment = [-1] * n
        
        # Start time for deadline guard
        start_time = time.time()
        deadline = start_time + 110.0

        def check_deadline():
            return time.time() > deadline

        # AC-3 Implementation
        # Queue of arcs (u, v) to check
        # If domain of v changes, add (u, v) for all u in adj[v]
        def revise(u, v):
            # Optimization: Local variable access is faster
            dv = domains[v]
            if dv & (dv - 1) == 0: # domains[v] is a power of 2 (singleton)
                du = domains[u]
                if du & dv:
                    domains[u] = du & ~dv
                    return True
            return False

        def ac3(initial_arcs=None):
            if initial_arcs is None:
                initial_arcs = []
            queue = list(initial_arcs)
            while queue:
                u, v = queue.pop()
                if revise(u, v):
                    if domains[u] == 0:
                        return False
                    for w in adj[u]:
                        if w != v:
                            queue.append((w, u))
            return True

        def backtrack(depth):
            if check_deadline():
                return False
            
            if depth == n:
                return True
            
            # Local references for speed
            doms = domains
            asg = assigned
            adjs = adj
            degs = degrees
            asn = assignment
            
            # MRV Heuristic: Find unassigned vertex with smallest domain size
            min_size = 10 # Max possible is 3 (list size) or 5 (palette). 10 is safe.
            best_v = -1
            best_deg = -1
            for v in range(n):
                if not asg[v]:
                    # Count bits in doms[v]
                    size = bin(doms[v]).count('1')
                    if size < min_size or (size == min_size and degs[v] > best_deg):
                        min_size = size
                        best_v = v
                        best_deg = degs[v]
            
            if best_v == -1:
                return True
            
            if min_size == 0:
                return False
            
            v = best_v
            mask_v = doms[v]
            
            # Get list of colours in domain
            colors_in_domain = []
            temp_mask = mask_v
            while temp_mask:
                lsb = temp_mask & -temp_mask
                c = lsb.bit_length() - 1
                colors_in_domain.append(c)
                temp_mask ^= lsb
            
            # Calculate LCV score
            scores = [0, 0, 0, 0, 0]
            for u in adjs[v]:
                if not asg[u]:
                    mu = doms[u]
                    # Check each color in scores? No, just iterate bits of mu?
                    # Since palette is small (0..4), we can just check bits.
                    if mu & 1: scores[0] += 1
                    if mu & 2: scores[1] += 1
                    if mu & 4: scores[2] += 1
                    if mu & 8: scores[3] += 1
                    if mu & 16: scores[4] += 1
            
            # Sort colors by score (Lowest Constraining Value)
            colors_in_domain.sort(key=lambda c: scores[c])
            
            for c in colors_in_domain:
                if check_deadline():
                    return False
                
                # Assign c to v
                asg[v] = True
                asn[v] = c
                old_domain_v = mask_v
                mask_bit = 1 << c
                doms[v] = mask_bit
                
                changes = []
                conflict = False
                
                for u in adjs[v]:
                    if not asg[u]:
                        old_dom_u = doms[u]
                        if doms[u] & mask_bit:
                            doms[u] &= ~mask_bit
                            changes.append((u, old_dom_u))
                            if doms[u] == 0:
                                conflict = True
                                break
                
                if not conflict:
                    arcs = []
                    for u, _ in changes:
                        for w in adjs[u]:
                            if w != v:
                                arcs.append((w, u))
                    
                    if arcs:
                        if not ac3(arcs):
                            conflict = True
                
                if conflict:
                    for u, old_dom in changes:
                        doms[u] = old_dom
                    doms[v] = old_domain_v
                    asg[v] = False
                    asn[v] = -1
                    continue
                
                if backtrack(depth + 1):
                    return True
                
                for u, old_dom in changes:
                    doms[u] = old_dom
                doms[v] = old_domain_v
                asg[v] = False
                asn[v] = -1
            
            return False

        # Run solver
        if backtrack(0):
            return [int(assignment[i]) for i in range(n)]
        else:
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = solve_instance(n, edges, lists)
        out.append(coloring)
    return out
