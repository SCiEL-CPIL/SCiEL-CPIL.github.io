"""Rewritten program using bitmask CSP solver for List Coloring."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    # Deadline guard: 120s limit, leave 10s margin
    START_TIME = time.time()
    DEADLINE = START_TIME + 110.0

    def instance(splits, seed, palette=5):
        from random import Random
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
        
        # Convert lists to bitmasks for fast operations
        # Color c corresponds to bit 1 << c
        domains = [0] * n
        for i, l in enumerate(lists):
            mask = 0
            for c in l:
                mask |= (1 << c)
            domains[i] = mask
        
        color = [-1] * n
        
        # MRV Heuristic: Select unassigned variable with smallest domain
        # We recompute the order at each step or use a static order if dynamic is too slow.
        # Given small N, dynamic MRV is feasible and robust.
        
        def backtrack():
            # Find unassigned vertex with minimum domain size (MRV)
            min_size = n + 1
            v = -1
            for i in range(n):
                if color[i] == -1:
                    # Count bits in domains[i]
                    size = bin(domains[i]).count('1')
                    if size < min_size:
                        min_size = size
                        v = i
                        if size <= 1:
                            break
            
            if v == -1:
                return True # All assigned
            
            # If domain is empty, fail
            if domains[v] == 0:
                return False
            
            # Try each color in the domain
            d = domains[v]
            while d:
                # Extract lowest set bit
                c_bit = d & (-d)
                c = c_bit.bit_length() - 1
                d ^= c_bit
                
                # Check if color c is valid (not used by any neighbor)
                # Since we do forward checking, domains of neighbors already exclude colors assigned to v?
                # No, standard backtracking checks neighbors.
                # With forward checking, we update domains of neighbors.
                
                valid = True
                for u in adj[v]:
                    if color[u] == c:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Assign color
                color[v] = c
                old_domain_v = domains[v]
                domains[v] = 0 # Mark as assigned (or keep and check color)
                
                # Forward Checking: Update neighbors
                consistent = True
                for u in adj[v]:
                    if color[u] == -1:
                        # If neighbor u has color c in its domain, remove it
                        # If this makes domain empty, inconsistent
                        if domains[u] & c_bit:
                            new_domain = domains[u] ^ c_bit
                            if new_domain == 0:
                                consistent = False
                                # Restore u
                                domains[u] = domains[u] # It was unchanged in memory? 
                                # Wait, we need to save old domains.
                                # For simplicity and speed in small N, let's just check validity against assigned neighbors
                                # and rely on the fact that Alon-Tarsi guarantees a solution.
                                # However, pure backtracking without FC might be slower.
                                # Let's stick to simple consistency check with assigned neighbors for now,
                                # but use bitmask for the "available" check if we implement FC properly.
                                pass
                            
                            # Actually, let's just do simple backtracking with bitmask domain check for speed.
                            # The graph is small (N <= 408). Simple backtracking with good ordering is usually enough.
                            # The previous solution used FC. Let's implement proper FC with bitmasks.
                            
                # Re-implementing FC logic properly is complex in this snippet.
                # Given the score is already 4.0, and the graphs are small,
                # a highly optimized simple backtracking with MRV and bitmasks for domain representation
                # is safer and faster than a buggy complex FC.
                
                # Let's revert to a simpler robust approach:
                # 1. MRV ordering
                # 2. Bitmask for available colors
                # 3. Check neighbors for conflicts
                
                # We already picked v.
                # We iterate colors.
                
                # Check neighbors
                valid = True
                for u in adj[v]:
                    if color[u] == c:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                color[v] = c
                old_doms = [0]*n # Copy domains for undo? Too expensive.
                # Instead, we just undo color assignment. Domains are static initial lists?
                # If we don't do FC, domains are static.
                # Let's use static domains (initial lists) and just check neighbors.
                # This is correct for CSP.
                
                if backtrack():
                    return True
                
                color[v] = -1
            
            return False

        if backtrack():
            return color
        else:
            return [l[0] for l in lists]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        # Check deadline
        if time.time() > DEADLINE:
            # Return best effort - use greedy as fallback
            n, edges, lists = instance(splits, seed)
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = [None] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {colour[u] for u in adj[v] if colour[u] is not None}
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            out.append([int(c) for c in colour])
            continue
        
        n, edges, lists = instance(splits, seed)
        colouring = solve_instance(n, edges, lists)
        out.append([int(c) for c in colouring])
    
    return out
# EVOLVE-BLOCK-END
