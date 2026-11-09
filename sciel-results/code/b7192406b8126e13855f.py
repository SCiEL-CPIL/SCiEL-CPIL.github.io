"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    from random import Random
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

    def solve_csp(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Initialize domains as bitmasks
        # Colors are 0..4, so we need 5 bits.
        # domain[v] is an integer where bit c is 1 if color c is possible.
        domains = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            domains[v] = mask
            
        # AC-3 implementation
        # Queue of (u, v) pairs to check.
        # We check if domain[u] is consistent with domain[v].
        # If we remove a value from domain[u], we must re-check all neighbors of u (except v? No, all neighbors).
        
        def ac3():
            # Initialize queue with all arcs
            queue = []
            for u in range(n):
                for v in adj[u]:
                    queue.append((u, v))
            
            while queue:
                u, v = queue.pop()
                # Check if any value in domain[u] is incompatible with domain[v]
                # Incompatible means: if u takes color c, v cannot take c.
                # So if c is in domain[u] AND c is in domain[v], that's fine?
                # Wait, constraint is color[u] != color[v].
                # So if u takes c, v cannot take c.
                # Thus, if c is in domain[u], we must ensure v does NOT have ONLY c? No.
                # We remove c from domain[u] if there is NO value in domain[v] that is != c.
                # i.e., if domain[v] is a subset of {c}.
                # Since domains have size >= 1, domain[v] is {c} only if domain[v] == (1 << c).
                
                # More simply: 
                # For each c in domain[u]:
                #   if there exists d in domain[v] such that d != c, then c is supported.
                #   if no such d exists, remove c from domain[u].
                
                # Optimization:
                # If domain[u] has a color c such that domain[v] == (1 << c), then c must be removed from domain[u].
                # Actually, if domain[v] has only one color c, then u cannot have c.
                # If domain[v] has multiple colors, u can have any color in domain[u] as long as it's not the ONLY color in domain[v]?
                # No. If domain[v] = {0, 1} and domain[u] = {0}, u can be 0 because v can be 1.
                # So c is supported if domain[v] is not a subset of {c}.
                # i.e., if domain[v] & ~(1 << c) != 0.
                
                new_domain_u = domains[u]
                changed = False
                # Iterate through bits in domain[u]
                temp = domains[u]
                while temp:
                    lsb = temp & -temp
                    c = lsb.bit_length() - 1
                    # Check if c is supported by v
                    # Supported if there is some d in domain[v] with d != c
                    # i.e., domain[v] has a bit set other than c
                    if domains[v] & ~(1 << c) == 0:
                        # c is not supported
                        new_domain_u &= ~lsb
                        changed = True
                    temp &= temp - 1
                    
                if changed:
                    domains[u] = new_domain_u
                    if new_domain_u == 0:
                        return False # Failure
                    # Re-add arcs for all neighbors of u except v? 
                    # Standard AC-3: add (w, u) for all w in adj[u] if w != v
                    for w in adj[u]:
                        if w != v:
                            queue.append((w, u))
            return True

        if not ac3():
            return None

        # Backtracking Search
        # State: domains (modified in place, need to restore or copy)
        # To be efficient, we'll pass a copy of domains or use a stack of changes.
        # Given small n, copying is fine.
        
        def backtrack(domains_copy):
            # Find unassigned variable with MRV
            unassigned = [v for v in range(n) if domains_copy[v] != 0 and (domains_copy[v] & (domains_copy[v]-1)) != 0]
            # Wait, assigned variables have domain size 1? 
            # In this formulation, we don't explicitly "assign" by setting domain to singleton immediately in the state passed to recursion?
            # Yes, we do. When we pick a color for v, we set domains[v] to (1 << c).
            # So assigned vars have domain size 1.
            # Unassigned vars have domain size > 1.
            
            if not unassigned:
                # All assigned
                return True

            # MRV: min domain size
            min_v = -1
            min_size = float('inf')
            for v in unassigned:
                size = bin(domains_copy[v]).count('1')
                if size < min_size:
                    min_size = size
                    min_v = v
            
            # Get colors for min_v
            colors = []
            temp = domains_copy[min_v]
            while temp:
                lsb = temp & -temp
                c = lsb.bit_length() - 1
                colors.append(c)
                temp &= temp - 1
                
            # Try each color
            for c in colors:
                # Check time limit
                if time.time() > deadline:
                    return False
                
                new_domains = domains_copy[:]
                new_domains[min_v] = (1 << c)
                
                # Propagate AC-3
                # We only need to propagate from min_v
                # Create a local AC-3 that starts with arcs (w, min_v) for w in adj[min_v]
                queue = []
                for w in adj[min_v]:
                    queue.append((w, min_v))
                    
                consistent = True
                while queue:
                    u, v = queue.pop()
                    # v is the one that changed (or was just assigned)
                    # Check consistency of u with v
                    # u must have a value supported by v
                    new_domain_u = new_domains[u]
                    changed = False
                    temp = new_domains[u]
                    while temp:
                        lsb = temp & -temp
                        cu = lsb.bit_length() - 1
                        # Check if cu is supported by v
                        # Supported if new_domains[v] has a bit set other than cu
                        if new_domains[v] & ~(1 << cu) == 0:
                            new_domain_u &= ~lsb
                            changed = True
                        temp &= temp - 1
                        
                    if changed:
                        new_domains[u] = new_domain_u
                        if new_domain_u == 0:
                            consistent = False
                            break
                        for w in adj[u]:
                            if w != v:
                                queue.append((w, u))
                
                if consistent:
                    if backtrack(new_domains):
                        return True
            
            return False

        # Start backtracking
        initial_domains = domains[:]
        if backtrack(initial_domains):
            # Extract solution
            solution = [0] * n
            for v in range(n):
                # Find the set bit
                c = initial_domains[v].bit_length() - 1
                # Verify it's a power of 2 (should be, as we assigned singletons)
                # If not, something went wrong, but backtrack ensures success
                solution[v] = c
            return solution
        else:
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # Fallback: return dummy valid structure to avoid crash, though score will be 0
            # We need to know n to return correct length.
            # Re-generate to get n? Or just break?
            # If we break, out is shorter than 4, which is invalid.
            # So we must generate all instances and try to solve.
            # If we run out of time, we return a "best effort" which is likely invalid coloring.
            pass
            
        n, edges, lists = instance(splits, seed)
        
        sol = solve_csp(n, edges, lists)
        
        if sol is None:
            # If solver failed (timeout or unsolvable), we must return something.
            # Returning a list of 0s will likely be invalid, but it's the only option to keep structure valid.
            # However, since we expect success, this is a safety net.
            sol = [0] * n
            
        # Ensure types are Python int
        sol = [int(c) for c in sol]
        out.append(sol)
        
    return out
# EVOLVE-BLOCK-END
