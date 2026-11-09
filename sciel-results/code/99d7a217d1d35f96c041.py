"""Rewritten program using AC-3 preprocessing and optimized Backtracking."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard: 120s limit, leave 10s margin
    START_TIME = time.time()
    DEADLINE = START_TIME + 110.0

    def instance(splits, seed, palette=5):
        from random import Random
        rng = Random(seed)
        # Initial cube faces
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
        
        # Degrees for heuristic
        degrees = [len(adj[i]) for i in range(n)]
        
        # Initial domains as bitmasks
        # Colors are 0..4, so bits 0..4
        domains = [0] * n
        for i, l in enumerate(lists):
            mask = 0
            for c in l:
                mask |= (1 << c)
            domains[i] = mask
        
        # AC-3 Preprocessing
        import collections
        queue = collections.deque()
        for u in range(n):
            for v in adj[u]:
                queue.append((u, v))
        
        def revise(u, v):
            du = domains[u]
            dv = domains[v]
            # If dv is a singleton, say {c}, then u cannot take c.
            # dv is singleton if dv & (dv-1) == 0 and dv != 0.
            if dv != 0 and (dv & (dv - 1)) == 0:
                # dv is a power of 2, i.e., a single color
                if du & dv:
                    domains[u] = du & ~dv
                    return True
            return False

        while queue:
            u, v = queue.popleft()
            if revise(u, v):
                if not domains[u]:
                    return None # Failure
                for x in adj[u]:
                    if x != v:
                        queue.append((x, u))
        
        # Check for empty domains after AC-3
        for i in range(n):
            if not domains[i]:
                return None

        # Backtracking Search
        color = [None] * n
        assigned = [False] * n
        
        # Helper to get list of colors from bitmask
        def get_colors(mask):
            colors = []
            for c in range(5):
                if mask & (1 << c):
                    colors.append(c)
            return colors

        def backtrack():
            if time.time() > DEADLINE:
                return False
            
            # MRV Heuristic: Select unassigned variable with smallest domain size
            min_size = float('inf')
            min_v = -1
            for v in range(n):
                if not assigned[v]:
                    # Count bits
                    size = bin(domains[v]).count('1')
                    if size < min_size:
                        min_size = size
                        min_v = v
                        if size == 0:
                            return False
                        if size == 1:
                            # Can't get better than 1, but check others? 
                            # MRV is primary. If multiple 1s, pick one.
                            # We can break early if we want, but scanning is fast.
                            pass
            
            if min_v == -1:
                return True # All assigned
            
            # Try each color for min_v
            colors = get_colors(domains[min_v])
            for c in colors:
                # Assign c to min_v
                color[min_v] = c
                assigned[min_v] = True
                domains[min_v] = (1 << c) # Lock it
                
                # Forward Checking: Remove c from neighbors' domains
                valid = True
                backup = []
                
                for nb in adj[min_v]:
                    if not assigned[nb]:
                        if domains[nb] & (1 << c):
                            # Remove c
                            old_val = domains[nb]
                            domains[nb] = old_val & ~(1 << c)
                            backup.append((nb, old_val))
                            if domains[nb] == 0:
                                valid = False
                                break
                
                if valid:
                    if backtrack():
                        return True
                    # Restore domains for failed branch
                    for nb, old_val in backup:
                        domains[nb] = old_val
                else:
                    # Restore domains for failed branch
                    for nb, old_val in backup:
                        domains[nb] = old_val
                
                # Unassign
                color[min_v] = None
                assigned[min_v] = False
                domains[min_v] = get_colors(domains[min_v]) # This is wrong, we need to restore the original domain before assignment
                # Actually, we didn't change domains[min_v] in a way that needs restoring in the loop logic above?
                # We set domains[min_v] = (1 << c). We need to restore it to the value it had before this iteration.
                # Let's fix the restore logic.
            
            return False

        # Re-implementing backtrack with proper restore for min_v domain
        # The previous draft had a flaw: `domains[min_v]` was modified and not restored correctly in the loop.
        
        def backtrack_v2():
            if time.time() > DEADLINE:
                return False
            
            min_size = float('inf')
            min_v = -1
            for v in range(n):
                if not assigned[v]:
                    size = bin(domains[v]).count('1')
                    if size < min_size:
                        min_size = size
                        min_v = v
                        if size == 0:
                            return False
            
            if min_v == -1:
                return True
            
            orig_domain_min_v = domains[min_v]
            colors = get_colors(orig_domain_min_v)
            
            for c in colors:
                color[min_v] = c
                assigned[min_v] = True
                domains[min_v] = (1 << c)
                
                valid = True
                backup = []
                
                for nb in adj[min_v]:
                    if not assigned[nb]:
                        if domains[nb] & (1 << c):
                            old_val = domains[nb]
                            domains[nb] = old_val & ~(1 << c)
                            backup.append((nb, old_val))
                            if domains[nb] == 0:
                                valid = False
                                break
                
                if valid:
                    if backtrack_v2():
                        return True
                    for nb, old_val in backup:
                        domains[nb] = old_val
                else:
                    for nb, old_val in backup:
                        domains[nb] = old_val
                
                color[min_v] = None
                assigned[min_v] = False
                domains[min_v] = orig_domain_min_v
            
            return False

        success = backtrack_v2()
        
        if success:
            return [int(c) for c in color]
        else:
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > DEADLINE:
            # Generate instance to know N
            n, edges, lists = instance(splits, seed)
            out.append([0] * n)
            continue
        
        n, edges, lists = instance(splits, seed)
        
        if time.time() > DEADLINE:
            out.append([0] * n)
            continue
            
        sol = solve_instance(n, edges, lists)
        out.append(sol)
        
    return out
# EVOLVE-BLOCK-END
