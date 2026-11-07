import time
import random
import sys

def construct():
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

    def solve(n, edges, lists, seed, time_limit=10.0):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Precompute static ordering by degree (descending) to guide backtracking
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        pos = [0] * n
        for i, v in enumerate(order):
            pos[v] = i
        
        # Initialize domains as bitmasks for efficient forward checking
        init_masks = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            init_masks[v] = mask
        
        color = [-1] * n
        domain_masks = init_masks[:]
        trail = []
        
        start_time = time.monotonic()
        bt_deadline = start_time + time_limit * 0.85
        sys.setrecursionlimit(10000)
        
        # MRV heuristic: dynamically select the unassigned vertex with smallest domain
        def get_mrv_vertex():
            min_size = 6
            best_v = -1
            for v in range(n):
                if color[v] == -1:
                    size = bin(domain_masks[v]).count('1')
                    if size < min_size:
                        min_size = size
                        best_v = v
                        if size <= 1:
                            break
            return best_v
        
        def backtrack(idx):
            # Time check
            if time.monotonic() > bt_deadline:
                return False
            
            if idx == n:
                return True
            
            # Use MRV to pick next vertex
            v = get_mrv_vertex()
            if v == -1:
                return True
            
            mask = domain_masks[v]
            
            # Try each available color
            for c in range(5):
                if not (mask & (1 << c)):
                    continue
                
                color[v] = c
                trail_len_before = len(trail)
                
                # Forward Checking: propagate constraint to uncolored neighbors
                conflict = False
                for u in adj[v]:
                    if color[u] == -1:
                        old_mask_u = domain_masks[u]
                        new_mask_u = old_mask_u & ~(1 << c)
                        if new_mask_u != old_mask_u:
                            domain_masks[u] = new_mask_u
                            trail.append((u, old_mask_u))
                            if new_mask_u == 0:
                                conflict = True
                                break
                
                if not conflict:
                    if backtrack(idx + 1):
                        return True
                
                # Undo forward checking changes
                while len(trail) > trail_len_before:
                    u, old_mask = trail.pop()
                    domain_masks[u] = old_mask
                color[v] = -1
            
            return False
        
        # Attempt exact backtracking with forward checking
        if backtrack(0):
            return [int(c) for c in color]
        
        # Fallback: Local Search (Min-Conflicts)
        rng = Random(seed)
        sol = [rng.choice(lists[i]) for i in range(n)]
        
        def count_conflicts(v):
            c = sol[v]
            cnt = 0
            for u in adj[v]:
                if sol[u] == c:
                    cnt += 1
            return cnt
        
        def total_conflicts():
            return sum(count_conflicts(v) for v in range(n)) // 2
        
        ls_deadline = start_time + time_limit
        best_sol = sol[:]
        best_conf = total_conflicts()
        
        while time.monotonic() < ls_deadline:
            conflicted = [v for v in range(n) if count_conflicts(v) > 0]
            if not conflicted:
                return [int(c) for c in sol]
            
            # Pick vertex with max conflicts (Max-Conflicts heuristic)
            v = max(conflicted, key=lambda x: count_conflicts(x))
            best_c = sol[v]
            best_conf_v = count_conflicts(v)
            
            for c in lists[v]:
                if c == sol[v]:
                    continue
                cnt = 0
                for u in adj[v]:
                    if sol[u] == c:
                        cnt += 1
                if cnt < best_conf_v:
                    best_conf_v = cnt
                    best_c = c
                    if best_conf_v == 0:
                        break
            
            sol[v] = best_c
            
            # Track best solution found
            tc = total_conflicts()
            if tc < best_conf:
                best_conf = tc
                best_sol = sol[:]
        
        # Return best solution found during local search
        return [int(c) for c in best_sol]

    out = []
    start_time = time.time()
    total_time_limit = 130.0  # Use most of the 150s budget with safety margin
    
    for idx, (splits, seed) in enumerate(((20, 701), (40, 702), (60, 703), (100, 704))):
        elapsed = time.time() - start_time
        remaining = total_time_limit - elapsed
        
        if remaining <= 0:
            # We are out of time. Generate instance to get n, return dummy.
            n, edges, lists = instance(splits, seed)
            out.append([0] * n)
            continue
        
        # Allocate time for this instance
        instances_remaining = 4 - idx
        per_instance_time = max(2.0, (remaining - 5.0) / instances_remaining)
        
        n, edges, lists = instance(splits, seed)
        sol = solve(n, edges, lists, seed, time_limit=per_instance_time)
        out.append([int(c) for c in sol])
    
    # Ensure length 4
    while len(out) < 4:
        out.append([0])
        
    return out
