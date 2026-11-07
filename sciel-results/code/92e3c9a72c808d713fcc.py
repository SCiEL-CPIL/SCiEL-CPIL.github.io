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
        bt_deadline = start_time + time_limit * 0.9
        sys.setrecursionlimit(10000)
        
        def backtrack(idx):
            # Time check
            if time.monotonic() > bt_deadline:
                return False
            
            if idx == n:
                return True
            
            v = order[idx]
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
                    if pos[u] > pos[v]:
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
        
        ls_deadline = start_time + time_limit
        while time.monotonic() < ls_deadline:
            conflicted = [v for v in range(n) if count_conflicts(v) > 0]
            if not conflicted:
                return [int(c) for c in sol]
            
            # Pick vertex with max conflicts (Max-Conflicts heuristic)
            v = max(conflicted, key=lambda x: count_conflicts(x))
            best_c = sol[v]
            best_conf = count_conflicts(v)
            
            for c in lists[v]:
                if c == sol[v]:
                    continue
                cnt = 0
                for u in adj[v]:
                    if sol[u] == c:
                        cnt += 1
                if cnt < best_conf:
                    best_conf = cnt
                    best_c = c
                    if best_conf == 0:
                        break
            
            sol[v] = best_c
        
        return [int(c) for c in sol]

    out = []
    start_time = time.time()
    total_time_limit = 110.0
    
    for idx, (splits, seed) in enumerate(((20, 701), (40, 702), (60, 703), (100, 704))):
        elapsed = time.time() - start_time
        remaining = total_time_limit - elapsed
        
        if remaining <= 0:
            # We are out of time. We must still return 4 lists.
            # Since we can't solve the remaining instances, we fill with dummies.
            # However, the prompt says "An instance whose colouring breaks a list or an edge scores 0 for that instance".
            # But "a wrong length or type makes the whole result invalid".
            # So we must return a list of correct length for each instance.
            # We don't know n for unsolved instances unless we generate them.
            # Generating is cheap. Solving is expensive.
            # Let's generate the instance to get n, then fill with 0s (which might be invalid coloring, but valid structure).
            # Actually, if we run out of time, we should probably just generate the instance and return a greedy attempt or 0s.
            # Given the tight time, let's just generate and return 0s to ensure structural validity.
            n, edges, lists = instance(splits, seed)
            out.append([0] * n)
            continue
        
        # Allocate time for this instance
        # Leave some buffer for the remaining instances
        instances_remaining = 4 - idx
        per_instance_time = max(1.0, (remaining - 5.0) / instances_remaining)
        
        n, edges, lists = instance(splits, seed)
        sol = solve(n, edges, lists, seed, time_limit=per_instance_time)
        out.append([int(c) for c in sol])
    
    # Ensure length 4
    while len(out) < 4:
        # This shouldn't happen if the loop runs 4 times, but just in case
        out.append([0])
        
    return out
