"""Local Search (Min-Conflicts) solver for List Coloring."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
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

    def solve_instance(n, edges, lists, deadline, rng):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initialize colors randomly from lists
        color = [rng.choice(lst) for lst in lists]
        
        # Calculate initial conflicts per vertex
        conflicts = [0] * n
        for u, v in edges:
            if color[u] == color[v]:
                conflicts[u] += 1
                conflicts[v] += 1
        
        # Set of vertices with conflicts
        conflicted_set = set(v for v in range(n) if conflicts[v] > 0)
        
        if not conflicted_set:
            return color

        last_check = time.time()
        
        while conflicted_set:
            # Time check
            if time.time() - last_check > 0.01:
                if time.time() > deadline:
                    return color # Return current best (likely invalid)
                last_check = time.time()
                
            # Pick a random conflicted vertex
            v = rng.choice(list(conflicted_set))
            
            # Find best color for v
            # We want to minimize conflicts with neighbors
            best_c = color[v]
            min_conf = conflicts[v]
            
            for c in lists[v]:
                if c == best_c:
                    continue
                conf = 0
                for u in adj[v]:
                    if color[u] == c:
                        conf += 1
                if conf < min_conf:
                    min_conf = conf
                    best_c = c
            
            # If best_c is same as current, we are stuck in a local minimum for this vertex.
            # To escape, pick a random different color.
            if best_c == color[v]:
                other_colors = [c for c in lists[v] if c != color[v]]
                if other_colors:
                    best_c = rng.choice(other_colors)
                else:
                    # Only one color? Impossible.
                    continue

            # Update
            old_c = color[v]
            color[v] = best_c
            
            # Update conflicts
            # Neighbors with old_c lose a conflict
            # Neighbors with new_c gain a conflict
            # v's conflict count is recalculated
            
            for u in adj[v]:
                if color[u] == old_c:
                    conflicts[u] -= 1
                    if conflicts[u] == 0:
                        conflicted_set.discard(u)
                    # Note: if conflicts[u] was >0, it stays in set (or was already there)
                elif color[u] == best_c:
                    conflicts[u] += 1
                    conflicted_set.add(u)
            
            # Recalculate conflicts for v
            new_conf_v = 0
            for u in adj[v]:
                if color[u] == best_c:
                    new_conf_v += 1
            
            conflicts[v] = new_conf_v
            if new_conf_v == 0:
                conflicted_set.discard(v)
            else:
                conflicted_set.add(v)
                
        return color

    out = []
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        
        remaining = DEADLINE - time.time()
        if remaining <= 1.0:
            out.append([0] * n)
            continue
        
        instance_deadline = min(time.time() + 25.0, DEADLINE - 1.0)
        
        solver_rng = random.Random(seed + 1000)
        
        sol = solve_instance(n, edges, lists, instance_deadline, solver_rng)
        
        # Verify
        valid = True
        for v in range(n):
            if sol[v] not in lists[v]:
                valid = False
                break
        if valid:
            for u, v in edges:
                if sol[u] == sol[v]:
                    valid = False
                    break
                    
        if valid:
            out.append([int(c) for c in sol])
        else:
            out.append([0] * n)
            
    return out
# EVOLVE-BLOCK-END
