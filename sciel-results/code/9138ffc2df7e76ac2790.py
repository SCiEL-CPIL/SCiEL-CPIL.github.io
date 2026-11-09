"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    START_TIME = time.time()
    DEADLINE = START_TIME + 110.0  # Safety margin before 120s kill

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

    def solve_instance(n, edges, lists, rng_solver, time_limit):
        # Build adjacency lists (both list and bitmask forms)
        adj = [[] for _ in range(n)]
        adj_mask = [0] * n
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # Convert lists to bitmask domains
        init_domains = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            init_domains[v] = mask
        
        # Check for empty initial domains
        for v in range(n):
            if init_domains[v] == 0:
                return None
        
        # Precompute popcount for small masks (0 to 31)
        popcount_table = [bin(i).count('1') for i in range(32)]
        
        # Precompute degree
        degree = [len(adj[v]) for v in range(n)]
        
        # AC-3 preprocessing: enforce arc consistency
        def ac3(domains):
            # Queue of arcs (u, v) where u's domain may need revision
            queue = []
            for u in range(n):
                for v in adj[u]:
                    queue.append((u, v))
            
            while queue:
                u, v = queue.pop()
                # Check if any value in domains[u] has no support in domains[v]
                du = domains[u]
                dv = domains[v]
                # For each color in du, check if there's a compatible color in dv
                # Since it's just inequality, a color c in du is supported if dv has any color != c
                # i.e., dv is not just {c}
                if du & dv:
                    # There's overlap; check if removing unsupported values changes du
                    # A value c in du is unsupported if dv == {c} (only c is in dv)
                    # More precisely: c is unsupported if (dv & ~(1<<c)) == 0, i.e., dv is subset of {c}
                    # Since domains are bitmasks of colors 0-4, dv has at most 3 bits
                    # c is unsupported iff dv == (1 << c)
                    new_du = 0
                    m = du
                    while m:
                        lb = m & (-m)
                        c = lb.bit_length() - 1
                        # c is supported if dv has some color != c
                        if dv & ~(1 << c):
                            new_du |= lb
                        m ^= lb
                    if new_du != du:
                        domains[u] = new_du
                        if new_du == 0:
                            return False
                        # Add all arcs (w, u) for w in adj[u], w != v
                        for w in adj[u]:
                            if w != v:
                                queue.append((w, u))
            return True

        # Apply AC-3 to initial domains
        ac3_domains = init_domains[:]
        if not ac3(ac3_domains):
            return None
        
        # Greedy Pass: Try to solve with a simple greedy strategy first
        def greedy_solve():
            assigned = [-1] * n
            # Order by degree (highest first), tie-break by MRV
            order = sorted(range(n), key=lambda x: (-degree[x], popcount_table[ac3_domains[x]]))
            
            for v in order:
                avail = ac3_domains[v]
                # Check assigned neighbors
                for u in adj[v]:
                    if assigned[u] != -1:
                        avail &= ~(1 << assigned[u])
                
                if avail == 0:
                    return None
                
                # Pick most constraining value: the one that removes most from neighbors
                best_val = None
                best_constraint = -1
                m = avail
                while m:
                    lb = m & (-m)
                    val = lb.bit_length() - 1
                    # Count how many neighbors would lose this value
                    constraint = 0
                    for u in adj[v]:
                        if assigned[u] == -1 and (ac3_domains[u] & lb):
                            constraint += 1
                    if constraint > best_constraint:
                        best_constraint = constraint
                        best_val = val
                    m ^= lb
                
                assigned[v] = best_val
                # Update domains of unassigned neighbors
                for u in adj[v]:
                    if assigned[u] == -1:
                        ac3_domains[u] &= ~(1 << best_val)
                        if ac3_domains[u] == 0:
                            return None
                
            return assigned

        greedy_sol = greedy_solve()
        if greedy_sol is not None:
            return greedy_sol

        # Helper to get list of values from mask
        def mask_to_list(mask):
            vals = []
            m = mask
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                vals.append(idx)
                m ^= lb
            return vals

        def backtrack(domains, assigned, unassigned, depth=0):
            if not unassigned:
                return True
            
            # Check time
            if time.time() > time_limit:
                return False
            
            # MRV: find vertex with minimum remaining values
            # Tie-breaker: highest degree
            min_count = 6
            best_idx = -1
            best_deg = -1
            candidates = []
            
            for i, v in enumerate(unassigned):
                c = popcount_table[domains[v]]
                if c == 0:
                    return False
                if c < min_count:
                    min_count = c
                    best_deg = degree[v]
                    candidates = [i]
                elif c == min_count:
                    d = degree[v]
                    if d > best_deg:
                        best_deg = d
                        candidates = [i]
                    elif d == best_deg:
                        candidates.append(i)
            
            if not candidates:
                return False

            best_idx = rng_solver.choice(candidates)
            v = unassigned[best_idx]
            unassigned.pop(best_idx)
            
            # Get values to try, ordered by most constraining
            mask = domains[v]
            vals = mask_to_list(mask)
            
            # Sort by most constraining (removes most from unassigned neighbors)
            def value_constraint(val):
                val_bit = 1 << val
                cnt = 0
                for u in adj[v]:
                    if assigned[u] == -1 and (domains[u] & val_bit):
                        cnt += 1
                return -cnt  # Negate so higher constraint comes first
            
            vals.sort(key=value_constraint)
            
            for val in vals:
                val_bit = 1 << val
                
                # Check consistency with assigned neighbors
                ok = True
                for u in adj[v]:
                    if assigned[u] == val:
                        ok = False
                        break
                if not ok:
                    continue
                
                assigned[v] = val
                
                # Forward checking
                saved = []
                ok_fc = True
                for u in adj[v]:
                    if assigned[u] == -1:
                        if domains[u] & val_bit:
                            saved.append((u, val_bit))
                            domains[u] &= ~val_bit
                            if domains[u] == 0:
                                ok_fc = False
                                break
                
                if ok_fc:
                    if backtrack(domains, assigned, unassigned, depth + 1):
                        return True
                
                # Restore forward checking changes
                for u, bit in saved:
                    domains[u] |= bit
                
                assigned[v] = -1
            
            unassigned.insert(best_idx, v)
            return False

        # Main solve loop with restarts
        max_restarts = 50
        for r in range(max_restarts):
            if time.time() > time_limit:
                break
            
            # Initialize state
            domains = ac3_domains[:]
            assigned = [-1] * n
            unassigned = list(range(n))
            rng_solver.shuffle(unassigned)
            
            if backtrack(domains, assigned, unassigned):
                return [assigned[i] for i in range(n)]
        
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        # Check deadline
        elapsed = time.time() - START_TIME
        if elapsed > DEADLINE - 5.0:
            # Not enough time left; use best effort fallback
            n, edges, lists = instance(splits, seed)
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            order = sorted(range(n), key=lambda x: -len(adj[x]))
            col = [0] * n
            for v in order:
                taken = set()
                for u in adj[v]:
                    taken.add(col[u])
                free = [c for c in lists[v] if c not in taken]
                if free:
                    col[v] = free[0]
                else:
                    col[v] = lists[v][0]
            out.append([int(c) for c in col])
            continue

        n, edges, lists = instance(splits, seed)
        
        # Solver RNG
        rng_solver = random.Random(seed + 42)
        
        # Allocate remaining time evenly among remaining instances
        remaining_instances = 4 - len(out)
        time_left = DEADLINE - time.time() - 2.0  # 2s safety margin
        time_per_instance = max(1.0, time_left / remaining_instances)
        
        coloring = solve_instance(n, edges, lists, rng_solver, time.time() + time_per_instance)
        
        if coloring is None:
            # Fallback: Greedy with degree ordering
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            order = sorted(range(n), key=lambda x: -len(adj[x]))
            col = [0] * n
            for v in order:
                taken = set()
                for u in adj[v]:
                    taken.add(col[u])
                free = [c for c in lists[v] if c not in taken]
                if free:
                    col[v] = free[0]
                else:
                    col[v] = lists[v][0]
            coloring = col
        
        # Ensure types are Python int
        out.append([int(c) for c in coloring])
    
    return out
# EVOLVE-BLOCK-END
