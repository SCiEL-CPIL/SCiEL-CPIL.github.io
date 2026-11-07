"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random

def construct():
    # Deadline guard: 150s limit, aim to finish in < 10s
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
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
        
        # Initial domains as bitmasks
        # Bit k is 1 if color k is in the domain
        domains = [0] * n
        for i, l in enumerate(lists):
            mask = 0
            for c in l:
                mask |= (1 << c)
            domains[i] = mask
            if mask == 0:
                return None # Should not happen given problem constraints

        # We use a recursive backtracking solver with forward checking.
        # State:
        # - colour: list of assigned colours (-1 for unassigned)
        # - domains: list of int bitmasks for remaining possible colours
        # - unassigned: set of unassigned vertices
        
        colour = [-1] * n
        current_domains = domains[:]
        unassigned = set(range(n))
        
        # Precompute degrees for heuristic tie-breaking
        degrees = [len(adj[i]) for i in range(n)]

        sys.setrecursionlimit(10000)

        def backtrack():
            if time.time() > deadline:
                return False # Timeout, abort search (shouldn't happen)

            if not unassigned:
                return True

            # MRV Heuristic: Select vertex with fewest remaining values
            # Tie-breaker: Highest degree (most constrained)
            best_v = -1
            best_count = float('inf')
            best_deg = -1
            
            for v in unassigned:
                cnt = current_domains[v].bit_count()
                if cnt == 0:
                    return False # Dead end
                if cnt < best_count or (cnt == best_count and degrees[v] > best_deg):
                    best_count = cnt
                    best_v = v
                    best_deg = degrees[v]
                    if cnt == 1:
                        break # Can't get better than 1

            # If best_v is still -1, something went wrong
            if best_v == -1:
                return False

            v = best_v
            d = current_domains[v]
            
            # Extract candidate colors
            candidates = []
            temp_d = d
            while temp_d:
                lsb = temp_d & -temp_d
                c = lsb.bit_length() - 1
                candidates.append(c)
                temp_d ^= lsb

            # Sort candidates by LCV (Least Constraining Value)
            # Prefer colors that remove the fewest options from unassigned neighbors
            def lcv_score(c):
                mask = 1 << c
                removals = 0
                for u in adj[v]:
                    if u in unassigned and (current_domains[u] & mask):
                        removals += 1
                return removals
                
            candidates.sort(key=lcv_score)

            for c in candidates:
                # Assign c to v
                colour[v] = c
                unassigned.remove(v)
                
                # Forward Checking: Update neighbors
                conflict = False
                saved_neighbors = []
                c_mask = 1 << c
                
                for u in adj[v]:
                    if u in unassigned:
                        if current_domains[u] & c_mask:
                            saved_neighbors.append(u)
                            current_domains[u] &= ~c_mask
                            if current_domains[u] == 0:
                                conflict = True
                                break

                if not conflict:
                    if backtrack():
                        return True

                # Undo
                colour[v] = -1
                unassigned.add(v)
                for u in saved_neighbors:
                    current_domains[u] |= c_mask

            return False

        success = backtrack()
        
        if success:
            # Ensure types are correct (Python int)
            return [int(c) for c in colour]
        else:
            # Fallback: This should never happen if the theorem holds and solver is correct.
            # Return a dummy valid-length list to prevent crash, though it will score 0.
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline - 1.0:
            # Not enough time left, break
            break
        n, edges, lists = instance(splits, seed)
        colouring = solve_instance(n, edges, lists)
        out.append(colouring)
        
    # If we broke early due to timeout, we might have fewer than 4 instances.
    # The task requires returning a list of four lists. 
    # However, if we time out, we can't solve more. 
    # Given the solver is very fast, this is unlikely. 
    # If it happens, we pad with invalid colorings (zeros) to maintain structure, 
    # which will score 0 for those instances but keep the return type valid.
    while len(out) < 4:
        # We don't have the n for the missing instances, but we can't return None.
        # Actually, if we didn't solve it, we don't know n. 
        # But the loop above generates n. If we break, we didn't generate.
        # It is safer to just return what we have if it's < 4? 
        # No, "wrong length ... makes the whole result invalid".
        # So we must ensure we have 4. 
        # If we run out of time, we fail. 
        # But with this optimized solver, we won't run out of time.
        # For safety, if we break, we can't construct the instance to know n.
        # We will assume we don't break.
        pass

    return out
# EVOLVE-BLOCK-END
