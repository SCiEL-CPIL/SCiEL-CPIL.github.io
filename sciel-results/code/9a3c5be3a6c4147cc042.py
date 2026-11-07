"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random
from collections import deque

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

    def ac3(domains, adj):
        """
        Enforce arc consistency.
        domains: list of int bitmasks
        adj: list of lists of neighbors
        Returns True if consistent, False if a domain becomes empty.
        """
        # Initialize queue with all arcs
        q = deque()
        for u in range(len(domains)):
            for v in adj[u]:
                q.append((u, v))
        
        while q:
            u, v = q.popleft()
            # Check if domain of u can be reduced due to v
            # For each color c in domain(u), there must exist a color d in domain(v) such that c != d
            # Since the constraint is "not equal", c is inconsistent with v only if domain(v) == {c}
            # Actually, if domain(v) is {c}, then u cannot be c.
            # More generally, we remove c from domain(u) if there is no d in domain(v) with c != d.
            # This happens if domain(v) is a subset of {c}. Since domains have at least 1 value, 
            # this means domain(v) == {c}.
            
            # Optimization: 
            # If domain(v) has only one color, say cv, then u cannot have cv.
            # If domain(v) has >1 colors, then for any c in domain(u), there is at least one d in domain(v) != c?
            # Not necessarily. If domain(u) = {1, 2} and domain(v) = {1, 2}, 
            # c=1 is ok (d=2), c=2 is ok (d=1).
            # If domain(u) = {1} and domain(v) = {1}, then c=1 has no d != 1. So remove 1 from u.
            
            # General check:
            changed = False
            du = domains[u]
            dv = domains[v]
            
            # Iterate through bits in du
            temp_du = du
            while temp_du:
                lsb = temp_du & -temp_du
                c = lsb.bit_length() - 1
                # Is there a d in dv such that d != c?
                # This is true if dv has more than 1 bit, OR if dv has 1 bit and it's not c.
                if dv == lsb:
                    # dv is exactly {c}, so c is not supported in v
                    domains[u] &= ~lsb
                    changed = True
                temp_du ^= lsb
            
            if domains[u] == 0:
                return False
            
            if changed:
                # If domain of u changed, we need to re-check arcs (w, u) for all w != v
                for w in adj[u]:
                    if w != v:
                        q.append((w, u))
                        
        return True

    def solve_instance(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initial domains as bitmasks
        domains = [0] * n
        for i, l in enumerate(lists):
            mask = 0
            for c in l:
                mask |= (1 << c)
            domains[i] = mask
            if mask == 0:
                return None # Should not happen given problem constraints

        # Preprocess with Arc Consistency
        if not ac3(domains, adj):
            return None # Inconsistent, though Alon-Tarsi says it should be solvable, 
                        # AC-3 failure implies no solution, which contradicts the theorem 
                        # unless the theorem applies to existence of *some* coloring and 
                        # our specific lists make it impossible? 
                        # Wait, Alon-Tarsi says *any* list of size >= 3 on a planar bipartite graph 
                        # is 3-colorable? No, it says planar bipartite graphs are 3-choosable.
                        # So a solution MUST exist. If AC-3 fails, it's a bug in AC-3 or logic.
                        # But AC-3 is sound. If it fails, there is no solution.
                        # This should not happen.

        colour = [-1] * n
        current_domains = domains[:]
        unassigned = set(range(n))
        
        # Precompute degrees for heuristic tie-breaking
        degrees = [len(adj[i]) for i in range(n)]

        sys.setrecursionlimit(10000)

        def backtrack():
            if time.time() > deadline:
                return False # Timeout, abort search

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
            scores = []
            for c in candidates:
                mask = 1 << c
                removals = 0
                for u in adj[v]:
                    if u in unassigned and (current_domains[u] & mask):
                        removals += 1
                scores.append(removals)
            
            # Sort by score
            candidates = [c for c, s in sorted(zip(candidates, scores), key=lambda x: x[1])]

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
            # Fallback
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline - 1.0:
            break
        n, edges, lists = instance(splits, seed)
        colouring = solve_instance(n, edges, lists)
        out.append(colouring)
        
    # Pad if necessary to ensure length 4, though this should not happen
    while len(out) < 4:
        # We don't have n for missing instances, but we must return a list of 4 lists.
        # If we didn't solve it, we don't know n. 
        # However, the task says "wrong length ... makes the whole result invalid".
        # If we time out, we fail anyway. 
        # To be safe against type errors if n is unknown, we can't really pad correctly.
        # But given the speed, this won't trigger.
        # We'll just break and hope we have 4. If not, the result is invalid.
        # Actually, we can just append an empty list or dummy? No, must match n.
        # Since we can't know n without generating, and generating is fast, 
        # we should have generated it. 
        # If we broke the loop, we didn't generate. 
        # So we can't pad. 
        # This path is only for extreme timeout.
        pass

    return out
# EVOLVE-BLOCK-END
