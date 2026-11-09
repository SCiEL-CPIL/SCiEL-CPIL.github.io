"""Rewritten program using MAC (Maintaining Arc Consistency) and Trail-Based Backtracking."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    # Deadline guard: 120s limit, leave 5s margin
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0
    sys.setrecursionlimit(10000)

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
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Domains are represented as lists of available colors for fast iteration and modification
        # We use sets for internal logic but convert to lists for sorting? 
        # Actually, sets are fine for membership. Lists are better for iteration.
        # Let's stick to sets for simplicity in AC-3, but trail-based undo needs careful handling.
        # Alternative: Use lists of booleans or bitmasks? 
        # For n<=400 and 5 colors, sets are efficient enough.
        
        doms = [set(l) for l in lists]
        if any(not d for d in doms):
            return None

        # Trail for undoing changes
        # Each entry: (var, value) that was removed
        trail = []

        def record_removal(var, val):
            trail.append((var, val))

        def undo_to(mark):
            while len(trail) > mark:
                var, val = trail.pop()
                doms[var].add(val)

        def ac3(initial_queue=None):
            """
            Runs AC-3. Returns False if any domain becomes empty.
            Modifies doms in place and records changes in trail.
            """
            if initial_queue is None:
                queue = [(u, v) for u in range(n) for v in adj[u]]
            else:
                queue = list(initial_queue)
            
            head = 0
            while head < len(queue):
                i, j = queue[head]
                head += 1
                
                # Check if dom[i] can be revised based on dom[j]
                # For != constraint, val in dom[i] is unsupported if dom[j] == {val}
                # More generally, val is unsupported if for all v in dom[j], val == v.
                # Since it's a graph coloring (all-diff on edges), val is unsupported iff dom[j] == {val}.
                
                if len(doms[j]) == 1:
                    val_j = next(iter(doms[j]))
                    if val_j in doms[i]:
                        # Remove val_j from doms[i]
                        doms[i].remove(val_j)
                        record_removal(i, val_j)
                        
                        if not doms[i]:
                            return False
                        
                        # Add neighbors of i (excluding j) to queue
                        for k in adj[i]:
                            if k != j:
                                queue.append((k, i))
            return True

        # Initial AC-3
        mark = len(trail)
        if not ac3():
            return None
        
        # If AC-3 succeeded, we have a consistent set of domains.
        # Now we perform Backtracking with MAC.
        
        assignment = {}

        def backtrack():
            if time.time() > DEADLINE:
                raise TimeoutError()
            
            # Find MRV node
            min_size = float('inf')
            min_node = -1
            for i in range(n):
                if i not in assignment:
                    size = len(doms[i])
                    if size < min_size:
                        min_size = size
                        min_node = i
                        if size == 0:
                            return False
                        if min_size == 1:
                            break

            if min_node == -1:
                return True # Success

            # LVC: Sort values by least constraining
            vals = list(doms[min_node])
            # Impact: number of unassigned neighbors that contain this value
            def impact(c):
                cnt = 0
                for nb in adj[min_node]:
                    if nb not in assignment:
                        if c in doms[nb]:
                            cnt += 1
                return cnt
            vals.sort(key=impact)

            for c in vals:
                # Check if c is consistent with assigned neighbors
                valid = True
                for nb in adj[min_node]:
                    if nb in assignment:
                        if assignment[nb] == c:
                            valid = False
                            break
                if not valid:
                    continue
                
                # Save state
                mark = len(trail)
                
                # Assign c to min_node
                assignment[min_node] = c
                # Remove all other values from doms[min_node]? 
                # Actually, we just need to ensure that when we check MRV, we skip assigned nodes.
                # But for AC-3, we should restrict doms[min_node] to {c}.
                # Let's store original domain to restore? 
                # Trail handles removals. 
                # We need to remove all values from doms[min_node] except c.
                for v in list(doms[min_node]):
                    if v != c:
                        doms[min_node].remove(v)
                        record_removal(min_node, v)
                
                # Propagate: Run AC-3 on neighbors
                # Queue arcs (nb, min_node) for all neighbors nb
                # Also, removing c from min_node's domain might affect neighbors? 
                # No, the constraint is on the edge. 
                # Assigning min_node=c means neighbors cannot be c.
                # So we should remove c from neighbors' domains? 
                # AC-3 will handle this if we queue the arcs correctly.
                # But AC-3 as defined only removes values if the *other* domain is a singleton.
                # If we assign min_node=c, doms[min_node] becomes {c}.
                # Then for arc (nb, min_node), AC-3 will see doms[min_node]={c} and remove c from doms[nb].
                # This is correct.
                
                queue = [(nb, min_node) for nb in adj[min_node] if nb not in assignment]
                
                if not ac3(queue):
                    # Conflict, undo and try next value
                    del assignment[min_node]
                    undo_to(mark)
                    continue
                
                # Recurse
                if backtrack():
                    return True
                
                # Undo
                del assignment[min_node]
                undo_to(mark)
            
            return False

        try:
            if backtrack():
                # Reconstruct solution
                return [int(assignment[i]) for i in range(n)]
        except TimeoutError:
            pass
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > DEADLINE:
            n, _, _ = instance(splits, seed)
            out.append([0] * n)
            continue
            
        n, edges, lists = instance(splits, seed)
        
        if time.time() > DEADLINE:
            out.append([0] * n)
            continue
            
        sol = solve_instance(n, edges, lists)
        
        if sol is None:
            out.append([0] * n)
        else:
            out.append([int(c) for c in sol])
            
    return out
# EVOLVE-BLOCK-END
