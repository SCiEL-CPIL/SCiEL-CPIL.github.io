"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    t_start = time.time()
    deadline = t_start + 120.0  # safety margin before 150s kill

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

    def solve_list_coloring(n, edges, lists):
        """
        Backtracking CSP solver with MRV + forward checking.
        Returns a valid colouring (list of ints) or None.
        """
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Initial domains: the list for each vertex
        domains = [list(lst) for lst in lists]

        # Order vertices by degree (descending) as a heuristic for initial ordering
        # We'll use dynamic MRV during search

        # We'll use a simple recursive backtracking with MRV
        # To avoid deep recursion issues, we limit depth or use iterative approach
        # For n up to ~400, recursion depth should be fine with sys.setrecursionlimit

        sys.setrecursionlimit(10000)

        # Track assignments
        assignment = [None] * n

        # For forward checking, we maintain remaining domains
        # We'll use a stack-based approach for efficiency

        # MRV heuristic: pick unassigned vertex with smallest domain
        # Tie-break by degree (descending)

        def get_mrv_vertex(assignment, domains):
            best_v = -1
            best_size = float('inf')
            best_degree = -1
            for v in range(n):
                if assignment[v] is None:
                    size = len(domains[v])
                    if size == 0:
                        return v  # dead end
                    if size < best_size or (size == best_size and len(adj[v]) > best_degree):
                        best_size = size
                        best_degree = len(adj[v])
                        best_v = v
            return best_v

        # Forward checking: reduce domains of unassigned neighbors
        def reduce_domains(v, c, assignment, domains):
            # c is the colour assigned to v
            # For each unassigned neighbor u of v, remove c from domains[u]
            # Return True if no domain becomes empty, False otherwise
            # We need to track what was removed for backtracking
            # For simplicity, we'll copy domains (n is small enough)
            # Actually, for efficiency, let's do it in-place with a trail
            # But for simplicity and correctness, let's use a simpler approach:
            # We'll just check if any neighbor's domain becomes empty
            for u in adj[v]:
                if assignment[u] is None:
                    if c in domains[u]:
                        # Check if removing c would make domain empty
                        if len(domains[u]) == 1:
                            return False
            return True

        # Let's use a cleaner backtracking with domain copying for small n
        # n is at most 8 + 4*100 = 408, so copying domains of size 3 each is fine

        def backtrack(assignment, domains):
            # Check if all assigned
            for v in range(n):
                if assignment[v] is None:
                    break
            else:
                return True  # all assigned

            # Find MRV vertex
            best_v = -1
            best_size = float('inf')
            best_degree = -1
            for v in range(n):
                if assignment[v] is None:
                    size = len(domains[v])
                    if size == 0:
                        return False  # dead end
                    if size < best_size or (size == best_size and len(adj[v]) > best_degree):
                        best_size = size
                        best_degree = len(adj[v])
                        best_v = v

            if best_v == -1:
                return True

            # Try each colour in the domain of best_v
            for c in domains[best_v]:
                # Check if c conflicts with any assigned neighbor
                conflict = False
                for u in adj[best_v]:
                    if assignment[u] is not None and assignment[u] == c:
                        conflict = True
                        break
                if conflict:
                    continue

                # Assign c to best_v
                assignment[best_v] = c

                # Forward checking: check neighbors' domains
                fc_ok = True
                new_domains = [None] * n
                for v in range(n):
                    if v == best_v:
                        new_domains[v] = domains[v]
                    elif assignment[v] is None:
                        # Remove c from domain if present
                        if c in domains[v]:
                            new_domains[v] = [x for x in domains[v] if x != c]
                        else:
                            new_domains[v] = domains[v]
                    else:
                        new_domains[v] = domains[v]

                # Check if any domain became empty
                for v in range(n):
                    if assignment[v] is None and len(new_domains[v]) == 0:
                        fc_ok = False
                        break

                if fc_ok:
                    if backtrack(assignment, new_domains):
                        return True

                # Undo
                assignment[best_v] = None

            return False

        # Start backtracking
        if backtrack(assignment, domains):
            return assignment
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline - 2.0:
            # Not enough time; use greedy fallback for remaining
            break

        n, edges, lists = instance(splits, seed)

        # Try backtracking solver
        colouring = solve_list_coloring(n, edges, lists)

        if colouring is None:
            # Fallback: greedy
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = [None] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {colour[u] for u in adj[v]}
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            colouring = colour

        # Convert to plain Python ints
        colouring = [int(c) for c in colouring]
        out.append(colouring)

    # If we ran out of time, fill remaining with greedy
    while len(out) < 4:
        splits, seed = ((20, 701), (40, 702), (60, 703), (100, 704))[len(out)]
        n, edges, lists = instance(splits, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        colour = [None] * n
        for v in sorted(range(n), key=lambda u: -len(adj[u])):
            taken = {colour[u] for u in adj[v]}
            free = [c for c in lists[v] if c not in taken]
            colour[v] = free[0] if free else lists[v][0]
        out.append([int(c) for c in colour])

    return out
# EVOLVE-BLOCK-END
