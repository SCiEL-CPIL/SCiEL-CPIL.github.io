import time

def construct():
    start = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    # Construct 8 disjoint K4s on 32 vertices.
    #
    # C5-free proof: Each connected component is a K4 (4 vertices, 6 edges).
    # A C5 requires 5 distinct vertices. Since no component has 5 vertices
    # and there are no edges between components, no C5 can exist.
    #
    # Alpha proof: Each K4 is a clique, so an independent set contains at
    # most 1 vertex per K4. With 8 K4s, alpha <= 8 < 9. Pigeonhole: any
    # 9 vertices include 2 from some K4, which are adjacent, so no
    # independent set of size 9 exists.
    #
    # This gives R(C5, K9) > 32.

    n = 32
    edges = []
    for block in range(8):
        base = block * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])

    # Fast structural validation (not brute-force)
    def validate_structural(n, edges):
        if time.time() > DEADLINE:
            return False
        # Verify basic properties
        if n < 1 or n > 35:
            return False
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            if u == v or u < 0 or u >= n or v < 0 or v >= n:
                return False
            if adj[u][v]:
                return False  # duplicate edge
            adj[u][v] = True
            adj[v][u] = True

        # Check C5-free via connected components:
        # If every component has <= 4 vertices, no C5 is possible.
        visited = [False] * n
        for i in range(n):
            if not visited[i]:
                # BFS to find component size
                comp = []
                stack = [i]
                visited[i] = True
                while stack:
                    u = stack.pop()
                    comp.append(u)
                    for v in range(n):
                        if adj[u][v] and not visited[v]:
                            visited[v] = True
                            stack.append(v)
                if len(comp) >= 5:
                    # Component has >= 5 vertices; need to check for C5
                    # For our construction, all components are K4s (size 4)
                    # so this branch should never trigger.
                    # But to be safe, do a quick C5 check on this component
                    if _has_c5_in_component(comp, adj):
                        return False

        # Check alpha < 9:
        # For our construction, each component is a clique of size 4.
        # Alpha = number of components = 8 < 9.
        # General check: use the fact that in a graph where each component
        # is a clique of size k, alpha = number of components.
        # Verify each component is a clique:
        visited2 = [False] * n
        num_components = 0
        for i in range(n):
            if not visited2[i]:
                num_components += 1
                comp = []
                stack = [i]
                visited2[i] = True
                while stack:
                    u = stack.pop()
                    comp.append(u)
                    for v in range(n):
                        if adj[u][v] and not visited2[v]:
                            visited2[v] = True
                            stack.append(v)
                # Check comp is a clique
                for a in range(len(comp)):
                    for b in range(a + 1, len(comp)):
                        if not adj[comp[a]][comp[b]]:
                            # Not a clique; need full alpha check
                            # Fallback: use a greedy upper bound
                            # For safety, just return False if not as expected
                            return False
        # Alpha = num_components for a graph of disjoint cliques
        if num_components >= 9:
            return False

        return True

    def _has_c5_in_component(comp, adj):
        """Check if a component with >= 5 vertices contains a C5.
        Only called if structural check fails (shouldn't happen for our graph)."""
        k = len(comp)
        if k < 5:
            return False
        # For small components, check all 5-subsets
        from itertools import combinations, permutations
        for subset in combinations(comp, 5):
            a = subset[0]
            rest = subset[1:]
            for perm in permutations(rest):
                x, y, z, w = perm
                if adj[a][x] and adj[x][y] and adj[y][z] and adj[z][w] and adj[w][a]:
                    return True
        return False

    if validate_structural(n, edges):
        return {'n': n, 'edges': edges}

    # Fallback: 7 disjoint K4s (n=28)
    n_base = 28
    edges_base = [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]
    return {'n': n_base, 'edges': edges_base}
