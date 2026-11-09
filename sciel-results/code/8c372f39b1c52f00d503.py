import itertools

def construct():
    def generate_paley(p):
        """Generates the Paley graph on p vertices (p must be 1 mod 4)."""
        if p % 4 != 1:
            return None
        residues = set()
        for x in range(1, p):
            residues.add((x * x) % p)
        edges = []
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in residues:
                    edges.append((i, j))
        return p, edges

    def is_5_5_free(n, edges):
        """Checks if the graph has no K5 and no independent set of size 5."""
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        # Check for K5
        for combo in itertools.combinations(range(n), 5):
            is_clique = True
            for i in range(5):
                for j in range(i + 1, 5):
                    if not adj[combo[i]][combo[j]]:
                        is_clique = False
                        break
                if not is_clique:
                    break
            if is_clique:
                return False
        
        # Check for Independent Set of size 5 (Clique in complement)
        for combo in itertools.combinations(range(n), 5):
            is_independent = True
            for i in range(5):
                for j in range(i + 1, 5):
                    if adj[combo[i]][combo[j]]:
                        is_independent = False
                        break
                if not is_independent:
                    break
            if is_independent:
                return False
        
        return True

    # Try Paley graph on 41 vertices
    p41 = generate_paley(41)
    if p41:
        n, edges = p41
        if is_5_5_free(n, edges):
            return n, edges

    # Fallback to Paley graph on 37 vertices (known valid)
    p37 = generate_paley(37)
    if p37:
        n37, edges37 = p37
        # Try to extend to 38 vertices by adding an isolated vertex
        # This is valid if alpha(P_37) <= 3
        n38 = n37 + 1
        edges38 = edges37[:]
        # No new edges for isolated vertex
        if is_5_5_free(n38, edges38):
            return n38, edges38
        
        # Try to extend to 38 vertices by adding a universal vertex
        # This is valid if omega(P_37) <= 3
        edges38_univ = edges37[:]
        for i in range(n37):
            edges38_univ.append((i, n37))
        if is_5_5_free(n38, edges38_univ):
            return n38, edges38_univ

        return p37

    # Fallback to 37 vertices if generation failed (should not happen)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    return q, [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
