import itertools
import time

def construct():
    # Try Paley graph on 41 vertices
    q = 41
    residues = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    # Build adjacency matrix
    adj = [[False] * q for _ in range(q)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    # Check for K5 and Independent 5
    # Using bitsets for faster checks?
    # adj_bits[i] is an integer where bit j is 1 if edge (i,j) exists.
    adj_bits = [0] * q
    for i in range(q):
        mask = 0
        for j in range(q):
            if adj[i][j]:
                mask |= (1 << j)
        adj_bits[i] = mask

    def has_k5():
        for combo in itertools.combinations(range(q), 5):
            # Check if all pairs are connected
            # For a clique, adj_bits[combo[0]] must have bits set for combo[1..4]
            # And adj_bits[combo[1]] for combo[2..4], etc.
            # Optimized check:
            # The intersection of neighborhoods of the first vertex restricted to the rest must contain the rest.
            # Simpler: just check pairs.
            v0, v1, v2, v3, v4 = combo
            if not (adj_bits[v0] & (1 << v1) and adj_bits[v0] & (1 << v2) and adj_bits[v0] & (1 << v3) and adj_bits[v0] & (1 << v4)):
                continue
            if not (adj_bits[v1] & (1 << v2) and adj_bits[v1] & (1 << v3) and adj_bits[v1] & (1 << v4)):
                continue
            if not (adj_bits[v2] & (1 << v3) and adj_bits[v2] & (1 << v4)):
                continue
            if not (adj_bits[v3] & (1 << v4)):
                continue
            return True
        return False

    def has_indep5():
        # Independent set: no edges between any pair.
        # This is equivalent to a clique in the complement graph.
        # We can use the same logic with ~adj_bits (masked to q bits)
        # Or just check if adj_bits[v0] has NO bits for v1..v4, etc.
        for combo in itertools.combinations(range(q), 5):
            v0, v1, v2, v3, v4 = combo
            if (adj_bits[v0] & (1 << v1)) or (adj_bits[v0] & (1 << v2)) or (adj_bits[v0] & (1 << v3)) or (adj_bits[v0] & (1 << v4)):
                continue
            if (adj_bits[v1] & (1 << v2)) or (adj_bits[v1] & (1 << v3)) or (adj_bits[v1] & (1 << v4)):
                continue
            if (adj_bits[v2] & (1 << v3)) or (adj_bits[v2] & (1 << v4)):
                continue
            if (adj_bits[v3] & (1 << v4)):
                continue
            return True
        return False

    # Time check
    start_time = time.time()
    
    # The checks are fast enough for q=41
    if not has_k5() and not has_indep5():
        return q, edges
    else:
        # Fallback to 37
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        return q, [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
