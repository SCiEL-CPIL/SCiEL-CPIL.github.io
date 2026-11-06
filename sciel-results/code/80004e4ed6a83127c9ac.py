import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin

    def find_violation(n, adj):
        """
        Finds a subset of size >= 6 that induces a regular subgraph.
        Returns the mask of the subset if found, else None.
        """
        # Iterate through all masks
        # Optimization: We can iterate by size if we want, but simple range is fine
        # if violations are common.
        
        for mask in range(1 << n):
            k = mask.bit_count()
            if k < 6:
                continue
            
            # Quick check: Sum of degrees must be divisible by k
            # Sum of degrees = 2 * E(mask)
            # We can compute E(mask) on the fly?
            # Or just check degrees directly.
            
            # Get degrees of all vertices in mask
            # We need to be efficient.
            
            # Pick first vertex
            lsb = mask & -mask
            v0 = lsb.bit_length() - 1
            d0 = (adj[v0] & mask).bit_count()
            
            # Check if 2 * E(mask) == k * d0
            # We need E(mask).
            # Calculating E(mask) by summing degrees / 2 is O(k).
            # So total O(k) per mask.
            
            # Let's just check if all degrees are d0.
            # If any degree != d0, it's not regular (with degree d0).
            # But it could be regular with a different degree?
            # No, if it's regular, all degrees are equal.
            # So if we pick d0, and find a vertex with degree != d0, it's NOT regular.
            # If all vertices have degree d0, it IS regular.
            
            is_reg = True
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if (adj[v] & mask).bit_count() != d0:
                    is_reg = False
                    break
                m ^= lsb
            
            if is_reg:
                return mask
        
        return None

    def generate_graph(n, p):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def adj_to_edges(adj, n):
        edges = []
        for i in range(n):
            m = adj[i]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                m ^= lsb
        return edges

    # Try n=21 first
    for n_target in [21, 20]:
        # Local search
        adj = generate_graph(n_target, 0.5)
        
        # Check initial
        violation = find_violation(n_target, adj)
        if violation is None:
            return {'n': n_target, 'edges': adj_to_edges(adj, n_target)}
        
        # If violation found, try to fix
        # We will do a simple hill climbing / random walk
        # Flip edges to break the violation
        
        best_adj = adj
        best_violation_count = 1 # Just tracking existence
        
        while time.time() < deadline:
            # Pick a violation
            violation = find_violation(n_target, adj)
            if violation is None:
                return {'n': n_target, 'edges': adj_to_edges(adj, n_target)}
            
            # Flip a random edge incident to the violation or inside it
            # To be more effective, flip an edge inside the violation
            verts = []
            m = violation
            while m:
                lsb = m & -m
                verts.append(lsb.bit_length() - 1)
                m ^= lsb
            
            # Pick two random vertices from the violation
            if len(verts) >= 2:
                v1, v2 = random.sample(verts, 2)
                # Flip edge (v1, v2)
                if adj[v1] & (1 << v2):
                    adj[v1] ^= (1 << v2)
                    adj[v2] ^= (1 << v1)
                else:
                    adj[v1] |= (1 << v2)
                    adj[v2] |= (1 << v1)
            else:
                # Should not happen since size >= 6
                break

    # Fallback: empty graph on 5 vertices (or known valid small graph)
    # Actually, n=10 is easy to find.
    # Let's just return a small valid graph to ensure score > 0
    # A 5-vertex graph has no 6-vertex subset, so it's always valid.
    # But we want to maximize n.
    # If n=20/21 fails, we should return the best found.
    # But we didn't keep track of "best" in terms of validity, only found valid ones.
    # So if we didn't find a valid one for 21/20, we return 5.
    
    # Better: Try n=10 quickly if 20/21 fails?
    # Or just return 5.
    
    return {'n': 5, 'edges': []}
