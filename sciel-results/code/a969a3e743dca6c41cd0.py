import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin for evaluator overhead (300s limit)
    
    def is_prime(x):
        if x < 2:
            return False
        if x == 2:
            return True
        if x % 2 == 0:
            return False
        for i in range(3, int(x**0.5) + 1, 2):
            if x % i == 0:
                return False
        return True
    
    def paley_graph(n):
        """Generate Paley graph on n vertices (n must be prime, n ≡ 1 mod 4)."""
        if n < 2:
            return []
        if not is_prime(n) or n % 4 != 1:
            return None
        
        # Quadratic residues mod n
        residues = set()
        for x in range(1, n):
            residues.add((x * x) % n)
        
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n in residues:
                    edges.append((i, j))
        return edges
    
    def build_adjacency(n, edges):
        """Build adjacency bitmasks from edge list."""
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def adj_to_edges(n, adj):
        """Convert adjacency bitmasks to edge list."""
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append((i, j))
                mask ^= lsb
        return edges

    def find_violation(n, adj):
        """
        Find a K5 or an independent set of size 5.
        Returns (type, vertices) where type is 'K5' or 'IND5', or (None, None) if valid.
        vertices is a tuple of 5 distinct vertices.
        """
        # Check for K5
        for u in range(n):
            mask_u = adj[u]
            while mask_u:
                lsb = mask_u & -mask_u
                v = lsb.bit_length() - 1
                if u < v:
                    common = mask_u & adj[v]
                    mask_common = common
                    while mask_common:
                        lsb_x = mask_common & -mask_common
                        x = lsb_x.bit_length() - 1
                        neighbors_x_in_common = adj[x] & common
                        mask_neigh = neighbors_x_in_common
                        while mask_neigh:
                            lsb_y = mask_neigh & -mask_neigh
                            y = lsb_y.bit_length() - 1
                            z_mask = adj[y] & neighbors_x_in_common
                            if z_mask:
                                lsb_z = z_mask & -z_mask
                                z = lsb_z.bit_length() - 1
                                return ('K5', (u, v, x, y, z))
                        mask_neigh ^= lsb_y
                mask_u ^= lsb
        
        # Check for Independent 5
        full_mask = (1 << n) - 1
        comp_adj = [0] * n
        for i in range(n):
            comp_adj[i] = full_mask ^ adj[i] ^ (1 << i)
        
        for u in range(n):
            mask_u = comp_adj[u]
            while mask_u:
                lsb = mask_u & -mask_u
                v = lsb.bit_length() - 1
                if u < v:
                    common = mask_u & comp_adj[v]
                    mask_common = common
                    while mask_common:
                        lsb_x = mask_common & -mask_common
                        x = lsb_x.bit_length() - 1
                        neighbors_x_in_common = comp_adj[x] & common
                        mask_neigh = neighbors_x_in_common
                        while mask_neigh:
                            lsb_y = mask_neigh & -mask_neigh
                            y = lsb_y.bit_length() - 1
                            z_mask = comp_adj[y] & neighbors_x_in_common
                            if z_mask:
                                lsb_z = z_mask & -z_mask
                                z = lsb_z.bit_length() - 1
                                return ('IND5', (u, v, x, y, z))
                        mask_neigh ^= lsb_y
                mask_u ^= lsb
        
        return (None, None)

    def is_valid(n, adj):
        """Check if graph is (5,5)-free."""
        v_type, _ = find_violation(n, adj)
        return v_type is None

    # Start with the best known valid Paley graph
    best_n = 37
    best_edges = paley_graph(37)
    best_adj = build_adjacency(37, best_edges)
    
    if not is_valid(37, best_adj):
        # Fallback
        return 37, best_edges

    best_score = 37
    best_result_edges = best_edges

    # Try to extend to larger graphs using local search
    # We will try to find a valid graph for N = 38, 39, ..., 46
    
    for target_n in range(38, 47):
        if time.time() > deadline - 10:
            break
            
        # Initialize graph for target_n
        # Strategy: Start with Paley 37, add random vertices, then local search
        
        found_valid = False
        
        for restart in range(100):
            if time.time() > deadline - 5:
                break
                
            # Initialize
            adj = [0] * target_n
            # Copy Paley 37 structure for first 37 vertices
            if target_n >= 37:
                for i in range(37):
                    adj[i] = best_adj[i]
            # Add random edges for new vertices and between old/new
            for i in range(37, target_n):
                for j in range(i):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local Search
            steps = 500
            for step in range(steps):
                if time.time() > deadline - 2:
                    break
                    
                v_type, verts = find_violation(target_n, adj)
                if v_type is None:
                    # Valid!
                    found_valid = True
                    break
                
                # Get the actual vertices of the violation
                u, v, x, y, z = verts
                candidates = [u, v, x, y, z]
                # Pick a random pair from candidates to flip
                i, j = random.sample(candidates, 2)
                
                # Flip edge (i, j)
                if adj[i] & (1 << j):
                    adj[i] ^= (1 << j)
                    adj[j] ^= (1 << i)
                else:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
            
            if found_valid:
                break
            
        if found_valid:
            # Update best
            best_score = target_n
            best_result_edges = adj_to_edges(target_n, adj)
            # Continue to next target_n?
            # We can use this valid graph as a seed for the next target?
            # For simplicity, we just record it.
            
    # Return the best found
    return best_score, best_result_edges
