"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import math

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    n = 18
    random.seed(42)
    full_mask = (1 << n) - 1
    
    # Bitmask-based adjacency for speed
    # adj_mask[v] is a bitmask of neighbors of v
    
    def is_k4_free_masks(masks):
        # Check no K4 using bitmasks
        for a in range(n):
            for b in range(a+1, n):
                if not (masks[a] & (1 << b)):
                    continue
                common_ab = masks[a] & masks[b]
                for c in range(b+1, n):
                    if not (common_ab & (1 << c)):
                        continue
                    common_abc = common_ab & masks[c]
                    # Check if any d > c is in common_abc
                    remaining = common_abc & ~((1 << (c+1)) - 1)
                    if remaining:
                        return False
        return True
    
    def get_triangles(masks):
        """Return list of triangle bitmasks."""
        triangles = []
        for u in range(n):
            v_mask = masks[u] & ~((1 << (u + 1)) - 1)
            while v_mask:
                lsb = v_mask & -v_mask
                v = lsb.bit_length() - 1
                v_mask ^= lsb
                common = masks[u] & masks[v] & ~((1 << (v + 1)) - 1)
                while common:
                    wlsb = common & -common
                    w = wlsb.bit_length() - 1
                    common ^= wlsb
                    triangles.append((1 << u) | (1 << v) | (1 << w))
        return triangles
    
    def max_triangle_free_induced_masks(masks):
        """
        Find the size of the largest triangle-free induced subgraph.
        Equivalent to finding the minimum triangle transversal (hitting set).
        Answer = n - min_transversal_size
        """
        triangles = get_triangles(masks)
        if not triangles:
            return n
        
        num_tris = len(triangles)
        
        # Precompute: for each vertex, which triangles it hits
        vertex_hits = [0] * n
        for i, tri in enumerate(triangles):
            bits = tri
            while bits:
                lsb = bits & -bits
                v = lsb.bit_length() - 1
                vertex_hits[v] |= (1 << i)
                bits ^= lsb
        
        all_tris_mask = (1 << num_tris) - 1
        
        def find_min_transversal():
            """Find minimum size of a set of vertices that hits all triangles."""
            # Try m = 1, 2, 3, ...
            # For each m, check if a transversal of size m exists
            # Using backtracking with pruning
            
            # Sort vertices by number of triangles they hit (descending) for better pruning
            vertices_by_hits = sorted(range(n), key=lambda v: bin(vertex_hits[v]).count('1'), reverse=True)
            
            def backtrack(remaining_tris, chosen_count, max_count, start_idx):
                if remaining_tris == 0:
                    return True
                if chosen_count >= max_count:
                    return False
                if start_idx >= n:
                    return False
                
                # Pruning: if remaining vertices can't hit all remaining triangles
                # (this is a weak bound, but helps)
                
                for i in range(start_idx, n):
                    v = vertices_by_hits[i]
                    hit = vertex_hits[v] & remaining_tris
                    if hit == 0:
                        continue
                    
                    # Try including v
                    if backtrack(remaining_tris & ~hit, chosen_count + 1, max_count, i + 1):
                        return True
                    
                    # Pruning: if we don't include v, some triangles must be hit by others
                    # But we need to be careful - this is where it gets complex
                    # For now, just continue
                    
                return False
            
            for m in range(1, n + 1):
                if backtrack(all_tris_mask, 0, m, 0):
                    return m
            
            return n  # Should never reach here
        
        min_trans = find_min_transversal()
        return n - min_trans
    
    def random_greedy_k4_free():
        # Build a random K4-free graph greedily
        masks = [0] * n
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            common = masks[u] & masks[v]
            conflict = False
            m = common
            while m:
                w_bit = m & -m
                w = w_bit.bit_length() - 1
                if masks[w] & (m ^ w_bit):
                    conflict = True
                    break
                m ^= w_bit
            if not conflict:
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
        return masks
    
    def can_add_edge(masks, u, v):
        """Check if adding edge (u,v) creates a K4."""
        if masks[u] & (1 << v):
            return True
        common = masks[u] & masks[v]
        m = common
        while m:
            w_bit = m & -m
            w = w_bit.bit_length() - 1
            if masks[w] & (m ^ w_bit):
                return False
            m ^= w_bit
        return True
    
    # Generate multiple random starting points
    best_masks = None
    best_score = n + 1
    
    # First try the tripartite K6,6,6 as a baseline
    tri_masks = [0] * n
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    tri_masks[a] |= (1 << b)
                    tri_masks[b] |= (1 << a)
    
    score = max_triangle_free_induced_masks(tri_masks)
    if score < best_score:
        best_score = score
        best_masks = tri_masks[:]
    
    # Try random greedy constructions
    attempts = 0
    while time.time() < deadline - 25:
        attempts += 1
        masks = random_greedy_k4_free()
        if not is_k4_free_masks(masks):
            continue
        score = max_triangle_free_induced_masks(masks)
        if score < best_score:
            best_score = score
            best_masks = masks[:]
            # Local search from this point
            while time.time() < deadline - 10:
                improved = False
                all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
                random.shuffle(all_edges)
                for u, v in all_edges:
                    if time.time() > deadline - 10:
                        break
                    # Try adding edge
                    if not (masks[u] & (1 << v)):
                        if can_add_edge(masks, u, v):
                            masks[u] |= (1 << v)
                            masks[v] |= (1 << u)
                            new_score = max_triangle_free_induced_masks(masks)
                            if new_score < best_score:
                                best_score = new_score
                                best_masks = masks[:]
                                improved = True
                            else:
                                masks[u] ^= (1 << v)
                                masks[v] ^= (1 << u)
                    # Try removing edge
                    else:
                        masks[u] ^= (1 << v)
                        masks[v] ^= (1 << u)
                        new_score = max_triangle_free_induced_masks(masks)
                        if new_score < best_score:
                            best_score = new_score
                            best_masks = masks[:]
                            improved = True
                        else:
                            masks[u] |= (1 << v)
                            masks[v] |= (1 << u)
                if not improved:
                    break
    
    # Convert best_masks to edge list
    result = []
    for u in range(n):
        for v in range(u+1, n):
            if best_masks[u] & (1 << v):
                result.append([u, v])
    return result
# EVOLVE-BLOCK-END
