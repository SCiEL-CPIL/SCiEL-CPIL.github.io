import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    def _build_edge_list(n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        return edges
    
    def _edge_index(n, i, j):
        # i < j, lexicographic order
        # edges before row i: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)//2
        # offset in row i: j - i - 1
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    def _has_c6_with_edge(adj, n, i, j, c):
        """Check if adding edge (i,j) in color c creates a C6 in G_c.
        C6: i-a-b-e-f-j where all edges are color c and all vertices distinct.
        """
        # Neighbors of i in color c (excluding j)
        mask_i = adj[c][i] & ~(1 << j)
        if mask_i == 0:
            return False
        
        # Neighbors of j in color c (excluding i)
        mask_j = adj[c][j] & ~(1 << i)
        
        a_mask = mask_i
        while a_mask:
            a_bit = a_mask & -a_mask
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            
            # Neighbors of a in color c, excluding i, j, a
            mask_a = adj[c][a] & ~(1 << i) & ~(1 << j) & ~a_bit
            if mask_a == 0:
                continue
            
            b_mask = mask_a
            while b_mask:
                b_bit = b_mask & -b_mask
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit
                
                # Neighbors of b in color c, excluding i, j, a, b
                mask_b = adj[c][b] & ~(1 << i) & ~(1 << j) & ~a_bit & ~b_bit
                if mask_b == 0:
                    continue
                
                e_mask = mask_b
                while e_mask:
                    e_bit = e_mask & -e_mask
                    e = e_bit.bit_length() - 1
                    e_mask ^= e_bit
                    
                    # Need f: neighbor of e in color c, neighbor of j in color c
                    # f not in {i, j, a, b, e}
                    mask_e = adj[c][e] & ~(1 << i) & ~(1 << j) & ~a_bit & ~b_bit & ~e_bit
                    # f must be in mask_e AND in mask_j
                    if mask_e & mask_j:
                        return True
        
        return False
    
    def _count_c6s(adj, n):
        """Count total C6 violations across all colors."""
        count = 0
        for c in range(4):
            for v in range(n):
                # For each pair of neighbors of v in color c, count paths of length 3 between them
                # through other vertices, forming C6s
                # Actually, let's just check all edges for potential C6
                pass
        # Better: for each color, check each edge if it's part of a C6
        # But that's expensive. Let's use a different approach.
        # For each color c, for each 6-tuple... no, too expensive.
        # Let's just check: for each edge (i,j) in color c, check if there's a path of length 5 between i and j in G_c \ {(i,j)}
        # That's the same as _has_c6_with_edge but without adding the edge.
        # Actually, for counting violations, we need to check if the current coloring has any C6.
        # A C6 exists if there are 6 distinct vertices v0..v5 with edges (v0,v1),(v1,v2),(v2,v3),(v3,v4),(v4,v5),(v5,v0) all in same color.
        # This is expensive to check directly.
        # For local search, we'll track violations incrementally.
        return 0  # Placeholder - we'll use incremental tracking
    
    def _local_search(n, max_time=100):
        """Random local search for a valid 4-coloring of K_n with no monochromatic C6."""
        edge_list = _build_edge_list(n)
        num_edges = len(edge_list)
        
        # Initialize with random coloring
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Build adjacency masks
        adj = [[0] * n for _ in range(4)]
        for idx, (i, j) in enumerate(edge_list):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
        
        def _rebuild_adj():
            for c in range(4):
                for v in range(n):
                    adj[c][v] = 0
            for idx, (i, j) in enumerate(edge_list):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        
        def _check_valid():
            """Check if current coloring has no monochromatic C6."""
            for c in range(4):
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[c][i] & (1 << j):
                            # Edge (i,j) is in color c. Check if it's part of a C6.
                            # Remove edge (i,j) temporarily and check for path of length 5
                            adj[c][i] &= ~(1 << j)
                            adj[c][j] &= ~(1 << i)
                            if _has_c6_with_edge(adj, n, i, j, c):
                                # Restore
                                adj[c][i] |= (1 << j)
                                adj[c][j] |= (1 << i)
                                return False
                            # Restore
                            adj[c][i] |= (1 << j)
                            adj[c][j] |= (1 << i)
            return True
        
        # Incremental violation tracking
        # For each edge, track how many C6s it participates in
        # This is complex. Let's use a simpler approach:
        # At each step, pick a random edge, try all 4 colors, pick the one that minimizes new violations.
        
        def _violations_if_edge(idx, new_color):
            """Count how many C6s would be created if we set edge idx to new_color."""
            i, j = edge_list[idx]
            old_color = colors[idx]
            if old_color == new_color:
                return 0
            
            # Temporarily change the edge
            adj[old_color][i] &= ~(1 << j)
            adj[old_color][j] &= ~(1 << i)
            adj[new_color][i] |= (1 << j)
            adj[new_color][j] |= (1 << i)
            
            # Check C6s involving this edge
            # C6s where (i,j) is one of the edges
            violations = 0
            # Check if (i,j) in new_color creates a C6
            if _has_c6_with_edge(adj, n, i, j, new_color):
                violations += 1
            
            # Also need to check C6s where (i,j) was in old_color and we removed it
            # But those C6s might still exist through other paths... this is complex.
            # For simplicity, let's just check the new edge.
            
            # Restore
            adj[new_color][i] &= ~(1 << j)
            adj[new_color][j] &= ~(1 << i)
            adj[old_color][i] |= (1 << j)
            adj[old_color][j] |= (1 << i)
            
            return violations
        
        # Simpler approach: at each step, pick a random edge, try all 4 colors,
        # and pick the one that minimizes total C6 count (approximated by checking the edge)
        
        best_score = float('inf')
        best_colors = None
        best_adj = None
        
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Random restart
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            _rebuild_adj()
            
            # Local search iterations
            for iteration in range(200000):
                if time.time() > deadline - 5:
                    break
                
                # Check if valid
                if _check_valid():
                    if best_colors is None or True:
                        best_colors = colors[:]
                        return best_colors, n
                
                # Pick a random edge
                idx = random.randint(0, num_edges - 1)
                i, j = edge_list[idx]
                
                # Try all 4 colors, pick best
                best_c = colors[idx]
                best_v = _violations_if_edge(idx, best_c)
                
                for c in range(4):
                    if c == colors[idx]:
                        continue
                    v = _violations_if_edge(idx, c)
                    if v < best_v or (v == best_v and random.random() < 0.3):
                        best_v = v
                        best_c = c
                
                if best_c != colors[idx]:
                    old_c = colors[idx]
                    colors[idx] = best_c
                    adj[old_c][i] &= ~(1 << j)
                    adj[old_c][j] &= ~(1 << i)
                    adj[best_c][i] |= (1 << j)
                    adj[best_c][j] |= (1 << i)
        
        return None, n
    
    # Try n=18 first, then fall back to n=17
    for n_target in [18, 17]:
        if time.time() > deadline - 10:
            break
        
        result, n_found = _local_search(n_target)
        if result is not None:
            return {'n': n_found, 'colors': result}
    
    # Fallback: return n=1
    return {'n': 1, 'colors': []}
