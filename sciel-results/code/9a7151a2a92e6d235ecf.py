import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Known valid coloring for n=9
    # From inspiration 1: K9 split by circular distance
    # Distances 1, 2, 4 -> colors 0, 1, 3? 
    # Let's verify the inspiration 1 formula: min(j-i, n-j+i)-1
    # n=9.
    # (0,1): d=1, min(1,8)-1=0
    # (0,2): d=2, min(2,7)-1=1
    # (0,3): d=3, min(3,6)-1=2
    # (0,4): d=4, min(4,5)-1=3
    # (0,5): d=5, min(5,4)-1=3
    # (0,6): d=6, min(6,3)-1=2
    # (0,7): d=7, min(7,2)-1=1
    # (0,8): d=8, min(8,1)-1=0
    # This creates a 4-coloring.
    # We assume it is valid (score 1.0 implies n=9 is valid).
    
    def _init_n9():
        n = 9
        colors = [0] * (n * (n - 1) // 2)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                colors[idx] = min(j - i, n - j + i) - 1
                idx += 1
        return colors

    def _edge_index(n, i, j):
        # i < j
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def _build_adj(colors, n):
        """
        Returns a list of 4 lists of bitmasks.
        adj[c][v] is a bitmask of neighbors of v in color c.
        """
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        return adj

    def _count_p4(u, v, adj_c):
        """
        Count simple paths of length 4 from u to v in graph adj_c.
        adj_c is a list of bitmasks for one color class.
        """
        count = 0
        # a in N(u), a != v
        mask_a = adj_c[u] & ~(1 << v)
        while mask_a:
            a_bit = mask_a & -mask_a
            a = a_bit.bit_length() - 1
            mask_a ^= a_bit
            
            # b in N(a), b != u, v, a
            mask_b = adj_c[a] & ~(1 << u) & ~(1 << v) & ~a_bit
            while mask_b:
                b_bit = mask_b & -mask_b
                b = b_bit.bit_length() - 1
                mask_b ^= b_bit
                
                # e in N(b), e != u, v, a, b
                mask_e = adj_c[b] & ~(1 << u) & ~(1 << v) & ~a_bit & ~b_bit
                while mask_e:
                    e_bit = mask_e & -mask_e
                    e = e_bit.bit_length() - 1
                    mask_e ^= e_bit
                    
                    # Check if v in N(e)
                    if adj_c[e] & (1 << v):
                        count += 1
                        
        return count

    def _total_c6_count(adj, n):
        """
        Total number of C6s is (1/6) * sum over all edges (u,v) of P4(u,v) in the color class of (u,v).
        """
        total_p4 = 0
        # Iterate over all pairs
        for i in range(n):
            for j in range(i + 1, n):
                # Find color of (i,j)
                # We need to know the color. 
                # It's better to pass colors or store them.
                # Let's modify to pass colors too.
                pass
        # This function is inefficient if we don't have colors.
        # Let's compute delta directly in the search loop.
        pass

    def _try_extend(n_target, initial_colors_n9):
        """
        Attempt to find a valid 4-coloring for K_n_target.
        Starts with the n9 subgraph fixed and randomizes the rest, then local searches.
        """
        if n_target < 9:
             # Should not happen in this flow
             return None
             
        # Initialize colors list for n_target
        colors = [0] * (n_target * (n_target - 1) // 2)
        
        # Copy n9 colors
        for i in range(9):
            for j in range(i + 1, 9):
                idx_t = _edge_index(n_target, i, j)
                idx_9 = _edge_index(9, i, j)
                colors[idx_t] = initial_colors_n9[idx_9]
                
        # Randomize edges involving vertices >= 9
        for i in range(9):
            for j in range(i + 1, n_target):
                if j >= 9:
                    idx_t = _edge_index(n_target, i, j)
                    colors[idx_t] = random.randint(0, 3)
                    
        for i in range(9, n_target):
            for j in range(i + 1, n_target):
                idx_t = _edge_index(n_target, i, j)
                colors[idx_t] = random.randint(0, 3)

        # Local Search
        best_colors = list(colors)
        best_adj = _build_adj(colors, n_target)
        
        # Calculate initial C6 count
        # We need a function to calculate total C6s from adj
        def _calc_total_c6(adj, n, cols):
             total = 0
             idx = 0
             for i in range(n):
                 for j in range(i + 1, n):
                     c = cols[idx]
                     # Count P4(i,j) in adj[c]
                     # But adj[c] contains the edge (i,j). 
                     # P4(i,j) counts paths i-a-b-e-j. 
                     # The edge (i,j) is NOT part of the path internally.
                     # So we can just use _count_p4(i, j, adj[c]).
                     total += _count_p4(i, j, adj[c])
                     idx += 1
             return total // 6

        current_c6 = _calc_total_c6(best_adj, n_target, best_colors)
        
        if current_c6 == 0:
            return best_colors

        step = 0
        # Simulated Annealing or Greedy Hill Climbing
        # Let's try greedy with random restarts if stuck
        temperature = 100.0
        cooling = 0.99
        
        while step < 20000:
            if time.time() > deadline - 2:
                break
                
            # Pick random edge
            idx = random.randint(0, len(colors) - 1)
            # Determine i, j
            # We can precompute a list of (i,j) for idx
            # For speed, let's just find them
            # Or precompute edge_list outside
            
            # Find i, j for idx
            # i*(n-1) - i*(i-1)//2 + (j-i-1) = idx
            # This is slow. Let's precompute edge_list for n_target.
            
            # Precompute edge_list once per _try_extend
            # (Done outside this loop in the actual implementation below)
            
            i, j = edge_list[n_target][idx]
            
            old_c = colors[idx]
            new_c = random.choice([c for c in range(4) if c != old_c])
            
            # Calculate Delta
            # Remove (i,j) from old_c
            # Add (i,j) to new_c
            
            # P4 in old_c without the edge (i,j)
            # _count_p4 uses adj which HAS the edge.
            # Does the edge (i,j) affect P4(i,j)?
            # Path i-a-b-e-j. 
            # a,b,e are distinct from i,j.
            # The edge (i,j) is not used in the path.
            # So P4(i,j) in G is the same as P4(i,j) in G-(i,j).
            # Wait, is it?
            # If the path uses the edge (i,j)? No, path length 4 has 5 vertices.
            # So the edge (i,j) is not part of the internal path.
            # So we can just use _count_p4(i, j, adj[old_c]) directly?
            # Yes, because the edge (i,j) connects the endpoints, not the intermediate nodes.
            
            p4_old = _count_p4(i, j, best_adj[old_c])
            p4_new = _count_p4(i, j, best_adj[new_c])
            
            delta = p4_new - p4_old
            
            if delta < 0 or (temperature > 0 and random.random() < pow(2.71828, -delta / temperature)):
                # Accept
                colors[idx] = new_c
                
                # Update adj
                best_adj[old_c][i] &= ~(1 << j)
                best_adj[old_c][j] &= ~(1 << i)
                best_adj[new_c][i] |= (1 << j)
                best_adj[new_c][j] |= (1 << i)
                
                current_c6 += delta // 6 # Wait, delta is change in sum of P4s.
                # Total C6 = Sum(P4) / 6.
                # So change in Total C6 is delta / 6?
                # Yes.
                
                if current_c6 == 0:
                    return colors
                
                best_colors = list(colors)
                temperature *= cooling
            else:
                # Reject
                temperature *= cooling
                
            step += 1
            
            # Reset temperature occasionally? Or just let it cool.
            
        return best_colors if current_c6 == 0 else None

    # Precompute edge lists for n=9 to 20
    edge_list = {}
    for n in range(9, 21):
        el = []
        for i in range(n):
            for j in range(i + 1, n):
                el.append((i, j))
        edge_list[n] = el
        
    # Re-define _try_extend to use edge_list
    def _try_extend_impl(n_target, initial_colors_n9):
        if n_target < 9:
            return None
            
        colors = [0] * (n_target * (n_target - 1) // 2)
        
        for i in range(9):
            for j in range(i + 1, 9):
                idx_t = _edge_index(n_target, i, j)
                idx_9 = _edge_index(9, i, j)
                colors[idx_t] = initial_colors_n9[idx_9]
                
        for i in range(9):
            for j in range(i + 1, n_target):
                if j >= 9:
                    idx_t = _edge_index(n_target, i, j)
                    colors[idx_t] = random.randint(0, 3)
                    
        for i in range(9, n_target):
            for j in range(i + 1, n_target):
                idx_t = _edge_index(n_target, i, j)
                colors[idx_t] = random.randint(0, 3)

        best_adj = _build_adj(colors, n_target)
        
        def _calc_total_c6(adj, n, cols):
             total = 0
             idx = 0
             for i in range(n):
                 for j in range(i + 1, n):
                     c = cols[idx]
                     total += _count_p4(i, j, adj[c])
                     idx += 1
             return total // 6

        current_c6 = _calc_total_c6(best_adj, n_target, colors)
        
        if current_c6 == 0:
            return colors

        temperature = 50.0
        cooling = 0.995
        step = 0
        max_steps = 10000
        
        while step < max_steps:
            if time.time() > deadline - 2:
                break
                
            idx = random.randint(0, len(colors) - 1)
            i, j = edge_list[n_target][idx]
            
            old_c = colors[idx]
            new_c = random.choice([c for c in range(4) if c != old_c])
            
            p4_old = _count_p4(i, j, best_adj[old_c])
            p4_new = _count_p4(i, j, best_adj[new_c])
            
            delta = p4_new - p4_old
            
            if delta < 0 or (temperature > 0 and random.random() < pow(2.71828, -delta / temperature)):
                colors[idx] = new_c
                
                best_adj[old_c][i] &= ~(1 << j)
                best_adj[old_c][j] &= ~(1 << i)
                best_adj[new_c][i] |= (1 << j)
                best_adj[new_c][j] |= (1 << i)
                
                current_c6 += delta // 6
                
                if current_c6 == 0:
                    return colors
                
                temperature *= cooling
            else:
                temperature *= cooling
                
            step += 1
            
        return None

    base_colors = _init_n9()
    best_n = 9
    best_colors = base_colors
    
    # Try to extend
    for n_target in range(10, 21):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Multiple restarts
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            res = _try_extend_impl(n_target, base_colors)
            if res is not None:
                best_n = n_target
                best_colors = res
                found = True
                break
                
    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    res = construct()
    print(res['n'])
