import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 110
    start_time = time.time()
    
    def count_violations(neighbors, n):
        """
        Counts the number of monochromatic K2,2,2 subgraphs in the graph defined by 'neighbors'.
        'neighbors' is a list of integers (bitmasks).
        Each K2,2,2 is counted 3 times (once for each pair of the 3 pairs acting as the 'base' K2,2).
        We return the raw count.
        """
        total = 0
        # Iterate over all pairs of vertices (i, j)
        for i in range(n):
            for j in range(i + 1, n):
                # Common neighbors of i and j
                # We need to find another pair (k, l) disjoint from {i, j}
                # such that {i, j, k, l} forms a K2,2.
                # This means k and l must be in N(i) & N(j).
                # And edge (k, l) must exist? No, K2,2 requires edges (i,k), (i,l), (j,k), (j,l).
                # It does NOT require edge (k,l).
                # So, for a fixed pair {i, j}, the set of candidates for {k, l} is S = N(i) & N(j).
                # We need to pick two distinct vertices k, l from S, such that k != i, l != j (already true if no self loops)
                # and k, l are distinct from i, j.
                # Actually, S contains i and j if there are self loops, but there aren't.
                # However, S might contain vertices that are i or j? No, N(i) doesn't contain i.
                # So S is disjoint from {i, j}? No, N(i) & N(j) can contain vertices other than i, j.
                # Wait, if k is in N(i) & N(j), then edges (i,k) and (j,k) exist.
                # For {i, j, k, l} to be a K2,2, we need edges (i,l), (j,l) as well.
                # So l must also be in N(i) & N(j).
                # So we just need to pick any pair {k, l} from S = N(i) & N(j).
                # BUT, k and l must be distinct from i and j.
                # Since N(i) does not contain i, and N(j) does not contain j, 
                # if the graph is simple, S does not contain i or j?
                # N(i) & N(j) contains x if x is neighbor of i AND x is neighbor of j.
                # i is not neighbor of i. So i is not in N(i). So i is not in N(i) & N(j).
                # Similarly j is not in S.
                # So S is automatically disjoint from {i, j}.
                
                mask = neighbors[i] & neighbors[j]
                
                # We need to count pairs {k, l} from mask.
                # But we must ensure that for EVERY pair {k, l} we pick, 
                # we are counting a K2,2,2?
                # No. The definition of K2,2,2 is 3 pairs.
                # My counting logic:
                # Fix pair 1: {i, j}.
                # Fix pair 2: {k, l}.
                # For {i, j, k, l} to be a K2,2, k, l must be in N(i) & N(j).
                # Let S_ij = N(i) & N(j).
                # Any pair {k, l} from S_ij forms a K2,2 with {i, j}.
                # Now, for this K2,2 to be part of a K2,2,2, we need a third pair {m, o}.
                # {m, o} must be in N(i) & N(j) & N(k) & N(l).
                # Let S_ijkl = N(i) & N(j) & N(k) & N(l).
                # The number of such third pairs is C(|S_ijkl|, 2).
                
                # This approach requires iterating over all pairs {k, l} in S_ij.
                # Size of S_ij can be up to n-2 = 28.
                # Number of pairs in S_ij is C(|S_ij|, 2).
                # Total iterations over i, j is C(n, 2).
                # This is O(n^2 * n^2) = O(n^4) in worst case.
                # For n=30, C(30,2)=435. Max |S|=28. C(28,2)=378.
                # 435 * 378 * (bit ops) ~ 160k operations per color.
                # This is very fast.
                
                # Extract vertices in mask
                # We can just iterate k, l
                verts = []
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    verts.append(idx)
                    m ^= lsb
                
                # Iterate over all pairs k, l in verts
                for a_idx in range(len(verts)):
                    k = verts[a_idx]
                    for b_idx in range(a_idx + 1, len(verts)):
                        l = verts[b_idx]
                        # Check if {i, j, k, l} is a K2,2.
                        # By construction, k, l are in N(i) & N(j).
                        # So edges (i,k), (i,l), (j,k), (j,l) exist.
                        # This IS a K2,2.
                        
                        # Now find common neighbors of all 4: i, j, k, l
                        common = neighbors[i] & neighbors[j] & neighbors[k] & neighbors[l]
                        # Remove i, j, k, l from common (though they shouldn't be there)
                        common &= ~(1 << i)
                        common &= ~(1 << j)
                        common &= ~(1 << k)
                        common &= ~(1 << l)
                        
                        # Count pairs in common
                        cnt = bin(common).count('1')
                        if cnt >= 2:
                            total += cnt * (cnt - 1) // 2
        return total

    def solve_for_n(n):
        n_edges = n * (n - 1) // 2
        if n_edges == 0:
            return [0] * n_edges
        
        # Try multiple random restarts
        max_restarts = 50
        restarts = 0
        
        while restarts < max_restarts and time.time() < deadline - 5:
            restarts += 1
            colors = [random.randint(0, 1) for _ in range(n_edges)]
            
            # Build initial neighbors
            neighbors0 = [0] * n
            neighbors1 = [0] * n
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if colors[idx] == 0:
                        neighbors0[i] |= (1 << j)
                        neighbors0[j] |= (1 << i)
                    else:
                        neighbors1[i] |= (1 << j)
                        neighbors1[j] |= (1 << i)
                    idx += 1
            
            current_violations = count_violations(neighbors0, n) + count_violations(neighbors1, n)
            
            if current_violations == 0:
                return colors
            
            improved = True
            no_improve_count = 0
            max_no_improve = 1000
            
            while improved and no_improve_count < max_no_improve and time.time() < deadline - 5:
                improved = False
                # Shuffle edges to avoid bias
                edge_indices = list(range(n_edges))
                random.shuffle(edge_indices)
                
                for e in edge_indices:
                    if time.time() > deadline - 5:
                        break
                    
                    # Try flipping edge e
                    # Determine i, j from e
                    # e is index in lexicographic order
                    # i = floor((-1 + sqrt(1 + 8e)) / 2) ?
                    # Simpler: precompute edge mapping
                    # Or just loop i, j to find e? No, too slow.
                    # Precompute a list of (i, j) for each edge index.
                    
                    # Let's precompute edge_map outside the loop for speed
                    # Actually, we can just re-calculate neighbors for the flipped edge
                    # and then re-calculate total violations.
                    
                    # To speed up, we can update neighbors incrementally.
                    # But count_violations is fast enough.
                    
                    # Flip
                    old_color = colors[e]
                    colors[e] = 1 - old_color
                    
                    # Update neighbors
                    # Find i, j for edge e
                    # We need a fast way to get i, j from e.
                    # e = i*(2n-i-1)/2 + (j-i-1) ?
                    # Let's just store the list of pairs.
                    pass # Placeholder for logic below

            # If we exit the while loop without finding 0 violations, we restart
            # But we should keep the best result?
            # The task asks for a valid coloring. If we don't find one, we return fallback.
            
        return None

    # Precompute edge pairs for fast access
    def get_edge_pairs(n):
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
        return pairs

    # Revised solve_for_n with precomputed pairs
    def solve_for_n_optimized(n):
        n_edges = n * (n - 1) // 2
        edge_pairs = get_edge_pairs(n)
        
        max_restarts = 20
        for _ in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            colors = [random.randint(0, 1) for _ in range(n_edges)]
            
            neighbors0 = [0] * n
            neighbors1 = [0] * n
            idx = 0
            for i, j in edge_pairs:
                if colors[idx] == 0:
                    neighbors0[i] |= (1 << j)
                    neighbors0[j] |= (1 << i)
                else:
                    neighbors1[i] |= (1 << j)
                    neighbors1[j] |= (1 << i)
                idx += 1
            
            current_violations = count_violations(neighbors0, n) + count_violations(neighbors1, n)
            
            if current_violations == 0:
                return colors
            
            improved = True
            stall_counter = 0
            max_stall = 500
            
            while improved and stall_counter < max_stall and time.time() < deadline - 5:
                improved = False
                stall_counter += 1
                
                # Try all edges? Or random?
                # Random is usually better for escaping local minima.
                # But if we try all edges and accept any improvement, it's steepest descent.
                # Let's try a few random edges per iteration.
                
                for _ in range(10):
                    if time.time() > deadline - 5:
                        break
                    e = random.randint(0, n_edges - 1)
                    i, j = edge_pairs[e]
                    
                    old_color = colors[e]
                    new_color = 1 - old_color
                    colors[e] = new_color
                    
                    # Update neighbors
                    if new_color == 0:
                        neighbors0[i] |= (1 << j)
                        neighbors0[j] |= (1 << i)
                        neighbors1[i] &= ~(1 << j)
                        neighbors1[j] &= ~(1 << i)
                    else:
                        neighbors1[i] |= (1 << j)
                        neighbors1[j] |= (1 << i)
                        neighbors0[i] &= ~(1 << j)
                        neighbors0[j] &= ~(1 << i)
                    
                    new_violations = count_violations(neighbors0, n) + count_violations(neighbors1, n)
                    
                    if new_violations < current_violations:
                        current_violations = new_violations
                        improved = True
                        stall_counter = 0
                        if current_violations == 0:
                            return colors
                    else:
                        # Revert
                        colors[e] = old_color
                        if old_color == 0:
                            neighbors0[i] |= (1 << j)
                            neighbors0[j] |= (1 << i)
                            neighbors1[i] &= ~(1 << j)
                            neighbors1[j] &= ~(1 << i)
                        else:
                            neighbors1[i] |= (1 << j)
                            neighbors1[j] |= (1 << i)
                            neighbors0[i] &= ~(1 << j)
                            neighbors0[j] &= ~(1 << i)
                            
        return None

    # Try n=30 down to 18
    for n in range(30, 17, -1):
        if time.time() > deadline - 10:
            break
        result = solve_for_n_optimized(n)
        if result is not None:
            return {'n': n, 'colors': result}
            
    # Fallback to Paley 17
    squares = {x * x % 17 for x in range(1, 17)}
    colors = [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]
    return {'n': 17, 'colors': colors}

# EVOLVE-BLOCK-END
