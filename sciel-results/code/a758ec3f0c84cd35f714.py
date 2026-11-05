import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 295
    
    # The grid is [0, 64)^2.
    # We construct a set of points in a sub-grid [0, p)^2 where p is prime.
    # Using a prime field allows us to use multiplicative group properties.
    # A known result is that for a prime p, the set {(x, x^k) : x in F_p} 
    # often has good "no-isosceles" properties for certain k, 
    # specifically related to the number of solutions to quadratic equations 
    # arising from the isosceles condition.
    
    # The isosceles condition for points P, Q, R is d(P,Q) = d(P,R) or similar.
    # d(P,Q)^2 = (x_p - x_q)^2 + (y_p - y_q)^2.
    # If y = x^k, then d^2 = (x-y)^2 + (x^k - y^k)^2.
    # We want to ensure that for any distinct x, y, z, the distances are distinct.
    
    # Let's try p=61 (largest prime < 64).
    p = 61
    
    # We will try different exponents k.
    # k=1 is a line (bad).
    # k=2 is a parabola.
    # k=p-1 is trivial (0 or 1).
    # We want k such that the map x -> x^k is a permutation (gcd(k, p-1)=1) 
    # or has good distribution.
    
    # Actually, a stronger construction comes from finite geometry.
    # But let's first test simple power functions on F_p.
    
    best_points = []
    best_score = 0
    
    # Precompute squared distances for the p x p grid to speed up verification
    # We only care about points in [0, p) x [0, p)
    xs = np.arange(p)
    ys = np.arange(p)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2_grid = (dx * dx + dy * dy).astype(np.int64)
    
    # Try various exponents k
    # k must be such that x^k mod p is well-defined.
    # We iterate k from 2 to p-2.
    
    for k in range(2, p-1):
        if time.time() > deadline - 5:
            break
            
        # Generate points
        # x ranges 0..p-1
        # y = x^k mod p
        # Note: 0^k = 0 for k>0.
        x_coords = np.arange(p)
        y_coords = np.array([pow(x, k, p) for x in x_coords])
        
        points = list(zip(x_coords, y_coords))
        
        # Verify no isosceles triangles
        # Condition: For every point i, distances to all other points j are distinct.
        # AND for every pair i, j, the distance d_ij is not equal to d_ik for any k != j.
        # This is equivalent to: The distance matrix D (p x p) has distinct entries in each row (excluding diagonal).
        
        # Let's build the distance matrix for these p points.
        # Indices in the p x p grid:
        # idx = x * p + y
        
        indices = x_coords * p + y_coords
        
        # Extract submatrix of dist2_grid
        # D[i, j] = dist2_grid[indices[i], indices[j]]
        D = dist2_grid[indices][:, indices]
        
        # Check if each row has distinct non-diagonal elements
        valid = True
        for i in range(p):
            row = D[i]
            # Exclude diagonal (distance 0)
            row_no_diag = np.delete(row, i)
            # Check for duplicates
            if len(np.unique(row_no_diag)) != len(row_no_diag):
                valid = False
                break
        
        if valid:
            # We found a valid set of size p.
            # Can we add more points?
            # The grid is 64x64. We used 61x61.
            # We can try to add points from the remaining 64x64 \ 61x61 area?
            # Or just return this set.
            # p=61 is a good score.
            
            # Let's see if we can extend it.
            # The current set S has p points.
            # We can try to greedily add points from the full 64x64 grid.
            
            selected = set(points)
            selected_indices = set(indices)
            
            # Maintain distance sets for verification
            # dist_sets[i] = set of distances from point i to others
            # This is O(p^2) memory and time to init, which is fine.
            
            # Actually, let's just use the full 64x64 dist2 matrix for the greedy phase.
            N = 64
            xs_full = np.repeat(np.arange(N), N)
            ys_full = np.tile(np.arange(N), N)
            dx_full = xs_full[:, None] - xs_full[None, :]
            dy_full = ys_full[:, None] - ys_full[None, :]
            dist2_full = (dx_full * dx_full + dy_full * dy_full).astype(np.int64)
            
            # Initialize greedy state
            si = list(indices)
            ud = [set() for _ in range(p)]
            
            # Populate ud
            for i in range(p):
                for j in range(p):
                    if i != j:
                        ud[i].add(dist2_full[si[i], si[j]])
            
            # Try to add more points
            all_indices = list(range(N*N))
            # Shuffle for randomization in greedy
            import random
            random.shuffle(all_indices)
            
            # We need to be careful with time.
            # Greedy addition is O(N^2 * p) per step? No, O(p) per candidate.
            # Total candidates ~ 4000. p ~ 60. 4000*60 = 240,000 ops. Fast.
            
            for p_idx in all_indices:
                if p_idx in selected_indices:
                    continue
                
                k_cur = len(si)
                
                # Vectorized distance to all selected
                dists = dist2_full[p_idx, si]
                
                # Check 1: Distinct distances from new point to existing
                if len(set(dists.tolist())) != k_cur:
                    continue
                
                # Check 2: Distances from existing to new point not already used
                valid = True
                for i in range(k_cur):
                    if dists[i] in ud[i]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Add point
                si.append(p_idx)
                selected_indices.add(p_idx)
                ud.append(set(dists.tolist()))
                
                for i in range(k_cur):
                    ud[i].add(dists[i])
                
                if time.time() > deadline - 10:
                    break
            
            current_score = len(si)
            if current_score > best_score:
                best_score = current_score
                # Convert indices to (x, y)
                best_points = [(idx // N, idx % N) for idx in si]
                
            # If we found a valid set of size p, it's a strong baseline.
            # We might try other k's to see if they allow larger extensions.
            
    # If no valid set found from algebraic construction (unlikely), fallback to random
    if not best_points:
        # Simple random greedy
        N = 64
        xs_full = np.repeat(np.arange(N), N)
        ys_full = np.tile(np.arange(N), N)
        dx_full = xs_full[:, None] - xs_full[None, :]
        dy_full = ys_full[:, None] - ys_full[None, :]
        dist2_full = (dx_full * dx_full + dy_full * dy_full).astype(np.int64)
        
        import random
        all_indices = list(range(N*N))
        random.shuffle(all_indices)
        
        si = []
        ud = []
        
        for p_idx in all_indices:
            if not si:
                si.append(p_idx)
                ud.append(set())
                continue
            
            k = len(si)
            dists = dist2_full[p_idx, si]
            
            if len(set(dists.tolist())) != k:
                continue
            
            valid = True
            for i in range(k):
                if dists[i] in ud[i]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            si.append(p_idx)
            ud.append(set(dists.tolist()))
            for i in range(k):
                ud[i].add(dists[i])
        
        best_points = [(idx // N, idx % N) for idx in si]
        best_score = len(si)
        
    return best_points

