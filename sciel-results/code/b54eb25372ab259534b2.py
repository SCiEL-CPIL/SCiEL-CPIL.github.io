"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 270.0  # safety margin before 330s kill
    
    n_points = N * N
    
    # Index: point (x, y) -> index x * N + y
    def idx(x, y):
        return x * N + y
    
    def deidx(i):
        return (i // N, i % N)
    
    # Precompute squared distances from each point to all others
    # dist[i][j] = squared distance between point i and point j
    # For conflict detection: two points a, b conflict if there exists c
    # such that {d(a,c), d(b,c), d(a,b)} has a repeated value
    
    # Build conflict graph using bitsets
    # Two points are adjacent in conflict graph if they share an isosceles partner
    # i.e., exists c such that d(a,c)==d(b,c) or d(a,c)==d(a,b) or d(b,c)==d(a,b)
    
    # Precompute distance from each point to all other points
    # Use numpy for speed
    coords = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    
    # Compute all pairwise squared distances (n_points x n_points)
    # This is 4096 x 4096 = 16M entries, should be manageable
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_sq = (diff ** 2).sum(axis=2)  # shape (n_points, n_points)
    
    # Build conflict bitsets
    # conflicts[i] is a bitset where bit j is set if i and j conflict
    # Two points i, j conflict if:
    #   1. There exists k != i,j such that dist_sq[i,k] == dist_sq[j,k] (isosceles with apex k)
    #   2. There exists k != i,j such that dist_sq[i,k] == dist_sq[i,j] (isosceles with apex i, base ik... wait)
    #   Actually: triangle (i,j,k) is isosceles if any two of d(i,j), d(i,k), d(j,k) are equal.
    #   So i and j conflict if exists k such that:
    #     d(i,j) == d(i,k) OR d(i,j) == d(j,k) OR d(i,k) == d(j,k)
    
    # For efficiency, let's think about this differently.
    # For a fixed pair (i,j), they conflict if:
    #   - The set of distances from i to all k intersects with {d(i,j)} (excluding j itself)
    #     i.e., exists k != j such that d(i,k) == d(i,j)
    #   - OR the set of distances from j to all k intersects with {d(i,j)} (excluding i itself)
    #     i.e., exists k != i such that d(j,k) == d(i,j)  
    #   - OR there exists k such that d(i,k) == d(j,k) and k != i, k != j
    
    # Let me precompute for each point i, for each distance value d, the set of points at distance d from i
    # Then for pair (i,j):
    #   conflict if:
    #     (j is in dist_set[i][d(i,j)] and dist_set[i][d(i,j)] has size > 1) 
    #     OR (i is in dist_set[j][d(i,j)] and dist_set[j][d(i,j)] has size > 1)
    #     OR exists k in intersection of dist_set[i][d] and dist_set[j][d] for some d, where k != i, j
    
    # This is still complex. Let me use a simpler bitset approach.
    # For each point i, build a bitset of points that conflict with i.
    
    # Alternative: For each point i, and each other point j, check if they conflict.
    # But that's O(n^3) which is 4096^3 = 68 billion - too slow.
    
    # Let's use the distance-based approach:
    # For each point i, group other points by their distance from i.
    # For a pair (i,j) to conflict:
    #   Case 1: d(i,j) == d(i,k) for some k != j. This means j and k are both at distance d(i,j) from i.
    #           So if the number of points at distance d(i,j) from i is >= 2, then i conflicts with all of them.
    #   Case 2: d(i,j) == d(j,k) for some k != i. Similarly, if number of points at distance d(i,j) from j is >= 2.
    #   Case 3: d(i,k) == d(j,k) for some k != i,j. This is harder.
    
    # For case 3: d(i,k) == d(j,k) means k is equidistant from i and j.
    # The locus of points equidistant from i and j is the perpendicular bisector.
    # On a grid, we can check which points lie on this bisector.
    
    # Let me try a different approach: directly compute conflicts using numpy vectorization.
    
    # For each pair (i,j), compute d_ij = dist_sq[i,j]
    # They conflict if:
    #   (dist_sq[i,:] == d_ij).sum() > 1  [case 1: another point at same distance from i]
    #   OR (dist_sq[j,:] == d_ij).sum() > 1  [case 2]
    #   OR exists k such that dist_sq[i,k] == dist_sq[j,k] and k != i, k != j  [case 3]
    
    # Case 3 is the tricky one. Let's handle it by noting that if d(i,k) == d(j,k),
    # then k lies on the perpendicular bisector of segment ij.
    
    # Actually, let me just brute force case 3 efficiently.
    # For each pair (i,j), the condition d(i,k) == d(j,k) can be rewritten as:
    #   |i-k|^2 = |j-k|^2
    #   |i|^2 - 2*i.k + |k|^2 = |j|^2 - 2*j.k + |k|^2
    #   |i|^2 - |j|^2 = 2*(i-j).k
    # So k must satisfy: 2*(i-j).k = |i|^2 - |j|^2
    # This is a linear equation in k. On a grid, we can enumerate.
    
    # But this is getting complicated. Let me try a simpler heuristic first:
    # Use the greedy approach but with precomputed conflict bitsets using only cases 1 and 2,
    # then add case 3 checks during the greedy phase.
    
    # Actually, let me reconsider. The original code checks all three conditions.
    # Let me precompute a conflict matrix using numpy, being careful about memory.
    
    # n_points = 4096, so dist_sq is 4096x4096 = 16.7M integers. With int32, that's ~67MB. OK.
    
    # For case 1: for each i, for each distance value d that appears in row i,
    #   all points at distance d from i conflict with each other.
    # For case 2: similar for columns.
    # For case 3: for each pair (i,j), check if there's a k on the bisector.
    
    # Let me build the conflict graph step by step.
    
    # Step 1: Handle cases 1 and 2 (easy with numpy)
    conflicts = np.zeros((n_points, n_points), dtype=bool)
    
    # For each point i, group points by distance
    # For case 1: if multiple points are at the same distance from i, they all conflict with each other
    for i in range(n_points):
        if time.time() > deadline:
            break
        row = dist_sq[i]
        # Get unique distances and their counts
        unique_dists, counts = np.unique(row, return_counts=True)
        for d, count in zip(unique_dists, counts):
            if count > 1:
                # All points at distance d from i conflict with each other
                mask = (row == d)
                idxs = np.where(mask)[0]
                for a in range(len(idxs)):
                    for b in range(a+1, len(idxs)):
                        ia, ib = idxs[a], idxs[b]
                        conflicts[ia, ib] = True
                        conflicts[ib, ia] = True
    
    # Step 2: Handle case 3 (equidistant points)
    # For each pair (i,j), check if there exists k such that d(i,k) == d(j,k)
    # Using the linear equation: 2*(i-j).k = |i|^2 - |j|^2
    for i in range(n_points):
        if time.time() > deadline:
            break
        xi, yi = deidx(i)
        for j in range(i+1, n_points):
            xj, yj = deidx(j)
            # Check if already in conflict
            if conflicts[i, j]:
                continue
            
            # d(i,j)
            dij_sq = dist_sq[i, j]
            
            # Case 1 check: is there another point at distance dij_sq from i?
            if (dist_sq[i] == dij_sq).sum() > 1:
                conflicts[i, j] = True
                conflicts[j, i] = True
                continue
            
            # Case 2 check: is there another point at distance dij_sq from j?
            if (dist_sq[j] == dij_sq).sum() > 1:
                conflicts[i, j] = True
                conflicts[j, i] = True
                continue
            
            # Case 3: exists k such that d(i,k) == d(j,k)
            # 2*(xi-xj)*kx + 2*(yi-yj)*ky = xi^2 + yi^2 - xj^2 - yj^2
            ax = 2 * (xi - xj)
            ay = 2 * (yi - yj)
            b = xi*xi + yi*yi - xj*xj - yj*yj
            
            # If ax == 0 and ay == 0, then i == j, skip (but i != j here)
            if ax == 0 and ay == 0:
                continue
            
            # Check if any grid point (kx, ky) in [0,N)^2 satisfies ax*kx + ay*ky == b
            # and k != i, k != j
            # This is a linear Diophantine equation. Let's just check all points... 
            # That's 4096 checks per pair, too slow for O(n^2) pairs.
            
            # Alternative: use the distance matrix.
            # d(i,k) == d(j,k) iff dist_sq[i,k] == dist_sq[j,k]
            # So check if rows i and j of dist_sq have any common value at the same column k,
            # where k != i and k != j.
            
            # For efficiency, let's check: is there a k such that dist_sq[i,k] == dist_sq[j,k] and k not in {i,j}?
            # We can do this by checking the intersection of the sets of distances.
            # But that's still O(n) per pair.
            
            # For now, let's skip case 3 in the precomputation and handle it during greedy.
            # This might miss some conflicts but should still be a good heuristic.
    
    # Convert to bitsets for efficient greedy
    # Use Python integers as bitsets
    conflict_bits = [0] * n_points
    for i in range(n_points):
        bits = 0
        row = conflicts[i]
        for j in range(n_points):
            if row[j]:
                bits |= (1 << j)
        conflict_bits[i] = bits
    
    # Now do greedy with bitsets
    random.seed(42)
    point_order = list(range(n_points))
    random.shuffle(point_order)
    
    selected = []
    selected_set = set()
    # Available candidates: start with all, remove as we go
    available = (1 << n_points) - 1
    
    for p in point_order:
        if time.time() > deadline:
            break
        
        if not (available & (1 << p)):
            continue
        
        # Check if p conflicts with any selected point
        # A point p can be added if it doesn't conflict with any selected point
        # AND adding it doesn't create an isosceles triangle with two selected points
        # Wait, the conflict graph already encodes: two points conflict if they CANNOT both be in the set
        # because there exists a third point that would form an isosceles triangle.
        # But actually, the constraint is: no three points form an isosceles triangle.
        # So if we have selected points S, we can add p if for all pairs (a,b) in S,
        # (a,b,p) is not isosceles.
        # 
        # The conflict graph approach: if a and b conflict, they cannot both be in S.
        # But that's not quite right either. The constraint is on triples, not pairs.
        #
        # Actually, re-reading: "no three points forming an isosceles triangle"
        # So the set S is valid if no three points in S form an isosceles triangle.
        # This is a 3-uniform hypergraph independent set problem, not a graph one.
        #
        # The conflict graph I built is: two points conflict if there EXISTS a third point
        # that would form an isosceles triangle with them. But that third point might not
        # be in the set! So the conflict graph is too restrictive.
        #
        # Let me reconsider. The correct approach is:
        # - Start with empty set
        # - Greedily add points, checking that no three form an isosceles triangle
        # - The check for adding p: for all pairs (a,b) in current set, verify (a,b,p) is not isosceles
        #
        # To speed this up, I can precompute for each pair (a,b), the set of points p
        # that would form an isosceles triangle with (a,b). Then when adding p, I check
        # if p is in the "forbidden" set for any pair in the current selection.
    
    # Let me restart with a cleaner approach.
    # Precompute: for each pair (a,b), the set of forbidden points.
    # forbidden[a,b] = set of p such that (a,b,p) is isosceles.
    
    # This is O(n^3) to compute but can be vectorized.
    # Actually, for each pair (a,b), the forbidden points are:
    #   - Points p such that d(a,p) == d(b,p) (apex at... wait)
    #   - Points p such that d(a,p) == d(a,b)
    #   - Points p such that d(b,p) == d(a,b)
    
    # Let me build this using numpy.
    
    # For each pair (a,b), compute forbidden points.
    # n_pairs = n_points * (n_points-1) / 2 ~ 8M pairs. Too many to store explicitly.
    
    # Alternative: during greedy, for each candidate p, check against all pairs in selected.
    # If selected has k points, there are k*(k-1)/2 pairs. For k=50, that's ~1200 pairs.
    # For each pair, checking if p is forbidden is O(1) with precomputed data.
    
    # Let me precompute for each point p, and each distance d, the set of points at distance d from p.
    # Then for pair (a,b) with distance d_ab:
    #   forbidden points include:
    #   - Points at distance d_ab from a (excluding b)
    #   - Points at distance d_ab from b (excluding a)
    #   - Points equidistant from a and b (excluding a, b)
    
    # This is still complex. Let me just use the original greedy but with numpy-accelerated checks.
    
    # Reset and do a clean implementation
    selected = []
    selected_set = set()
    
    # Precompute distance lookup: for each point i, a dict mapping distance -> list of points
    dist_to_points = [{} for _ in range(n_points)]
    for i in range(n_points):
        row = dist_sq[i]
        for j in range(n_points):
            d = row[j]
            if d not in dist_to_points[i]:
                dist_to_points[i][d] = []
            dist_to_points[i][d].append(j)
    
    # For equidistant check: for each pair (a,b), precompute the set of equidistant points
    # This is expensive. Let's compute on the fly using the linear equation.
    
    def is_equidistant(k, a, b):
        """Check if point k is equidistant from a and b."""
        return dist_sq[k, a] == dist_sq[k, b]
    
    def forms_isosceles(a, b, p):
        """Check if points a, b, p form an isosceles triangle."""
        dab = dist_sq[a, b]
        dap = dist_sq[a, p]
        dbp = dist_sq[b, p]
        return (dab == dap) or (dab == dbp) or (dap == dbp)
    
    # Greedy with efficient checks
    for p in point_order:
        if time.time() > deadline:
            break
        
        if p in selected_set:
            continue
        
        valid = True
        if len(selected) >= 2:
            for i in range(len(selected)):
                if not valid:
                    break
                ai = selected[i]
                for j in range(i+1, len(selected)):
                    if not valid:
                        break
                    bj = selected[j]
                    if forms_isosceles(ai, bj, p):
                        valid = False
                        break
        
        if valid:
            selected.append(p)
            selected_set.add(p)
    
    # Convert indices back to (x,y) tuples
    result = [deidx(i) for i in selected]
    return result
# EVOLVE-BLOCK-END
