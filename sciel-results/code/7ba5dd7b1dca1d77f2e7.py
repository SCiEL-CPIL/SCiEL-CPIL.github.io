import time
import random
import numpy as np
from collections import defaultdict

N = 64
NUM_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Precompute all squared distances between grid points
    xs = np.arange(N)
    ys = np.arange(N)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    x_flat = X.flatten()
    y_flat = Y.flatten()
    
    # Compute squared distance matrix
    diff_x = x_flat[:, None] - x_flat[None, :]
    diff_y = y_flat[:, None] - y_flat[None, :]
    dist2 = (diff_x**2 + diff_y**2).astype(np.int32)
    
    # Build conflict graph:
    # Two points i, j conflict if there exists a third point k such that
    # {i,j,k} forms an isosceles triangle.
    # This happens when:
    #   d(i,j) == d(i,k) or d(i,j) == d(j,k) or d(i,k) == d(j,k)
    # 
    # More precisely, for a fixed pair (i,j), they conflict if there exists k≠i,j such that:
    #   d(i,j) == d(i,k)  [isosceles with apex at i]
    #   d(i,j) == d(j,k)  [isosceles with apex at j]
    #   d(i,k) == d(j,k)  [isosceles with apex at k]
    #
    # We can precompute for each point i, which squared distances it has to other points.
    # Then for pair (i,j) with d = d(i,j):
    #   - Conflict if any other point k has d(i,k) == d (apex at i)
    #   - Conflict if any other point k has d(j,k) == d (apex at j)  
    #   - Conflict if any other point k has d(i,k) == d(j,k) ≠ 0 (apex at k)
    #
    # Let's precompute distance frequency maps.
    
    # For each point i, count how many points have each squared distance
    dist_counts = [defaultdict(int) for _ in range(NUM_POINTS)]
    for i in range(NUM_POINTS):
        for j in range(NUM_POINTS):
            if i != j:
                d = int(dist2[i, j])
                dist_counts[i][d] += 1
    
    # Build adjacency list for conflict graph
    # Two points i,j are connected (conflict) if they can't both be in a valid set
    # because they'd form an isosceles triangle with some third point.
    
    # Actually, let me reconsider. The constraint is: no THREE points form an isosceles triangle.
    # So points i and j can both be selected as long as there's no k such that {i,j,k} is isosceles.
    # But this depends on k, which may or may not be selected.
    #
    # This is NOT a simple pairwise constraint! It's a 3-wise constraint.
    # So the conflict graph approach needs to be:
    # i and j "conflict" if for EVERY possible k, {i,j,k} would be isosceles? No, that's too strong.
    #
    # Actually, i and j can be selected together only if there's NO k in the selected set such that {i,j,k} is isosceles.
    # This is a hypergraph independent set problem, not a graph independent set.
    #
    # Let me go back to the greedy approach but make it much faster.
    # The current approach is O(|S|^2) per insertion check. Let me optimize.
    
    best_set = set()
    rng = random.Random(42)
    
    def greedy(order):
        selected = []
        selected_set = set()
        # For each selected point q, track the set of squared distances from q to other selected points
        dist_sets = {}
        
        for p in order:
            if p in selected_set:
                continue
            
            if not selected:
                can_add = True
                new_dists = set()
            else:
                # Get distances from p to all selected points
                d_to_sel = dist2[p, selected]
                
                # Check 1: No duplicate distances from p (apex at p)
                if len(set(d_to_sel)) != len(d_to_sel):
                    continue
                
                # Check 2: For each selected point q, the distance d(p,q) must not already be in dist_sets[q]
                # (which would mean there's another point r such that d(q,r) == d(q,p), forming isosceles with apex q)
                can_add = True
                for i, q in enumerate(selected):
                    d = int(d_to_sel[i])
                    if d in dist_sets[q]:
                        can_add = False
                        break
                
                if not can_add:
                    continue
                new_dists = set(int(d) for d in d_to_sel)
            
            # Add p
            selected.append(p)
            selected_set.add(p)
            dist_sets[p] = new_dists
            
            # Update dist_sets for existing points
            for q in selected[:-1]:  # all except p
                dist_sets[q].add(int(dist2[p, q]))
        
        return selected_set
    
    # Try multiple strategies
    orders = []
    
    # 1. Random orders
    for _ in range(3):
        o = list(range(NUM_POINTS))
        rng.shuffle(o)
        orders.append(o)
    
    # 2. Sorted by x
    orders.append(list(range(NUM_POINTS)))
    
    # 3. Spiral order (from corners inward)
    def spiral_order():
        result = []
        left, right = 0, N - 1
        top, bottom = 0, N - 1
        while left <= right and top <= bottom:
            for x in range(left, right + 1):
                result.append(top * N + x)
            top += 1
            for y in range(top, bottom + 1):
                result.append(y * N + right)
            right -= 1
            if top <= bottom:
                for x in range(right, left - 1, -1):
                    result.append(bottom * N + x)
                bottom -= 1
            if left <= right:
                for y in range(bottom, top - 1, -1):
                    result.append(y * N + left)
                left += 1
        return result
    
    orders.append(spiral_order())
    
    # 4. Checkerboard pattern
    def checkerboard_order():
        result = []
        for y in range(N):
            for x in range(N):
                if (x + y) % 2 == 0:
                    result.append(y * N + x)
        for y in range(N):
            for x in range(N):
                if (x + y) % 2 == 1:
                    result.append(y * N + x)
        return result
    
    orders.append(checkerboard_order())
    
    # Run greedy on each order
    for o in orders:
        if time.time() > deadline:
            break
        s = greedy(o)
        if len(s) > len(best_set):
            best_set = s
    
    # More random restarts with the remaining time
    while time.time() < deadline:
        o = list(range(NUM_POINTS))
        rng.shuffle(o)
        s = greedy(o)
        if len(s) > len(best_set):
            best_set = s
    
    return [(p // N, p % N) for p in best_set]
