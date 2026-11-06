import random
import time
import numpy as np

N = 64
TOTAL_POINTS = N * N
MAX_TIME = 300.0
SAFETY_MARGIN = 10.0

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME - SAFETY_MARGIN
    
    # Generate all grid points
    points = [(x, y) for x in range(N) for y in range(N)]
    num_points = TOTAL_POINTS
    
    # Precompute squared distances
    # dist_matrix[i][j] = squared distance between points[i] and points[j]
    xs = np.array([p[0] for p in points], dtype=np.int32)
    ys = np.array([p[1] for p in points], dtype=np.int32)
    
    # For each point, compute distances to all other points
    # We'll use a different approach: for each point, store a dictionary of distance -> count
    # and also store the set of points at each distance
    
    # Actually, let's think about the constraint differently.
    # No three points form an isosceles triangle.
    # This means: for any point p, and any other points q, r, 
    # if dist(p,q) == dist(p,r), then we have an isosceles triangle with apex at p.
    # But also, if dist(p,q) == dist(q,r), we have an isosceles triangle with apex at q.
    # And if dist(p,r) == dist(q,r), we have an isosceles triangle with apex at r.
    
    # So the constraint is: for ANY three distinct points p, q, r,
    # all three distances dist(p,q), dist(p,r), dist(q,r) must be distinct.
    
    # This is equivalent to: for any two points p, q, and any third point r,
    # dist(p,r) != dist(q,r) unless p=q (impossible for distinct points).
    # Wait, that's not right either.
    
    # Let me re-examine. An isosceles triangle has at least two equal sides.
    # So we need: for any three distinct points p, q, r,
    # NOT (dist(p,q) == dist(p,r)) AND NOT (dist(p,q) == dist(q,r)) AND NOT (dist(p,r) == dist(q,r))
    
    # This means: for any point p, all distances from p to other selected points must be distinct.
    # AND for any pair of points p, q, all distances from p to other selected points 
    # must be different from all distances from q to other selected points? No.
    
    # Actually, let's think about it this way:
    # If we select a set S, then for each point p in S, 
    # the set of distances {dist(p, q) : q in S, q != p} must have all distinct values.
    # AND for any two points p, q in S, and any third point r in S, r != p, q,
    # dist(p, r) != dist(q, r) is NOT required by itself.
    
    # Wait, let me reconsider. If dist(p,q) == dist(p,r), that's an isosceles triangle with apex at p.
    # If dist(p,q) == dist(q,r), that's an isosceles triangle with apex at q.
    # If dist(p,r) == dist(q,r), that's an isosceles triangle with apex at r.
    
    # So the condition is: for any three distinct points p, q, r in S,
    # dist(p,q) != dist(p,r) AND dist(p,q) != dist(q,r) AND dist(p,r) != dist(q,r).
    
    # This is equivalent to: for any two distinct points p, q in S,
    # and any third point r in S with r != p and r != q,
    # dist(p, r) != dist(q, r).
    
    # Because if dist(p,r) == dist(q,r), then we have an isosceles triangle with apex at r.
    # And similarly for the other cases.
    
    # So the key constraint is: for any two points p, q in S, 
    # the sets of distances from p to S\{p} and from q to S\{q} 
    # must not share any common value (except possibly the distance between p and q itself, 
    # but that's handled separately).
    
    # Actually, more precisely: for any two points p, q in S, and any r in S\{p,q},
    # dist(p,r) != dist(q,r).
    
    # Let's use a greedy approach with backtracking or local search.
    
    # Strategy: Use a greedy algorithm that tries to add points one by one,
    # checking if adding a point violates the constraint.
    
    # For efficiency, we'll maintain:
    # - selected: set of selected point indices
    # - For each selected point, we track which distances are "used"
    
    # When we want to add a new point p, we need to check:
    # 1. For each existing selected point q, dist(p, q) must not equal any 
    #    distance from q to another selected point (other than p).
    # 2. For each pair of existing selected points q1, q2, 
    #    dist(p, q1) != dist(p, q2) is automatically satisfied if we ensure 
    #    that for each q, all distances from q to selected points are distinct.
    
    # Actually, let's simplify. When adding point p to a valid set S:
    # - For each q in S, dist(p, q) must be a new distance not seen before from q.
    #   I.e., dist(p, q) != dist(q, r) for all r in S, r != q, r != p.
    # - Additionally, for any two points q1, q2 in S, dist(p, q1) != dist(p, q2)
    #   is not required by the constraint directly, but it follows from the fact
    #   that if dist(p, q1) == dist(p, q2), then p, q1, q2 form an isosceles triangle
    #   with apex at p. So we DO need dist(p, q1) != dist(p, q2) for all q1 != q2 in S.
    
    # So the check for adding p to S is:
    # 1. All distances from p to points in S must be distinct.
    # 2. For each q in S, dist(p, q) must not equal dist(q, r) for any r in S, r != q.
    
    # Let's implement this.
    
    # Precompute distance from each point to all others
    # dist[i][j] = squared distance between point i and point j
    dist = np.zeros((num_points, num_points), dtype=np.int32)
    for i in range(num_points):
        xi, yi = points[i]
        for j in range(num_points):
            xj, yj = points[j]
            dx = xi - xj
            dy = yi - yj
            dist[i][j] = dx*dx + dy*dy
    
    # To check if we can add point p to set S:
    # We need to track for each point q in S, which distances are "used" by q to other points in S.
    
    # Let's use a local search approach with random restarts.
    
    def can_add(p, S, used_dists):
        """
        Check if point p can be added to set S.
        used_dists[q] is a set of distances from q to other points in S.
        """
        if len(S) == 0:
            return True
        
        # Check 1: All distances from p to points in S must be distinct
        dists_from_p = {}
        for q in S:
            d = dist[p][q]
            if d in dists_from_p:
                return False
            dists_from_p[d] = True
        
        # Check 2: For each q in S, dist(p, q) must not be in used_dists[q]
        for q in S:
            d = dist[p][q]
            if d in used_dists[q]:
                return False
        
        return True
    
    def add_point(p, S, used_dists):
        """
        Add point p to set S.
        """
        S.add(p)
        used_dists[p] = set()
        for q in S:
            if p == q:
                continue
            d = dist[p][q]
            used_dists[p].add(d)
            used_dists[q].add(d)
    
    def remove_point(p, S, used_dists):
        """
        Remove point p from set S.
        """
        for q in S:
            if p == q:
                continue
            d = dist[p][q]
            used_dists[q].remove(d)
        S.remove(p)
        del used_dists[p]
    
    best_size = 0
    best_set = set()
    best_used_dists = {}
    
    # Local search with random restarts
    max_restarts = 50
    restart_count = 0
    
    while time.time() < deadline and restart_count < max_restarts:
        restart_count += 1
        S = set()
        used_dists = {}
        
        # Randomly initialize with a few points
        available = list(range(num_points))
        random.shuffle(available)
        
        # Try to add points greedily in random order
        for p in available:
            if can_add(p, S, used_dists):
                add_point(p, S, used_dists)
        
        # Record best so far
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            best_used_dists = {q: set(used_dists[q]) for q in used_dists}
        
        # Now try to improve by swapping points
        # Local search: try removing one point and adding a different one
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list = list(S)
            random.shuffle(S_list)
            
            for p_remove in S_list:
                if time.time() > deadline:
                    break
                
                # Remove p_remove
                remove_point(p_remove, S, used_dists)
                
                # Try to add a new point
                added = False
                for p_add in range(num_points):
                    if p_add in S:
                        continue
                    if can_add(p_add, S, used_dists):
                        add_point(p_add, S, used_dists)
                        added = True
                        if len(S) > best_size:
                            best_size = len(S)
                            best_set = set(S)
                            best_used_dists = {q: set(used_dists[q]) for q in used_dists}
                        break
                
                # If we didn't add a point, put p_remove back
                if not added:
                    add_point(p_remove, S, used_dists)
                else:
                    improved = True
        
    return [points[i] for i in best_set]
