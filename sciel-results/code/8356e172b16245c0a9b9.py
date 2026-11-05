import time
import random
import numpy as np
from collections import defaultdict

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 300s limit
    
    num_points = N * N
    xs = np.arange(N, dtype=np.int32)
    ys = np.arange(N, dtype=np.int32)
    
    all_x = np.repeat(xs, N)
    all_y = np.tile(ys, N)
    
    # Compute squared distance matrix
    dx = all_x[:, None] - all_x[None, :]
    dy = all_y[:, None] - all_y[None, :]
    dist2 = (dx * dx + dy * dy).astype(np.int32)
    
    # Precompute conflict bitsets:
    # conflict[P] = bitset of all points Q such that there exists R with {P,Q,R} isosceles
    # We use Python integers as arbitrary-precision bitsets
    # conflict[P] has bit Q set if P and Q conflict (i.e., there's some R making isosceles)
    
    # For each point P, find all Q such that P,Q conflict
    # P and Q conflict if:
    #   - There exists R != P,Q such that dist(P,R) == dist(Q,R) (R is apex)
    #   - OR dist(P,Q) == dist(P,R) for some R != P,Q (P is apex, but this means P,Q,R isosceles with P as apex)
    #   - OR dist(P,Q) == dist(Q,R) for some R != P,Q (Q is apex)
    # Actually, P and Q conflict if there EXISTS any R != P,Q such that {P,Q,R} is isosceles.
    # This happens iff:
    #   dist(P,Q) == dist(P,R) OR dist(P,Q) == dist(Q,R) OR dist(P,R) == dist(Q,R) for some R
    
    # More efficient: for each P, build a map from distance to count of points at that distance
    # Then P conflicts with Q if:
    #   - count of points at distance d=PQ from P is >= 2 (so there's R with dist(P,R)=d, making P apex)
    #   - OR count of points at distance d=PQ from Q is >= 2 (Q apex)
    #   - OR there exists R with dist(P,R) == dist(Q,R) (R apex)
    
    # Let's precompute for each point P:
    #   dist_counts[P][d] = number of points at distance d from P
    #   For R-apex check: for each P, group other points by their distance to P
    #     P conflicts with Q if there exists R such that dist(P,R) == dist(Q,R)
    #     This means: for some distance d, both P and Q have at least one point at distance d from them
    #     More precisely: the sets of distances from P and Q to other points must overlap
    #     But that's expensive. Let's think differently.
    
    # Alternative approach: 
    # For each triple (P, Q, R), check if isosceles. But that's O(n^3) which is too much.
    
    # Better: For each P, compute conflict[P] as follows:
    # 1. For each Q != P, let d = dist(P,Q)
    # 2. P conflicts with Q if:
    #    a) There exists R != P,Q with dist(P,R) == d (P is apex)
    #       -> This is true if count_P[d] >= 2 (including Q itself, so need >= 2 others besides Q... 
    #          actually count_P[d] counts all points at distance d from P, including Q.
    #          So we need count_P[d] >= 2 means there's at least one other point R at distance d from P)
    #    b) There exists R != P,Q with dist(Q,R) == d (Q is apex)
    #       -> count_Q[d] >= 2
    #    c) There exists R != P,Q with dist(P,R) == dist(Q,R) (R is apex)
    #       -> The sets of distances from P and Q to other points (excluding each other) must have a common value
    #          that is achieved by at least one point R != P,Q
    
    # This is still complex. Let me use a different strategy:
    # Precompute for each point P a bitset of points that are at the same distance from P as some other point.
    # Actually, let's just directly compute conflicts using the distance matrix.
    
    # For efficiency, let's use a different representation:
    # For each point P, we'll store a set of points that conflict with P.
    # We compute this by iterating over all pairs (P, Q) and checking if they conflict.
    
    # Two points P and Q conflict if there exists R != P, Q such that the triangle PQR is isosceles.
    # This is equivalent to: the multiset {dist(P,Q), dist(P,R), dist(Q,R)} has at least two equal elements
    # for some R != P, Q.
    
    # Efficient check: P and Q conflict iff:
    #   (i) There exists R != P,Q with dist(P,R) == dist(P,Q)  [P is apex]
    #   (ii) There exists R != P,Q with dist(Q,R) == dist(P,Q) [Q is apex]  
    #   (iii) There exists R != P,Q with dist(P,R) == dist(Q,R) [R is apex]
    
    # Precompute for each P: for each distance d, how many points are at distance d from P
    # And also, for the R-apex case, we need to check if the distance sets overlap.
    
    # Let's precompute:
    # dist_from_P[P] = dict mapping distance -> list of points at that distance from P
    # Then for P,Q with d = dist(P,Q):
    #   (i) true if len(dist_from_P[P][d]) >= 2
    #   (ii) true if len(dist_from_P[Q][d]) >= 2
    #   (iii) true if there exists a distance d' such that d' appears in both dist_from_P[P] and dist_from_P[Q]
    #          and the corresponding point R is not P or Q.
    
    # Actually for (iii), we need: exists R != P,Q with dist(P,R) == dist(Q,R).
    # This is equivalent to: the sets of distances from P (to points other than Q) and from Q (to points other than P) 
    # have a common value, AND that value is achieved by a point R that is neither P nor Q.
    
    # This is getting complex. Let me use a simpler but potentially slower approach:
    # For each P, compute conflict[P] by checking all Q.
    
    # Precompute distance maps
    dist_maps = []
    for P in range(num_points):
        dm = {}
        for Q in range(num_points):
            if P == Q:
                continue
            d = int(dist2[P, Q])
            if d not in dm:
                dm[d] = []
            dm[d].append(Q)
        dist_maps.append(dm)
    
    # Precompute for each P, a set of distances that P has to other points
    # For the R-apex check, we need to know if P and Q share a common distance to some R
    # Let's precompute for each P: set of distances from P to all other points
    P_dist_sets = [set(dm.keys()) for dm in dist_maps]
    
    # Now compute conflict bitsets
    # conflict[P] will be a Python int with bit Q set if P conflicts with Q
    conflict = [0] * num_points
    
    for P in range(num_points):
        for Q in range(num_points):
            if P == Q:
                continue
            d = int(dist2[P, Q])
            
            # Check (i): P is apex - is there R != P,Q with dist(P,R) == d?
            if d in dist_maps[P] and len(dist_maps[P][d]) >= 2:
                # There's at least one other point at distance d from P
                # But we need to make sure R != Q. Since Q is at distance d from P,
                # len >= 2 means there's at least one other point R != Q at distance d from P.
                conflict[P] |= (1 << Q)
                conflict[Q] |= (1 << P)
                continue
            
            # Check (ii): Q is apex
            if d in dist_maps[Q] and len(dist_maps[Q][d]) >= 2:
                conflict[P] |= (1 << Q)
                conflict[Q] |= (1 << P)
                continue
            
            # Check (iii): R is apex - exists R != P,Q with dist(P,R) == dist(Q,R)
            # This means there's a distance d' such that P has some point at distance d'
            # and Q has some point at distance d', and that point is the same R (or at least 
            # there exists such R).
            # More precisely: exists R != P,Q with dist(P,R) == dist(Q,R)
            # Let d' = dist(P,R) = dist(Q,R)
            # So d' must be in both P_dist_sets[P] and P_dist_sets[Q]
            # And there must be a point R at distance d' from both P and Q, with R != P,Q.
            
            # For efficiency, let's check if the distance sets overlap
            common_dists = P_dist_sets[P] & P_dist_sets[Q]
            if common_dists:
                # For each common distance d', check if there's a point R at distance d' from both P and Q
                # with R != P and R != Q
                found = False
                for d_prime in common_dists:
                    # Get points at distance d_prime from P
                    pts_P = dist_maps[P].get(d_prime, [])
                    # Get points at distance d_prime from Q
                    pts_Q = dist_maps[Q].get(d_prime, [])
                    # Check if any point R is in both, and R != P and R != Q
                    # Use set intersection for efficiency
                    if len(pts_P) < len(pts_Q):
                        smaller = set(pts_P)
                        larger = pts_Q
                    else:
                        smaller = set(pts_Q)
                        larger = pts_P
                    for R in smaller:
                        if R == P or R == Q:
                            continue
                        # Check if R is also at distance d_prime from the other point
                        if R in larger:  # This isn't quite right, larger is a list
                            # Need to check properly
                            pass
                    # Let's do this more carefully
                    set_P = set(pts_P)
                    set_Q = set(pts_Q)
                    common_pts = set_P & set_Q
                    for R in common_pts:
                        if R != P and R != Q:
                            found = True
                            break
                    if found:
                        break
            
            if found:
                conflict[P] |= (1 << Q)
                conflict[Q] |= (1 << P)
    
    # Now we have conflict bitsets. 
    # A set S is valid if for all P, Q in S, bit Q is not set in conflict[P].
    # Equivalently, the set S must be an independent set in the conflict graph.
    
    # For local search, we maintain:
    # - S: list of points in the set
    # - in_S: boolean array
    # - To check if P can be added: check if conflict[P] & S_bitset == 0
    #   where S_bitset is the bitset of all points in S
    
    # But we also need to maintain that S is independent, which it will be if we only add
    # points that don't conflict with existing points.
    
    def can_add(P, S_bitset):
        """Check if P can be added to the set represented by S_bitset."""
        return (conflict[P] & S_bitset) == 0
    
    def greedy_construction(order):
        S = []
        S_bitset = 0
        in_S = np.zeros(num_points, dtype=bool)
        
        for P in order:
            if in_S[P]:
                continue
            if can_add(P, S_bitset):
                in_S[P] = True
                S.append(P)
                S_bitset |= (1 << P)
        
        return S, S_bitset, in_S
    
    def local_search(S, S_bitset, in_S, time_limit):
        best_S = S[:]
        best_S_bitset = S_bitset
        best_size = len(S)
        
        no_improve = 0
        max_no_improve = 30
        
        while time.time() < time_limit and no_improve < max_no_improve:
            # Try to add a random point not in S
            added = False
            
            # Try a batch of random candidates
            batch_size = min(500, num_points - len(S))
            if batch_size > 0:
                candidates = random.sample(range(num_points), batch_size)
                for P in candidates:
                    if in_S[P]:
                        continue
                    if can_add(P, S_bitset):
                        in_S[P] = True
                        S.append(P)
                        S_bitset |= (1 << P)
                        
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = S[:]
                            best_S_bitset = S_bitset
                            no_improve = 0
                        else:
                            no_improve += 1
                        
                        added = True
                        break
            
            if not added:
                no_improve += 1
                
                # Perturbation: remove 1-2 random points, try to add 2-3
                if len(S) > 3:
                    # Remove 1 or 2 points
                    num_remove = 1 if random.random() < 0.5 else 2
                    num_remove = min(num_remove, len(S) - 2)  # Keep at least 2 points
                    if num_remove > 0:
                        rem_indices = random.sample(range(len(S)), num_remove)
                        for rem_idx in sorted(rem_indices, reverse=True):
                            rem_P = S[rem_idx]
                            in_S[rem_P] = False
                            S_bitset &= ~(1 << rem_P)
                            S.pop(rem_idx)
                        
                        # Try to add 2-3 new points
                        num_add = num_remove + 1
                        added_count = 0
                        batch_size = min(300, num_points - len(S))
                        if batch_size > 0:
                            candidates = random.sample(range(num_points), batch_size)
                            for P in candidates:
                                if in_S[P]:
                                    continue
                                if can_add(P, S_bitset):
                                    in_S[P] = True
                                    S.append(P)
                                    S_bitset |= (1 << P)
                                    added_count += 1
                                    
                                    if len(S) > best_size:
                                        best_size = len(S)
                                        best_S = S[:]
                                        best_S_bitset = S_bitset
                                        no_improve = 0
                                    
                                    if added_count >= num_add:
                                        break
                        
                        # Reset no_improve after perturbation
                        no_improve = 0
        
        return best_S, best_S_bitset
    
    # Main loop: run multiple rounds of greedy + local search
    best_set = []
    best_size = 0
    
    # Initial greedy construction with a few different orderings
    for _ in range(5):
        order = list(range(num_points))
        random.shuffle(order)
        
        S, S_bitset, in_S = greedy_construction(order)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
        
        # Run local search
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            search_time = min(30, remaining_time / 2)
            S, S_bitset = local_search(S, S_bitset, in_S, time.time() + search_time)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
        
        if time.time() > deadline - 10:
            break
    
    # Continue with more rounds if time permits
    while time.time() < deadline - 10:
        order = list(range(num_points))
        random.shuffle(order)
        
        S, S_bitset, in_S = greedy_construction(order)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
        
        # Run local search
        remaining_time = deadline - time.time()
        if remaining_time > 10:
            search_time = min(20, remaining_time / 3)
            S, S_bitset = local_search(S, S_bitset, in_S, time.time() + search_time)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
        
        # Early exit if we found a very good set
        if best_size > 55:
            break
    
    # Convert indices back to (x, y) coordinates
    result = [(idx % N, idx // N) for idx in best_set]
    
    return result

if __name__ == "__main__":
    points = construct()
    print(f"Found set of size {len(points)}")
    print(points[:10])
