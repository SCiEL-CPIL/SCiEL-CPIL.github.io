import time
import random
import numpy as np

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin for 300s limit
    
    num_points = N * N
    
    # Precompute coordinates
    xs = np.arange(N, dtype=np.int32)
    ys = np.arange(N, dtype=np.int32)
    all_x = np.repeat(xs, N)
    all_y = np.tile(ys, N)
    
    # Compute squared distance matrix
    dx = all_x[:, None] - all_x[None, :]
    dy = all_y[:, None] - all_y[None, :]
    dist2 = (dx * dx + dy * dy).astype(np.int32)
    
    # Build conflict graph using bitsets
    # For each point P, conflicts[P] is a bitset of points that conflict with P
    # P and Q conflict if they cannot both be in a valid set
    # This happens if there exists a third point R such that {P,Q,R} forms an isosceles triangle
    # with P or Q as the apex (since we check pairwise)
    
    # More precisely: P and Q conflict if:
    # 1. dist(P,Q) appears twice in distances from some R to {P,Q} (R is apex)
    # 2. dist(R,P) == dist(R,Q) for some R (P and Q are base vertices, R is apex)
    # 3. dist(P,Q) == dist(P,R) for some R (P is apex)
    # 4. dist(P,Q) == dist(Q,R) for some R (Q is apex)
    
    # Actually, the simplest formulation: P and Q are incompatible if there exists R != P,Q
    # such that {P,Q,R} forms an isosceles triangle.
    
    # For efficiency, let's precompute for each pair (P,Q) whether they conflict.
    # But that's O(n^3) which is too slow. Instead, we'll use a different approach.
    
    # Key insight: P and Q conflict if and only if there exists R such that:
    # - dist(R,P) == dist(R,Q)  [R is apex, P,Q are base]
    # - dist(P,Q) == dist(P,R)  [P is apex, Q,R are base]  
    # - dist(P,Q) == dist(Q,R)  [Q is apex, P,R are base]
    
    # Let's precompute for each point P, the set of distances from P to all other points.
    # Then P and Q conflict if:
    # - There exists R with dist(R,P) == dist(R,Q) [check via distance sets]
    # - dist(P,Q) is in dist_set[P] (P already has another point at that distance)
    # - dist(P,Q) is in dist_set[Q] (Q already has another point at that distance)
    
    # For the bitset approach, we'll precompute conflict[P] as a bitset.
    # conflict[P] has bit Q set if P and Q cannot both be in a valid set.
    
    # Number of bits needed per point
    num_bits = num_points
    num_uint64 = (num_bits + 63) // 64
    
    # Precompute distance sets for each point
    # dist_set[P] = set of squared distances from P to all other points
    dist_sets_all = [set() for _ in range(num_points)]
    for P in range(num_points):
        for Q in range(num_points):
            if P != Q:
                dist_sets_all[P].add(int(dist2[P, Q]))
    
    # Build conflict bitsets
    # conflict[P] is a list of uint64, where bit Q is set if P and Q conflict
    conflicts = [np.zeros(num_uint64, dtype=np.uint64) for _ in range(num_points)]
    
    def set_bit(arr, idx):
        arr[idx // 64] |= (1 << (idx % 64))
    
    def get_bit(arr, idx):
        return bool(arr[idx // 64] & (1 << (idx % 64)))
    
    # For each pair (P, Q), check if they conflict
    # This is O(n^3) but with vectorization we can speed it up
    # Actually, let's think more carefully.
    
    # P and Q conflict if there exists R (R != P, R != Q) such that {P,Q,R} is isosceles.
    # This means one of:
    # a) dist(P,Q) == dist(P,R)  -> R is in the circle centered at P with radius dist(P,Q)
    # b) dist(P,Q) == dist(Q,R)  -> R is in the circle centered at Q with radius dist(P,Q)
    # c) dist(R,P) == dist(R,Q)  -> R is on the perpendicular bisector of PQ
    
    # For (a): R must satisfy dist(P,R) == dist(P,Q). So R is at distance dist(P,Q) from P.
    # For (b): R must satisfy dist(Q,R) == dist(P,Q). So R is at distance dist(P,Q) from Q.
    # For (c): R must satisfy dist(R,P) == dist(R,Q).
    
    # Let's precompute for each point P and each distance d, the set of points at distance d from P.
    # This allows O(1) lookup for cases (a) and (b).
    
    # dist_to_points[P][d] = list of points R such that dist(P,R) == d
    # But d can be up to 2*N^2, so this might use a lot of memory.
    # Alternative: for each P, group points by distance.
    
    # Let's use a different approach: for each P, create a dict mapping distance -> list of points
    dist_map = [{} for _ in range(num_points)]
    for P in range(num_points):
        for Q in range(num_points):
            if P != Q:
                d = int(dist2[P, Q])
                if d not in dist_map[P]:
                    dist_map[P][d] = []
                dist_map[P][d].append(Q)
    
    # Now build conflicts
    # For each pair (P, Q):
    #   Check if there exists R != P, Q with dist(P,R) == dist(P,Q)
    #   Check if there exists R != P, Q with dist(Q,R) == dist(P,Q)
    #   Check if there exists R != P, Q with dist(R,P) == dist(R,Q)
    
    # The third condition is harder to check quickly. Let's handle it differently.
    # dist(R,P) == dist(R,Q) means R is equidistant from P and Q.
    # This is equivalent to R lying on the perpendicular bisector of segment PQ.
    
    # For efficiency, let's precompute for each unordered pair {P,Q}, whether they conflict.
    # We'll use a 2D boolean array or a set of conflicting pairs.
    
    # Actually, let's just precompute the full conflict matrix as a bitset.
    # conflict[P] is a bitset where bit Q is set if P and Q conflict.
    
    # For each P:
    #   For each Q > P:
    #     Check if P and Q conflict
    
    # To check if P and Q conflict:
    #   d = dist(P,Q)
    #   Case (a): Is there R != P,Q with dist(P,R) == d?
    #     Look at dist_map[P][d]. If it contains any R != Q, then yes.
    #     Note: Q is in dist_map[P][d] by definition, so we need at least one other.
    #   Case (b): Is there R != P,Q with dist(Q,R) == d?
    #     Look at dist_map[Q][d]. If it contains any R != P, then yes.
    #   Case (c): Is there R != P,Q with dist(R,P) == dist(R,Q)?
    #     This requires checking all R, which is O(n). 
    #     But we can precompute this.
    
    # Let's precompute for each pair (P,Q) whether they are equidistant from some R.
    # Actually, let's just do the O(n) check for case (c) since it's only done once.
    
    # Precompute equidistant: for each R, for each pair (P,Q) equidistant from R, mark them.
    # equidistant[P][Q] = True if there exists R with dist(R,P) == dist(R,Q)
    
    # This is O(n^3) which for n=4096 is about 68 billion operations. Too slow.
    
    # Alternative: for each R, the points equidistant from R form a set.
    # For each R, group points by distance from R. For each group, all pairs in that group
    # are equidistant from R.
    
    # Let's precompute: for each R, for each distance d, the list of points at distance d from R.
    # Then for each such list, all pairs are equidistant from R.
    
    # But we need to know if P and Q are equidistant from ANY R, not just a specific one.
    
    # Let's use a different strategy: precompute a bitset equi_from_R[R] where bit P is set
    # if... no, that doesn't help directly.
    
    # Actually, let's just precompute the conflict matrix using the following:
    # P and Q conflict if:
    #   1. len(dist_map[P][d]) > 1 where d = dist(P,Q)  [case a: P has another point at same distance as Q]
    #   2. len(dist_map[Q][d]) > 1 where d = dist(P,Q)  [case b: Q has another point at same distance as P]
    #   3. There exists R != P,Q with dist(R,P) == dist(R,Q)  [case c]
    
    # For case 3, let's precompute a 2D array or use a different approach.
    
    # Let's precompute for each R, the equivalence classes of points by distance from R.
    # For each R and each distance d, let S(R,d) = {P : dist(R,P) = d}.
    # Then for any P, Q in S(R,d), P and Q are equidistant from R.
    # So we need: exists R, d such that P, Q both in S(R,d).
    
    # We can precompute this by iterating over all R and all d, and for each S(R,d),
    # marking all pairs in S(R,d) as equidistant.
    
    # But the total number of pairs across all R,d could be large.
    # For each R, the sum of |S(R,d)|^2 over all d is at most n^2 (since sum |S(R,d)| = n-1).
    # So total work is O(n^3) in the worst case, but typically less.
    
    # For n=4096, n^3 = 68 billion, which is too slow in Python.
    
    # Let's use a smarter approach: for each pair (P,Q), check if they are equidistant from any R.
    # This is equivalent to: the perpendicular bisector of PQ contains at least one grid point R != P,Q.
    
    # The perpendicular bisector of P=(x1,y1) and Q=(x2,y2) is the line:
    # (x2-x1)(x - (x1+x2)/2) + (y2-y1)(y - (y1+y2)/2) = 0
    # Or: (x2-x1)x + (y2-y1)y = (x2^2-x1^2 + y2^2-y1^2)/2
    
    # For integer grid points, we need to check if this line contains any integer point (x,y) in [0,N)^2
    # with (x,y) != P and (x,y) != Q.
    
    # This is getting complex. Let's use a simpler but potentially slower approach for now,
    # and optimize later if needed.
    
    # Alternative approach: instead of precomputing all conflicts, use a faster validity check
    # that doesn't require the full conflict graph.
    
    # Let's go back to the original approach but optimize it.
    
    # Key optimization: instead of using Python sets for dist_sets, use a different data structure.
    # Also, vectorize the validity check where possible.
    
    # New approach: maintain for each point in the set, the set of distances to other points in the set.
    # When adding point P, check:
    # 1. All distances from P to points in S are distinct (P is not apex)
    # 2. For each point Q in S, dist(P,Q) is not already in dist_set[Q] (Q is not apex)
    
    # This is what the original code does. The bottleneck is the O(|S|) check for condition 2.
    
    # Let's try to vectorize this.
    
    # Actually, let's try a completely different approach: use a SAT-like or greedy approach
    # with a different validity criterion.
    
    # Observation: if we ensure that all pairwise distances in the set are distinct, then
    # we automatically have no isosceles triangles. This is a sufficient but not necessary condition.
    # The original code relaxes this to allow shared distances between disjoint pairs.
    
    # Let's try to improve the local search by using a better perturbation strategy.
    
    # For now, let's keep the original approach but with some optimizations:
    # 1. Use numpy for the distance check in can_add
    # 2. Better randomization in local search
    # 3. More aggressive restarts
    
    # Let's rewrite with these optimizations.
    
    def can_add_fast(P, S, dist_sets):
        """Check if P can be added to set S. Returns True if valid."""
        n = len(S)
        if n == 0:
            return True
        if n == 1:
            return True
        
        # Get distances from P to all points in S
        d_vals = dist2[P, S]
        
        # Check if all distances are distinct (P is not apex)
        # Use sorting for O(n log n) instead of O(n) unique
        if n > 1:
            sorted_d = np.sort(d_vals)
            if np.any(sorted_d[:-1] == sorted_d[1:]):
                return False
        
        # Check if any point in S has P at a distance it already has
        # This is the expensive part
        for i in range(n):
            d = int(d_vals[i])
            if d in dist_sets[i]:
                return False
        
        return True
    
    def add_point_fast(P, S, dist_sets):
        """Add point P to set S, updating dist_sets."""
        S.append(P)
        new_dist_set = set()
        if len(S) > 1:
            d_vals = dist2[P, S[:-1]]
            for i, d in enumerate(d_vals):
                d_int = int(d)
                dist_sets[i].add(d_int)
                new_dist_set.add(d_int)
        dist_sets.append(new_dist_set)
    
    def remove_point_fast(P, S, dist_sets):
        """Remove point P from set S, updating dist_sets."""
        idx = S.index(P)
        for i in range(len(S)):
            if i == idx:
                continue
            d = int(dist2[P, S[i]])
            dist_sets[i].discard(d)
        S.pop(idx)
        dist_sets.pop(idx)
    
    def greedy_construction(order, time_limit):
        """Greedy construction with time limit."""
        S = []
        dist_sets = []
        in_S = np.zeros(num_points, dtype=bool)
        
        for P in order:
            if time.time() > time_limit:
                break
            if in_S[P]:
                continue
            
            if can_add_fast(P, S, dist_sets):
                in_S[P] = True
                add_point_fast(P, S, dist_sets)
        
        return S, dist_sets, in_S
    
    def local_search(S, dist_sets, in_S, time_limit):
        """Local search with add/remove moves."""
        best_S = S[:]
        best_dist_sets = [s.copy() for s in dist_sets]
        best_size = len(S)
        
        no_improve = 0
        max_no_improve = 100
        
        while time.time() < time_limit and no_improve < max_no_improve:
            added = False
            
            # Try to add a random point
            complement_size = num_points - len(S)
            if complement_size > 0:
                batch_size = min(500, complement_size)
                candidates = random.sample(range(num_points), batch_size)
                for P in candidates:
                    if in_S[P]:
                        continue
                    
                    if can_add_fast(P, S, dist_sets):
                        in_S[P] = True
                        add_point_fast(P, S, dist_sets)
                        
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = S[:]
                            best_dist_sets = [s.copy() for s in dist_sets]
                            no_improve = 0
                        else:
                            no_improve += 1
                        
                        added = True
                        break
            
            if not added:
                no_improve += 1
                
                # Perturbation: remove 1-2 random points, try to add 2-3
                if len(S) > 3:
                    num_remove = random.choice([1, 1, 2])
                    removed = []
                    for _ in range(num_remove):
                        if not S:
                            break
                        rem_idx = random.randrange(len(S))
                        rem_P = S[rem_idx]
                        in_S[rem_P] = False
                        remove_point_fast(rem_P, S, dist_sets)
                        removed.append(rem_P)
                    
                    # Try to add points
                    added_count = 0
                    target_add = num_remove + 1
                    complement_size = num_points - len(S)
                    if complement_size > 0:
                        batch_size = min(200, complement_size)
                        candidates = random.sample(range(num_points), batch_size)
                        for P in candidates:
                            if in_S[P]:
                                continue
                            
                            if can_add_fast(P, S, dist_sets):
                                in_S[P] = True
                                add_point_fast(P, S, dist_sets)
                                added_count += 1
                                
                                if len(S) > best_size:
                                    best_size = len(S)
                                    best_S = S[:]
                                    best_dist_sets = [s.copy() for s in dist_sets]
                                    no_improve = 0
                                
                                if added_count >= target_add:
                                    break
                    
                    no_improve = 0
        
        return best_S, best_dist_sets
    
    # Main loop
    best_set = []
    best_size = 0
    rng = np.random.default_rng(42)
    
    # Multiple restarts with different random seeds
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Create random order
        order = rng.permutation(num_points).tolist()
        
        # Greedy construction
        greedy_time = time.time() + min(5, (deadline - time.time()) / 3)
        S, dist_sets, in_S = greedy_construction(order, greedy_time)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
        
        # Local search
        remaining = deadline - time.time()
        if remaining > 10:
            search_time = min(20, remaining / 3)
            S, dist_sets = local_search(S, dist_sets, in_S, time.time() + search_time)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
        
        # Early exit
        if best_size > 55:
            break
    
    # Convert indices to coordinates
    result = [(idx % N, idx // N) for idx in best_set]
    
    return result

if __name__ == "__main__":
    points = construct()
    print(f"Found set of size {len(points)}")
    print(points[:10])
