import time
import random
import math

N = 64
NUM_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    # Precompute coordinates and squared distances
    # To save memory and time, we precompute squared distances between all pairs?
    # 4096 * 4096 = 16M pairs. 
    # Storing a full matrix of int16 or int32 is 64-128MB. 
    # Python lists of ints are huge. 
    # Let's use numpy for the distance matrix if available, or just compute on fly but cache per-point.
    # Actually, computing dx*dx + dy*dy is very fast. The bottleneck is the set lookups and list iterations.
    
    # Strategy:
    # 1. Greedy initialization with multiple random seeds/orderings.
    # 2. Local search: 
    #    - Try to add points. If fail, identify conflicts.
    #    - Try to swap points: Remove a point that is "blocking" many others, add a new point.
    #    - Or simpler: Random restarts of a "greedy with backtracking" or "local repair".
    
    # Given the time limit (280s), we can afford several iterations of a local search.
    
    # Data structures:
    # S: set of active point indices
    # dist_count: array of size NUM_POINTS, where dist_count[i] is a set of squared distances from i to other points in S.
    # This allows O(1) check if a distance is already used by a neighbor.
    # But checking "can_add(p)" requires checking against ALL points in S.
    # For each a in S, check if dist(p, a) is in dist_count[a].
    # If S is large (~1000), this is 1000 checks per candidate.
    # 4096 candidates * 1000 checks = 4M checks. 
    # If we do this in a loop, it might be slow in Python.
    
    # Optimization: 
    # Instead of iterating all S, we can maintain a "conflict" counter?
    # Or just use a fast loop.
    
    # Let's try a different approach: 
    # The problem is equivalent to finding a large set of points such that all pairwise distances are distinct?
    # No, "no three points form an isosceles triangle" means that for any point A, the distances from A to any other two points B, C must be distinct.
    # It does NOT require that dist(A,B) != dist(C,D) for disjoint pairs.
    # It only requires that for any fixed vertex, the outgoing edges have unique lengths.
    # So, for each point p in S, the set of distances {dist(p, q) for q in S, q != p} must have no duplicates.
    
    # This is a "Sidon set" in terms of distances from each vertex.
    
    # Algorithm:
    # 1. Start with empty set.
    # 2. Add points greedily.
    # 3. Local search: 
    #    - Pick a random point p in S.
    #    - Try to remove p and replace it with a new point q.
    #    - Or, try to add q. If it conflicts with some p, try to remove p and add q.
    
    # Let's implement a fast "can_add" and "add" using lists.
    
    def get_coords(idx):
        return (idx // N, idx % N)
        
    # Precompute all squared distances?
    # dist2[i][j]
    # Using a flat array of uint16? Max dist is 63^2 + 63^2 = 3969 + 3969 = 7938. Fits in uint16.
    # 4096 * 4096 * 2 bytes = 32 MB. This is acceptable.
    import numpy as np
    
    # Generate grid
    xs = np.repeat(np.arange(N), N)
    ys = np.tile(np.arange(N), N)
    # dist2[i, j] = (xs[i]-xs[j])^2 + (ys[i]-ys[j])^2
    # This is 4096x4096 matrix.
    # Computation might take a few seconds.
    
    # Let's check if numpy is fast enough for initialization.
    # If not, we compute on the fly. 
    # On the fly: 4096 points. 
    # In local search, we might do 10,000 swaps. 
    # Each swap involves checking ~1000 neighbors for the new point. 
    # 10,000 * 1000 = 10M distance calculations. 
    # 10M * (few ops) is fast in numpy, but slow in pure Python loops.
    # So, precomputing the distance matrix is crucial for speed.
    
    try:
        # Create distance matrix
        # To save memory, we only need upper triangle or full?
        # Full is easier.
        # Use float32 or uint16? uint16 is enough.
        # But numpy operations on uint16 might be slower than float32?
        # Let's use float32 for simplicity and speed of arithmetic, then cast to int for set lookups?
        # Or just use a list of lists of ints? 4096 lists of 4096 ints. 
        # Python int objects are 28 bytes. 4096*4096*28 = 470 MB. Too much.
        # So we MUST use numpy or array module.
        
        # Let's use a 2D numpy array of uint16.
        # dist2[i, j]
        
        # Efficient computation:
        # diff_x = xs[:, None] - xs[None, :]
        # diff_y = ys[:, None] - ys[None, :]
        # dist2 = diff_x**2 + diff_y**2
        # This creates intermediate arrays of size 4096x4096. 
        # 4096*4096*4 bytes = 64MB per array. 
        # We have diff_x, diff_y, dist2. ~200MB. Acceptable.
        
        xs_np = xs.astype(np.float32)
        ys_np = ys.astype(np.float32)
        
        diff_x = xs_np[:, None] - xs_np[None, :]
        diff_y = ys_np[:, None] - ys_np[None, :]
        dist2 = (diff_x * diff_x + diff_y * diff_y).astype(np.uint16)
        
        # Free memory
        del diff_x, diff_y, xs_np, ys_np
        
    except:
        # Fallback if memory fails or numpy not available (though it is)
        dist2 = None

    def can_add(S_set, dist_used, p):
        # S_set: set of indices
        # dist_used: list of sets. dist_used[i] contains dists from i to S
        # p: point to add
        # Check if p conflicts with any a in S
        # For each a in S:
        #   d = dist2[p, a]
        #   if d in dist_used[a]: return False
        #   if d in dist_used[p] (which is being built): return False
        
        # Optimization: Iterate over S.
        # If S is a list, iteration is fast.
        # Let's keep S as a list for fast iteration, and a set for O(1) membership.
        
        if not S_set:
            return True
            
        # We need to check against existing points in S.
        # Also, we need to ensure p doesn't have duplicate distances to OTHER points in S.
        # i.e., if dist(p, a) == dist(p, b) for a != b in S, then p creates an isosceles triangle (p, a, b) with p as apex.
        # So we must track distances FROM p as well.
        
        seen_dists_from_p = set()
        
        # Iterate over S
        # Using list is faster for iteration than set in Python?
        # Let's assume S is a list.
        for a in S_list:
            d = dist2[p, a]
            if d in dist_used[a]:
                return False
            if d in seen_dists_from_p:
                return False
            seen_dists_from_p.add(d)
            
        return True

    def add_point(S_list, S_set, dist_used, p):
        S_list.append(p)
        S_set.add(p)
        dist_used[p] = set()
        for a in S_list[:-1]:
            d = dist2[p, a]
            dist_used[a].add(d)
            dist_used[p].add(d)

    def remove_point(S_list, S_set, dist_used, p):
        S_set.remove(p)
        # Remove from list
        # List removal is O(N). 
        # To make it O(1), we can use a swap-remove if order doesn't matter.
        # But we need to iterate S_list in can_add. Order doesn't matter.
        # So swap-remove is good.
        
        idx = S_list.index(p)
        last = S_list[-1]
        S_list[idx] = last
        S_list.pop()
        
        # Remove distances
        for a in S_list:
            d = dist2[p, a]
            dist_used[a].discard(d)
        dist_used[p] = set()

    # Greedy Initialization
    def greedy_init(order):
        S_list = []
        S_set = set()
        dist_used = [set() for _ in range(NUM_POINTS)]
        
        for p in order:
            if can_add(S_set, dist_used, p):
                add_point(S_list, S_set, dist_used, p)
                
        return S_list, S_set, dist_used

    # Local Search
    def local_search(S_list, S_set, dist_used, max_iters=10000):
        # Try to improve size
        # Strategy:
        # 1. Try to add a random point.
        # 2. If fail, try a swap: remove a random point from S, add a random new point.
        #    If the new point fits, we keep it (size same, but might unlock additions later).
        #    Actually, we want to INCREASE size.
        #    So, we look for a point p that CAN be added.
        #    If none, we look for a swap that allows a future addition?
        #    Simpler: Just try to add. If stuck, do random swaps to change the landscape.
        
        size = len(S_list)
        all_points = list(range(NUM_POINTS))
        
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Try to add
            added = False
            # Try a few random points
            for _ in range(10):
                p = random.choice(all_points)
                if p not in S_set:
                    if can_add(S_set, dist_used, p):
                        add_point(S_list, S_set, dist_used, p)
                        size += 1
                        added = True
                        break
                        
            if added:
                continue
                
            # If no add, do a swap
            # Remove a random point, add a random point
            if S_list:
                q = random.choice(S_list)
                remove_point(S_list, S_set, dist_used, q)
                
                # Try to add a different point
                p = random.choice(all_points)
                if p != q and p not in S_set:
                    if can_add(S_set, dist_used, p):
                        add_point(S_list, S_set, dist_used, p)
                        # Size is same, but configuration changed
                    else:
                        # Revert? Or keep q out?
                        # If we keep q out, size decreases.
                        # Better to revert if we can't add p.
                        add_point(S_list, S_set, dist_used, q)
                else:
                    add_point(S_list, S_set, dist_used, q)
                    
        return S_list, S_set, dist_used

    best_size = 0
    best_S_list = []
    
    # Run multiple restarts with different orderings
    # 1. Random
    # 2. Spiral?
    # 3. Row-major
    
    orders = []
    
    # Random
    rand_order = list(range(NUM_POINTS))
    random.shuffle(rand_order)
    orders.append(rand_order)
    
    # Spiral
    spiral_order = []
    left, right = 0, N-1
    top, bottom = 0, N-1
    while left <= right and top <= bottom:
        for i in range(left, right+1):
            spiral_order.append(top * N + i)
        top += 1
        for i in range(top, bottom+1):
            spiral_order.append(i * N + right)
        right -= 1
        if top <= bottom:
            for i in range(right, left-1, -1):
                spiral_order.append(bottom * N + i)
            bottom -= 1
        if left <= right:
            for i in range(bottom, top-1, -1):
                spiral_order.append(i * N + left)
            left += 1
    orders.append(spiral_order)
    
    # Row-major
    orders.append(list(range(NUM_POINTS)))
    
    # Run greedy + local search for each order
    for order in orders:
        S_list, S_set, dist_used = greedy_init(order)
        
        # Local search
        S_list, S_set, dist_used = local_search(S_list, S_set, dist_used, max_iters=5000)
        
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_S_list = S_list[:]
            
        if time.time() > deadline - 60:
            break
            
    # Final polish on best
    # We need to reconstruct dist_used for best_S_list to do final local search?
    # Or just return best_S_list.
    # Let's do one more local search pass on the best found.
    
    # Rebuild dist_used for best_S_list
    S_set = set(best_S_list)
    dist_used = [set() for _ in range(NUM_POINTS)]
    for p in best_S_list:
        for a in best_S_list:
            if p != a:
                dist_used[p].add(dist2[p, a])
                
    # Run local search again
    best_S_list, S_set, dist_used = local_search(best_S_list, S_set, dist_used, max_iters=5000)
    
    # Convert to tuples
    return [get_coords(i) for i in best_S_list]

# Ensure numpy is imported at top if used inside function
# It is imported inside construct.
# But if dist2 is None, can_add will fail.
# The try-except handles it.
# If dist2 is None, we should fallback to on-the-fly calculation.
# Given the constraints, numpy should work.

if __name__ == "__main__":
    pts = construct()
    print(len(pts))
