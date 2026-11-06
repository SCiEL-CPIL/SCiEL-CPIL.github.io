import random
import time
from collections import defaultdict

N = 64
NUM_POINTS = N * N

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin

    import numpy as np

    # 1. Precompute Conflict Graph
    # Two points i, j conflict if there exists k such that dist(k,i) == dist(k,j).
    # This means k is the apex of an isosceles triangle with base ij.
    # We iterate over all possible apexes k.
    
    # Create grid coordinates
    coords = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    
    # conflicts[i] will be a set of indices j that conflict with i
    # Using sets for construction, then converting to bitmasks
    conflicts = [set() for _ in range(NUM_POINTS)]
    
    # Optimization: We only need to check apexes k. 
    # For a fixed k, group all other points by their squared distance to k.
    # Any two points in the same group conflict.
    
    # To speed this up, we can precompute squared distances between all points? 
    # No, that's 16M floats/ints, which is fine, but let's try the grouping approach directly.
    
    # Actually, a full pairwise distance matrix is 4096x4096 = 16.7M entries.
    # Memory: 16.7M * 8 bytes (int64) = 134 MB. This is acceptable.
    # Time to compute: vectorized numpy subtraction and squaring.
    
    # Compute pairwise squared distances
    # diff[i, j] = coords[i] - coords[j]
    # dist_sq[i, j] = sum(diff[i, j]^2)
    
    # We can do this in chunks to avoid memory spikes if needed, but 134MB is okay.
    # Let's use int32 for coords to save memory in intermediate steps if possible, 
    # but distances can be up to 2*63^2 = 7938, so int16 is enough for distances? 
    # Max dist sq: (63-0)^2 + (63-0)^2 = 3969 + 3969 = 7938. 
    # Wait, N=64, so max coord is 63.
    # Yes, int16 is sufficient for squared distances.
    
    # However, numpy broadcasting for 4096x4096x2 might be heavy.
    # Let's compute dist_sq row by row to keep memory lower and cache friendly.
    
    dist_sq = np.zeros((NUM_POINTS, NUM_POINTS), dtype=np.int32)
    
    for i in range(NUM_POINTS):
        # Vectorized distance from point i to all others
        diff = coords[i] - coords
        dist_sq[i] = (diff[:, 0] ** 2 + diff[:, 1] ** 2).astype(np.int32)
        
        if time.time() > deadline - 10:
            break # Emergency exit if precomp is too slow (shouldn't happen)

    # Now build conflict sets
    # For each apex k (row k in dist_sq), group points by distance.
    # If multiple points have the same distance > 0, they all conflict with each other.
    
    for k in range(NUM_POINTS):
        # Get distances from k
        dks = dist_sq[k]
        # We only care about points j != k (d > 0)
        # Group indices by distance value
        
        # Using a dictionary is O(N) per k, so O(N^2) total.
        # 4096 * 4096 = 16M operations. Python loops are slow.
        # Let's use numpy to group.
        
        # Get unique distances and inverse indices
        # Exclude distance 0 (self)
        mask = dks > 0
        indices = np.where(mask)[0]
        values = dks[mask]
        
        # Sort by value to group
        # This is O(N log N) per k. 4096 * 4096 * log(4096) ~ 16M * 12 = 192M ops.
        # Might be tight in Python.
        
        # Alternative: Use a dictionary with numpy arrays?
        # Or just use the fact that distances are small integers (max ~8000).
        # We can use a list of lists where index is distance.
        
        max_dist = int(np.max(dks)) if NUM_POINTS > 1 else 0
        # max_dist is small (~8000).
        groups = [[] for _ in range(max_dist + 1)]
        
        # This python loop is 4096 iterations per k. 4096*4096 = 16M iterations.
        # This is likely the bottleneck.
        
        # Let's try to vectorize the grouping logic?
        # np.unique with return_inverse is C-optimized.
        unique_dists, inverse = np.unique(values, return_inverse=True)
        
        # Now for each unique distance, get the points.
        # This still requires iterating over unique distances.
        
        for d_idx in range(len(unique_dists)):
            # Get points with this distance
            # Using boolean mask on inverse
            # This is also slow in pure Python loop if done naively.
            
            # Let's stick to the dictionary approach but optimize the inner loop
            # by using numpy to find indices for a specific distance?
            
            pass

    # Re-evaluating precomputation speed.
    # 16M iterations in Python is ~1-2 seconds.
    # The inner loop adds to conflicts.
    # If a group has size M, we add M*(M-1)/2 edges.
    # On average, how many points share a distance from a center?
    # For a grid, distances are not uniformly distributed.
    # But total edges in conflict graph is bounded by N^3?
    # Actually, for each k, sum of (M_d^2) over all distances d.
    # This can be large.
    
    # Let's try a different precomputation:
    # Iterate over all pairs (i, j)? No, O(N^4).
    
    # Let's optimize the k-loop.
    # Instead of Python loops for grouping, we can use a dictionary mapping distance to a list of points.
    # Building this dict for each k takes O(N).
    # Then for each distance, if list size > 1, add edges.
    
    # Let's rewrite the precomputation more carefully.
    
    conflicts = [set() for _ in range(NUM_POINTS)]
    
    for k in range(NUM_POINTS):
        dks = dist_sq[k]
        # Group points by distance
        # Using a dict is efficient for sparse data, but here we have all points.
        # Let's use numpy's unique.
        
        # Exclude self
        mask = dks > 0
        idxs = np.where(mask)[0]
        vals = dks[mask]
        
        # Sort by value
        # argsort is O(N log N)
        sorted_indices = np.argsort(vals)
        sorted_vals = vals[sorted_indices]
        sorted_idxs = idxs[sorted_indices]
        
        # Iterate through sorted values to find groups
        # This is O(N) per k.
        
        current_val = -1
        group_start = 0
        
        for pos in range(len(sorted_vals)):
            if sorted_vals[pos] != current_val:
                # Process previous group
                if current_val != -1 and pos > group_start + 1:
                    # Group is sorted_idxs[group_start:pos]
                    # Add all pairs in this group to conflicts
                    group_points = sorted_idxs[group_start:pos]
                    # Add edges
                    # M = len(group_points)
                    # For a in group, for b in group, if a!=b: conflicts[a].add(b)
                    # This is M^2.
                    # We can do: for a in group_points: conflicts[a].update(group_points)
                    # But we must not add self-conflict? The problem says "three points", so distinct.
                    # If a==b, it's not a triangle. But conflict graph usually has no self-loops.
                    # If we add self, it doesn't hurt MIS much (just ignores self).
                    
                    for a in group_points:
                        # Update with all points in group
                        # Using set.update is fast
                        conflicts[a].update(group_points)
                        
                # Start new group
                current_val = sorted_vals[pos]
                group_start = pos
                
        # Process last group
        if current_val != -1 and len(sorted_vals) > group_start + 1:
            group_points = sorted_idxs[group_start:]
            for a in group_points:
                conflicts[a].update(group_points)

    # Remove self-conflicts (though they shouldn't affect MIS if we mask them out)
    # And ensure symmetry (update already did this? No, if a is in group, a.update(group) adds b to a.
    # But does it add a to b? Yes, because b is also in group and b.update(group) adds a.
    # Wait, the loop `for a in group_points: conflicts[a].update(group_points)` ensures
    # that for every a in group, all other points in group are added to conflicts[a].
    # So yes, symmetry is maintained.

    # Convert to bitmasks for fast B&B
    # Bit i is set if i is in the set.
    # conflict_mask[i] has bit j set if i and j conflict.
    
    conflict_masks = [0] * NUM_POINTS
    for i in range(NUM_POINTS):
        mask = 0
        for j in conflicts[i]:
            mask |= (1 << j)
        conflict_masks[i] = mask

    # 2. Branch and Bound for Maximum Independent Set
    
    best_size = 0
    best_set_mask = 0
    
    # Upper bound heuristic: Greedy Coloring
    # We color the subgraph induced by available vertices.
    # The number of colors is an upper bound on the size of the MIS in that subgraph.
    
    def greedy_coloring(available_mask):
        """
        Returns the number of colors needed for the subgraph induced by available_mask.
        We use a simple greedy algorithm:
        1. Order vertices (e.g., by degree in the subgraph, or just index order).
        2. Assign smallest available color.
        
        For speed, we use a simple heuristic:
        Iterate through vertices in the available mask.
        Keep track of colors used by neighbors.
        
        Since we need this in the inner loop of B&B, it must be fast.
        A common fast heuristic is to use the number of vertices as a loose bound,
        or a very fast greedy.
        
        Let's use a simple greedy:
        colors = {}
        for each vertex v in available:
            forbidden = set of colors of neighbors of v that are in available.
            color[v] = smallest non-negative integer not in forbidden.
        return max(color.values()) + 1
        """
        if available_mask == 0:
            return 0
            
        # Get list of vertices
        # Extracting bits from a large integer is O(N)
        # We can iterate through bits.
        
        # Optimization: If the number of vertices is small, just do it.
        # If large, maybe use a simpler bound?
        
        # Let's extract vertices
        verts = []
        m = available_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
            
        if not verts:
            return 0
            
        # Simple greedy coloring
        # We need to check neighbors.
        # For each v, check which colors are used by its neighbors in verts.
        
        # To make it faster, we can use a dictionary for colors assigned so far.
        # And for each v, check conflicts[v] & available_mask to see neighbors.
        
        color_map = {}
        max_color = 0
        
        for v in verts:
            # Get neighbors of v in available set
            neighbors_mask = conflict_masks[v] & available_mask
            # Remove v itself (should be 0 anyway if no self loops, but just in case)
            neighbors_mask &= ~(1 << v)
            
            # Get colors of these neighbors
            # We need to map neighbor index to color.
            # Since we process in order, some neighbors might not be colored yet?
            # Greedy coloring usually requires an ordering. If we use natural order,
            # neighbors with higher index might not be colored.
            # Standard greedy: color v based on colors of previously colored neighbors.
            
            forbidden = set()
            n_mask = neighbors_mask
            while n_mask:
                lsb = n_mask & -n_mask
                n_idx = lsb.bit_length() - 1
                if n_idx in color_map:
                    forbidden.add(color_map[n_idx])
                n_mask ^= lsb
                
            # Find smallest available color
            c = 0
            while c in forbidden:
                c += 1
                
            color_map[v] = c
            if c > max_color:
                max_color = c
                
        return max_color + 1

    def branch_and_bound(available_mask, current_mask, current_size):
        nonlocal best_size, best_set_mask
        
        if time.time() > deadline:
            return
            
        # Pruning: if current_size + upper_bound(available_mask) <= best_size, prune
        # Upper bound: number of vertices in available_mask
        # Better upper bound: greedy coloring of available_mask
        
        # Quick check: if available_mask is 0, update best
        if available_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_mask
            return
            
        # Heuristic upper bound
        # Using greedy coloring is expensive if done at every node.
        # Let's use a simpler bound first: popcount(available_mask)
        # If current_size + popcount(available_mask) <= best_size: return
        
        avail_count = available_mask.bit_count()
        if current_size + avail_count <= best_size:
            return
            
        # Try a better bound: Greedy Coloring
        # To save time, we only use coloring if avail_count is small enough or 
        # if the simple bound doesn't prune.
        
        # Actually, let's just use the simple bound for now to ensure speed,
        # and rely on the search order to find good solutions early.
        
        # Search Order:
        # Pick a vertex v from available.
        # Branch 1: Include v.
        # Branch 2: Exclude v.
        
        # To improve pruning, we should pick v with high degree or something?
        # Or just pick the first one?
        
        # Let's pick a vertex.
        # We can try to pick the vertex that appears in the most conflicts?
        # Or just the lowest index for determinism.
        
        # Let's pick the vertex with the highest degree in the available subgraph?
        # That requires calculating degrees, which is O(N^2) per node. Too slow.
        
        # Let's just pick the first set bit.
        lsb = available_mask & -available_mask
        v = lsb.bit_length() - 1
        
        # Branch 1: Include v
        # New available: available_mask & ~conflict_masks[v] & ~(1 << v)
        # New current: current_mask | (1 << v)
        
        # We remove v and all its neighbors from available.
        new_available = available_mask & ~conflict_masks[v] & ~(1 << v)
        
        branch_and_bound(new_available, current_mask | (1 << v), current_size + 1)
        
        # Branch 2: Exclude v
        # New available: available_mask & ~(1 << v)
        # New current: current_mask
        
        branch_and_bound(available_mask & ~(1 << v), current_mask, current_size)

    # Initialize B&B
    # Start with all points available
    full_mask = (1 << NUM_POINTS) - 1
    
    # To help B&B find a good solution quickly, we can run a greedy first.
    # Greedy: Pick point with max degree? Or min degree?
    # For MIS, picking a vertex with high degree removes many neighbors,
    # which might be bad if we want a large set.
    # Picking a vertex with low degree is often better for greedy MIS.
    
    # Let's run a simple greedy to initialize best_size
    # Greedy:
    # 1. Start with empty set.
    # 2. While available:
    #    Pick vertex v in available with minimum degree in the induced subgraph?
    #    Or just pick the first one.
    #    Add v to set.
    #    Remove v and neighbors from available.
    
    def greedy_mis():
        available = full_mask
        selected = 0
        count = 0
        while available:
            # Pick first vertex
            lsb = available & -available
            v = lsb.bit_length() - 1
            
            selected |= (1 << v)
            count += 1
            
            # Remove v and neighbors
            available &= ~conflict_masks[v]
            available &= ~(1 << v)
            
        return count, selected

    init_size, init_mask = greedy_mis()
    best_size = init_size
    best_set_mask = init_mask
    
    # Now run B&B
    # We need to be careful with recursion depth. NUM_POINTS is 4096.
    # Python recursion limit is usually 1000.
    # We must increase it or use an iterative approach.
    # sys.setrecursionlimit(10000)
    
    import sys
    sys.setrecursionlimit(10000)
    
    # Run B&B
    branch_and_bound(full_mask, 0, 0)
    
    # Extract result
    result_points = []
    m = best_set_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result_points.append((coords[idx][0], coords[idx][1]))
        m ^= lsb
        
    return result_points

# EVOLVE-BLOCK-END
