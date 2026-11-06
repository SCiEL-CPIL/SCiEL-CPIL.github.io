import time
import numpy as np

N = 64
GRID_SIZE = N * N

# Pre-compute conflict graph
# An edge exists between point i and point j if there exists a point k
# such that {i, j, k} forms an isosceles triangle.
# This is equivalent to:
# 1. d(i, j) == d(i, k)  => d(i, k)^2 == d(i, j)^2
# 2. d(i, j) == d(j, k)  => d(i, k)^2 == d(i, j)^2
# 3. d(i, k) == d(j, k)  => d(i, k)^2 == d(j, k)^2 ... wait.
# Let's re-verify the condition for "no three points forming an isosceles triangle".
# A set S is valid if for all distinct a, b, c in S, the triangle abc is not isosceles.
# Triangle abc is isosceles if:
# d(a,b) == d(a,c) OR d(a,b) == d(b,c) OR d(a,c) == d(b,c).
#
# If we want to select a subset of vertices such that NO triple is isosceles,
# we can define a graph G where an edge (u, v) exists if u and v CANNOT be in the same valid set.
# When does u and v conflict?
# u and v conflict if there exists ANY w in the grid such that {u, v, w} is isosceles.
# If we select u and v, and later select w, we violate the condition.
# So, if we restrict ourselves to sets where no pair (u, v) has a "witness" w that is also selected, 
# is that sufficient? 
# Actually, if we define the conflict graph H where (u, v) is an edge if there exists w 
# such that {u, v, w} is isosceles, then an Independent Set in H is a set of vertices 
# such that for any pair u, v in the set, there is NO w in the grid such that {u, v, w} is isosceles.
# This is STRONGER than the requirement. The requirement is that no w IN THE SET forms an isosceles triangle.
# If we use Independent Set on H, we guarantee that for any u, v in S, there is no w IN THE GRID 
# (let alone in S) that forms an isosceles triangle. 
# This is a valid sufficient condition, but might be too restrictive?
# Let's check. If {u, v, w} is isosceles, we cannot have all three. 
# The condition "no three points in S form an isosceles triangle" means S is a 3-uniform hypergraph independent set.
# Mapping to a graph independent set: 
# If we define edges (u, v) for every pair that participates in ANY isosceles triangle with ANY third point, 
# then any independent set in this graph will not contain any such pair. 
# Consequently, it will not contain any triple that is isosceles, because any isosceles triple contains at least one pair 
# that is "conflicting" (in fact, all pairs in an isosceles triangle are part of that isosceles triangle).
# Wait, if {u, v, w} is isosceles, does (u, v) have to be an edge in H? 
# Yes, because w is a grid point. So (u, v) is an edge. 
# Thus, an independent set in H is a valid no-isosceles set.
# Is it optimal? Probably not, but it's a valid relaxation that yields a valid solution.
# Actually, this is exactly the definition of the "2-packing" or similar, but here it's just "no pair is part of any isosceles triangle".
# This is a standard reduction for this specific problem type in some contexts, or at least a very strong heuristic.
# Let's verify: If S is an independent set in H, then for any u, v in S, there is no w in Grid such that {u, v, w} is isosceles.
# Therefore, there is no w in S such that {u, v, w} is isosceles.
# So S is a valid no-isosceles set.
#
# The graph H is dense. 
# How to build H efficiently?
# For each pair (u, v), check if there exists w such that d(u,v) == d(u,w) or d(u,v) == d(v,w) or d(u,w) == d(v,w).
# This is O(N^6) if done naively. N=64, N^2=4096. (4096)^3 is too big.
# We need a smarter way to build the conflict graph or the heuristic.
#
# Alternative: 
# The "lowest degree" greedy on the conflict graph requires the graph.
# Building the full conflict graph H:
# For each point u, let D_u be the multiset of distances from u to all other points.
# If a distance d appears in D_u with count > 0, it means there are points at distance d from u.
# For a pair (u, v) to be an edge in H:
# 1. There exists w such that d(u,w) = d(u,v). This is true if count of d(u,v) in D_u is >= 2 (one is v, so at least one other w).
#    Wait, if d(u,v) is the distance, and there is another point w with d(u,w)=d(u,v), then (u,v) is an edge.
#    So, if the number of points at distance d(u,v) from u is >= 2, then (u,v) is an edge? 
#    No, if d(u,v) is the distance, and w is distinct from v, then yes.
#    So condition 1: freq_u(d(u,v)) >= 2.
# 2. There exists w such that d(v,w) = d(u,v). Similarly, freq_v(d(u,v)) >= 2.
# 3. There exists w such that d(u,w) = d(v,w). This means w is equidistant from u and v.
#    The set of points equidistant from u and v is the perpendicular bisector of segment uv.
#    If there is any grid point w on the perpendicular bisector of uv (and w != u, v), then (u,v) is an edge.
#
# So, (u, v) is an edge in H if:
# - freq_u(d(u,v)) >= 2  (u is the apex of an isosceles triangle with base involving v and some w)
# - OR freq_v(d(u,v)) >= 2 (v is the apex)
# - OR there exists a grid point w on the perpendicular bisector of uv.
#
# This can be precomputed.
#
# Let's precompute:
# 1. For each point u, and each possible squared distance d, the frequency of that distance from u.
#    Actually, we just need to know for a specific pair (u,v) if freq_u(d_uv) >= 2.
#    We can precompute a 2D array `freq_count[u][d]`? No, d can be large.
#    Better: For each u, create a dictionary or array of counts for each distance.
#    Max distance is 64^2 + 64^2 = 8192.
#    So `dist_counts[u][d]` where d is in [0, 8192].
#    Size: 4096 * 8193 integers. ~33 million integers. 33M * 4 bytes = 132 MB. Acceptable.
#    
# 2. For the perpendicular bisector condition:
#    For each pair (u, v), check if there is a point w on the bisector.
#    This is harder to precompute globally without iterating pairs.
#    However, we can iterate over all pairs (u, v) and check the bisector condition.
#    Number of pairs is ~8 million.
#    For each pair, checking if any of 4096 points is on the bisector is O(N^2) per pair -> O(N^4) total. Too slow.
#    
#    Optimization for bisector:
#    The condition d(u,w) == d(v,w) is equivalent to w being on the perpendicular bisector.
#    We can precompute for each point w, the set of pairs (u,v) for which w is on the bisector? 
#    That's also complex.
#    
#    Let's look at the condition again.
#    If we skip the bisector condition, we only catch isosceles triangles where the equal sides meet at u or v.
#    We miss triangles where the equal sides meet at w (the base is uv).
#    If we ignore the "base" case, our conflict graph H' will have fewer edges.
#    An independent set in H' is NOT necessarily a valid no-isosceles set, because it might contain u, v, w where d(u,w)=d(v,w).
#    So we MUST include the bisector condition to ensure validity.
#    
#    Alternative Strategy:
#    Instead of precomputing the full graph, use the greedy selection but check validity on the fly?
#    The prompt asks for a "greedy heuristic that iteratively selects the vertex with the lowest degree in the remaining graph".
#    This implies we need the graph.
#    
#    Let's try to build the graph efficiently.
#    N=64.
#    Precompute `dist_counts[u][d]`.
#    For the bisector part:
#    For each point w, consider all pairs (u, v) such that d(u,w) == d(v,w).
#    This means u and v are on a circle centered at w.
#    For a fixed w, group all other points by their distance to w.
#    For each group (circle), all pairs (u, v) in that group have d(u,w) == d(v,w).
#    So for each w, and for each distance d, let C(w, d) be the set of points at distance d from w.
#    Then for every pair (u, v) in C(w, d), we add an edge (u, v) to the conflict graph.
#    
#    Total edges added this way:
#    Sum over all w and all d of |C(w, d)| choose 2.
#    This is much smaller than N^4.
#    
#    Algorithm to build Conflict Graph H (adjacency matrix or set of edges):
#    1. Initialize an empty set of edges or a boolean matrix.
#       Given N^2 = 4096, a boolean matrix is 4096x4096 = 16M booleans. 16MB. OK.
#       Or a list of sets.
#    2. For each point u (0 to 4095):
#       - Compute distances to all other points.
#       - Count frequencies of each distance.
#       - For each other point v:
#           - d = dist(u, v)
#           - If freq_u[d] >= 2: add edge (u, v). (Apex at u)
#    3. For each point w (0 to 4095):
#       - Group all other points v by dist(w, v).
#       - For each group (list of points at same distance from w):
#           - For each pair (u, v) in this group:
#               - Add edge (u, v). (Apex at w, base is uv)
#    
#    This covers all cases:
#    - Apex at u: handled in step 2.
#    - Apex at v: handled in step 2 when u=v is swapped? No, step 2 handles "u is apex". 
#      When we process point v in step 2, we check if freq_v[d(v,u)] >= 2. If so, add edge (v,u).
#      Since graph is undirected, adding (u,v) is same as (v,u).
#      So step 2 covers "Apex at u" and "Apex at v".
#    - Apex at w: handled in step 3.
#    
#    Complexity:
#    Step 2: For each u, iterate v. O(N^4) distance calculations? No, we can precompute distances.
#            Precompute DIST matrix: O(N^4).
#            Then Step 2 is O(N^4) to check frequencies.
#    Step 3: For each w, group points. 
#            For each w, we look at 4095 points. Grouping is O(N^2).
#            Then for each group, add edges.
#            Total work in Step 3 is sum of |C(w,d)|^2.
#            In a grid, points on a circle are sparse. Max points on a circle in 64x64 grid is small (maybe < 100).
#            So Step 3 is fast.
#            
#    Total Precomputation: O(N^4) ~ 4096^2 * constant. 16M operations. Fast in Python? 
#    16M simple operations might take 1-2 seconds. Acceptable.
#    
#    Then, Greedy MIS on the remaining graph:
#    We maintain a set of available vertices.
#    We need to find the vertex with the lowest degree in the induced subgraph of available vertices.
#    Naively, this is O(V^2) per step. V=4096. V^2 = 16M.
#    Total steps = size of set. If size is ~100, total 1.6B operations. Too slow in Python.
#    
#    Optimization:
#    We can use a priority queue or just re-scan.
#    Or, we can approximate the degree.
#    Or, we can use a different heuristic: 
#    "Greedy by lowest degree" is standard.
#    To speed up, we can maintain the current degree of each node.
#    When we select a node u, we remove u and all its neighbors.
#    We need to update degrees of remaining nodes.
#    
#    Let's try to implement this with NumPy for speed if possible, or optimized Python.
#    Given the 300s limit, and precomputation taking ~2s, we have ~290s for search.
#    If we do one greedy pass, it should be fast enough if optimized.
#    
#    Let's refine the "Lowest Degree" selection.
#    Instead of exact lowest degree in the remaining graph (which is expensive to maintain dynamically),
#    we can use the initial degree in the full conflict graph as a proxy?
#    No, the prompt says "lowest degree in the remaining graph".
#    
#    Let's try a simpler approach first: 
#    Randomized Greedy with Conflict Graph?
#    The prompt specifically asks for "greedy heuristic that iteratively selects the vertex with the lowest degree in the remaining graph".
#    
#    Let's implement the exact algorithm but optimize the degree update.
#    We maintain a boolean array `active` of size V.
#    We maintain an integer array `deg` of size V, representing the number of active neighbors.
#    
#    Step:
#    1. Find u in active with min deg.
#    2. Add u to solution.
#    3. Mark u as inactive.
#    4. For each neighbor v of u:
#       - If v is active, mark v as inactive.
#       - Decrement deg of all active neighbors of v? 
#         Wait, if v is removed, it no longer contributes to the degree of its neighbors.
#         So for each active neighbor w of v, decrement deg[w].
#    
#    This is O(D^2) per step in the worst case, where D is the max degree.
#    In this graph, D is likely high (dense graph).
#    If D is ~1000, D^2 = 1M. 100 steps -> 100M ops. Might be tight in Python.
#    
#    Let's check the density.
#    Most points are "conflicting" with many others.
#    For a random point, how many points form an isosceles triangle with it?
#    Likely most of the grid.
#    So the graph is very dense.
#    If the graph is dense, the "lowest degree" vertex will still have high degree.
#    And the independent set will be small.
#    
#    If the graph is dense, maybe we should just pick the vertex with the globally lowest degree first, 
#    and then remove its neighbors.
#    
#    Let's write the code and see.
#    If it's too slow, we can fall back to a simpler heuristic or limit the search.
#    
#    One more thing: The prompt says "Keep the function names and signatures of the seed program."
#    Seed program has `construct()`.
#    
#    Precomputation at module load time.

N = 64
GRID_SIZE = N * N
MAX_DIST_SQ = N*N + N*N

# Precompute distances
# DIST[i][j]
DIST = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)

def precompute_distances():
    xs = np.arange(N)
    ys = np.arange(N)
    # Create grid of points
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    points = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Compute pairwise squared distances
    # diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # DIST = np.sum(diff**2, axis=-1)
    # This creates a 4096x4096x2 array, which is 32MB. OK.
    
    # To save memory/time, we can do it in chunks or just let numpy handle it.
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    DIST[:] = np.sum(diff**2, axis=-1).astype(np.int32)

precompute_distances()

# Precompute conflict graph
# adj[i] is a set of neighbors j such that (i,j) is an edge in conflict graph.
# Using a list of sets for flexibility, or a boolean matrix for speed.
# Boolean matrix: 4096x4096. 16M bytes.
ADJ_MATRIX = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

def build_conflict_graph():
    # Step 1: Apex at u or v
    # For each u, count distances to all v.
    # If count of d(u,v) from u is >= 2, then (u,v) is an edge.
    
    # We can vectorize this.
    # For each u, we have DIST[u]. 
    # We need to know for each v, if DIST[u, v] appears at least twice in DIST[u].
    
    # This is a bit tricky to fully vectorize without loops over u.
    # Loop over u (4096 iterations).
    for u in range(GRID_SIZE):
        d_u = DIST[u]
        # Count occurrences of each distance
        # Max distance is ~8192.
        # Use np.bincount
        counts = np.bincount(d_u, minlength=MAX_DIST_SQ + 1)
        
        # For each v, if counts[d_u[v]] >= 2, then edge (u,v)
        # We want to set ADJ_MATRIX[u, v] = True for all v where counts[d_u[v]] >= 2
        # Note: d_u[u] is 0. counts[0] is 1 (only u itself). So no self-loop from this condition.
        
        # Create a boolean mask for valid v
        # valid_v = counts[d_u] >= 2
        # ADJ_MATRIX[u, valid_v] = True
        
        # Optimization: 
        # distances_present = np.unique(d_u)
        # For each d in distances_present:
        #    if counts[d] >= 2:
        #       mask = (d_u == d)
        #       ADJ_MATRIX[u, mask] = True
        
        # This is faster if number of unique distances is small.
        unique_dists = np.unique(d_u)
        for d in unique_dists:
            if counts[d] >= 2:
                ADJ_MATRIX[u, d_u == d] = True
        
    # Step 2: Apex at w (base is u,v)
    # For each w, group points by distance to w.
    # For each group, all pairs are edges.
    
    for w in range(GRID_SIZE):
        d_w = DIST[w]
        unique_dists = np.unique(d_w)
        for d in unique_dists:
            if d == 0: continue # w itself
            # Get indices of points at distance d from w
            mask = (d_w == d)
            indices = np.where(mask)[0]
            if len(indices) < 2:
                continue
            
            # Add edges between all pairs in indices
            # We can use a submatrix of ADJ_MATRIX
            # ADJ_MATRIX[np.ix_(indices, indices)] = True
            # But we only need to set the upper triangle or just the whole block?
            # Since it's undirected, setting the whole block is fine, but redundant.
            # Setting the whole block is O(k^2).
            
            # Optimization: 
            # We can just set ADJ_MATRIX[indices[:, None], indices[None, :]] = True
            ADJ_MATRIX[indices[:, None], indices[None, :]] = True
            
    # Ensure symmetry (though the operations above should maintain it if done carefully, 
    # Step 1 sets row u. Step 2 sets blocks. 
    # Step 1: For u, sets ADJ[u, v]. Does it set ADJ[v, u]?
    # When we process v in Step 1, we check counts from v. 
    # If (u,v) is an edge due to u being apex, is it necessarily an edge due to v being apex?
    # No. 
    # Example: u is apex. d(u,v) = d(u,w). 
    # Does this imply d(v,u) = d(v,something)? Not necessarily.
    # So the graph might not be symmetric if we only do Step 1 per row.
    # But the conflict relation IS symmetric. 
    # If (u,v) is an edge because u is apex, then (u,v) is an edge. 
    # The edge is undirected. 
    # So we must ensure ADJ_MATRIX is symmetric.
    
    ADJ_MATRIX = ADJ_MATRIX | ADJ_MATRIX.T

build_conflict_graph()

def construct():
    start_time = time.time()
    deadline = start_time + 280.0 # Safety margin
    
    # Greedy MIS
    active = np.ones(GRID_SIZE, dtype=bool)
    # deg[i] is the number of active neighbors of i
    # Initially, deg is the row sum of ADJ_MATRIX
    deg = ADJ_MATRIX.sum(axis=1).astype(np.int32)
    
    selected = []
    
    while time.time() < deadline:
        if not active.any():
            break
        
        # Find index with min degree among active
        # We can use np.where(active) and then min, or just min on a masked array
        # To speed up, we can keep a list of active indices and scan it.
        # But np.argmin on a masked array is fast.
        
        # Create a large value for inactive nodes
        deg_masked = np.where(active, deg, GRID_SIZE + 1)
        u = np.argmin(deg_masked)
        
        if not active[u]:
            break # Should not happen if active.any()
            
        # Select u
        selected.append(u)
        
        # Remove u and its neighbors
        # Neighbors of u are ADJ_MATRIX[u]
        # We need to set active[u] = False and active[neighbors] = False
        
        neighbors = np.where(ADJ_MATRIX[u])[0]
        
        # Update active
        active[u] = False
        active[neighbors] = False
        
        # Update degrees
        # For each node v that is still active:
        # deg[v] should be decreased by the number of neighbors of v that were just removed.
        # The removed nodes are {u} U neighbors.
        # Let R = {u} U neighbors.
        # For each v in active:
        #    deg[v] -= sum(ADJ_MATRIX[v, R])
        
        # This is O(V * |R|) per step.
        # |R| can be large.
        
        # Optimization: 
        # We can compute the contribution of removed nodes to all active nodes at once.
        # removed_mask = np.zeros(GRID_SIZE, dtype=bool)
        # removed_mask[u] = True
        # removed_mask[neighbors] = True
        
        # We want to subtract ADJ_MATRIX[:, removed_mask].sum(axis=1) from deg, 
        # but only for active nodes? 
        # Actually, we can just subtract from ALL deg, and ignore inactive ones later.
        # But we must be careful not to subtract for nodes that are already inactive?
        # It doesn't matter if deg of inactive nodes becomes negative, as long as we don't select them.
        
        # So:
        # removed_count_per_node = ADJ_MATRIX[:, [u] + list(neighbors)].sum(axis=1)
        # deg -= removed_count_per_node
        
        # This is O(V * |R|) memory and compute.
        # V=4096. |R| ~ 1000. 4M ops.
        # 100 steps -> 400M ops. Might be slow in Python loop, but NumPy is fast.
        
        # Let's try this.
        
        # Create a boolean mask for removed nodes
        # Actually, we can just use the indices.
        # removed_indices = np.concatenate(([u], neighbors))
        # deg -= ADJ_MATRIX[:, removed_indices].sum(axis=1)
        
        # But wait, if a neighbor was already inactive, we shouldn't double count?
        # No, ADJ_MATRIX is static. 
        # If a node was removed in a previous step, it's in `removed_indices`? 
        # No, `neighbors` are current neighbors in the ORIGINAL graph.
        # But we are removing them NOW.
        # If they were already inactive, `active[neighbors]` would have been False, 
        # but we are setting them to False again (no-op).
        # However, if they were already inactive, they were already removed.
        # Did we subtract their contribution to deg before?
        # Yes, when they were removed.
        # So if we subtract their contribution AGAIN, we are subtracting too much.
        
        # We only want to subtract the contribution of nodes that are being removed NOW 
        # and were previously active.
        
        # So:
        # newly_removed = neighbors[active[neighbors]] # Neighbors that were active
        # newly_removed = np.concatenate(([u], newly_removed)) # u was active
        
        # deg -= ADJ_MATRIX[:, newly_removed].sum(axis=1)
        
        # Let's refine:
        # active_neighbors = neighbors[active[neighbors]]
        # newly_removed = np.concatenate(([u], active_neighbors))
        
        # deg -= ADJ_MATRIX[:, newly_removed].sum(axis=1)
        
        # This is correct.
        
        # Update deg
        # Note: ADJ_MATRIX[:, newly_removed] creates a view or copy? 
        # Summing it is O(V * len(newly_removed)).
        
        # Let's implement this.
        
        # Check time again
        if time.time() > deadline - 10:
            break
            
        active_neighbors = neighbors[active[neighbors]]
        newly_removed = np.concatenate(([u], active_neighbors))
        
        # Subtract contributions
        # deg is int32. Sum might overflow? Max degree 4096. Sum of 1000 degrees ~ 4M. Fits in int32.
        deg -= ADJ_MATRIX[:, newly_removed].sum(axis=1)
        
        # Clip deg to 0 to avoid negatives? 
        # If we subtract correctly, it shouldn't go negative for active nodes.
        # But due to integer arithmetic or bugs, it might.
        # We only care about min for active nodes.
        
    return [ (i // N, i % N) for i in selected ]

# Test locally? No, just return.
# The function construct is defined.

# One potential issue: 
# The "lowest degree" greedy on a dense graph might pick a vertex that is "central" 
# but has slightly lower degree than others. 
# This is standard.

# Let's double check the "Apex at w" logic.
# If d(u,w) == d(v,w), then (u,v) is an edge.
# My loop:
# for w:
#   for d:
#     indices = points at dist d from w
#     ADJ[indices, indices] = True
# This sets ALL pairs in the group. Correct.

# Let's double check "Apex at u" logic.
# If freq_u(d(u,v)) >= 2, then (u,v) is an edge.
# My loop:
# for u:
#   counts = bincount(DIST[u])
#   for d in unique:
#     if counts[d] >= 2:
#       ADJ[u, DIST[u]==d] = True
# This sets (u,v) for all v at distance d. Correct.

# Symmetry:
# ADJ = ADJ | ADJ.T
# This ensures if (u,v) is set, (v,u) is set. Correct.

# The resulting graph H is the conflict graph.
# Independent set in H is a valid no-isosceles set.

# The greedy algorithm finds a large independent set.

# Final check on performance.
# Precomputation:
# DIST: 4096x4096. ~0.5s.
# Build Graph:
# Loop 4096. Inside, bincount and unique. ~1s.
# Loop 4096. Inside, unique and blocks. ~1s.
# Total precomp ~3s.
# Greedy:
# 100 iterations.
# Each iteration: 
#   argmin: 4096.
#   neighbor update: 
#     active_neighbors: ~1000.
#     deg update: 4096x1000 sum. ~4M ops.
#   Total per iter: ~5M ops.
#   100 iters: 500M ops.
#   In NumPy, 500M simple ops might take 5-10 seconds.
#   So total time ~15s. Well within 300s.

# We can even do multiple restarts if we want, but the greedy is deterministic given the graph.
# Unless we randomize the tie-breaking in argmin.
# Let's add random tie-breaking to allow multiple runs.

def construct():
    start_time = time.time()
    deadline = start_time + 280.0
    
    best_set = []
    best_size = 0
    
    # Run multiple times with random tie-breaking
    # Since the graph is static, we just re-run the greedy.
    
    while time.time() < deadline:
        active = np.ones(GRID_SIZE, dtype=bool)
        deg = ADJ_MATRIX.sum(axis=1).astype(np.int32)
        
        selected = []
        
        while time.time() < deadline - 5:
            if not active.any():
                break
            
            deg_masked = np.where(active, deg, GRID_SIZE + 1)
            
            # Find min degree
            min_deg = np.min(deg_masked)
            
            # Get all indices with min degree
            candidates = np.where(deg_masked == min_deg)[0]
            
            # Pick one randomly
            if len(candidates) > 0:
                u = candidates[np.random.randint(len(candidates))]
            else:
                break
                
            if not active[u]:
                continue
                
            selected.append(u)
            
            neighbors = np.where(ADJ_MATRIX[u])[0]
            
            active[u] = False
            active[neighbors] = False
            
            active_neighbors = neighbors[active[neighbors]]
            newly_removed = np.concatenate(([u], active_neighbors))
            
            deg -= ADJ_MATRIX[:, newly_removed].sum(axis=1)
            
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            
        # If we found a perfect set (unlikely), break
        if best_size == GRID_SIZE:
            break
            
    return [ (i // N, i % N) for i in best_set ]

