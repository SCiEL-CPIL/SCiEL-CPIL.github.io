import time
import numpy as np
import random

N = 64
GRID_SIZE = N * N

# Pre-compute all points
points_arr = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
xs = points_arr[:, 0].astype(np.int64)
ys = points_arr[:, 1].astype(np.int64)

# Pre-compute squared distances
dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
DIST = (dx * dx + dy * dy)  # shape (GRID_SIZE, GRID_SIZE)

# Convert to list of tuples for return
ALL_POINTS = [tuple(p) for p in points_arr]

# Build Conflict Graph as Bitsets
# Two points u, v conflict if there exists a point w such that:
# 1. d(u, w) == d(v, w)  (u and v are equidistant from w)
# 2. d(u, w) == d(v, u)  (w is equidistant from u and v? No, isosceles means two sides equal.
#    Triangle (u, v, w) is isosceles if:
#    d(u, v) == d(u, w) OR d(u, v) == d(v, w) OR d(u, w) == d(v, w)
#
# Let's re-verify the conflict definition.
# A set S is valid if NO three points in S form an isosceles triangle.
# This means for any u, v in S, there is NO w in S such that (u, v, w) is isosceles.
# So, if we add w to S, and u, v are already in S, we must ensure (u, v, w) is NOT isosceles.
# This implies that for a fixed pair (u, v), the set of points w that conflict with this pair is:
# W_conflict(u, v) = { w | d(u, v) == d(u, w) OR d(u, v) == d(v, w) OR d(u, w) == d(v, w) }
#
# However, the "Conflict Graph" approach usually defines an edge between u and v if they CANNOT coexist.
# But u and v CAN coexist if there is no w in the set that completes an isosceles triangle with them.
# The conflict is actually a 3-way constraint.
#
# Wait, the prompt says: "Model the problem as a maximum independent set on a precomputed conflict graph where an edge connects two points if they share a common distance to any third grid point".
# Let's analyze this specific definition: "share a common distance to any third grid point".
# This means: exists w such that d(u, w) == d(v, w).
# If u and v share a distance to w, then (u, v, w) is isosceles with base (u, v) and equal sides (u, w) and (v, w).
# So if u and v are in the set, and w is in the set, it's invalid.
# But if u and v are in the set, and w is NOT in the set, it's fine.
# So a simple graph edge between u and v based on "exists w" is too strong?
# If u and v have an edge, it means they CANNOT be in the set together?
# No, they can be in the set together AS LONG AS no w that completes the triangle is also in the set.
#
# The "Conflict Graph" interpretation in the directive is: "an edge connects two points if they share a common distance to any third grid point".
# If we treat this as a graph G where u-v is an edge if exists w s.t. d(u,w)==d(v,w), then an Independent Set in G is a set of points such that NO two points in the set share a distance to ANY grid point w (not just points in the set).
# This is a MUCH stricter condition than the task requires. The task only requires that no three points IN THE SET form an isosceles triangle.
# If we use this strict graph, the resulting set will be a valid no-isosceles set (since if u,v in S, they don't share distance to ANY w, so certainly not to any w in S).
# So solving MIS on this strict graph yields a valid solution. It might be smaller than the optimal, but it's a valid structural change.
#
# Let's verify: If u, v in S, and edge (u,v) exists in G, then there is w such that d(u,w)=d(v,w).
# If S is an independent set in G, then for any u,v in S, there is NO w (any grid point) such that d(u,w)=d(v,w).
# Therefore, for any w in S, d(u,w) != d(v,w).
# Also, we need to check the other two conditions for isosceles:
# d(u,v) == d(u,w) and d(u,v) == d(v,w).
# Does the graph definition cover these?
# The definition "share a common distance to any third grid point" specifically covers d(u,w) == d(v,w).
# It does NOT cover d(u,v) == d(u,w).
# So an Independent Set in this graph only guarantees that the "apex" is not w (i.e., sides u-w and v-w are not equal).
# It does NOT guarantee that sides u-v and u-w are not equal.
# So this graph is insufficient to guarantee validity on its own unless we define the conflict graph differently.
#
# Let's re-read the directive carefully: "where an edge connects two points if they share a common distance to any third grid point".
# This specific definition might be flawed for full validity if interpreted strictly as a 2-node conflict.
# However, if I implement a graph where u and v are connected if they form an isosceles triangle with ANY third point w (i.e., if there exists w such that (u,v,w) is isosceles), then an independent set in THIS graph would mean that no two points in S form an isosceles triangle with ANY grid point w.
# This implies that for any u, v in S, and any w in S, (u,v,w) is not isosceles.
# This IS a sufficient condition for validity.
#
# So, the correct "Conflict Graph" for this strategy is:
# G = (V, E) where V is the grid points.
# (u, v) in E if there exists ANY w in V such that (u, v, w) forms an isosceles triangle.
#
# Let's compute this graph.
# For a pair (u, v), when is there a w such that (u, v, w) is isosceles?
# Case 1: d(u, w) == d(v, w). (w is on the perpendicular bisector of uv).
# Case 2: d(u, v) == d(u, w). (w is on the circle centered at u with radius d(u, v)).
# Case 3: d(u, v) == d(v, w). (w is on the circle centered at v with radius d(u, v)).
#
# So (u, v) is an edge if:
# 1. There is a grid point w on the perpendicular bisector of uv.
# 2. There is a grid point w on the circle centered at u with radius dist(u, v).
# 3. There is a grid point w on the circle centered at v with radius dist(u, v).
#
# Note: w can be u or v? No, triangle requires distinct points.
# If w=u, d(v, u) == d(u, u)=0? No.
# If w=v, d(u, v) == d(v, v)=0? No.
# So w must be distinct from u and v.
#
# Let's optimize the graph construction.
# Instead of iterating all pairs (u, v) and checking all w, we can invert the logic.
# For each w, and each distance d, find pairs (u, v) such that d(u, w) == d(v, w) == d.
# All such pairs (u, v) become edges.
# For each u, and each distance d, find pairs (v, w) such that d(u, v) == d(u, w) == d.
# All such pairs (v, w) become edges. (Note: this covers Case 2 where u is the apex).
# Similarly for v as apex.
#
# Actually, it's simpler to just iterate pairs (u, v) and check if they are "bad".
# A pair (u, v) is "bad" (i.e., edge exists) if:
# - There exists w != u, v such that d(u, w) == d(v, w).
# - There exists w != u, v such that d(u, v) == d(u, w).
# - There exists w != u, v such that d(u, v) == d(v, w).
#
# Let's precompute for each point u, the counts of points at each distance.
# dist_counts[u] is a dictionary or array mapping distance to count.
#
# Condition 2: d(u, v) == d(u, w).
# This means w is at distance d(u, v) from u.
# So if count_u[d(u, v)] >= 2 (one is v, at least one other w), then there exists such a w.
# Wait, if w=v, that's not a triangle. So we need count_u[d(u, v)] >= 2?
# If count_u[d(u, v)] == 1, then the only point at that distance is v. No other w.
# If count_u[d(u, v)] >= 2, then there is at least one w != v at that distance.
# So Condition 2 holds if dist_counts[u][d(u, v)] >= 2.
#
# Condition 3: d(u, v) == d(v, w).
# Similarly, holds if dist_counts[v][d(u, v)] >= 2.
#
# Condition 1: d(u, w) == d(v, w).
# This means w is equidistant from u and v.
# This is harder to check via counts.
# We need to know if there is a grid point w such that d(u, w) == d(v, w).
# This is equivalent to: the set of distances from u and the set of distances from v intersect?
# No, it's about specific points.
# Let's precompute for each pair of distances? No.
# Let's just iterate w for Condition 1? That's O(N^3) which is 64^3 = 262k. That's fast.
# Or even better, for each w, the points equidistant from u and v...
# Actually, we can precompute a matrix `equidistant[u, v]` which is True if there exists w such that d(u, w) == d(v, w).
# How to compute `equidistant` efficiently?
# For each w, for each distance d, let S_d(w) be the set of points at distance d from w.
# All pairs (u, v) in S_d(w) have d(u, w) == d(v, w) == d.
# So for each w and each d, all pairs in S_d(w) are "equidistant via w".
# We can set a bitset or boolean matrix for these pairs.
#
# Total operations: Sum over w of Sum over d of |S_d(w)|^2.
# This might be large.
# Alternative: Just check the three conditions for each pair (u, v) in O(1) or O(log N) time using precomputed data.
#
# Let's stick to the O(1) checks for Cond 2 and 3.
# For Cond 1, we can precompute a boolean matrix `has_common_distance[u, v]`.
# We can fill this by iterating w.
# For each w:
#   Get list of (dist, points) groups.
#   For each group of points P at distance d from w:
#     For all pairs (u, v) in P:
#       has_common_distance[u, v] = True
#
# This is efficient enough for N=64.
#
# Once we have the graph, we solve MIS.
# The graph is likely very dense. Most pairs will be connected.
# Why?
# Cond 2: d(u, v) == d(u, w). For most pairs, there are many points at distance d(u, v) from u.
# So Cond 2 is true for almost all pairs.
# If Cond 2 is true for almost all pairs, the graph is nearly complete.
# MIS in a nearly complete graph is small.
#
# Wait, if the graph is nearly complete, the MIS size will be very small (like 2 or 3).
# The current score is 9.375.
# If the new strategy yields a set of size 3, the score will be 3.
# This is a significant drop.
#
# Is my interpretation of the directive's graph correct?
# "an edge connects two points if they share a common distance to any third grid point"
# This specifically refers to Cond 1: d(u, w) == d(v, w).
# It does NOT include Cond 2 and 3.
# If we use ONLY this graph, the independent set will satisfy Cond 1 for all pairs.
# But it will NOT satisfy Cond 2 and 3.
# So the resulting set might contain isosceles triangles where the equal sides involve the base (u, v).
# So the solution would be INVALID.
#
# Therefore, I MUST use the full conflict graph (Cond 1 OR Cond 2 OR Cond 3) to ensure validity.
# And if that graph is nearly complete, the MIS will be small.
#
# Let's check the density.
# For a random pair (u, v), what is the probability that Cond 2 holds?
# Cond 2: There exists w != u, v such that d(u, w) == d(u, v).
# This is true if the number of points at distance d(u, v) from u is >= 2.
# In a 64x64 grid, for most distances, the count is small.
# But for small distances (like 1, 2, 5), the count is higher.
# However, for large distances, the count is 1 or 0.
# If d(u, v) is large, count_u[d(u, v)] is likely 1 (just v).
# So Cond 2 might be false for many pairs.
#
# Let's check Cond 1.
# Cond 1: There exists w such that d(u, w) == d(v, w).
# This is true if u and v are in the same "distance shell" from some w.
# For a fixed w, points are grouped by distance.
# If u and v are in the same group, Cond 1 holds.
#
# Let's estimate the edge probability.
# If the graph is sparse enough, MIS can be large.
# If the graph is dense, MIS is small.
#
# Given the risk of a low score, I should implement the MIS solver carefully.
# If the graph is too dense, I can fall back to the previous strategy?
# No, the directive asks for a structural change.
# I will implement the MIS on the full conflict graph.
#
# Solver: Bitset-accelerated greedy coloring heuristic.
# 1. Represent graph adjacency as bitsets (list of Python ints or numpy uint64 arrays).
# 2. Greedy MIS:
#    - Maintain a set of available nodes.
#    - Pick a node v with minimum degree in the induced subgraph on available nodes? Or random?
#    - Add v to MIS.
#    - Remove v and its neighbors from available nodes.
#    - Repeat.
# 3. Random restarts:
#    - Randomize the order of node selection or the graph (by permuting node IDs)?
#    - Or just randomize the tie-breaking in the greedy step.
#
# Let's implement this.

# Precompute distance counts for each point
# dist_counts[i] is a dict: distance -> count
dist_counts = [{} for _ in range(GRID_SIZE)]
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        if i != j:
            d = DIST[i, j]
            dist_counts[i][d] = dist_counts[i].get(d, 0) + 1

# Precompute has_common_distance matrix (Cond 1)
# has_common_distance[i, j] = True if exists w != i, j such that d(i, w) == d(j, w)
has_common_distance = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

# For each w, group points by distance
for w in range(GRID_SIZE):
    # Get distances from w to all other points
    dists = DIST[w, :]
    # Create groups
    groups = {}
    for u in range(GRID_SIZE):
        if u == w:
            continue
        d = dists[u]
        if d not in groups:
            groups[d] = []
        groups[d].append(u)
    
    # For each group, all pairs are equidistant from w
    for d, points_in_group in groups.items():
        if len(points_in_group) < 2:
            continue
        # All pairs in points_in_group have has_common_distance True
        for i in range(len(points_in_group)):
            u = points_in_group[i]
            for j in range(i + 1, len(points_in_group)):
                v = points_in_group[j]
                has_common_distance[u, v] = True
                has_common_distance[v, u] = True

# Now, determine edges for the conflict graph
# Edge (u, v) exists if:
# 1. has_common_distance[u, v] is True
# 2. dist_counts[u].get(DIST[u, v], 0) >= 2  (Cond 2)
# 3. dist_counts[v].get(DIST[u, v], 0) >= 2  (Cond 3)

# Build adjacency bitsets
# Use Python integers for bitsets (arbitrary precision)
adjacency = [0] * GRID_SIZE

for u in range(GRID_SIZE):
    for v in range(u + 1, GRID_SIZE):
        d_uv = DIST[u, v]
        
        cond1 = has_common_distance[u, v]
        cond2 = dist_counts[u].get(d_uv, 0) >= 2
        cond3 = dist_counts[v].get(d_uv, 0) >= 2
        
        if cond1 or cond2 or cond3:
            # Add edge
            adjacency[u] |= (1 << v)
            adjacency[v] |= (1 << u)

# Solver: Greedy MIS with random restarts
def solve_mis_random_restarts(adjacency, num_restarts, time_limit):
    best_set = set()
    best_size = 0
    start_time = time.time()
    
    nodes = list(range(GRID_SIZE))
    
    for _ in range(num_restarts):
        if time.time() > time_limit:
            break
            
        # Randomize node order
        random.shuffle(nodes)
        
        available = set(nodes)
        current_set = set()
        
        # Greedy step
        while available:
            # Pick a node with minimum degree in the induced subgraph
            # To speed up, we can just pick the first one in the shuffled list?
            # Or pick random?
            # Minimum degree heuristic is better for MIS size.
            # But calculating degree for all available nodes is O(N^2) per step.
            # Total O(N^3) per restart. 64^3 = 262k ops. Fast.
            
            min_deg = float('inf')
            best_node = -1
            
            for u in available:
                # Count neighbors in available
                # Bitset trick: (adjacency[u] & available_mask).bit_count()
                # But available is a set. Converting to mask is O(N).
                # Let's just iterate neighbors?
                # Or maintain a mask.
                pass
            
            # Efficient implementation with masks
            # available_mask is an int
            # We need to update it.
            
        # Let's implement a simpler greedy:
        # 1. Start with all nodes available.
        # 2. While available:
        #    a. Pick a node v. (Strategy: min degree, or random)
        #    b. Add v to set.
        #    c. Remove v and its neighbors from available.
        
        # Using Python sets for available is slow for bit ops.
        # Let's use a list of booleans or an integer mask.
        
        # Mask approach:
        mask = (1 << GRID_SIZE) - 1
        current_set = []
        
        while mask:
            if time.time() > time_limit:
                break
            
            # Find node with min degree in induced subgraph
            # Iterate through bits in mask
            # This is O(N) per step, total O(N^2) per restart.
            
            min_deg = GRID_SIZE
            best_node = -1
            
            # Get list of available nodes
            # We can iterate through range and check bit
            for u in range(GRID_SIZE):
                if mask & (1 << u):
                    # Count neighbors in mask
                    # (adjacency[u] & mask).bit_count()
                    deg = (adjacency[u] & mask).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        best_node = u
                        # If deg is 0, we can break early?
                        # No, we want min degree. 0 is min possible.
                        if deg == 0:
                            break
            
            if best_node == -1:
                break
                
            # Add best_node to set
            current_set.append(best_node)
            
            # Remove best_node and its neighbors from mask
            # Neighbors are in adjacency[best_node]
            # We want to keep nodes that are NOT neighbors and NOT best_node
            # remove_mask = (1 << best_node) | adjacency[best_node]
            # mask &= ~remove_mask
            remove_mask = (1 << best_node) | adjacency[best_node]
            mask &= ~remove_mask
            
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Solve MIS
    best_set = solve_mis_random_restarts(adjacency, num_restarts=10000, time_limit=deadline)
    
    # Convert to list of tuples
    result = [ALL_POINTS[i] for i in best_set]
    
    return result

