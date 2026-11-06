import random
import math
import time

N = 64
TOTAL = N * N
all_points = [(x, y) for x in range(N) for y in range(N)]
point_to_idx = {p: i for i, p in enumerate(all_points)}

# Precompute conflict graph: adj[i] is a bitset (int) of points that conflict with point i
# Two points i, j conflict if there exists a third point k such that {i, j, k} forms an isosceles triangle.
# This is equivalent to: i and j are in the "conflict set" of some pair (i, k) or (j, k) or (k, l) where k,l in {i,j}?
# Actually, the definition from the inspiration is:
# "edges connect pairs that share a common 'isosceles partner' (i.e., two points are adjacent if there exists a third point forming an isosceles triangle with them)"
# So, i and j are adjacent if there exists k such that {i, j, k} is isosceles.
# This means:
# 1. dist(i,k) == dist(j,k) (k is apex)
# 2. dist(i,k) == dist(i,j) (i is apex)
# 3. dist(j,k) == dist(i,j) (j is apex)

# We can build the adjacency list by iterating over all pairs (i, j) and checking if they form an isosceles triangle with any k.
# But that's O(N^6) which is too slow (4096^3 ~ 68 billion).
# 
# Better approach: For each pair (i, j), determine the set of points k that form an isosceles triangle with i and j.
# If this set is non-empty, then i and j are adjacent in the conflict graph.
# The set of such k is the union of:
# - Perpendicular bisector of ij intersected with grid (for apex k)
# - Circle centered at i with radius dist(i,j) intersected with grid (for apex i)
# - Circle centered at j with radius dist(i,j) intersected with grid (for apex j)
#
# We can precompute this for all pairs (i, j) where i < j.
# There are ~8 million pairs. For each pair, we do O(N) work to check the bisector and circles.
# Total work: 8M * 64 = 512M operations. This might be too slow in Python (5-10 minutes).
#
# Alternative: Build the conflict graph incrementally during the search? No, we need it for branch and bound.
#
# Let's try to optimize the precomputation.
# We can represent the grid as a 2D array of indices.
# For the perpendicular bisector:
#   Equation: 2(dx)x + 2(dy)y = C
#   We can iterate over x and check if y is integer and in bounds. O(N).
# For the circles:
#   Iterate over x, check if rem is a perfect square. O(N).
#
# Total 3 * N operations per pair. 8M * 192 = 1.5B operations. Still likely too slow in pure Python.
#
# Let's use a different strategy for the graph construction.
# Instead of building the full graph, we can use the "forbidden points" idea from inspiration 1, but apply it in a branch-and-bound framework.
#
# Actually, the directive says: "Model the problem as a maximum independent set search on a conflict graph... then solve this using a heuristic branch-and-bound algorithm with bitset-accelerated neighborhood pruning".
#
# If we cannot precompute the graph fast enough, we might need to compute conflicts on the fly or use a different graph representation.
# However, for N=64, the graph is fixed.
# Let's try to speed up the precomputation using NumPy or just accept a longer startup time if it fits in the budget.
# 300s budget. If precomputation takes 60s, we have 240s for search.
#
# Let's try to optimize the conflict check.
# We can precompute for each point i, the set of points j that are at a specific squared distance from i.
# dist_sq[i][d] = set of points j such that dist(i,j)^2 == d.
#
# Then, for a pair (i, j):
# d_ij = dist(i,j)^2
#
# 1. Apex k: dist(k,i) == dist(k,j).
#    This means k is in the intersection of dist_sq[i][d] and dist_sq[j][d] for some d? No.
#    It means dist(k,i) == dist(k,j).
#    We can check this by iterating k? No, that's O(N^2) per pair.
#    
#    Let's go back to the geometric check. It's O(N) per pair.
#    Maybe we can vectorize the bisector check?
#    
#    Let's try a hybrid:
#    1. Precompute all squared distances between all pairs? 4096x4096 matrix = 16M entries. Fits in memory.
#    2. Use the matrix to check conflicts?
#       For pair (i, j), we need to know if there exists k such that:
#       - dist[i][k] == dist[j][k]  (Apex k)
#       - dist[i][k] == dist[i][j]  (Apex i)
#       - dist[j][k] == dist[i][j]  (Apex j)
#       
#       If we have the distance matrix D (NxN), where D[i][j] is squared distance:
#       - Apex i: Is there k != i, j such that D[i][k] == D[i][j]?
#         This is equivalent to: count of k with D[i][k] == D[i][j] is > 1 (excluding j itself? No, k can be any point).
#         Actually, if D[i][k] == D[i][j], then triangle (i, j, k) is isosceles with apex i.
#         So if there exists ANY k (k != i, k != j) such that D[i][k] == D[i][j], then (i, j) is a conflict pair.
#         Note: k could be j? If k=j, D[i][j] == D[i][j] is true, but that's degenerate. We need a distinct third point.
#         So we need to check if the set of points at distance D[i][j] from i has size > 1.
#         Let's precompute for each i and each distance d, the count of points at that distance.
#         If count[i][d] > 1, then any pair (i, j) where D[i][j] == d is a conflict pair (due to apex i).
#         Wait, if count[i][d] > 1, it means there are at least 2 points j, k at distance d from i.
#         Then for pair (i, j), k is the third point. So yes, (i, j) is a conflict.
#         
#       - Apex j: Similarly, if count[j][d] > 1 where d = D[i][j], then (i, j) is a conflict.
#       
#       - Apex k: D[i][k] == D[j][k].
#         This is harder. It means k is on the perpendicular bisector.
#         We can check this by iterating k for each pair (i, j)? That's O(N^3) total. 68 billion. Too slow.
#         
#         Can we optimize Apex k check?
#         For a fixed pair (i, j), we need to know if there is a k such that D[i][k] == D[j][k].
#         This is equivalent to: The intersection of the sets of distances from i and j is non-empty? No.
#         It means there exists a point k where the distance to i equals the distance to j.
#         
#         Let's precompute for each pair (i, j) if there is such a k.
#         We can do this by iterating k and updating a structure?
#         For each k, it is on the perpendicular bisector of (i, j) if D[i][k] == D[j][k].
#         So for each k, we can find all pairs (i, j) such that D[i][k] == D[j][k].
#         This is like grouping points by their distance to k.
#         For each k, let groups be based on D[i][k].
#         Any two points i, j in the same group (same distance to k) form a pair (i, j) that has k as an apex.
#         So (i, j) is a conflict pair.
#         
#         Algorithm:
#         1. Initialize adjacency bitsets adj[i] = 0 for all i.
#         2. For each k in 0..N^2-1:
#            - Compute distances from k to all other points i.
#            - Group indices i by distance d.
#            - For each group with size > 1, for each pair (i, j) in the group, set adj[i] |= (1 << j) and adj[j] |= (1 << i).
#            
#         Complexity:
#         For each k, we have N^2 points. Grouping takes O(N^2).
#         For each group of size m, we add m^2/2 edges.
#         Sum of m^2 over groups is at most (N^2)^2 if all in one group, but usually smaller.
#         Worst case: all points at same distance from k (impossible for grid, but let's say large groups).
#         Actually, for a fixed k, the number of points at a specific distance is small (O(sqrt(d))).
#         So the total work per k is O(N^2 + sum(m_g^2)).
#         Sum over all k: N^2 * (N^2 + ...).
#         This might be feasible if implemented efficiently.
#         
#         Let's estimate:
#         N^2 = 4096.
#         For each k, we iterate 4096 points to compute distances and group.
#         Then we process groups.
#         Total iterations for k: 4096 * 4096 = 16M.
#         Inside the inner loop, we update bitsets.
#         If we use Python integers as bitsets, ORing is fast.
#         But finding pairs in a group of size m takes O(m^2).
#         Average m is small.
#         Let's try this.

import numpy as np

def build_conflict_graph():
    """
    Builds the conflict graph adjacency list as a list of integers (bitsets).
    adj[i] is a bitset of points j such that i and j form an isosceles triangle with some k.
    """
    adj = [0] * TOTAL
    
    # Precompute coordinates as numpy arrays for faster distance calculation?
    # Or just use list of tuples.
    # Let's use numpy for distance calculation to speed up the "for each k" loop.
    pts = np.array(all_points, dtype=np.int32)
    
    for k_idx in range(TOTAL):
        # Compute squared distances from k to all points
        diff = pts - pts[k_idx]
        dists_sq = np.sum(diff * diff, axis=1)
        
        # Group points by distance
        # We need to find all pairs (i, j) such that dists_sq[i] == dists_sq[j]
        # And i != j, i != k, j != k.
        
        # Use a dictionary to group indices by distance
        groups = {}
        for i in range(TOTAL):
            if i == k_idx:
                continue
            d = dists_sq[i]
            if d in groups:
                groups[d].append(i)
            else:
                groups[d] = [i]
        
        # For each group with size > 1, add edges between all pairs in the group
        for d, group in groups.items():
            if len(group) < 2:
                continue
            # For each i in group, add all other j in group to adj[i]
            # We can do this by ORing the bitsets of the other members.
            # But constructing the bitset for the group and then subtracting self?
            # Let's just iterate.
            # Optimization: If the group is large, this is O(m^2).
            # But m is usually small.
            
            # Create a bitset for the group
            group_mask = 0
            for idx in group:
                group_mask |= (1 << idx)
            
            for i in group:
                # Add all j in group to adj[i], except i itself
                # adj[i] |= (group_mask ^ (1 << i))
                # But we only want to add edges for isosceles with apex k.
                # Yes, this is correct.
                adj[i] |= (group_mask ^ (1 << i))
    
    # Now we have handled the "Apex k" case (k is the third point, i and j are base).
    # Wait, the condition for (i, j) to be a conflict is that there EXISTS a k.
    # The loop above finds all (i, j) such that there is a k with dist(i,k)==dist(j,k).
    # This covers the case where k is the apex.
    
    # Now we need to handle the cases where i is the apex or j is the apex.
    # Case: i is apex. dist(i, j) == dist(i, k).
    # This means for a fixed i, if there are two points j, k at the same distance from i, then (i, j) and (i, k) are conflict pairs?
    # No, the pair is (j, k)? No.
    # The pair of points in the graph are the vertices of the independent set.
    # If i is apex, the isosceles triangle is (i, j, k) with dist(i,j)==dist(i,k).
    # The pairs involved are (i, j), (i, k), and (j, k).
    # In the independent set context, we cannot have both i and j in the set if there is a k such that...
    # Wait, the conflict graph definition: "two points are adjacent if there exists a third point forming an isosceles triangle with them".
    # So if (i, j, k) is isosceles, then:
    # - i and j are adjacent (because k is the third point)
    # - i and k are adjacent (because j is the third point)
    # - j and k are adjacent (because i is the third point)
    # So ALL three pairs are edges in the conflict graph.
    
    # My previous loop handled:
    # For each k, find pairs (i, j) such that dist(i,k)==dist(j,k).
    # This adds edges (i, j).
    # This covers the case where k is the apex.
    # But it does NOT add edges (i, k) or (j, k).
    # For (i, k) to be an edge, we need a third point j such that (i, k, j) is isosceles.
    # If (i, j, k) is isosceles with apex i (dist(i,j)==dist(i,k)), then:
    # - Pair (j, k) has apex i. This was handled when we processed apex i?
    #   No, my loop processed "for each k, find i,j with dist(i,k)==dist(j,k)".
    #   This finds edges (i, j) where k is the apex.
    #   It does NOT find edges (i, k) where j is the apex.
    
    # So I need to also handle:
    # For each apex a, find pairs (u, v) such that dist(a, u) == dist(a, v).
    # Then (u, v) is an edge? No.
    # If dist(a, u) == dist(a, v), then triangle (a, u, v) is isosceles with apex a.
    # The pairs are (a, u), (a, v), (u, v).
    # So (u, v) is an edge (apex a).
    # (a, u) is an edge (apex v? No, apex is a. The third point is v? No, the third point for pair (a, u) is v).
    # Yes, for pair (a, u), the third point is v. Since (a, u, v) is isosceles, (a, u) is a conflict pair.
    # Similarly (a, v) is a conflict pair.
    
    # So, for each apex a:
    # 1. Group points by distance to a.
    # 2. For each group {u, v, ...}, add edges (u, v), (u, w), ... (all pairs in group).
    # 3. Also add edges (a, u), (a, v), ... for all u in the group?
    #    Wait. If u and v are in the same group (dist(a,u)==dist(a,v)), then (u, v) is an edge.
    #    Is (a, u) an edge? Yes, because v is the third point.
    #    So a is connected to ALL points in the group.
    #    And all points in the group are connected to each other.
    
    # Let's re-evaluate the loop.
    # My loop: "for each k, group i by dist(k, i). Add edges between all pairs in group."
    # This adds edges (i, j) for i, j in group.
    # This covers the "base" of the isosceles triangle where k is the apex.
    # It does NOT add edges (k, i).
    
    # So I need a second pass or modify the first.
    # For each apex k:
    #   For each group of points at distance d from k:
    #     Let G be the group.
    #     Add edges between all pairs in G. (Done in current loop)
    #     Add edges between k and all points in G. (Missing)
    
    # Let's add the missing edges.
    # We can do this in the same loop.
    
    # Let's restart the build function with this correction.
    
    adj = [0] * TOTAL
    pts = np.array(all_points, dtype=np.int32)
    
    for k_idx in range(TOTAL):
        diff = pts - pts[k_idx]
        dists_sq = np.sum(diff * diff, axis=1)
        
        groups = {}
        for i in range(TOTAL):
            if i == k_idx:
                continue
            d = dists_sq[i]
            if d in groups:
                groups[d].append(i)
            else:
                groups[d] = [i]
        
        for d, group in groups.items():
            if len(group) < 1:
                continue
            
            # 1. Add edges between all pairs in group (k is apex)
            if len(group) >= 2:
                group_mask = 0
                for idx in group:
                    group_mask |= (1 << idx)
                for i in group:
                    adj[i] |= (group_mask ^ (1 << i))
            
            # 2. Add edges between k and all points in group (group members are apex? No, k is connected to them because they form isosceles with each other? No.)
            # If i, j in group, then (k, i, j) is isosceles with apex k.
            # So (k, i) is a conflict pair (third point j).
            # So k should be connected to i.
            # Yes.
            for i in group:
                adj[k_idx] |= (1 << i)
                adj[i] |= (1 << k_idx)
    
    return adj

# Precompute the conflict graph
# This might take a while. Let's measure it.
print("Building conflict graph...")
start_time = time.time()
adj = build_conflict_graph()
print(f"Graph built in {time.time() - start_time:.2f}s")

def construct():
    """Construct a set of grid points in [0, N)^2 with no three forming an isosceles triangle."""
    start_time = time.time()
    deadline = start_time + 320  # Safety margin
    
    best_set = []
    best_size = 0
    
    # Branch and Bound for Maximum Independent Set
    # Vertices: 0..TOTAL-1
    # adj[i] is bitset of neighbors.
    
    # Heuristic: Branch on vertices with highest degree? Or lowest?
    # Usually, branching on high degree vertices prunes more.
    
    # We will use a recursive DFS with bitsets.
    # State:
    # - candidates: bitset of vertices that can still be added.
    # - current_set: list of indices in the current independent set.
    # - bound: upper bound on the size of the independent set we can form from candidates.
    #   A simple bound is the number of bits in candidates.
    #   A better bound: Greedy coloring bound or just |candidates|.
    
    # To speed up, we can use a greedy heuristic to get a good initial solution.
    
    def greedy_iset(candidates):
        """Greedy independent set on bitset candidates."""
        S = []
        c = candidates
        while c:
            # Pick vertex with highest degree in the remaining subgraph?
            # Or just the first one?
            # Let's pick the one with the lowest index for determinism, or highest degree for better pruning.
            # Finding highest degree in bitset is expensive.
            # Let's just pick the lowest set bit.
            i = (c & -c).bit_length() - 1
            S.append(i)
            # Remove i and its neighbors from c
            c &= ~(adj[i] | (1 << i))
        return S
    
    # Get an initial solution
    initial_candidates = (1 << TOTAL) - 1
    init_set = greedy_iset(initial_candidates)
    best_set = init_set[:]
    best_size = len(best_set)
    
    # Branch and Bound
    # We need to be careful with recursion depth and Python overhead.
    # Iterative DFS might be better, but recursive is easier to write.
    # Max depth is TOTAL (4096). Python recursion limit is 1000.
    # We need to increase recursion limit or use iterative.
    import sys
    sys.setrecursionlimit(10000)
    
    def dfs(candidates, current_size, current_set):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        # Upper bound: current_size + popcount(candidates)
        # If this is <= best_size, prune.
        if current_size + candidates.bit_count() <= best_size:
            return
        
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return
        
        # Heuristic: Pick a vertex to branch on.
        # Choose vertex with highest degree in the candidate subgraph?
        # This is expensive to compute.
        # Let's pick the vertex with the lowest index in candidates.
        v = (candidates & -candidates).bit_length() - 1
        
        # Branch 1: Include v
        # New candidates: candidates & ~adj[v] & ~(1 << v)
        new_candidates = candidates & ~(adj[v] | (1 << v))
        current_set.append(v)
        dfs(new_candidates, current_size + 1, current_set)
        current_set.pop()
        
        # Branch 2: Exclude v
        # New candidates: candidates & ~(1 << v)
        new_candidates = candidates & ~(1 << v)
        dfs(new_candidates, current_size, current_set)
    
    # Run B&B
    # We might need to run multiple times with different heuristics or restarts if the first one is slow.
    # But B&B with bitsets should be fast if the bound is tight.
    
    # Let's add a time check in the main loop as well, just in case.
    # The dfs function already checks time.
    
    dfs((1 << TOTAL) - 1, 0, [])
    
    return [all_points[i] for i in best_set]

# Test
if __name__ == "__main__":
    res = construct()
    print(len(res))
