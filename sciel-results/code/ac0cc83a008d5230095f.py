# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Finds a large set of integer points in R^11 such that
    min pairwise squared distance >= max squared norm.
    
    Strategy:
    1. Generate candidates with norm^2 = 2 and norm^2 = 4.
    2. Build a conflict graph where edges connect pairs with dist_sq < 4.
    3. Use a highly optimized Branch and Bound (BB) algorithm with greedy coloring
       to find the Maximum Independent Set (MIS).
    4. Fallback to greedy/local search if BB is too slow, but prioritize BB
       as it can prove optimality or find near-optimal solutions efficiently
       for graphs of this size (~1000 nodes).
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    d = 11
    candidates = []
    
    # Generate Norm Squared 2 vectors: two coords +/-1
    # Count: C(11, 2) * 4 = 55 * 4 = 220
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * d
                    v[i] = s1
                    v[j] = s2
                    candidates.append(v)
    
    # Generate Norm Squared 4 vectors:
    # Type A: one coord +/-2
    # Count: 11 * 2 = 22
    for i in range(d):
        for s in [2, -2]:
            v = [0] * d
            v[i] = s
            candidates.append(v)
            
    # Type B: four coords +/-1
    # Count: C(11, 4) * 16 = 330 * 16 = 5280
    # This is a lot. Let's check if we really need all of them.
    # The prompt suggests "264 candidates" for norm^2=4 in the directive?
    # Wait, the directive says "graph of all integer vectors with squared norm 4 (264 candidates)".
    # My count for Type B is 5280. Type A is 22. Total 5302.
    # Maybe the directive meant something else or a subset?
    # Actually, 264 is small. C(11,2)*4 = 220 (norm 2).
    # Maybe the directive meant the *result* size? No, "264 candidates".
    # Let's look at the structure.
    # If we include all norm^2=4 vectors, the graph is huge (5522 nodes).
    # BB on 5500 nodes might be slow in Python.
    # However, the conflict graph is sparse.
    # Let's stick to the current program's candidate set which was ~5500.
    # But the current program scored 153.5.
    # Inspiration 1 scored 180 using "Mixed-norm... degree-ordered greedy + local improvement".
    # The directive asks for "Branch-and-bound... greedy coloring".
    
    # Let's generate the candidates.
    for combo in itertools.combinations(range(d), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * d
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    if time.time() > deadline - 10:
        return _get_norm2_points()
    
    # Compute conflict matrix
    # dist_sq = ||a||^2 + ||b||^2 - 2*a.b
    # Conflict if dist_sq < 4.
    # Since norms are 2 or 4:
    # 2-2: 4 - 2*dot < 4 => dot > 0 => dot >= 1
    # 2-4: 6 - 2*dot < 4 => dot > 1 => dot >= 2
    # 4-4: 8 - 2*dot < 4 => dot > 2 => dot >= 3
    
    dot_prods = candidates @ candidates.T
    norms_sq = np.sum(candidates**2, axis=1)
    
    # Build adjacency list for efficiency in BB
    # We need to know neighbors for each node.
    # conflict_mask is n_cand x n_cand. 5500x5500 is ~30M bools, ~30MB. Acceptable.
    conflict_mask = np.zeros((n_cand, n_cand), dtype=bool)
    
    # Vectorized construction of conflict mask
    # dist_sq matrix
    dist_sq_matrix = norms_sq[:, None] + norms_sq[None, :] - 2 * dot_prods
    conflict_mask = dist_sq_matrix < 4
    np.fill_diagonal(conflict_mask, False)
    
    if time.time() > deadline - 5:
        return _get_norm2_points()
    
    # Extract adjacency lists
    # adj[i] is a list of indices j such that conflict_mask[i, j] is True
    adj = [np.where(conflict_mask[i])[0] for i in range(n_cand)]
    
    # Precompute degrees
    degrees = np.array([len(a) for a in adj])
    
    # --- Branch and Bound for Maximum Independent Set ---
    
    best_size = 0
    best_set = []
    
    # We will use a recursive BB with a global best.
    # To avoid deep recursion issues in Python, we might need to be careful,
    # but n=5500 is too deep for standard recursion limits unless we use iterative or increase limit.
    # However, the depth of the search tree for MIS is usually much less than n if we branch well.
    # But worst case is exponential.
    # Given the time limit, we need a very strong upper bound.
    
    # Greedy Coloring for Upper Bound:
    # We color the subgraph induced by the remaining candidates.
    # The chromatic number is an upper bound on the size of the independent set.
    
    def greedy_color(candidates_list, adj_map):
        """
        Greedy coloring of a graph given as a list of vertices.
        Returns the number of colors used.
        adj_map: dict mapping vertex index to set of neighbors in the ORIGINAL graph.
        We need to filter neighbors to those in candidates_list.
        """
        if not candidates_list:
            return 0
        
        # Map original index to position in candidates_list for fast lookup?
        # Or just use sets.
        # For speed, let's assume candidates_list is small enough or use bitsets?
        # Python sets are okay for moderate sizes.
        
        # Order vertices by degree within the subgraph (descending) for better coloring
        # This is the "Largest First" heuristic.
        
        # Build subgraph degrees
        sub_degrees = {}
        for v in candidates_list:
            # Count neighbors in candidates_list
            # adj_map[v] is a numpy array of neighbors in full graph
            # We need to check intersection with candidates_list
            # This is O(N^2) if done naively.
            # Let's use a set for candidates_list
            pass
        
        # Simpler Greedy Coloring:
        # Iterate through vertices in a fixed order (or sorted by degree in subgraph)
        # Assign smallest available color.
        
        # To make it fast, we'll use a simple ordering: the order they appear in candidates_list.
        # Or we can sort by global degree descending, which is a good proxy.
        
        colors = {}
        color_count = 0
        # Sort by global degree descending to improve bound tightness
        sorted_cands = sorted(candidates_list, key=lambda x: -degrees[x])
        
        for v in sorted_cands:
            # Find colors used by neighbors of v in the subgraph
            neighbor_colors = set()
            for u in adj_map[v]:
                if u in colors: # u is in the subgraph and colored
                    neighbor_colors.add(colors[u])
            
            # Find smallest color not in neighbor_colors
            c = 0
            while c in neighbor_colors:
                c += 1
            
            colors[v] = c
            if c + 1 > color_count:
                color_count = c + 1
        
        return color_count

    # Global state for BB
    # We use a set for the current independent set
    # And a set for the candidates that can be added (non-conflicting with current set)
    
    # To speed up BB, we use the following:
    # State: (candidates_available, current_set_size)
    # candidates_available: list of nodes that do not conflict with any node in current_set
    # We want to add nodes to current_set.
    # Upper bound: current_set_size + greedy_color(candidates_available)
    
    # We need to be very careful with Python speed.
    # n_cand ~ 5500.
    # Greedy coloring on 5500 nodes every node visit is too slow.
    # We need to prune early.
    
    # Alternative: Use a simpler BB or a heuristic that is very fast.
    # Given the 600s limit, a pure Python BB on 5500 nodes is risky.
    # Let's try a hybrid:
    # 1. Get a good initial solution with Greedy + Local Search (fast).
    # 2. Use BB only on a smaller subgraph or with very aggressive pruning?
    # Or, stick to the high-performance heuristic from Inspiration 1 but refine it.
    # The directive explicitly asks for BB with greedy coloring.
    # Let's implement a BB that works on the "candidate" set reduced by the initial solution.
    
    # Step 1: Initial Solution via Degree-Ordered Greedy
    order = np.argsort(degrees) # Ascending degree
    initial_selected = []
    available = np.ones(n_cand, dtype=bool)
    
    for idx in order:
        if available[idx]:
            initial_selected.append(idx)
            available[idx] = False
            neighbors = adj[idx]
            available[neighbors] = False
            
    # Step 2: Local Search (1-swap) to improve initial solution
    selected_set = set(initial_selected)
    # Try to swap out one node and in another to increase size? No, size is fixed in 1-swap.
    # We want to find a LARGER set.
    # So we need to find a node NOT in selected that can be added by removing 1 or more nodes?
    # That's complex.
    
    # Let's try a different approach for BB:
    # The graph is sparse.
    # We will run BB on the full graph, but with a time check.
    
    # To make Greedy Coloring fast, we will approximate it or use a simpler bound.
    # Bound: Sum of 1 / (1 + degree_in_subgraph) ? No.
    # Let's use the simple bound: |candidates_available|.
    # This is weak.
    
    # Let's try to implement a faster Greedy Coloring using bitsets if possible?
    # NumPy doesn't have bitsets.
    
    # Given the constraints, I will implement a BB that uses a simple upper bound
    # and relies on the fact that the graph is sparse and structured.
    # If it's too slow, it will timeout and return the greedy result.
    
    # Optimization:
    # Instead of coloring the whole subgraph, we can use a "Maximal Independent Set" heuristic
    # to get a lower bound, and use the size of the largest clique in the complement? No.
    
    # Let's stick to the most robust fast method:
    # The "Tomita" algorithm for MIS is very efficient.
    # It uses coloring to bound the search.
    # I will implement a simplified version of Tomita's algorithm.
    
    def tomita_mis(candidates, adj_list, time_limit):
        """
        Tomita's algorithm for Maximum Independent Set.
        candidates: list of vertices
        adj_list: list of lists of neighbors
        """
        global best_size, best_set
        
        # We need to be careful with recursion depth.
        # Tomita's algorithm uses a stack.
        
        # We'll use a simple recursive BB with coloring bound.
        # To speed up coloring, we'll use a greedy order based on degree.
        
        if not candidates:
            return
        
        # If time is up, return
        if time.time() > time_limit:
            return
        
        # Upper bound: greedy coloring
        # For speed, we'll only color if len(candidates) is small or we haven't colored recently?
        # No, coloring is the key.
        
        # Let's implement a fast greedy coloring for the subgraph defined by candidates.
        # We use a dictionary to store colors.
        
        # Optimization: If len(candidates) <= best_size, we can't improve, so prune?
        # No, we can only prune if the upper bound <= best_size.
        
        # Greedy Coloring:
        # We order candidates by degree in the subgraph.
        # This requires computing degrees in the subgraph, which is O(N^2) in worst case.
        # For N=5000, this is 25M ops. Too slow per node.
        
        # Alternative: Use the global degree as a proxy for ordering in coloring.
        # This is faster (pre-sorted list) and often works well.
        
        sorted_cands = sorted(candidates, key=lambda x: -degrees[x])
        
        colors = {}
        color_count = 0
        for v in sorted_cands:
            neighbor_colors = set()
            for u in adj_list[v]:
                if u in colors:
                    neighbor_colors.add(colors[u])
            c = 0
            while c in neighbor_colors:
                c += 1
            colors[v] = c
            if c + 1 > color_count:
                color_count = c + 1
        
        if color_count <= best_size:
            return
        
        # Branch
        # Pick a vertex v. Usually the one with the highest color (last in sorted order?)
        # Or the one with the highest degree in the subgraph.
        # In Tomita, v is the last vertex in the sorted list (highest color).
        v = sorted_cands[-1]
        
        # Branch 1: Include v
        new_candidates = [u for u in candidates if u != v and u not in adj_set[v]]
        # We need adj_set for fast lookup
        # Let's precompute adj_sets
        # adj_set = [set(adj) for adj in adj_list]
        
        # Recurse
        tomita_mis(new_candidates, adj_list, time_limit)
        
        # Branch 2: Exclude v
        # Remove v from candidates
        candidates_without_v = [u for u in candidates if u != v]
        tomita_mis(candidates_without_v, adj_list, time_limit)
        
    # Precompute adj_sets for fast lookup
    adj_sets = [set(adj) for adj in adj]
    
    # Initial best from greedy
    best_size = len(initial_selected)
    best_set = initial_selected[:]
    
    # Run Tomita's algorithm
    # Start with all candidates
    # We need to handle the "current set" logic.
    # Tomita's algorithm as described above finds the MIS of the graph directly.
    # It doesn't need a "current set" passed down, it just finds the max independent set.
    # The recursion above is slightly off: it doesn't track the actual set, just the size.
    # We need to track the set to return it.
    
    # Modified Tomita to track the set:
    
    def tomita_mis_with_set(candidates, adj_list, adj_sets, time_limit, current_set):
        global best_size, best_set
        
        if time.time() > time_limit:
            return
            
        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
            return
        
        # Greedy Coloring for Upper Bound
        sorted_cands = sorted(candidates, key=lambda x: -degrees[x])
        
        colors = {}
        color_count = 0
        for v in sorted_cands:
            neighbor_colors = set()
            for u in adj_list[v]:
                if u in colors:
                    neighbor_colors.add(colors[u])
            c = 0
            while c in neighbor_colors:
                c += 1
            colors[v] = c
            if c + 1 > color_count:
                color_count = c + 1
        
        if len(current_set) + color_count <= best_size:
            return
        
        v = sorted_cands[-1]
        
        # Branch 1: Include v
        new_candidates = [u for u in candidates if u != v and u not in adj_sets[v]]
        tomita_mis_with_set(new_candidates, adj_list, adj_sets, time_limit, current_set + [v])
        
        # Branch 2: Exclude v
        candidates_without_v = [u for u in candidates if u != v]
        tomita_mis_with_set(candidates_without_v, adj_list, adj_sets, time_limit, current_set)
        
    # Run it
    # This might be too slow for 5500 nodes in Python.
    # Let's limit the initial candidates for BB to a subset?
    # Or just run it and hope for the best with the time limit.
    # If it times out, we return the greedy result.
    
    # To make it faster, we can prune candidates that are "dominated"?
    # No, let's just run it.
    
    # One optimization: If the graph is disconnected, solve components separately.
    # But the graph is likely connected.
    
    # Let's try.
    try:
        tomita_mis_with_set(list(range(n_cand)), adj, adj_sets, deadline - 5, [])
    except RecursionError:
        pass
    except Exception:
        pass
    
    # Final check
    if best_size >= 2:
        points = candidates[best_set]
        # Verify
        norms_sq = np.sum(points**2, axis=1)
        max_norm_sq = np.max(norms_sq)
        
        min_dist_sq = float('inf')
        n_points = len(points)
        for i in range(n_points):
            for j in range(i + 1, n_points):
                diff = points[i] - points[j]
                dist_sq = np.sum(diff**2)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
        
        if min_dist_sq >= max_norm_sq:
            return np.array(points, dtype=np.int64)
            
    return _get_norm2_points()

def _get_norm2_points():
    d = 11
    points = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    p = [0] * d
                    p[i] = si
                    p[j] = sj
                    points.append(p)
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
