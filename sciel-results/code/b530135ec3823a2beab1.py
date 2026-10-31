import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # 1. Generate candidates
    # Type 1: Weight 4, entries +/-1
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Type 2: Weight 1, entries +/-2
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    all_vectors = np.array(v4_candidates + v1_candidates, dtype=np.int64)
    N = len(all_vectors)
    
    # 2. Compute Conflict Matrix
    # Dot products
    dots = all_vectors @ all_vectors.T
    # Conflict if dot > 2
    # Note: Diagonal is 4, so no self-conflict in logic, but we mask it out
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to adjacency lists or bitsets?
    # For Tomita, we need to efficiently find neighbors.
    # Let's use a list of sets or sorted lists of neighbors for each vertex.
    # Or, better for B&B with bitsets:
    # Since N is ~5300, we can use Python integers as bitsets.
    # adj[i] is a bitmask of neighbors of i.
    
    adj = [0] * N
    for i in range(N):
        # Get indices where conflicts[i] is True
        # This is O(N^2) total, which is ~28M ops. Might be slow in pure Python loop.
        # Let's vectorize the creation of bitmasks or just use numpy indices.
        pass

    # Faster way to build adjacency for B&B:
    # We will maintain a set of available vertices P as a numpy boolean array or list of indices.
    # And use a greedy coloring heuristic.
    
    # Let's try a standard recursive B&B with numpy.
    
    # Pre-compute degrees for initial ordering
    degrees = conflicts.sum(axis=1)
    
    # Initial ordering: sort by degree descending? Or use a greedy initial solution.
    
    # Greedy Initial Solution
    def greedy_is(conflicts, order):
        selected = []
        available = set(order)
        for v in order:
            if v in available:
                selected.append(v)
                # Remove v and its neighbors
                to_remove = set(np.where(conflicts[v])[0])
                to_remove.add(v)
                available -= to_remove
        return selected

    # Try a few greedy orders to get a good lower bound
    best_is = []
    # Order 1: Random
    rng = np.random.default_rng(42)
    for _ in range(10):
        order = rng.permutation(N)
        is_found = greedy_is(conflicts, order)
        if len(is_found) > len(best_is):
            best_is = is_found
            
    # Order 2: Degree ascending (Min-degree heuristic)
    order_deg = np.argsort(degrees)
    is_found = greedy_is(conflicts, order_deg)
    if len(is_found) > len(best_is):
        best_is = is_found

    current_best_size = len(best_is)
    
    # Branch and Bound
    # We will use a recursive approach.
    # State: P (list of available vertices), S (current selected set, size tracked)
    # To speed up, we use a greedy coloring bound on P.
    
    def greedy_coloring(P, conflicts):
        # P is a list of vertices
        # Returns list of colors for each vertex in P
        # and the number of colors used
        if not P:
            return [], 0
        
        colors = {}
        color_class = 0
        uncolored = set(P)
        
        while uncolored:
            color_class += 1
            # Find a maximal independent set in the induced subgraph of uncolored
            # Greedy: pick vertex with min degree in uncolored?
            # For simplicity, just iterate and pick if no neighbor in current color class
            current_color_set = []
            candidates = list(uncolored)
            # Sort candidates by degree in uncolored?
            # To keep it fast, just iterate
            for v in candidates:
                if v not in uncolored: continue
                # Check if v conflicts with any in current_color_set
                # This is slow. 
                # Better: maintain a set of "blocked" vertices.
                pass
        
        # This naive coloring is too slow.
        # Let's use a simpler bound: |P| is a trivial bound.
        # Or, use a pre-computed greedy coloring on the *whole* graph? No, it changes.
        
        # Alternative: Use the "Tomita" coloring which is efficient.
        # But implementing it correctly in Python within time limits is tricky.
        
        # Let's try a different strategy: Local Search (Simulated Annealing or Hill Climbing) on the MIS.
        # Start from the best greedy solution.
        # Try to swap one vertex out for two in, or one in for one out (if it improves density? No, we want size).
        # Actually, the seed already did some local search.
        
        # Let's try a more aggressive B&B with a simple bound:
        # Bound = number of vertices in P.
        # This is weak.
        
        # Let's use the "Maximal Independent Set" property.
        # If we select a vertex, we remove its neighbors.
        
        # Given the time constraint, let's implement a specific B&B variant:
        # 1. Order vertices by degree.
        # 2. Recurse.
        # 3. Bound: We can use a simple greedy coloring of P to get a better bound.
        
        def color_bound(P):
            # P is a numpy array of indices
            if len(P) == 0: return 0
            # Greedy coloring
            # We want to assign colors such that adjacent nodes have different colors.
            # The max color index + 1 is the upper bound.
            # We process nodes in P.
            # To be fast, we don't need optimal coloring, just a valid one.
            # Simple greedy: 
            # 1. Create a list of uncolored nodes.
            # 2. While uncolored:
            #    a. Pick a node (e.g., first one).
            #    b. Assign it the smallest available color.
            #    c. Remove it and its neighbors from "available for this color"? No.
            #    Standard greedy coloring:
            #    Iterate through nodes. For each node, check neighbors already colored.
            #    Assign smallest color not used by neighbors.
            
            P_list = P.tolist()
            # Order P_list to help coloring? e.g., high degree first?
            # Let's just use the order they are in.
            
            node_color = {}
            for v in P_list:
                # Find colors used by neighbors of v in P
                # neighbors in P: np.intersect1d(conflicts[v], P)
                # This is slow.
                
                # Optimization: Pre-compute neighbor lists as sets?
                # N=5300, edges ~ ?
                
                pass
            
            # This is getting complicated. Let's stick to a robust Greedy + Local Search which is easier to optimize in Python.
            
            return len(P) # Trivial bound

    # Let's refine the Local Search approach from the seed, but make it more powerful.
    # The seed did "swap 1 for 2".
    # We can do "swap 1 for k" where k is the max independent set in the available neighbors of the removed vertex.
    
    selected = np.array(best_is)
    selected_set = set(selected)
    
    # Precompute conflict with selected
    # conflict_with_selected[v] = True if v conflicts with any in selected
    
    def get_conflict_mask(vertices):
        mask = np.zeros(N, dtype=bool)
        for v in vertices:
            mask |= conflicts[v]
        return mask

    # Initial mask
    blocked = get_conflict_mask(selected)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try to remove one vertex from selected
        for i in range(len(selected)):
            if time.time() > deadline - 5:
                break
            v_remove = selected[i]
            
            # Temporarily remove v_remove
            # New blocked mask: blocked without v_remove's contribution?
            # It's hard to subtract.
            # Recalculate blocked for remaining selected?
            # Remaining = selected without i
            rem_selected = np.delete(selected, i)
            new_blocked = get_conflict_mask(rem_selected)
            
            # Candidates: vertices not in rem_selected and not in new_blocked
            # Candidates can be added if they don't conflict with rem_selected
            # And they don't conflict with each other.
            
            # Available candidates:
            # ~new_blocked & ~isin(arange(N), rem_selected)
            # Note: v_remove is now available if it doesn't conflict with rem_selected (it doesn't, it was in selected)
            # But we want to find a set of candidates that can replace v_remove.
            # Ideally, we find an Independent Set in the subgraph of candidates that is larger than 1.
            
            # Let's find the best independent set in the "Available" set using a greedy method.
            # Available = vertices that do not conflict with rem_selected.
            avail_mask = ~new_blocked & ~np.isin(np.arange(N), rem_selected)
            avail_indices = np.where(avail_mask)[0]
            
            if len(avail_indices) == 0:
                continue
                
            # Run a greedy MIS on avail_indices
            # We want to maximize the size.
            # Greedy:
            temp_selected = []
            temp_avail = set(avail_indices)
            
            # Order by degree in the induced subgraph?
            # For speed, just random or original order.
            # Let's try multiple random orders for the greedy step.
            best_temp = []
            for _ in range(5):
                order = rng.permutation(avail_indices)
                curr_temp = []
                curr_avail = set(avail_indices)
                for v in order:
                    if v in curr_avail:
                        curr_temp.append(v)
                        to_remove = set(np.where(conflicts[v])[0])
                        to_remove.add(v)
                        curr_avail -= to_remove
                if len(curr_temp) > len(best_temp):
                    best_temp = curr_temp
            
            # If we can replace 1 vertex with > 1 vertices, do it.
            if len(best_temp) > 1:
                # Update selected
                selected = np.concatenate([rem_selected, np.array(best_temp)])
                selected_set = set(selected)
                improved = True
                # break to restart the loop with new state
                break
                
    return all_vectors[selected]
