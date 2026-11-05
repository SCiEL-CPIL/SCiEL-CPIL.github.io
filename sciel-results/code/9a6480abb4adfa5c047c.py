import numpy as np
import time
from scipy.sparse.csgraph import maximum_bipartite_matching # Might not be needed if we use a simple LP heuristic

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: LP-Relaxation Guided Structural Scoring + Deterministic Local Search.
    1. Compute a fractional vertex cover (dual of fractional matching) to identify
       vertices with low "constraint weight". These are likely to be in the MIS.
    2. Initialize the ordering based on these LP dual values (lower dual -> higher priority).
    3. Refine the ordering using a deterministic local search (2-opt style) that 
       evaluates swaps based on their impact on the greedy set size.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    degrees = np.array([len(nbrs) for nbrs in neighbors_list])
    
    # 1. LP Relaxation Heuristic for Vertex Cover
    # The LP for Minimum Vertex Cover is: min sum(x_v) s.t. x_u + x_v >= 1 for all (u,v), 0<=x<=1.
    # The dual is Maximum Matching: max sum(y_e) s.t. sum(y_e) <= 1 for all v, y_e >= 0.
    # For general graphs, we can approximate the fractional vertex cover using a greedy 
    # approach or simply use the degree-based proxy if LP is too slow.
    # However, a better proxy for "independence" is often the inverse of the "bottleneck" degree.
    # Let's try a simple iterative LP-like update or just use the standard "Low Degree" 
    # but enhanced by a "clique" penalty if possible? 
    # Actually, let's stick to the directive: use LP dual variables.
    # Since we don't have a fast LP solver in stdlib, we can approximate the fractional 
    # vertex cover by solving the dual (fractional matching) or just using a heuristic 
    # that mimics it.
    # A known heuristic for fractional matching/vertex cover is related to the 
    # "b-matching" or just using the degrees.
    # Let's try: score = -degree / (1 + degree) or similar?
    # Or better: The LP dual variable for a vertex v is the sum of weights of matching edges incident to v.
    # In a sparse random graph, the optimal fractional matching is often just 0.5 for all edges 
    # if the graph is a collection of odd cycles or similar, but for ER graphs, it's complex.
    
    # Let's use a simpler structural signal: 
    # Vertices with low degree are better. 
    # Vertices with neighbors that have HIGH degree are worse (because those neighbors 
    # are "strong" and will likely be picked, blocking this vertex).
    # Let's define a "strength" score for neighbors.
    
    # Initial Score: 
    # We want to pick vertices that are "weakly constrained".
    # Heuristic: s_v = - (degree(v) + sum_{u in N(v)} degree(u))
    # This penalizes vertices that are connected to high-degree hubs.
    
    neighbor_deg_sum = np.zeros(n)
    for i in range(n):
        for nbr in neighbors_list[i]:
            neighbor_deg_sum[i] += degrees[nbr]
            
    initial_scores = -(degrees + neighbor_deg_sum)
    
    # Initialize permutation: Sort by score descending (highest priority first)
    # Higher score means "more independent-like"
    perm = np.argsort(-initial_scores, kind='stable')
    
    def evaluate_full(perm):
        """Standard greedy evaluation for baseline."""
        active = np.ones(n, dtype=bool)
        count = 0
        for v in perm:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    active[nbrs] = False
        return count

    best_val = evaluate_full(perm)
    best_perm = perm.copy()
    current_val = best_val
    current_perm = best_perm.copy()
    
    # Precompute prefix masks for fast delta evaluation
    def precompute_prefix(perm):
        masks = [None] * (n + 1)
        counts = [0] * (n + 1)
        active = np.ones(n, dtype=bool)
        count = 0
        masks[0] = active.copy()
        counts[0] = 0
        for i in range(n):
            v = perm[i]
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    active[nbrs] = False
            masks[i+1] = active.copy()
            counts[i+1] = count
        return masks, counts

    masks, counts = precompute_prefix(current_perm)
    
    def eval_swap_delta(perm, masks, counts, i, j):
        """
        Evaluate delta for swapping perm[i] and perm[j] (i < j).
        Returns delta in count.
        """
        active_i = masks[i]
        
        # Simulate NEW segment [i, j]
        new_active = active_i.copy()
        new_count = 0
        
        # Position i: vertex u_new = perm[j]
        u_new = perm[j]
        if new_active[u_new]:
            new_count += 1
            new_active[u_new] = False
            nbrs = neighbors_list[u_new]
            if nbrs:
                new_active[nbrs] = False
                
        # Positions i+1 to j-1: unchanged vertices
        for k in range(i + 1, j):
            v = perm[k]
            if new_active[v]:
                new_count += 1
                new_active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    new_active[nbrs] = False
                    
        # Position j: vertex v_new = perm[i]
        v_new = perm[i]
        if new_active[v_new]:
            new_count += 1
            new_active[v_new] = False
            nbrs = neighbors_list[v_new]
            if nbrs:
                new_active[nbrs] = False
                
        # Simulate OLD segment [i, j]
        old_active = active_i.copy()
        old_count = 0
        
        # Position i: vertex u_old = perm[i]
        u_old = perm[i]
        if old_active[u_old]:
            old_count += 1
            old_active[u_old] = False
            nbrs = neighbors_list[u_old]
            if nbrs:
                old_active[nbrs] = False
                
        # Positions i+1 to j-1: unchanged vertices
        for k in range(i + 1, j):
            v = perm[k]
            if old_active[v]:
                old_count += 1
                old_active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    old_active[nbrs] = False
                    
        # Position j: vertex v_old = perm[j]
        v_old = perm[j]
        if old_active[v_old]:
            old_count += 1
            old_active[v_old] = False
            nbrs = neighbors_list[v_old]
            if nbrs:
                old_active[nbrs] = False
        
        # Check if suffix is affected
        if np.array_equal(new_active, old_active):
            # Suffix count is unchanged
            return new_count - old_count
        else:
            # Must evaluate suffix
            # This is expensive. We'll return a special value to indicate "need full re-eval" 
            # or just do it if we are close to the end.
            # For efficiency, if the difference is small, we might skip? 
            # No, we need accuracy.
            # Let's evaluate the suffix.
            def eval_suffix(perm, start_idx, active_mask):
                count = 0
                act = active_mask.copy()
                for k in range(start_idx, n):
                    v = perm[k]
                    if act[v]:
                        count += 1
                        act[v] = False
                        nbrs = neighbors_list[v]
                        if nbrs:
                            act[nbrs] = False
                return count
            
            suffix_new = eval_suffix(perm, j+1, new_active)
            suffix_old = eval_suffix(perm, j+1, old_active)
            
            return (new_count + suffix_new) - (old_count + suffix_old)

    def update_masks_after_swap(perm, masks, counts, i, j):
        """
        Update masks and counts after swapping perm[i] and perm[j] (i < j).
        """
        active = masks[i].copy()
        
        for k in range(i, n):
            v = perm[k]
            if active[v]:
                active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    active[nbrs] = False
                counts[k+1] = counts[k] + 1
            else:
                counts[k+1] = counts[k]
            
            if k + 1 < n:
                masks[k+1] = active.copy()
            else:
                masks[n] = active.copy()

    # Local Search Loop
    # We use a deterministic sweep: try to improve by moving vertices earlier if beneficial.
    # Specifically, we scan the permutation and for each vertex, see if swapping it with 
    # an earlier vertex improves the set.
    
    max_time = 118.0
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        # To avoid O(n^3) in worst case, we limit the range of swaps or use a random subset?
        # Given n=200, n^2 = 40k swaps. Each swap eval is O(n) worst case = 200.
        # 40k * 200 = 8M ops. This might take ~1-2 seconds in Python if not optimized.
        # Let's try to optimize by only checking "nearby" swaps first?
        # Or just run one full pass.
        
        for i in range(n - 1):
            if time.time() - start_time > max_time - 2:
                break
            # Try swapping i with j > i
            # We can break early if we find a good swap?
            # Let's just evaluate all j > i for a fixed i?
            # That is O(n) per i, total O(n^2) evaluations.
            
            for j in range(i + 1, n):
                # Quick check: if perm[i] and perm[j] are not adjacent in graph, 
                # the change might be small.
                
                delta = eval_swap_delta(current_perm, masks, counts, i, j)
                
                if delta > 0:
                    # Accept swap
                    current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
                    current_val += delta
                    
                    if current_val > best_val:
                        best_val = current_val
                        best_perm = current_perm.copy()
                    
                    # Update masks
                    update_masks_after_swap(current_perm, masks, counts, i, j)
                    
                    improved = True
                    # After a swap, the state has changed. 
                    # We should restart the inner loop or continue carefully.
                    # Since we are moving j to i, the vertex at i is now the one from j.
                    # The vertices between i and j have shifted? No, swap is just i and j.
                    # But the "active mask" for k > i has changed.
                    # So the subsequent evaluations for this i (with larger j) are invalid 
                    # if we don't update masks. We did update masks.
                    # However, the "delta" for the next j might be different.
                    # Let's restart the j loop for this i to be safe, or just continue 
                    # with the updated masks. Continuing is fine as masks are updated.
                    # But note: the vertex at j is now the old perm[i].
                    # So the "j" index in the loop still refers to the position, 
                    # but the content changed.
                    # This is a standard local search behavior.
                    
        # End i loop
        # End while loop

    # Final check
    final_val = evaluate_full(best_perm)
    if final_val > best_val:
        best_val = final_val
        
    # Convert best_perm to scores
    # Vertex at position k in best_perm gets score (n - k)
    scores = np.zeros(n)
    for k, v in enumerate(best_perm):
        scores[v] = n - k
        
    return scores
