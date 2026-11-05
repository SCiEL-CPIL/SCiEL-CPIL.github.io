"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

DEADLINE_MARGIN = 5.0

def partition(adjacency):
    start_time = time.time()
    deadline = start_time + 120.0 - DEADLINE_MARGIN
    
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)
    
    # 1. Spectral Initialization
    # Compute the Fiedler vector (eigenvector corresponding to the second smallest eigenvalue)
    # of the graph Laplacian. This provides a globally informed initial partition
    # that is often much better than random or alternating starts.
    try:
        # L = D - A
        # Use sparse laplacian for efficiency if n is large, though 100 is small.
        # eigsh finds k smallest eigenvalues.
        # We want the eigenvector for lambda_1 (second smallest, since lambda_0=0).
        # k=2 to get lambda_0 and lambda_1.
        L = laplacian(adj)
        
        # For small graphs, eigsh might fail or be overkill, but 100 nodes is fine.
        # If the graph is disconnected, lambda_1 might be 0, but Fiedler vector still useful or we fallback.
        try:
            w, v = eigsh(L, k=2, which='SM')
            # Sort eigenvalues to ensure we pick the correct one
            idx = np.argsort(w)
            fiedler_vec = v[:, idx[1]]
        except Exception:
            # Fallback to random or alternating if eigsh fails
            fiedler_vec = np.random.randn(n)
            
        # Partition based on the sign of the Fiedler vector
        labels = (fiedler_vec >= 0).astype(int)
        
        # Ensure both partitions are non-empty if possible (though not strictly required for cut size, 
        # it avoids trivial 0-cut if graph is connected and vector is constant-ish, which shouldn't happen for Fiedler)
        if labels.sum() == 0 or labels.sum() == n:
            # Fallback to alternating if spectral init degenerated
            labels = np.arange(n) % 2
            
    except Exception:
        # Fallback to alternating if spectral init fails completely
        labels = np.arange(n) % 2

    # 2. Vectorized Greedy Local Search
    # We perform multiple passes of greedy flips.
    # A vertex v is flipped if the number of neighbors in the opposite partition 
    # is greater than the number of neighbors in the same partition.
    # Gain = 2 * (neighbors_opposite) - degree(v)
    # If Gain > 0, flip.
    
    # Precompute degrees
    degrees = adj.sum(axis=1)
    
    best_labels = labels.copy()
    best_cut = 0
    
    # Helper to calculate cut size efficiently
    def calc_cut(lab):
        # Cut size = sum of edges where endpoints differ
        # diff = (lab[None, :] != lab[:, None])
        # return np.sum(adj * diff)
        # More efficient: 
        # Sum of adj rows for nodes labeled 0, multiplied by labels? No.
        # Let's use: 
        # cut = sum_{u,v: u<v, adj[u,v]=1, lab[u]!=lab[v]} 1
        # Using matrix multiplication:
        # Let L be label vector.
        # Cut = 0.5 * sum_{u,v} adj[u,v] * (1 - (2*lab[u]-1)*(2*lab[v]-1)) ? 
        # Simpler: 
        # diff = (lab[:, None] != lab[None, :])
        # return int(np.sum(adj * diff))
        diff = (lab[:, None] != lab[None, :])
        return int(np.sum(adj * diff))
        
    best_cut = calc_cut(best_labels)
    
    # Iterate until no improvement or deadline
    while time.time() < deadline:
        improved = False
        # Calculate gain for flipping each vertex
        # For vertex v, if label is l, neighbors with label 1-l contribute to cut.
        # current_cut_contribution_of_v = (neighbors with label 1-l)
        # new_cut_contribution_of_v = (neighbors with label l)
        # Gain = new - current = neighbors_with_label_l - neighbors_with_label_1-l
        # Let S_l be set of vertices with label l.
        # neighbors_with_label_l = adj[v, labels == l].sum()
        # neighbors_with_label_1-l = adj[v, labels == 1-l].sum()
        
        # Vectorized calculation:
        # For all v, compute sum of adj[v, :] where labels match v, and where they don't.
        # match_mask = (labels[None, :] == labels[:, None]) # n x n boolean
        # This is O(n^2) memory and time per pass. n=100 is 10k ops, very fast.
        
        labels_current = best_labels
        match_mask = (labels_current[:, None] == labels_current[None, :])
        
        # same_side_neighbors[v] = sum(adj[v, match_mask[v]])
        # other_side_neighbors[v] = sum(adj[v, ~match_mask[v]])
        
        same_side = (adj * match_mask).sum(axis=1)
        other_side = (adj * (~match_mask)).sum(axis=1)
        
        # Gain if flipped:
        # New same side count becomes current other_side count.
        # New other side count becomes current same_side count.
        # Gain = (new_other_side - current_other_side) ? No.
        # Cut size change = (new_edges_crossing - old_edges_crossing)
        # old_edges_crossing_for_v = other_side[v]
        # new_edges_crossing_for_v = same_side[v]
        # Gain = same_side[v] - other_side[v]
        
        gains = same_side - other_side
        
        # Find vertices with positive gain
        flip_candidates = np.where(gains > 0)[0]
        
        if len(flip_candidates) == 0:
            break
            
        # Flip all candidates that have positive gain.
        # Note: Simultaneous flipping can be tricky if they are connected, but for n=100 and simple graphs,
        # a "synchronous" update or just flipping the best ones is often sufficient.
        # To be safe and standard, we can flip them one by one or in batches.
        # Given the time budget, flipping all positive gain vertices simultaneously is a valid heuristic 
        # (similar to parallel local search). However, it might oscillate.
        # Let's flip the one with the MAXIMUM gain first, then recalculate? 
        # That's O(n^2) per flip, O(n^3) per pass. 100^3 = 1e6. 120s is plenty for millions of passes.
        # But vectorized simultaneous flip is faster. Let's try simultaneous first.
        # If it oscillates, the loop will terminate when gains become <= 0.
        
        best_labels[flip_candidates] = 1 - best_labels[flip_candidates]
        improved = True
        
        new_cut = calc_cut(best_labels)
        if new_cut > best_cut:
            best_cut = new_cut
        else:
            # If simultaneous flip didn't improve (or decreased) the cut, revert and try sequential?
            # Or just break? 
            # Actually, if we flipped vertices with positive gain, the cut *should* increase locally.
            # But interactions might reduce it globally.
            # Let's keep the best found so far.
            # If it decreased, we might want to revert this specific batch and try sequential.
            # For simplicity and speed, if it decreased, we revert and break, assuming we are at a local optimum 
            # or the simultaneous update was too aggressive.
            # However, reverting and breaking might miss a better local optimum.
            # Let's try: if new_cut < best_cut, revert labels to pre-flip state and break?
            # Or just continue? The next iteration will likely flip them back.
            # To avoid oscillation, if new_cut <= best_cut, we revert the changes and break.
            best_labels[flip_candidates] = 1 - best_labels[flip_candidates] # Revert
            break
            
    return best_labels
# EVOLVE-BLOCK-END
