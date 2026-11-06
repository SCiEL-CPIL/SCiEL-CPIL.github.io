import numpy as np
import time

def partition(adjacency):
    start_time = time.time()
    DEADLINE = 145.0  # Leave 5s safety margin for 150s limit
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy float32 for faster ops
    adj = np.array(adjacency, dtype=np.float32)
    degrees = adj.sum(axis=1)
    total_edges = 0.5 * adj.sum()
    
    def cut_size(labels):
        # Efficient cut size calculation using s-vector
        s = 2.0 * labels.astype(np.float32) - 1.0
        val = s @ (adj @ s)
        return 0.5 * total_edges - 0.25 * val

    def second_order_local_search(labels, max_passes=200):
        """
        Perform local search allowing flips of single vertices or pairs of vertices.
        This helps escape local optima where no single flip is beneficial, but a pair flip is.
        """
        labels = labels.copy()
        current_cut = cut_size(labels)
        
        for _ in range(max_passes):
            if time.time() > DEADLINE - 2:
                break
                
            improved = False
            
            # 1. Try single flips (First order)
            # Compute gain for flipping each vertex v
            # gain[v] = 2 * (sum_{u in N(v), label[u] != label[v]} 1) - degree[v]
            # Let S_v = sum_{u in N(v), label[u] == label[v]} 1
            # gain[v] = 2 * (degree[v] - S_v) - degree[v] = degree[v] - 2*S_v
            # Wait, standard gain formula:
            # Current contribution to cut from v: D_v (edges to opposite side)
            # New contribution: S_v (edges to same side, which become crossing)
            # Change = S_v - D_v = S_v - (degree[v] - S_v) = 2*S_v - degree[v]
            
            mask0 = (labels == 0).astype(np.float32)
            mask1 = (labels == 1).astype(np.float32)
            sum0 = adj @ mask0
            sum1 = adj @ mask1
            
            S = np.where(labels == 0, sum0, sum1)
            gain_single = 2.0 * S - degrees
            
            # Find best single flip
            best_idx = np.argmax(gain_single)
            if gain_single[best_idx] > 1e-6:
                labels[best_idx] = 1 - labels[best_idx]
                current_cut += gain_single[best_idx]
                improved = True
                continue # Restart loop to recompute gains
            
            # 2. Try pair flips (Second order) if no single flip helps
            # Gain for flipping u and v simultaneously:
            # Let A be adjacency.
            # Change in cut = 
            #   (Change in contribution of u) + (Change in contribution of v)
            #   - 2 * (Change in edge u-v status) ?
            
            # Let's derive carefully.
            # Let L be the label vector.
            # Cut(L) = 0.5 * (E - 0.25 * s^T A s) ? No, let's use the edge counting.
            # Cut = sum_{i<j} A[i,j] * (L[i] != L[j])
            
            # If we flip u and v:
            # Edges not involving u or v: unchanged.
            # Edges (u, w) where w != v:
            #   If L[w] == L[u], they were same, now diff. Cut increases by 1.
            #   If L[w] != L[u], they were diff, now same. Cut decreases by 1.
            #   Net change for u (ignoring v) = S_u - D_u = 2*S_u - deg[u] = gain_single[u]
            #   Net change for v (ignoring u) = S_v - D_v = 2*S_v - deg[v] = gain_single[v]
            
            # Edge (u, v):
            #   If A[u,v] == 1:
            #     If L[u] == L[v], they were same. After flip, L[u]' = 1-L[u], L[v]' = 1-L[v].
            #     Since L[u]==L[v], L[u]'==L[v]'. Still same. No change.
            #     If L[u] != L[v], they were diff. After flip, still diff. No change.
            #     Wait. If L[u]=0, L[v]=1. Flip -> L[u]=1, L[v]=0. Still diff.
            #     So flipping both endpoints of an edge does NOT change whether that edge is cut.
            
            # Therefore, the gain for flipping a pair {u, v} is simply:
            # gain_pair(u, v) = gain_single[u] + gain_single[v]
            
            # We want to find max_{u < v} (gain_single[u] + gain_single[v])
            # This is maximized by taking the two largest positive gains?
            # Not necessarily, because we only flip if the sum is > 0.
            # Since we already checked that max(gain_single) <= 0 (otherwise we would have flipped),
            # all gains are <= 0.
            # So sum of any two gains is <= 0.
            
            # Wait, this implies that if no single flip is beneficial, no pair flip is beneficial either?
            # Let's re-verify the edge (u,v) logic.
            # Cut = sum_{i<j} A[i,j] * (L[i] != L[j])
            # Term for edge (u,v) is A[u,v] * (L[u] != L[v]).
            # If we flip both, L[u] -> 1-L[u], L[v] -> 1-L[v].
            # (1-L[u]) != (1-L[v]) is equivalent to L[u] != L[v].
            # So the term for edge (u,v) is invariant.
            
            # Term for edge (u,w), w!=v:
            # Original: A[u,w] * (L[u] != L[w])
            # New: A[u,w] * (1-L[u] != L[w])
            # If L[u] != L[w], then 1-L[u] == L[w]. So term goes 1 -> 0. Change -1.
            # If L[u] == L[w], then 1-L[u] != L[w]. So term goes 0 -> 1. Change +1.
            # Sum of changes for u (excluding v) = (Number of neighbors w!=v with L[w]==L[u]) - (Number of neighbors w!=v with L[w]!=L[u])
            # Let S_u_excl_v = sum_{w!=v} A[u,w] * (L[w]==L[u])
            # Let D_u_excl_v = sum_{w!=v} A[u,w] * (L[w]!=L[u])
            # Change_u = S_u_excl_v - D_u_excl_v
            
            # We know gain_single[u] = S_u - D_u = (S_u_excl_v + A[u,v]*(L[v]==L[u])) - (D_u_excl_v + A[u,v]*(L[v]!=L[u]))
            # If L[u]==L[v]: A[u,v]*(L[v]==L[u]) = A[u,v]. A[u,v]*(L[v]!=L[u]) = 0.
            #   gain_single[u] = S_u_excl_v - D_u_excl_v + A[u,v]
            #   Change_u = S_u_excl_v - D_u_excl_v = gain_single[u] - A[u,v]
            # If L[u]!=L[v]: A[u,v]*(L[v]==L[u]) = 0. A[u,v]*(L[v]!=L[u]) = A[u,v].
            #   gain_single[u] = S_u_excl_v - D_u_excl_v - A[u,v]
            #   Change_u = S_u_excl_v - D_u_excl_v = gain_single[u] + A[u,v]
            
            # So Change_u = gain_single[u] - A[u,v] * (1 if L[u]==L[v] else -1) ?
            # Let's check signs.
            # If same: gain_single = Change + A. So Change = gain_single - A.
            # If diff: gain_single = Change - A. So Change = gain_single + A.
            
            # Total Change for pair {u,v} = Change_u + Change_v
            # = (gain_single[u] - A[u,v]*I_same) + (gain_single[v] - A[u,v]*I_same)
            # where I_same is 1 if L[u]==L[v] else -1? No.
            # If same: I=1. Change_u = gain_u - A. Change_v = gain_v - A. Total = gain_u + gain_v - 2A.
            # If diff: I=-1. Change_u = gain_u + A. Change_v = gain_v + A. Total = gain_u + gain_v + 2A.
            
            # So gain_pair(u,v) = gain_single[u] + gain_single[v] + 2 * A[u,v] * (1 if L[u]==L[v] else -1)
            
            # We need to check if any pair has gain_pair > 0.
            # Since max gain_single <= 0, gain_single[u] + gain_single[v] <= 0.
            # The term 2*A[u,v]*(...) can be +2 or -2.
            # So we can only improve if:
            # 1. L[u] != L[v] and A[u,v] == 1. Then term is -2. Total <= -2. No improvement.
            # 2. L[u] == L[v] and A[u,v] == 1. Then term is +2. Total = gain_u + gain_v + 2.
            #    This can be > 0 if gain_u + gain_v > -2.
            #    Since gains are integers (or close to), if gain_u = -1 and gain_v = -1, sum = -2, total = 0. No strict improvement.
            #    If gain_u = -1 and gain_v = 0, sum = -1, total = 1. Improvement!
            
            # So we only need to check pairs (u,v) such that:
            # - L[u] == L[v]
            # - A[u,v] == 1
            # - gain_single[u] + gain_single[v] + 2 > 0
            
            # Given that max gain_single <= 0, the best candidates are vertices with gain_single close to 0.
            # Specifically, if max gain is 0, we look for neighbors with gain 0 or -1.
            # If max gain is -1, we look for neighbors with gain -1 (sum -2, +2 = 0, no strict improve) or 0 (sum -1, +2 = 1, improve).
            
            # Optimization:
            # Sort vertices by gain_single descending.
            # Check top K vertices. For each u in top K, check neighbors v in top K?
            # Or simply:
            # Let U0 = {u | gain_single[u] == 0}
            # Let U1 = {u | gain_single[u] == -1}
            
            # Pairs to check:
            # 1. u in U0, v in U0, L[u]==L[v], A[u,v]==1. Gain = 0+0+2 = 2.
            # 2. u in U0, v in U1, L[u]==L[v], A[u,v]==1. Gain = 0-1+2 = 1.
            # 3. u in U1, v in U1, L[u]==L[v], A[u,v]==1. Gain = -1-1+2 = 0. (No strict improvement)
            
            # So we only need to check pairs where at least one vertex has gain 0.
            # If there are no vertices with gain 0, no pair flip can strictly improve (since max sum is -2, +2=0).
            
            if not np.any(gain_single > -1e-6):
                # All gains <= -1. No pair flip can improve.
                break
                
            # Find vertices with gain close to 0
            zero_gain_indices = np.where(gain_single > -1e-6)[0]
            
            # For each u in zero_gain_indices, check neighbors v
            # We need L[u] == L[v] and A[u,v] == 1
            # And we want to maximize gain_single[u] + gain_single[v] + 2
            # Since gain_single[u] is approx 0, we want max gain_single[v] among valid neighbors.
            
            best_pair_gain = 0
            best_u = -1
            best_v = -1
            
            # To be efficient, we can iterate over zero_gain_indices
            # For each u, look at its neighbors.
            # n=100, so this is fast.
            for u in zero_gain_indices:
                # Get neighbors of u
                neighbors = np.where(adj[u] > 0)[0]
                if len(neighbors) == 0:
                    continue
                    
                # Filter neighbors v such that L[v] == L[u]
                same_label_neighbors = neighbors[labels[neighbors] == labels[u]]
                if len(same_label_neighbors) == 0:
                    continue
                    
                # Among these, find the one with max gain_single[v]
                # We also require A[u,v] == 1, which is true by definition of neighbors.
                max_gain_v = np.max(gain_single[same_label_neighbors])
                best_v_in_group = same_label_neighbors[np.argmax(gain_single[same_label_neighbors])]
                
                potential_gain = gain_single[u] + max_gain_v + 2.0
                
                if potential_gain > best_pair_gain + 1e-6:
                    best_pair_gain = potential_gain
                    best_u = u
                    best_v = best_v_in_group
            
            if best_u != -1 and best_pair_gain > 1e-6:
                labels[best_u] = 1 - labels[best_u]
                labels[best_v] = 1 - labels[best_v]
                current_cut += best_pair_gain
                improved = True
                continue # Restart loop
            
            if not improved:
                break
                
        return labels, current_cut

    best_labels = None
    best_cut = -1
    
    # Phase 1: Spectral Initialization (Fiedler Vector)
    try:
        D = np.diag(degrees)
        L = D - adj
        # Use scipy.linalg.eigh for symmetric matrices
        from scipy.linalg import eigh
        vals, vecs = eigh(L, subset_by_index=[1, 1])
        fiedler = vecs[:, 0]
        init_fiedler = (fiedler > 0).astype(int)
        if np.all(init_fiedler == 0) or np.all(init_fiedler == 1):
            med = np.median(fiedler)
            init_fiedler = (fiedler > med).astype(int)
    except Exception:
        init_fiedler = np.zeros(n, dtype=int)
        init_fiedler[:n//2] = 1

    # Phase 2: Second Order Local Search from Spectral Init
    improved_labels, cut_val = second_order_local_search(init_fiedler)
    if cut_val > best_cut:
        best_cut = cut_val
        best_labels = improved_labels.copy()

    # Phase 3: Multi-start Random with Second Order Local Search
    rng = np.random.default_rng(42)
    
    # We can afford many restarts because second order search converges fast or stops quickly.
    while time.time() < DEADLINE - 5:
        labels = rng.integers(0, 2, size=n)
        improved_labels, cut_val = second_order_local_search(labels, max_passes=50)
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = improved_labels.copy()
            
        # Optional: Try flipping one random vertex and running search again?
        # Or just rely on random restarts.
        
    if best_labels is None:
        best_labels = np.arange(n) % 2

    return best_labels
