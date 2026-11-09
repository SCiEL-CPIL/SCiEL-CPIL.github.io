"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    adjacency = np.asarray(adjacency, dtype=np.int8)
    
    best_labels = np.zeros(n, dtype=np.int8)
    best_score = -1
    
    def get_cut_size(labels):
        return np.sum(adjacency * (labels[:, None] != labels[None, :]))

    def local_search_kl(labels):
        """
        Performs Kernighan-Lin style local search.
        Tries to flip pairs of vertices if it improves the cut.
        """
        current_score = get_cut_size(labels)
        improved = True
        max_iterations = 1000 # Safety limit
        
        while improved and max_iterations > 0:
            improved = False
            max_iterations -= 1
            
            best_delta = 0
            best_u = -1
            best_v = -1
            
            # Try all pairs? O(n^2) per iteration. n=100 -> 10,000 pairs.
            # Each pair evaluation O(1) if we maintain counts? 
            # Or O(n) to calculate delta.
            # Delta for flipping u and v:
            # Let L_u, L_v be labels.
            # Edges affected: (u, *), (v, *).
            # Specifically edges (u, v), (u, w) for w != v, (v, w) for w != u.
            
            # Efficient Delta Calculation:
            # Let S_u be set of neighbors of u.
            # Let S_v be set of neighbors of v.
            
            # Current cut contribution involving u and v:
            # For edge (u, v): contributes 1 if L_u != L_v.
            # For edge (u, w): contributes 1 if L_u != L_w.
            # For edge (v, w): contributes 1 if L_v != L_w.
            
            # If we flip u and v:
            # New L_u' = 1 - L_u, L_v' = 1 - L_v.
            
            # Change in cut for edge (u, v):
            # Old: 1 if L_u != L_v. New: 1 if L_u' != L_v' => 1 if (1-L_u) != (1-L_v) => 1 if L_u != L_v.
            # So edge (u, v) contribution does NOT change.
            
            # Change for edge (u, w) where w != v:
            # Old: 1 if L_u != L_w. New: 1 if L_u' != L_w => 1 if L_u != L_w.
            # Wait.
            # L_u' != L_w  <=>  (1-L_u) != L_w  <=>  L_u == L_w.
            # So if L_u != L_w (old cut 1), new cut is 0. Delta -1.
            # If L_u == L_w (old cut 0), new cut is 1. Delta +1.
            # So delta for (u, w) is +1 if L_u == L_w, -1 if L_u != L_w.
            # This is equivalent to: delta_u_w = 1 if L_w == L_u, else -1.
            # Sum over w in N(u) \ {v}: 
            # Let same_u = count of neighbors of u with label L_u.
            # Let opp_u = count of neighbors of u with label 1-L_u.
            # Delta from u's edges (excluding v) = same_u - opp_u.
            # BUT we must exclude v if v is a neighbor of u.
            
            # Similarly for v's edges (excluding u):
            # Let same_v = count of neighbors of v with label L_v.
            # Let opp_v = count of neighbors of v with label 1-L_v.
            # Delta from v's edges (excluding u) = same_v - opp_v.
            # Exclude u if u is a neighbor of v.
            
            # Total Delta = (same_u - opp_u) + (same_v - opp_v) - correction_for_uv_edge.
            
            # Let's compute same/opp counts for all vertices first?
            # That's O(n^2) per iteration if we recompute.
            # But n=100, n^2=10,000. 1000 iterations * 10,000 pairs * O(1) delta?
            # We can precompute same/opp counts for each vertex in O(n^2) per iteration.
            # Then evaluating a pair (u, v) is O(1) with adjustments.
            
            # Precompute counts for current labels
            same_counts = np.zeros(n)
            opp_counts = np.zeros(n)
            for i in range(n):
                # Count neighbors with same label
                same_counts[i] = np.sum(adjacency[i, labels == labels[i]])
                # Count neighbors with opposite label
                opp_counts[i] = np.sum(adjacency[i, labels != labels[i]])
            
            # Now iterate pairs
            # To speed up, we can just iterate all pairs
            for u in range(n):
                for v in range(u + 1, n):
                    # Calculate delta
                    delta = 0
                    
                    # Contribution from u's edges
                    # Base delta if u flips: same_counts[u] - opp_counts[u]
                    # But we are flipping u AND v.
                    # The edge (u, v) is handled separately.
                    # The other edges of u: 
                    # If v is a neighbor of u, the count of "same" or "opp" for u changes because v's label flips?
                    # No, we are evaluating the delta based on the CURRENT labels.
                    # The formula delta_u = same_u - opp_u assumes only u flips.
                    # If both flip, the interaction on edge (u, v) cancels out (as shown above, edge (u,v) cut status doesn't change).
                    # However, the counts same_u and opp_u include the edge to v if it exists.
                    # If u and v are connected:
                    # In same_u, v is counted if L_v == L_u.
                    # In opp_u, v is counted if L_v != L_u.
                    # When we flip u and v:
                    # The status of edge (u, v) doesn't change.
                    # The status of edges (u, w) for w != v changes based on L_u vs L_w.
                    # The status of edges (v, w) for w != u changes based on L_v vs L_w.
                    
                    # Let's derive the exact delta for flipping u and v.
                    # Let L be current labels.
                    # Cut(L) = sum_{(i,j) in E} [L_i != L_j]
                    # Cut(L') where L'_u = 1-L_u, L'_v = 1-L_v, L'_w = L_w for w!=u,v.
                    # Delta = Cut(L') - Cut(L)
                    # Only edges incident to u or v change.
                    # Edge (u, v): [L_u != L_v] vs [(1-L_u) != (1-L_v)] => Same. Delta 0.
                    # Edge (u, w) for w != v: [L_u != L_w] vs [(1-L_u) != L_w].
                    # If L_u == L_w: Old 0, New 1. Delta +1.
                    # If L_u != L_w: Old 1, New 0. Delta -1.
                    # So delta_u_w = 1 if L_u == L_w else -1.
                    # Sum over w in N(u) \ {v}: 
                    # Let S_u = {w in N(u) | w != v and L_w == L_u}
                    # Let O_u = {w in N(u) | w != v and L_w != L_u}
                    # Delta_u = |S_u| - |O_u|
                    
                    # Similarly for v:
                    # Let S_v = {w in N(v) | w != u and L_w == L_v}
                    # Let O_v = {w in N(v) | w != u and L_w != L_v}
                    # Delta_v = |S_v| - |O_v|
                    
                    # Total Delta = Delta_u + Delta_v
                    
                    # We have same_counts[u] = |{w in N(u) | L_w == L_u}|.
                    # If adjacency[u,v] == 1:
                    #   If L_u == L_v: v is in S_u (for u's perspective) and S_v (for v's perspective)?
                    #   Wait, S_u excludes v.
                    #   So if adj[u,v]==1 and L_u==L_v:
                    #      same_counts[u] includes v. So |S_u| = same_counts[u] - 1.
                    #      opp_counts[u] does not include v. So |O_u| = opp_counts[u].
                    #   If adj[u,v]==1 and L_u!=L_v:
                    #      same_counts[u] does not include v. So |S_u| = same_counts[u].
                    #      opp_counts[u] includes v. So |O_u| = opp_counts[u] - 1.
                    #   If adj[u,v]==0:
                    #      |S_u| = same_counts[u], |O_u| = opp_counts[u].
                    
                    # Let's calculate Delta_u and Delta_v efficiently.
                    
                    # Delta_u
                    if adjacency[u, v] == 1:
                        if labels[u] == labels[v]:
                            delta_u = (same_counts[u] - 1) - opp_counts[u]
                            delta_v = (same_counts[v] - 1) - opp_counts[v]
                        else:
                            delta_u = same_counts[u] - (opp_counts[u] - 1)
                            delta_v = same_counts[v] - (opp_counts[v] - 1)
                    else:
                        delta_u = same_counts[u] - opp_counts[u]
                        delta_v = same_counts[v] - opp_counts[v]
                    
                    delta = delta_u + delta_v
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_u = u
                        best_v = v
            
            if best_delta > 0:
                labels[best_u] = 1 - labels[best_u]
                labels[best_v] = 1 - labels[best_v]
                current_score += best_delta
                improved = True
        
        return current_score, labels

    # Spectral Initialization
    degrees = np.sum(adjacency, axis=1)
    D = np.diag(degrees)
    L = D - adjacency
    
    try:
        vals, vecs = eigsh(L, k=5, which='SM')
        idx = np.argsort(vals)
        vecs = vecs[:, idx]
        
        for i in range(1, 5):
            if i >= vecs.shape[1]:
                break
            v = vecs[:, i]
            labels = (v > 0).astype(np.int8)
            score, labels = local_search_kl(labels)
            
            if score > best_score:
                best_score = score
                best_labels = labels.copy()
    except Exception:
        pass

    # Random Restart Fallback / Enhancement
    for _ in range(10):
        labels = np.random.randint(0, 2, size=n).astype(np.int8)
        score, labels = local_search_kl(labels)
        if score > best_score:
            best_score = score
            best_labels = labels.copy()

    return best_labels
# EVOLVE-BLOCK-END
