"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # labels are 0/1. cut = sum over v of (number of neighbors of v with label != labels[v])
    # = sum_v adj[v] @ (1 - labels) where we count each edge twice, so divide by 2? 
    # Actually adj[v] @ (1-labels) counts edges from v to opposite side.
    # Summing over all v gives 2 * cut_size (since each crossing edge is counted from both ends).
    # Wait: if labels are 0/1, then (1-labels) is 1 where label is 0 and 0 where label is 1.
    # adj[v] @ (1-labels) = sum_u adj[v,u] * (1-labels[u]) = number of neighbors u of v with labels[u]=0.
    # If labels[v]=1, this is the number of crossing edges incident to v.
    # If labels[v]=0, this is the number of non-crossing edges incident to v (edges to other 0s).
    # So sum_v adj[v] @ (1-labels) is NOT 2*cut.
    # Let's use: cut = 0.5 * sum_v sum_u adj[v,u] * (labels[v] != labels[u])
    # In matrix form with 0/1 labels:
    # labels != labels' is 1 when different.
    # (labels - labels')^2 = 1 when different, 0 when same.
    # cut = 0.5 * sum_v sum_u adj[v,u] * (labels[v] - labels[u])^2
    #     = 0.5 * (sum_v sum_u adj[v,u] * labels[v]^2 - 2 sum_v sum_u adj[v,u] labels[v] labels[u] + sum_v sum_u adj[v,u] labels[u]^2)
    # Since labels are 0/1, labels[v]^2 = labels[v].
    # sum_v sum_u adj[v,u] labels[v]^2 = sum_v labels[v] * (sum_u adj[v,u]) = sum_v labels[v] * degree[v]
    # Similarly for the other term.
    # sum_v sum_u adj[v,u] labels[v] labels[u] = labels^T A labels
    # So cut = 0.5 * (2 * labels^T D labels - 2 * labels^T A labels) where D is degree matrix?
    # Wait: sum_v labels[v] * degree[v] = labels^T D 1? No, D is diagonal with degrees.
    # sum_v labels[v] * degree[v] is a scalar. Let's call it L.
    # Then cut = 0.5 * (2*L - 2*labels^T A labels) = L - labels^T A labels.
    # But L = sum_v labels[v] * degree[v].
    # This seems asymmetric. Let's re-derive.
    # cut = 0.5 * sum_{v,u} A_{vu} (x_v - x_u)^2 where x is 0/1.
    # (x_v - x_u)^2 = x_v^2 + x_u^2 - 2 x_v x_u = x_v + x_u - 2 x_v x_u (since x is 0/1).
    # sum_{v,u} A_{vu} (x_v + x_u - 2 x_v x_u) = sum_v x_v (sum_u A_{vu}) + sum_u x_u (sum_v A_{vu}) - 2 x^T A x
    # = sum_v x_v deg(v) + sum_u x_u deg(u) - 2 x^T A x
    # = 2 * sum_v x_v deg(v) - 2 x^T A x
    # So cut = 0.5 * (2 * sum_v x_v deg(v) - 2 x^T A x) = sum_v x_v deg(v) - x^T A x.
    # Let's verify with a small example.
    # Graph: 0-1-2. A = [[0,1,0],[1,0,1],[0,1,0]]. deg = [1,2,1].
    # x = [1,0,0]. cut should be 1 (edge 0-1).
    # sum_v x_v deg(v) = 1*1 + 0*2 + 0*1 = 1.
    # x^T A x = [1,0,0] [[0,1,0],[1,0,1],[0,1,0]] [1,0,0]^T = [1,0,0] [0,1,0]^T = 0.
    # cut = 1 - 0 = 1. Correct.
    # x = [1,1,0]. cut should be 1 (edge 1-2).
    # sum_v x_v deg(v) = 1*1 + 1*2 + 0*1 = 3.
    # x^T A x = [1,1,0] A [1,1,0]^T = [1,1,0] [1,1,1]^T = 1+1+0 = 2.
    # cut = 3 - 2 = 1. Correct.
    # So cut = x @ (adj @ x) ... no, x^T A x is x @ (A @ x).
    # And sum_v x_v deg(v) = x @ (adj @ ones)? No, deg = adj @ ones.
    # So sum_v x_v deg(v) = x @ deg = x @ (adj @ ones).
    # cut = x @ (adj @ ones) - x @ (adj @ x).
    # Let's implement this.
    deg = adj.sum(axis=1)
    return x @ deg - x @ (adj @ x)


def _spectral_init(adj):
    """Use Fiedler vector to get initial partition."""
    n = len(adj)
    # Laplacian L = D - A
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    # Find eigenvector corresponding to second smallest eigenvalue
    # Use scipy if available, else numpy
    try:
        from scipy.linalg import eigh
        # L is symmetric
        eigvals, eigvecs = eigh(L)
        # Fiedler vector is the eigenvector for the second smallest eigenvalue
        fiedler = eigvecs[:, 1]
    except ImportError:
        # Fallback to numpy
        eigvals, eigvecs = np.linalg.eigh(L)
        fiedler = eigvecs[:, 1]
    
    # Partition based on median or mean
    median = np.median(fiedler)
    labels = (fiedler > median).astype(int)
    return labels


def _greedy_flip_improve(adj, labels):
    """Improve partition by flipping single vertices."""
    n = len(adj)
    labels = labels.copy()
    improved = True
    while improved:
        improved = False
        for v in range(n):
            # Calculate gain from flipping v
            # Current contribution of v to cut:
            # neighbors in opposite partition
            same_side = (labels == labels[v]).sum()
            # Actually, let's compute the change in cut size.
            # Let S be the set of vertices with label 1.
            # cut = sum_{u in S, v not in S} A_{uv}
            # If we flip v from 0 to 1:
            # New cut = cut - (edges from v to S) + (edges from v to not S, excluding v)
            # Edges from v to S = sum_{u in S} A_{vu}
            # Edges from v to not S (excluding v) = deg(v) - sum_{u in S} A_{vu} - A_{vv} (which is 0)
            # Change = -sum_{u in S} A_{vu} + deg(v) - sum_{u in S} A_{vu} = deg(v) - 2 * sum_{u in S} A_{vu}
            # If v is 1, flipping to 0:
            # Change = -deg(v) + 2 * sum_{u in S \ {v}} A_{vu} = -deg(v) + 2 * (sum_{u in S} A_{vu} - A_{vv}) = -deg(v) + 2 * sum_{u in S} A_{vu}
            
            if labels[v] == 0:
                # Flip 0 -> 1
                # Gain = deg(v) - 2 * (adj[v] @ labels)
                gain = deg_v = adj[v].sum() - 2 * (adj[v] @ labels)
            else:
                # Flip 1 -> 0
                # Gain = -deg(v) + 2 * (adj[v] @ labels)
                gain = -adj[v].sum() + 2 * (adj[v] @ labels)
            
            if gain > 0:
                labels[v] = 1 - labels[v]
                improved = True
    return labels


def _two_swap_improve(adj, labels):
    """Improve partition by swapping pairs of vertices."""
    n = len(adj)
    labels = labels.copy()
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i+1, n):
                if labels[i] == labels[j]:
                    continue
                # Try swapping labels[i] and labels[j]
                # Compute gain
                # Let's compute the change in cut size.
                # Original cut contribution involving i and j:
                # This is complex, so let's just compute the new cut size and compare.
                # But that's O(n^2) per swap, which is O(n^4) total. Too slow for n=100?
                # 100^4 = 100,000,000. Might be okay in Python if optimized, but risky.
                # Let's derive the gain formula.
                # Let S be the set with label 1.
                # i is in S, j is not in S (WLOG).
                # Swap: i leaves S, j enters S.
                # Change in cut:
                # - Edges from i to S \ {i} become non-cut. (Loss)
                # - Edges from i to not S (including j) become cut. (Gain)
                # - Edges from j to not S \ {j} become non-cut. (Loss)
                # - Edges from j to S (including i) become cut. (Gain)
                # - Edge (i,j) was cut, now non-cut. (Loss)
                # Let N_S(v) be neighbors of v in S.
                # Let N_notS(v) be neighbors of v in not S.
                # i in S, j in not S.
                # Loss from i: |N_S(i) \ {i}| (edges to S that become non-cut)
                # Gain from i: |N_notS(i) \ {j}| (edges to not S that become cut)
                # Loss from j: |N_notS(j) \ {j}| (edges to not S that become non-cut)
                # Gain from j: |N_S(j) \ {i}| (edges to S that become cut)
                # Loss from (i,j): 1 (was cut, now not)
                # Total gain = Gain_i + Gain_j - Loss_i - Loss_j - 1
                # |N_S(i) \ {i}| = (adj[i] @ labels) - 1 (since i is in S, and A_ii=0)
                # |N_notS(i) \ {j}| = deg(i) - (adj[i] @ labels) - 1 (if j is in not S, adj[i,j]=1, so we exclude it)
                #   Actually, deg(i) = |N_S(i)| + |N_notS(i)|.
                #   |N_notS(i)| = deg(i) - |N_S(i)|.
                #   We exclude j, so |N_notS(i) \ {j}| = deg(i) - |N_S(i)| - 1.
                # |N_notS(j) \ {j}| = (deg(j) - (adj[j] @ labels)) - 1 (j is in not S, so adj[j]@labels counts N_S(j))
                #   Wait, j is in not S. N_notS(j) are neighbors in not S.
                #   deg(j) = |N_S(j)| + |N_notS(j)|.
                #   |N_notS(j)| = deg(j) - |N_S(j)|.
                #   We exclude j itself? No, j is a vertex, not an edge. We exclude the edge (i,j) separately.
                #   Actually, when j moves to S, its edges to not S \ {j} become non-cut.
                #   But j is not in not S anymore, so N_notS(j) is defined with respect to the old partition.
                #   Let's stick to the formula:
                #   Gain = [deg(i) - (adj[i]@labels) - 1] + [(adj[j]@labels) - 1] - [(adj[i]@labels) - 1] - [(deg(j) - (adj[j]@labels)) - 1] - 1
                #   Wait, this is getting confusing. Let's use a simpler approach:
                #   Compute the cut size before and after the swap for the affected vertices.
                #   Since n=100, we can afford to recompute the cut size for the whole graph
                #   if we do it efficiently, but O(n^2) swaps * O(n^2) cut size = O(n^4) = 10^8.
                #   This might be too slow in Python.
                #   Let's try to implement the gain formula carefully.
                
                # Let's define:
                # s_i = labels[i] (1 if in S, 0 otherwise)
                # s_j = labels[j]
                # Assume s_i=1, s_j=0.
                # deg_i = adj[i].sum()
                # deg_j = adj[j].sum()
                # cut_i = adj[i] @ labels  (number of neighbors of i in S)
                # cut_j = adj[j] @ labels  (number of neighbors of j in S)
                
                # After swap:
                # i is in not S, j is in S.
                # New cut contribution from i's edges:
                #   i's neighbors in not S (new) become cut.
                #   Old not S neighbors of i: deg_i - cut_i.
                #   But j was in not S, and now j is in S. So i's new not S neighbors = (deg_i - cut_i) - 1 (exclude j).
                #   So i contributes (deg_i - cut_i - 1) to the cut.
                #   Previously i contributed cut_i to the cut (edges to S).
                #   Change for i: (deg_i - cut_i - 1) - cut_i = deg_i - 2*cut_i - 1.
                
                # New cut contribution from j's edges:
                #   j's neighbors in S (new) become cut.
                #   Old S neighbors of j: cut_j.
                #   But i was in S, and now i is in not S. So j's new S neighbors = cut_j - 1 (exclude i).
                #   So j contributes (cut_j - 1) to the cut.
                #   Previously j contributed (deg_j - cut_j) to the cut (edges to not S).
                #   Change for j: (cut_j - 1) - (deg_j - cut_j) = 2*cut_j - deg_j - 1.
                
                # Edge (i,j):
                #   Was cut (i in S, j in not S). Now not cut (i in not S, j in S).
                #   Change: -1.
                
                # Total gain = (deg_i - 2*cut_i - 1) + (2*cut_j - deg_j - 1) - 1
                #           = deg_i - 2*cut_i + 2*cut_j - deg_j - 3
                
                # Let's verify with an example.
                # Graph: 0-1, 1-2. S={0}, not S={1,2}. labels=[1,0,0].
                # Swap 0 and 1.
                # deg_0=1, cut_0=0 (no neighbors in S).
                # deg_1=2, cut_1=1 (neighbor 0 in S).
                # Gain = 1 - 0 + 2 - 2 - 3 = -2.
                # New labels: [0,1,0]. Cut: edge 0-1 (0 in not S, 1 in S) is cut. Edge 1-2 (1 in S, 2 in not S) is cut. Cut=2.
                # Old cut: edge 0-1 is cut. Cut=1.
                # Change = 2-1=1. But formula says -2. Something is wrong.
                
                # Let's re-evaluate.
                # Old cut: 1.
                # New cut: 2.
                # Change: +1.
                # My formula gave -2.
                
                # Let's re-derive.
                # i=0 (in S), j=1 (in not S).
                # i's old cut contribution: edges to S. Neighbors of 0: {1}. 1 is in not S. So 0.
                # i's new cut contribution: edges to not S. Neighbors of 0: {1}. 1 is now in S. So 0.
                # Change for i: 0 - 0 = 0.
                # Formula: deg_i - 2*cut_i - 1 = 1 - 0 - 1 = 0. Correct.
                
                # j's old cut contribution: edges to not S. Neighbors of 1: {0,2}. 0 in S, 2 in not S. So 1.
                # j's new cut contribution: edges to S. Neighbors of 1: {0,2}. 0 now in not S, 2 in not S. So 0.
                # Change for j: 0 - 1 = -1.
                # Formula: 2*cut_j - deg_j - 1 = 2*1 - 2 - 1 = -1. Correct.
                
                # Edge (i,j): was cut, now not cut. Change: -1.
                
                # Total change: 0 + (-1) + (-1) = -2.
                # But actual change is +1.
                # Where is the discrepancy?
                # Ah, the edge (i,j) is counted in both i's and j's contributions.
                # In the old cut, edge (0,1) is counted once.
                # In my calculation:
                # i's old contribution: 0.
                # j's old contribution: 1 (this includes edge (1,0)).
                # Total old cut from i and j: 0 + 1 = 1. Correct.
                # In the new cut:
                # i's new contribution: 0.
                # j's new contribution: 0.
                # Total new cut from i and j: 0.
                # But the actual new cut is 2 (edges (0,1) and (1,2)).
                # Edge (0,1) is between i and j. It is now not cut?
                # i=0 is in not S, j=1 is in S. So edge (0,1) IS cut.
                # My mistake: when I said "j's new cut contribution: edges to S", I included i.
                # j's neighbors: {0,2}. i=0 is now in not S. So j's new S neighbors: none.
                # But edge (0,1) is cut because 0 is in not S and 1 is in S.
                # So j's new cut contribution should include edge (0,1)?
                # No, the cut is defined as edges between S and not S.
                # Edge (0,1): 0 in not S, 1 in S. So it is a cut edge.
                # It is incident to j=1. So it should be counted in j's contribution.
                # But j's new S neighbors are those in S. 0 is in not S. So 0 is not in S.
                # So j's new cut contribution is edges to not S? No, j is in S, so its cut edges are to not S.
                # j's neighbors: 0 (not S), 2 (not S). So both are cut edges.
                # So j's new cut contribution is 2.
                # My formula for j's new contribution was "edges to S", which is wrong.
                # j is in S, so its cut edges are to not S.
                # So j's new cut contribution = number of neighbors in not S.
                # Old: j was in not S, so its cut edges were to S.
                # So:
                # i (was in S, now in not S):
                #   Old cut edges: to S. Count = cut_i.
                #   New cut edges: to not S. Count = deg_i - cut_i (but j was in not S, now in S, so subtract 1 if i-j is an edge).
                #   Actually, i's new not S neighbors = (old not S neighbors) - {j} + {i's new S neighbors that are now not S? No.}
                #   Let's just say:
                #   i's new cut edges = neighbors of i that are in not S (new partition).
                #   New not S = (old not S) \ {j} \ {i} + {i} ... this is messy.
                
                # Let's use a different approach. Compute the change in the objective function directly.
                # Objective: cut = 0.5 * sum_{v,u} A_{vu} (x_v - x_u)^2.
                # This is equivalent to x^T L x / 2 where L is the Laplacian? No.
                # cut = x^T (D - A) x / 2? No, that's for the QAP.
                # Let's just recompute the cut size for the whole graph after the swap.
                # Since n=100, and we have O(n^2) swaps, and each cut size computation is O(n^2),
                # total is O(n^4) = 10^8. In Python, this might take a few seconds.
                # Let's try it and see if it's fast enough.
                
                old_cut = _cut_size(adj, labels)
                labels[i], labels[j] = labels[j], labels[i]
                new_cut = _cut_size(adj, labels)
                if new_cut > old_cut:
                    # Keep the swap
                    improved = True
                else:
                    # Revert the swap
                    labels[i], labels[j] = labels[j], labels[i]
    return labels


def _tabu_search(adj, labels, max_iter=1000, tabu_tenure=7):
    """Tabu search to escape local optima."""
    n = len(adj)
    labels = labels.copy()
    best_labels = labels.copy()
    best_cut = _cut_size(adj, labels)
    tabu_list = [0] * n  # iteration when vertex was last flipped
    current_iter = 0
    
    while current_iter < max_iter:
        best_gain = -1
        best_v = -1
        for v in range(n):
            if current_iter - tabu_list[v] < tabu_tenure:
                continue
            # Calculate gain from flipping v
            if labels[v] == 0:
                gain = adj[v].sum() - 2 * (adj[v] @ labels)
            else:
                gain = -adj[v].sum() + 2 * (adj[v] @ labels)
            
            # Aspiration: allow tabu move if it improves the best solution
            if gain > best_gain or (current_iter - tabu_list[v] < tabu_tenure and gain > best_cut):
                best_gain = gain
                best_v = v
        
        if best_v == -1:
            break
        
        labels[best_v] = 1 - labels[best_v]
        tabu_list[best_v] = current_iter
        current_iter += 1
        
        current_cut = _cut_size(adj, labels)
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
    
    return best_labels


def partition(adjacency):
    """
    Partition the graph into two sets to maximize the cut size.
    Uses spectral initialization, greedy flip, two-swap, and tabu search.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    adj = np.array(adjacency, dtype=np.int32)
    n = len(adj)
    
    # Spectral initialization
    labels = _spectral_init(adj)
    
    # Greedy flip improvement
    labels = _greedy_flip_improve(adj, labels)
    
    # Two-swap improvement (might be slow, so check time)
    if time.time() < deadline - 30:
        labels = _two_swap_improve(adj, labels)
    
    # Tabu search
    if time.time() < deadline - 10:
        labels = _tabu_search(adj, labels, max_iter=2000, tabu_tenure=10)
    
    # Final greedy flip pass
    if time.time() < deadline - 5:
        labels = _greedy_flip_improve(adj, labels)
    
    return labels
# EVOLVE-BLOCK-END
