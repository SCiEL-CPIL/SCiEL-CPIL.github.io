"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import random


def partition(adjacency):
    """
    Solve Maximum Cut using simulated annealing with multi-start and
    local search refinement.
    
    Args:
        adjacency: bool matrix (n x n) representing the graph.
        
    Returns:
        np.array of 0/1 labels per vertex.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for efficient operations
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Precompute degrees
    degrees = adj.sum(axis=1)
    
    def cut_size(labels):
        """Compute the number of edges crossing the cut."""
        # Edges crossing: one endpoint 0, other 1
        # Sum over all edges where labels differ
        # Efficient: for each vertex, count neighbors with different label
        # cut = sum_v (sum_u adj[v,u] * (labels[v] != labels[u])) / 2
        # But we can just compute sum of adj[v, labels != labels[v]] for all v / 2
        # Actually, let's use a vectorized approach
        # For each vertex v, the number of crossing edges incident to v is:
        #   if labels[v] == 0: sum(adj[v, labels == 1])
        #   if labels[v] == 1: sum(adj[v, labels == 0])
        # Total cut size = sum of these / 2 (since each edge counted twice)
        
        # Vectorized:
        # crossing[v] = sum_u adj[v,u] * (1 - labels[v]*labels[u] - (1-labels[v])*(1-labels[u]))
        # = sum_u adj[v,u] * (labels[v] != labels[u])
        # Let's use: labels_diff = 1 - 2*abs(labels - 0.5) ... no
        # Simpler: for binary labels, labels[v] != labels[u] is 1 - labels[v]*labels[u] - (1-labels[v])*(1-labels[u])
        # = 1 - labels[v]*labels[u] - 1 + labels[v] + labels[u] - labels[v]*labels[u]
        # = labels[v] + labels[u] - 2*labels[v]*labels[u]
        # So crossing[v] = sum_u adj[v,u] * (labels[v] + labels[u] - 2*labels[v]*labels[u])
        # = labels[v] * sum_u adj[v,u] + sum_u adj[v,u]*labels[u] - 2*labels[v]*sum_u adj[v,u]*labels[u]
        # = labels[v] * degrees[v] + (adj @ labels)[v] - 2*labels[v]*(adj @ labels)[v]
        
        adj_labels = adj @ labels  # shape (n,)
        crossing = labels * degrees + adj_labels - 2 * labels * adj_labels
        return int(np.sum(crossing) / 2)
    
    def flip_gain(v, labels):
        """
        Compute the change in cut size if we flip vertex v.
        If we flip v, the change is:
        new_cut - old_cut
        For vertex v, let s = sum of labels of neighbors (adj_labels[v])
        Let d = degrees[v]
        If labels[v] = 0:
          Currently contributes: s (since labels[v]=0, crossing from v's perspective = s)
          After flip to 1: contributes d - s
          Change = (d - s) - s = d - 2*s
        If labels[v] = 1:
          Currently contributes: d - s
          After flip to 0: contributes s
          Change = s - (d - s) = 2*s - d = -(d - 2*s)
        So in general, delta = (2*labels[v] - 1) * (2*adj_labels[v] - degrees[v])
        Wait, let me re-derive.
        
        Let L = labels[v], S = sum_{u~v} labels[u] = adj_labels[v], D = degrees[v]
        Contribution to cut from v's incident edges (counted once per edge from v's side):
        If L=0: S edges cross (neighbors with label 1)
        If L=1: D-S edges cross (neighbors with label 0)
        
        If we flip L to 1-L:
        If L=0 -> 1: new contribution = D-S, old = S, delta = D-S-S = D-2S
        If L=1 -> 0: new contribution = S, old = D-S, delta = S-(D-S) = 2S-D
        
        So delta = (1-2L)*... no.
        If L=0: delta = D - 2S
        If L=1: delta = 2S - D = -(D - 2S)
        So delta = (1 - 2*L) * (D - 2*S)
        """
        adj_labels = adj @ labels
        s = adj_labels[v]
        d = degrees[v]
        l = labels[v]
        return (1 - 2 * l) * (d - 2 * s)
    
    def local_search(labels, max_iters=10000):
        """Greedy local search: keep flipping vertices that improve the cut."""
        labels = labels.copy()
        current_cut = cut_size(labels)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try each vertex
            for v in range(n):
                delta = flip_gain(v, labels)
                if delta > 0:
                    labels[v] = 1 - labels[v]
                    current_cut += delta
                    improved = True
        return labels, current_cut
    
    def simulated_annealing(labels, t_start=1.0, t_end=0.01, cooling=0.999, max_iters=50000):
        """Simulated annealing to escape local optima."""
        labels = labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        t = t_start
        
        for _ in range(max_iters):
            if t < t_end:
                break
            # Pick a random vertex
            v = random.randint(0, n - 1)
            delta = flip_gain(v, labels)
            if delta > 0 or random.random() < np.exp(delta / t):
                labels[v] = 1 - labels[v]
                current_cut += delta
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            t *= cooling
        
        return best_labels, best_cut
    
    # Multi-start with different initializations
    best_labels = None
    best_cut = -1
    
    # Start 1: Random initialization
    for trial in range(5):
        if trial == 0:
            # Greedy initial: assign each vertex to the side with fewer same-side neighbors
            labels = np.zeros(n, dtype=int)
            order = list(range(n))
            random.shuffle(order)
            for v in order:
                # Count neighbors already assigned
                neighbor_labels = labels[adj[v] == 1]
                if len(neighbor_labels) == 0:
                    labels[v] = random.randint(0, 1)
                else:
                    same_0 = np.sum(neighbor_labels == 0)
                    same_1 = np.sum(neighbor_labels == 1)
                    labels[v] = 0 if same_0 > same_1 else 1
        else:
            # Random initialization
            labels = np.random.randint(0, 2, n)
        
        # Local search
        labels_ls, cut_ls = local_search(labels, max_iters=10000)
        
        # Simulated annealing from the local optimum
        labels_sa, cut_sa = simulated_annealing(labels_ls, t_start=0.5, t_end=0.01, cooling=0.9995, max_iters=30000)
        
        # Final local search
        labels_final, cut_final = local_search(labels_sa, max_iters=10000)
        
        if cut_final > best_cut:
            best_cut = cut_final
            best_labels = labels_final.copy()
    
    # One more round of local search to ensure we're at a local optimum
    best_labels, _ = local_search(best_labels, max_iters=10000)
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
