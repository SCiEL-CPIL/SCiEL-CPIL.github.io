"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian


def partition(adjacency):
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
    n = adjacency.shape[0]
    
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s kill
    
    adj = adjacency.astype(np.int8)
    degrees = adj.sum(axis=1)
    total_edges = int(adj.sum()) // 2
    
    # Build sparse adjacency for fast neighbor lookups
    adj_sparse = csr_matrix(adj)
    # Precompute neighbor lists for faster iteration
    neighbors_list = [np.nonzero(adj[i])[0] for i in range(n)]
    
    def cut_size(labels):
        """Compute cut size efficiently."""
        opp = (adj @ (1 - labels))
        return int(np.sum(opp)) // 2
    
    def improve_single(labels, max_passes=200):
        """Fast local search using incremental gain updates with precomputed neighbors."""
        labels = labels.copy()
        
        adj_labels = adj @ labels
        opp_side = labels * degrees + (1 - 2 * labels) * adj_labels
        gain = degrees - 2 * opp_side
        
        for _pass in range(max_passes):
            if time.time() >= deadline - 2.0:
                break
            improved = False
            for v in range(n):
                if gain[v] > 0:
                    old_label = labels[v]
                    labels[v] = 1 - old_label
                    gain[v] = -gain[v]
                    
                    for u in neighbors_list[v]:
                        if labels[u] == old_label:
                            gain[u] -= 2
                        else:
                            gain[u] += 2
                    
                    improved = True
            
            if not improved:
                break
        
        return labels
    
    def fiedler_vector(adj_matrix):
        """Compute Fiedler vector (second eigenvector of Laplacian) for spectral initialization."""
        L = laplacian(csr_matrix(adj_matrix.astype(np.float64)))
        try:
            from scipy.sparse.linalg import eigsh
            # Find smallest non-zero eigenvalue
            # For connected graphs, eigenvalues are 0 < lambda_1 <= lambda_2 <= ...
            # We want the eigenvector for lambda_1 (Fiedler vector)
            # Use shift-invert to find smallest eigenvalues
            vals, vecs = eigsh(L, k=2, sigma=0.001, which='LM')
            # Sort by eigenvalue
            idx = np.argsort(vals)
            # Fiedler vector is the second smallest eigenvalue's eigenvector
            fiedler = vecs[:, idx[1]]
            # Normalize to 0/1
            labels = (fiedler > 0).astype(int)
            return labels
        except Exception:
            return None
    
    def perturb(labels, num_flips=5):
        """Randomly flip a few vertices to escape local optima."""
        new_labels = labels.copy()
        for _ in range(num_flips):
            v = np.random.randint(n)
            new_labels[v] = 1 - new_labels[v]
        return new_labels
    
    def simulated_annealing(labels, max_iters=5000):
        """SA polish to escape local optima."""
        labels = labels.copy()
        current_cut = cut_size(labels)
        best_cut = current_cut
        best_labels = labels.copy()
        
        T = 5.0
        cooling = 0.999
        
        adj_labels = adj @ labels
        opp_side = labels * degrees + (1 - 2 * labels) * adj_labels
        gain = degrees - 2 * opp_side
        
        for it in range(max_iters):
            if time.time() >= deadline - 2.0:
                break
            
            v = np.random.randint(n)
            if gain[v] > 0:
                # Always accept improving move
                old_label = labels[v]
                labels[v] = 1 - old_label
                gain[v] = -gain[v]
                
                for u in neighbors_list[v]:
                    if labels[u] == old_label:
                        gain[u] -= 2
                    else:
                        gain[u] += 2
                
                current_cut += gain[v] // 2  # approximate, recompute if needed
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            else:
                # Accept worsening move with probability exp(-delta/T)
                delta = -gain[v]  # positive value we'd lose
                if delta > 0:
                    prob = np.exp(-delta / T)
                    if np.random.random() < prob:
                        old_label = labels[v]
                        labels[v] = 1 - old_label
                        gain[v] = -gain[v]
                        
                        for u in neighbors_list[v]:
                            if labels[u] == old_label:
                                gain[u] -= 2
                            else:
                                gain[u] += 2
                        
                        current_cut -= delta // 2
                else:
                    # gain[v] == 0, accept with some probability
                    if np.random.random() < 0.1:
                        old_label = labels[v]
                        labels[v] = 1 - old_label
                        gain[v] = -gain[v]
                        
                        for u in neighbors_list[v]:
                            if labels[u] == old_label:
                                gain[u] -= 2
                            else:
                                gain[u] += 2
            
            T *= cooling
        
        return best_labels, best_cut
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Spectral initialization
    spectral_labels = fiedler_vector(adj)
    if spectral_labels is not None:
        labels = improve_single(spectral_labels)
        c = cut_size(labels)
        if c > best_cut:
            best_cut = c
            best_labels = labels.copy()
    
    # Phase 2: Multiple restarts with different initializations
    num_restarts = 25
    for restart in range(num_restarts):
        if time.time() >= deadline - 15.0:
            break
        
        if restart == 0:
            labels = np.arange(n) % 2
        elif restart == 1:
            labels = np.ones(n, dtype=int)
        elif restart == 2:
            labels = np.zeros(n, dtype=int)
        elif restart < 5 and spectral_labels is not None:
            # Perturbed spectral
            labels = perturb(spectral_labels, num_flips=restart)
        else:
            # Random with varying bias
            p = 0.5 + 0.3 * np.sin(restart * 0.7)
            labels = (np.random.random(n) < p).astype(int)
        
        # Single-vertex local search
        labels = improve_single(labels)
        c = cut_size(labels)
        if c > best_cut:
            best_cut = c
            best_labels = labels.copy()
        
        # SA polish if time permits
        if time.time() < deadline - 10.0:
            sa_labels, sa_cut = simulated_annealing(labels, max_iters=3000)
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()
        
        if best_cut >= total_edges:
            break
    
    # Final polish: SA on best solution
    if best_labels is not None and time.time() < deadline - 5.0:
        final_labels, final_cut = simulated_annealing(best_labels, max_iters=10000)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels
        
        # One more single-flip pass
        if time.time() < deadline - 2.0:
            final_labels2 = improve_single(best_labels, max_passes=300)
            final_cut2 = cut_size(final_labels2)
            if final_cut2 > best_cut:
                best_cut = final_cut2
                best_labels = final_labels2
    
    return best_labels if best_labels is not None else np.arange(n) % 2
# EVOLVE-BLOCK-END
