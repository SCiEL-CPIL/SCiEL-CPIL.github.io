"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition using vectorized ops."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _spectral_partition(adjacency):
    """Get an initial partition using spectral method (Fiedler vector)."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    D = A.sum(axis=1)
    L = np.diag(D) - A
    eigvals, eigvecs = np.linalg.eigh(L)
    fiedler = eigvecs[:, 1]
    labels = (fiedler < 0).astype(np.int64)
    return labels


def _vectorized_local_search(A_float, labels, max_passes=20):
    """Vectorized greedy local search: flip all improving vertices simultaneously per pass."""
    n = len(labels)
    labels = labels.copy()
    
    for _ in range(max_passes):
        # For each vertex v, compute gain of flipping:
        # gain[v] = (# neighbors on same side) - (# neighbors on other side)
        # = 2 * (sum of A[v,:]*labels) - (sum of A[v,:])  ... wait
        # If labels[v]=0: same side = neighbors with label 0, other side = label 1
        #   count_same = A[v,:] @ (labels==0) but excluding self... 
        # Let's use: s = 2*labels - 1 (so -1 for label 0, +1 for label 1)
        # For vertex v with sign s[v]:
        #   neighbors on same side: A[v,:] @ (s == s[v]) 
        #   neighbors on other side: A[v,:] @ (s != s[v])
        #   gain = count_same - count_other = 2*count_same - degree[v]
        
        s = 2.0 * labels.astype(np.float64) - 1.0  # -1 or +1
        # For each v: count_same[v] = sum_j A[v,j] * (s[j] == s[v])
        # = sum_j A[v,j] * (1 + s[v]*s[j])/2
        # = (degree[v] + s[v] * (A @ s)[v]) / 2
        As = A_float @ s  # A @ s
        degrees = A_float.sum(axis=1)
        count_same = (degrees + s * As) / 2.0
        count_other = degrees - count_same
        gain = count_same - count_other  # = 2*count_same - degrees
        
        # Flip all vertices with positive gain
        flip_mask = gain > 0
        if not np.any(flip_mask):
            break
        labels[flip_mask] = 1 - labels[flip_mask]
    
    return labels


def _simulated_annealing(A_float, labels, n_steps=5000, t_start=1.0, t_end=0.01, seed=None):
    """Simulated annealing for Max-Cut with incremental cut tracking."""
    n = len(labels)
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    
    labels = labels.copy()
    s = 2.0 * labels.astype(np.float64) - 1.0  # -1 or +1
    
    # Current cut size: 0.5 * (total_edges - s^T A s)
    # where total_edges = sum(A) / 2 for undirected... actually for 0/1 adjacency (symmetric),
    # s^T A s = sum_{i,j} s[i]*A[i,j]*s[j] = 2*sum_{i<j} A[i,j]*s[i]*s[j]
    # cut = sum_{i<j} A[i,j] * (1 - s[i]*s[j])/2 = (total_weighted - s^T A s) / 4
    # where total_weighted = sum_{i,j} A[i,j] = 2 * num_edges
    total_weighted = A_float.sum()
    current_cut = (total_weighted - s @ (A_float @ s)) / 4.0
    
    best_labels = labels.copy()
    best_cut = current_cut
    
    # Precompute degrees
    degrees = A_float.sum(axis=1)
    
    for step in range(n_steps):
        # Adaptive temperature
        progress = step / max(n_steps - 1, 1)
        t = t_start * (t_end / t_start) ** progress
        
        # Pick a random vertex to flip
        v = rng.randint(0, n - 1)
        
        # Gain of flipping v: 
        # Before flip: contribution to s^T A s from row v and col v
        # After flip: s[v] -> -s[v]
        # Delta(s^T A s) = -2 * s[v] * (A @ s)[v]  ... let's derive:
        # s^T A s = sum_i sum_j s[i] A[i,j] s[j]
        # Flipping s[v] -> -s[v]:
        # New: sum_i sum_j s'[i] A[i,j] s'[j] where s'[v] = -s[v], s'[i]=s[i] for i!=v
        # = old - 2*s[v]*(sum_{j!=v} A[v,j]*s[j] + sum_{i!=v} A[i,v]*s[i]) + 2*A[v,v]*s[v]^2
        # Since A is symmetric and A[v,v]=0:
        # = old - 4*s[v]*(A @ s)[v] + 2*A[v,v]*s[v]^2
        # With A[v,v]=0: delta = -4*s[v]*(A @ s)[v]
        # cut_new = (total - s_new^T A s_new)/4 = cut_old - delta/4 = cut_old + s[v]*(A@s)[v]
        
        As_v = (A_float @ s)[v]
        delta_cut = s[v] * As_v  # change in cut if we flip v
        
        if delta_cut > 0 or rng.random() < np.exp(delta_cut / max(t, 1e-10)):
            # Accept the flip
            s[v] = -s[v]
            labels[v] = 1 - labels[v]
            current_cut += delta_cut
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
    
    return best_labels, best_cut


def partition(adjacency):
    """
    Partition vertices into two groups to maximize the number of cut edges.
    Uses spectral initialization, vectorized local search, and multi-start SA.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (5s buffer before 120s limit)
    
    n = len(adjacency)
    A_bool = np.array(adjacency, dtype=bool)
    A_float = A_bool.astype(np.float64)
    
    random.seed(42)
    np.random.seed(42)
    
    # Phase 1: Spectral initialization + vectorized local search
    labels = _spectral_partition(A_bool)
    labels = _vectorized_local_search(A_float, labels, max_passes=30)
    best_labels = labels.copy()
    best_cut = _cut_size(A_bool, labels)
    
    # Collect top solutions from multiple spectral-like starts
    candidates = [(best_cut, best_labels.copy())]
    
    # Try a few random initializations + local search to get diverse starts
    for trial in range(3):
        if time.time() > deadline - 30.0:
            break
        rand_labels = np.random.randint(0, 2, size=n).astype(np.int64)
        ls_labels = _vectorized_local_search(A_float, rand_labels, max_passes=30)
        ls_cut = _cut_size(A_bool, ls_labels)
        candidates.append((ls_cut, ls_labels))
    
    # Sort by cut size descending, take top 3
    candidates.sort(key=lambda x: -x[0])
    top_candidates = candidates[:3]
    
    # Phase 2: Multi-start simulated annealing from top candidates
    sa_configs = [
        (3000, 1.0, 0.01),
        (5000, 0.8, 0.005),
        (8000, 0.6, 0.002),
    ]
    
    for ci, (cand_cut, cand_labels) in enumerate(top_candidates):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb the candidate
        perturbed = cand_labels.copy()
        num_flips = max(1, n // 8)
        flip_indices = random.sample(range(n), num_flips)
        for idx in flip_indices:
            perturbed[idx] = 1 - perturbed[idx]
        
        # Run SA with different configs
        for si, (n_steps, t_start, t_end) in enumerate(sa_configs):
            if time.time() > deadline - 5.0:
                break
            
            sa_labels, sa_cut = _simulated_annealing(
                A_float, perturbed, n_steps=n_steps, t_start=t_start, t_end=t_end,
                seed=42 + ci * 100 + si
            )
            
            # Polish with vectorized local search
            polished = _vectorized_local_search(A_float, sa_labels, max_passes=20)
            pol_cut = _cut_size(A_bool, polished)
            
            if pol_cut > best_cut:
                best_cut = pol_cut
                best_labels = polished.copy()
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 2.0:
        final_labels = _vectorized_local_search(A_float, best_labels, max_passes=50)
        final_cut = _cut_size(A_bool, final_labels)
        if final_cut > best_cut:
            best_labels = final_labels
            best_cut = final_cut
    
    return best_labels
# EVOLVE-BLOCK-END
