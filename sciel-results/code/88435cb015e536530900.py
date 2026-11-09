"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _cut_size_fast(adjacency, labels):
    """Faster cut size using matrix multiplication trick."""
    s = 2.0 * labels.astype(np.float64) - 1.0
    total_edges = adjacency.sum() / 2.0
    sAs = s @ adjacency @ s
    cut = (total_edges - sAs / 2.0)
    return int(round(cut))


def _spectral_partition(adjacency):
    """Use spectral method (Fiedler vector) to get initial partition."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    d = A.sum(axis=1)
    L = np.diag(d) - A
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return np.random.randint(0, 2, size=n), np.array([]), np.array([])
    
    idx = np.argsort(eigvals)
    fiedler = eigvecs[:, idx[1]]
    labels = (fiedler >= 0).astype(np.int32)
    
    if labels.sum() == 0 or labels.sum() == n:
        labels = np.random.randint(0, 2, size=n)
    
    return labels, eigvals, eigvecs


def _gwg_spectral_partition(adjacency, eigvals, eigvecs, n_hyperplanes=100):
    """
    Generate high-quality initial partitions using Goemans-Williamson style
    random hyperplane cuts from Laplacian eigenvectors.
    """
    n = len(adjacency)
    
    idx = np.argsort(eigvals)
    eigvals_sorted = eigvals[idx]
    eigvecs_sorted = eigvecs[:, idx]
    
    k = min(20, n - 1)
    embedding = eigvecs_sorted[:, 1:k+1]
    
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embedding_normalized = embedding / norms
    
    best_labels = None
    best_cut = -1
    
    A_int = adjacency.astype(np.int32)
    
    for _ in range(n_hyperplanes):
        normal = np.random.randn(k)
        normal /= (np.linalg.norm(normal) + 1e-10)
        
        projections = embedding_normalized @ normal
        labels = (projections >= 0).astype(np.int32)
        
        if labels.sum() == 0 or labels.sum() == n:
            continue
        
        cut = _cut_size_fast(adjacency, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels
    
    if best_labels is None:
        fiedler = eigvecs_sorted[:, 1]
        best_labels = (fiedler >= 0).astype(np.int32)
        if best_labels.sum() == 0 or best_labels.sum() == n:
            best_labels = np.random.randint(0, 2, size=n)
    
    return best_labels


def _random_labels(n):
    """Generate random initial labels."""
    return np.random.randint(0, 2, size=n)


def _local_search_fast(adjacency, labels, deadline, max_rounds=300):
    """
    Iterative local search with incremental cut tracking.
    Flipping vertex v changes cut by: 2 * (neighbors in opposite partition) - degree[v]
    """
    n = len(labels)
    labels = labels.copy().astype(np.int32)
    A = adjacency.astype(np.int32)
    degree = A.sum(axis=1)
    
    current_cut = _cut_size_fast(adjacency, labels)
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        # Shuffle order for diversity
        order = np.random.permutation(n)
        
        for v in order:
            if time.time() > deadline:
                break
            
            lv = labels[v]
            # Count neighbors in opposite partition
            opp_count = A[v, labels != lv].sum()
            
            # Delta if we flip v:
            # Currently v contributes: opp_count edges crossing (from v to opposite side)
            # After flip: same_count edges become crossing = degree[v] - opp_count
            # New contribution from v: degree[v] - opp_count
            # Delta = (degree[v] - opp_count) - opp_count = degree[v] - 2*opp_count
            delta = int(degree[v]) - 2 * int(opp_count)
            
            if delta > 0:
                labels[v] = 1 - lv
                current_cut += delta
                improved = True
        
        if not improved:
            break
    
    return labels, current_cut


def _simulated_annealing_fast(adjacency, labels, deadline, max_iterations=20000):
    """Simulated annealing with fast cut evaluation."""
    n = len(labels)
    labels = labels.copy().astype(np.int32)
    A = adjacency.astype(np.int32)
    degree = A.sum(axis=1)
    
    current_cut = _cut_size_fast(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    T_start = 5.0
    T_end = 0.01
    
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        
        progress = it / max_iterations
        T = T_start * (T_end / T_start) ** progress
        
        v = np.random.randint(n)
        old_label = labels[v]
        
        lv = old_label
        opp_count = A[v, labels != lv].sum()
        delta = int(degree[v]) - 2 * int(opp_count)
        
        if delta > 0 or (T > 0.01 and np.random.random() < np.exp(delta / T)):
            labels[v] = 1 - old_label
            current_cut += delta
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        else:
            # Don't change, but we already computed delta so no need to revert
            pass
    
    return best_labels, best_cut


def _walksat(adjacency, labels, deadline, max_iterations=30000, p_flip_best=0.75):
    """
    WalkSAT-style stochastic local search for Max-Cut.
    
    Formulates Max-Cut as Max-2-SAT where each edge (i,j) is a clause
    that is satisfied iff labels[i] != labels[j]. Unsatisfied clauses
    are edges within the same partition (uncut edges).
    
    Variable selection is biased toward vertices that appear in unsatisfied
    clauses (i.e., vertices with same-label neighbors). With probability
    p_flip_best, we flip the variable with best delta; otherwise we pick
    a random variable from the unsatisfied set (random walk to escape).
    """
    n = len(labels)
    labels = labels.copy().astype(np.int32)
    A = adjacency.astype(np.int32)
    degree = A.sum(axis=1)
    
    current_cut = _cut_size_fast(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    # Precompute neighbor lists for faster access
    neighbors = [np.where(A[v] > 0)[0] for v in range(n)]
    
    stall_count = 0
    stall_threshold = 2000
    
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Compute opp_count for all vertices (neighbors in opposite partition)
        # For vertex v: opp_count[v] = sum of A[v, u] where labels[u] != labels[v]
        # We can compute this efficiently:
        # For label-0 vertices: opp_count = A @ (labels==1)
        # For label-1 vertices: opp_count = A @ (labels==0)
        
        label1_mask = (labels == 1).astype(np.int32)
        label0_mask = (labels == 0).astype(np.int32)
        
        # Count neighbors in opposite partition for each vertex
        opp_counts = np.zeros(n, dtype=np.int32)
        opp_counts[labels == 0] = A[labels == 0] @ label1_mask
        opp_counts[labels == 1] = A[labels == 1] @ label0_mask
        
        # delta[v] = degree[v] - 2 * opp_count[v]
        # If delta > 0, flipping v increases the cut
        deltas = degree - 2 * opp_counts
        
        # Find vertices with positive delta (improving moves)
        improving_mask = deltas > 0
        
        if improving_mask.any():
            # With probability p_flip_best, pick best improving move
            # Otherwise pick a random improving move (diversity)
            if np.random.random() < p_flip_best:
                best_idx = np.where(improving_mask)[0]
                best_deltas = deltas[best_idx]
                v = best_idx[np.argmax(best_deltas)]
            else:
                candidates = np.where(improving_mask)[0]
                v = candidates[np.random.randint(len(candidates))]
            
            old_label = labels[v]
            labels[v] = 1 - old_label
            current_cut += int(deltas[v])
            stall_count = 0
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        else:
            # No improving move: random walk
            # Bias toward vertices in "unsatisfied" clauses (same-label neighbors)
            # A vertex is in an unsatisfied clause if it has any same-label neighbor
            # i.e., if opp_count[v] < degree[v] (it has at least one same-label neighbor)
            
            same_label_counts = degree - opp_counts  # neighbors with same label
            unsat_mask = same_label_counts > 0
            
            if unsat_mask.any():
                # Weight selection by number of unsatisfied clauses (same-label neighbors)
                weights = same_label_counts[unsat_mask].astype(np.float64)
                weights /= weights.sum()
                unsat_indices = np.where(unsat_mask)[0]
                v = unsat_indices[np.random.choice(len(unsat_indices), p=weights)]
            else:
                # All edges are cut - we're at the optimum!
                break
            
            old_label = labels[v]
            labels[v] = 1 - old_label
            current_cut += int(deltas[v])
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
            
            stall_count += 1
            
            # If stuck for too long, restart from best solution
            if stall_count > stall_threshold:
                labels = best_labels.copy()
                current_cut = best_cut
                stall_count = 0
    
    return best_labels, best_cut


def _polish(adjacency, labels, deadline):
    """Final high-precision polish: multiple passes of local search with random restarts of order."""
    n = len(labels)
    labels = labels.copy().astype(np.int32)
    A = adjacency.astype(np.int32)
    degree = A.sum(axis=1)
    
    current_cut = _cut_size_fast(adjacency, labels)
    
    # Do several full passes with different orderings
    for _ in range(20):
        if time.time() > deadline:
            break
        
        improved = False
        order = np.random.permutation(n)
        
        for v in order:
            if time.time() > deadline:
                break
            
            lv = labels[v]
            opp_count = A[v, labels != lv].sum()
            delta = int(degree[v]) - 2 * int(opp_count)
            
            if delta > 0:
                labels[v] = 1 - lv
                current_cut += delta
                improved = True
        
        if not improved:
            break
    
    return labels, current_cut


def _vectorized_local_search(adjacency, labels, deadline, max_rounds=100):
    """
    Vectorized local search that computes all gains at once and flips all positive gains.
    This is faster per round than sequential but may oscillate; use fewer rounds.
    """
    n = len(labels)
    labels = labels.copy().astype(np.int32)
    A = adjacency.astype(np.int32)
    degree = A.sum(axis=1).astype(np.int32)
    
    current_cut = _cut_size_fast(adjacency, labels)
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Compute gain for each vertex in one vectorized pass
        # gain[v] = degree[v] - 2 * opp_count[v]
        # opp_count[v] = number of neighbors with label != labels[v]
        
        # For each v, opp_count[v] = sum_j A[v,j] * (labels[j] != labels[v])
        # We can compute this as: for label 0 vertices, opp_count = sum of A[v, label1]
        # For label 1 vertices, opp_count = sum of A[v, label0]
        
        label0_mask = (labels == 0)
        label1_mask = (labels == 1)
        
        # For vertices with label 0: opp_count = A @ label1_mask
        # For vertices with label 1: opp_count = A @ label0_mask
        
        As_label1 = A @ label1_mask.astype(np.int32)
        As_label0 = A @ label0_mask.astype(np.int32)
        
        opp_counts = np.where(label0_mask, As_label1, As_label0)
        gains = degree - 2 * opp_counts
        
        # Flip all vertices with positive gain
        flip_mask = gains > 0
        if not flip_mask.any():
            break
        
        # Update cut: sum of gains for flipped vertices
        current_cut += int(gains[flip_mask].sum())
        labels[flip_mask] = 1 - labels[flip_mask]
    
    return labels, current_cut


def partition(adjacency):
    """Main partition function using spectral initialization, GW hyperplanes, local search, and SA."""
    n = len(adjacency)
    
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    # 1. Spectral partition (also get eigenvectors for GW)
    spec_labels, eigvals, eigvecs = _spectral_partition(adjacency)
    
    candidates = [spec_labels]
    
    # 2. GW-style spectral partitions with many random hyperplanes
    if time.time() < deadline - 5.0:
        gw_labels = _gwg_spectral_partition(adjacency, eigvals, eigvecs, n_hyperplanes=100)
        candidates.append(gw_labels)
    
    # 3. Random partitions
    for _ in range(4):
        candidates.append(_random_labels(n))
    
    # 4. Alternating
    candidates.append(np.arange(n) % 2)
    
    # 5. Degree-based: put high-degree nodes in one side initially
    d = adjacency.sum(axis=1)
    med = np.median(d)
    deg_labels = (d >= med).astype(np.int32)
    if deg_labels.sum() > 0 and deg_labels.sum() < n:
        candidates.append(deg_labels)
    
    # Run local search on each candidate
    best_labels = None
    best_cut = -1
    
    for cand in candidates:
        if time.time() > deadline - 30.0:
            break
        
        # Use vectorized local search first (faster per round)
        vls_labels, vls_cut = _vectorized_local_search(adjacency, cand, deadline - 25.0, max_rounds=50)
        
        # Then refine with sequential local search
        ls_labels, ls_cut = _local_search_fast(adjacency, vls_labels, deadline - 20.0, max_rounds=100)
        
        if ls_cut > best_cut:
            best_cut = ls_cut
            best_labels = ls_labels
    
    # If we have time, run simulated annealing from the best local search result
    if time.time() < deadline - 20.0 and best_labels is not None:
        sa_labels, sa_cut = _simulated_annealing_fast(adjacency, best_labels, deadline - 15.0)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
        
        # After SA, do a polish pass
        if time.time() < deadline - 8.0:
            final_labels, final_cut = _polish(adjacency, best_labels, deadline - 5.0)
            if final_cut > best_cut:
                best_cut = final_cut
                best_labels = final_labels
    
    # Run WalkSAT from the best solution
    if time.time() < deadline - 10.0 and best_labels is not None:
        ws_labels, ws_cut = _walksat(adjacency, best_labels, deadline - 5.0, max_iterations=25000)
        if ws_cut > best_cut:
            best_cut = ws_cut
            best_labels = ws_labels
        
        # Polish after WalkSAT
        if time.time() < deadline - 3.0:
            final_ws_labels, final_ws_cut = _polish(adjacency, best_labels, deadline - 1.0)
            if final_ws_cut > best_cut:
                best_cut = final_ws_cut
                best_labels = final_ws_labels
    
    # If we still have time, try from a different starting point
    if time.time() < deadline - 20.0:
        # Try from a random start with WalkSAT
        rand_start = _random_labels(n)
        vls2_labels, vls2_cut = _vectorized_local_search(adjacency, rand_start, deadline - 18.0, max_rounds=50)
        ls2_labels, ls2_cut = _local_search_fast(adjacency, vls2_labels, deadline - 15.0, max_rounds=100)
        if ls2_cut > best_cut:
            best_cut = ls2_cut
            best_labels = ls2_labels
        
        if time.time() < deadline - 10.0:
            ws2_labels, ws2_cut = _walksat(adjacency, best_labels, deadline - 5.0, max_iterations=15000)
            if ws2_cut > best_cut:
                best_cut = ws2_cut
                best_labels = ws2_labels
            
            if time.time() < deadline - 3.0:
                final2_labels, final2_cut = _polish(adjacency, best_labels, deadline - 1.0)
                if final2_cut > best_cut:
                    best_cut = final2_cut
                    best_labels = final2_labels
    
    # Ensure valid output
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
