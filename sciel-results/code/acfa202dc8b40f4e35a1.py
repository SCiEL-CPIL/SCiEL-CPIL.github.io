"""
Maximum Cut solver using spectral/SDP-inspired initialization + multi-start local search
with pairwise swaps, plus simulated annealing for escaping local optima.
"""
import numpy as np
import time


def partition(adjacency):
    """
    Find a maximum cut in an unweighted graph using:
    1. Spectral (Fiedler vector) initialization to get high-quality starting points
    2. Multiple diverse random restarts
    3. Greedy single-flip local search
    4. Pairwise swap local search
    5. Simulated annealing for escaping local optima
    6. Explicit deadline guard to ensure we finish in time
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)

    if n == 0:
        return np.array([], dtype=np.int8)
    if n == 1:
        return np.array([0], dtype=np.int8)

    # Precompute degrees
    degrees = adj.sum(axis=1).astype(np.int32)
    total_edges = int(degrees.sum()) // 2

    def compute_cut(labels):
        """Compute cut size efficiently using vectorized operations."""
        # Edge (i,j) is in cut if labels[i] != labels[j]
        # For each vertex, count neighbors with different label, sum, divide by 2
        diff = (labels[:, None] != labels[None, :])
        return int((diff & adj).sum() // 2)

    def greedy_flip_improve(labels, max_passes=100):
        """Greedy local improvement: flip vertices if it increases cut size."""
        labels = labels.copy()
        improved = True
        passes = 0
        while improved and passes < max_passes:
            if time.time() > deadline:
                break
            improved = False
            passes += 1
            for v in range(n):
                # Count neighbors in same group vs other group
                same_mask = labels == labels[v]
                same_count = int(adj[v][same_mask].sum())
                other_count = int(adj[v][~same_mask].sum())
                if same_count > other_count:
                    labels[v] = 1 - labels[v]
                    improved = True
        return labels

    def pairwise_swap_search(labels, max_passes=20):
        """Try swapping pairs of vertices to improve cut size."""
        labels = labels.copy()
        improved = True
        passes = 0
        while improved and passes < max_passes:
            if time.time() > deadline:
                break
            improved = False
            passes += 1
            # Try all pairs where swapping improves the cut
            # For efficiency, only try pairs with different labels
            group0 = np.where(labels == 0)[0]
            group1 = np.where(labels == 1)[0]
            if len(group0) == 0 or len(group1) == 0:
                break
            # Sample some pairs to avoid O(n^2) full search
            n_samples = min(len(group0) * len(group1), 2000)
            if n_samples <= 0:
                break
            for _ in range(n_samples):
                if time.time() > deadline:
                    break
                i = group0[np.random.randint(len(group0))]
                j = group1[np.random.randint(len(group1))]
                # Compute gain of swapping i and j
                # Current contribution of i: edges from i to same group (waste)
                # After swap: i goes to group 1, j goes to group 0
                # Gain = change in cut size
                # For vertex i moving from 0 to 1:
                #   loses cut edges to group 1 neighbors, gains cut edges to group 0 neighbors (excluding j)
                # For vertex j moving from 1 to 0:
                #   loses cut edges to group 0 neighbors, gains cut edges to group 1 neighbors (excluding i)
                # Net gain = (neighbors of i in group 0 excluding j) - (neighbors of i in group 1 excluding j)
                #          + (neighbors of j in group 1 excluding i) - (neighbors of j in group 0 excluding i)
                #          - 2 * adj[i,j] (the edge between i and j changes from cut to non-cut or vice versa)
                # Actually let's just compute directly
                old_cut = compute_cut(labels)
                labels[i], labels[j] = labels[j], labels[i]
                new_cut = compute_cut(labels)
                if new_cut > old_cut:
                    improved = True
                else:
                    labels[i], labels[j] = labels[j], labels[i]
        return labels

    def spectral_init():
        """
        Use the Fiedler vector (second eigenvector of the Laplacian) to create
        a high-quality initial partition. This is inspired by the Goemans-Williamson
        SDP relaxation where the optimal solution lies in the span of eigenvectors.
        """
        # Compute the graph Laplacian L = D - A
        L = np.diag(degrees.astype(np.float64)) - adj.astype(np.float64)
        
        try:
            # Get eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
            fiedler = eigenvectors[:, 1]
            
            # Use median split of the Fiedler vector
            median_val = np.median(fiedler)
            labels = (fiedler > median_val).astype(np.int8)
            
            # Ensure both groups are non-empty
            if labels.sum() == 0 or labels.sum() == n:
                labels = np.random.randint(0, 2, size=n)
            
            return labels
        except Exception:
            return np.random.randint(0, 2, size=n)

    def sdp_hyperplane_init():
        """
        Approximate SDP-based initialization using spectral embedding.
        Project onto a random hyperplane in the embedded space.
        """
        # Compute Laplacian
        L = np.diag(degrees.astype(np.float64)) - adj.astype(np.float64)
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            
            # Use the first few non-trivial eigenvectors for embedding
            # Skip the first eigenvector (constant)
            k = min(5, n - 1)
            embedding = eigenvectors[:, 1:1+k]
            
            # Random hyperplane projection
            random_vec = np.random.randn(k)
            projections = embedding @ random_vec
            median_val = np.median(projections)
            labels = (projections > median_val).astype(np.int8)
            
            if labels.sum() == 0 or labels.sum() == n:
                labels = np.random.randint(0, 2, size=n)
            
            return labels
        except Exception:
            return np.random.randint(0, 2, size=n)

    def simulated_annealing(labels, max_iter=3000):
        """Simulated annealing to escape local optima."""
        current = labels.copy()
        current_cut = compute_cut(current)
        best = current.copy()
        best_cut = current_cut
        
        T = 2.0
        T_min = 0.05
        cooling = 0.9995
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random vertex to flip
            v = np.random.randint(n)
            new_labels = current.copy()
            new_labels[v] = 1 - new_labels[v]
            new_cut = compute_cut(new_labels)
            
            delta = new_cut - current_cut
            
            if delta >= 0:
                current = new_labels
                current_cut = new_cut
                if current_cut > best_cut:
                    best = current.copy()
                    best_cut = current_cut
            else:
                if np.random.random() < np.exp(delta / T):
                    current = new_labels
                    current_cut = new_cut
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best, best_cut

    # Strategy: multiple diverse initializations + local search + optional annealing
    best_labels = None
    best_cut = -1

    def try_candidate(labels):
        nonlocal best_labels, best_cut
        if time.time() > deadline:
            return
        labels = greedy_flip_improve(labels)
        c = compute_cut(labels)
        if c > best_cut:
            best_cut = c
            best_labels = labels.copy()

    # 1. Spectral initialization
    try_candidate(spectral_init())

    # 2. SDP-inspired hyperplane initialization
    try_candidate(sdp_hyperplane_init())

    # 3. Multiple random restarts
    n_restarts = 30
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        labels = np.random.randint(0, 2, size=n)
        try_candidate(labels)

    # 4. Biased random restarts (slightly favor one group)
    for restart in range(10):
        if time.time() > deadline - 10:
            break
        p = np.random.uniform(0.3, 0.7)
        labels = (np.random.random(n) < p).astype(np.int8)
        try_candidate(labels)

    # 5. Pairwise swap search from best
    if best_labels is not None and time.time() < deadline - 15:
        swapped = pairwise_swap_search(best_labels)
        c = compute_cut(swapped)
        if c > best_cut:
            best_cut = c
            best_labels = swapped.copy()

    # 6. Simulated annealing from best solution
    if best_labels is not None and time.time() < deadline - 20:
        sa_labels, sa_cut = simulated_annealing(best_labels, max_iter=5000)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
        # Final greedy polish after SA
        if time.time() < deadline - 5:
            polished = greedy_flip_improve(best_labels)
            pc = compute_cut(polished)
            if pc > best_cut:
                best_cut = pc
                best_labels = polished.copy()

    # 7. One more round of pairwise swaps if time permits
    if best_labels is not None and time.time() < deadline - 5:
        swapped = pairwise_swap_search(best_labels, max_passes=10)
        c = compute_cut(swapped)
        if c > best_cut:
            best_cut = c
            best_labels = swapped.copy()

    # Final greedy polish
    if best_labels is not None and time.time() < deadline - 1:
        final_labels = greedy_flip_improve(best_labels)
        final_cut = compute_cut(final_labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels

    if best_labels is None:
        # Fallback: alternating pattern
        best_labels = np.arange(n) % 2

    return best_labels.astype(np.int8)
