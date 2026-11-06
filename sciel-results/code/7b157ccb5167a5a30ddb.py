"""Rewritten program for Maximum Cut task using SDP relaxation + diverse initializations + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adjacency, labels):
    """Compute the cut size (number of edges crossing the partition)."""
    labels_int = labels.astype(np.float64)
    degrees = adjacency.sum(axis=1)
    cut = np.dot(labels_int, degrees) - labels_int @ adjacency @ labels_int
    return cut


def _greedy_improve(adjacency, labels, deadline):
    """Perform greedy local search: flip vertices that improve the cut."""
    n = len(labels)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        order = list(range(n))
        random.shuffle(order)
        for v in order:
            if time.time() >= deadline:
                return labels
            same = np.sum(adjacency[v] & (labels == labels[v]))
            diff = np.sum(adjacency[v] & (labels != labels[v]))
            if diff < same:
                labels[v] = 1 - labels[v]
                improved = True
    return labels


def _simulated_annealing(adjacency, labels, deadline, start_temp=10.0, cooling=0.999):
    """Perform simulated annealing to escape local optima."""
    n = len(labels)
    temp = start_temp
    best_labels = labels.copy()
    best_cut = _cut_size(adjacency, labels)
    
    while time.time() < deadline and temp > 0.01:
        v = random.randint(0, n - 1)
        same = np.sum(adjacency[v] & (labels == labels[v]))
        diff = np.sum(adjacency[v] & (labels != labels[v]))
        delta = same - diff
        
        if delta > 0:
            labels[v] = 1 - labels[v]
            new_cut = _cut_size(adjacency, labels)
            if new_cut > best_cut:
                best_cut = new_cut
                best_labels = labels.copy()
        else:
            if random.random() < np.exp(delta / temp):
                labels[v] = 1 - labels[v]
        
        temp *= cooling
    
    return best_labels


def _spectral_init(adjacency):
    """Generate initial partition using the Fiedler vector."""
    n = len(adjacency)
    adj = adjacency.astype(np.float64)
    degrees = adj.sum(axis=1)
    L = np.diag(degrees) - adj
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        fiedler = eigvecs[:, 1]
    except:
        fiedler = np.random.randn(n)
    
    threshold = np.mean(fiedler)
    labels = (fiedler > threshold).astype(int)
    
    if labels.sum() == 0 or labels.sum() == n:
        threshold = np.median(fiedler)
        labels = (fiedler > threshold).astype(int)
        if labels.sum() == 0 or labels.sum() == n:
            labels = np.random.randint(0, 2, size=n)
    
    return labels


def _sdp_relaxation(adjacency, n, deadline, n_steps=2000):
    """
    Solve the Goemans-Williamson SDP relaxation via projected gradient ascent.
    Returns the relaxed matrix X where X[i,j] = v_i^T v_j.
    """
    adj = adjacency.astype(np.float64)
    degrees = adj.sum(axis=1)
    
    # Objective: max sum_{i,j} A[i,j] * (1 - X[i,j]) / 2
    #           = (1/2) * (sum A - sum A * X)
    # So we want to MINIMIZE sum A * X, i.e., maximize -sum A * X
    # Equivalently, maximize trace(W, X) where W = -A (off-diagonal)
    # Actually the standard GW objective is: max (1/4) * sum_{i,j} A[i,j] (1 - X[i,j])
    # = (1/4)(sum_{i,j} A[i,j] - sum_{i,j} A[i,j] X[i,j])
    # So we minimize sum_{i,j} A[i,j] X[i,j]
    
    # Gradient of sum A*X w.r.t. X is 2*A (symmetric), but we only care about the
    # direction. For projected gradient ascent on max -sum A*X:
    # gradient of f(X) = -sum A*X is -A (for the symmetric part, it's -2A but
    # we can just use -A as the gradient direction for the symmetric matrix)
    
    # Actually, let's be more careful. The objective is:
    # f(X) = (1/2) * sum_{i != j} A[i,j] (1 - X[i,j])
    # = (1/2) * (sum_{i != j} A[i,j] - sum_{i != j} A[i,j] X[i,j])
    # Gradient w.r.t. X[i,j] for i != j: -(1/2) * (A[i,j] + A[j,i]) = -A[i,j] (since A is symmetric)
    # For i == j: constraint X[i,i] = 1, so we project.
    
    # Projected gradient ascent: X <- project(X + lr * grad)
    # grad[i,j] = -A[i,j] for i != j, and we fix X[i,i] = 1
    
    # Initialize with random orthonormal vectors
    rng = np.random.RandomState(42)
    V = rng.randn(n, n)
    # Orthonormalize
    Q, R = np.linalg.qr(V)
    # Handle sign flips from QR
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1
    V = Q * signs
    
    X = V @ V.T  # This is PSD with X[i,i] = 1
    
    lr = 0.1
    best_X = X.copy()
    best_val = -np.sum(adj * X)
    
    for step in range(n_steps):
        if time.time() >= deadline:
            break
        
        # Gradient: for i != j, grad[i,j] = -A[i,j]; for i==j, 0 (will be projected)
        grad = -adj.copy()
        np.fill_diagonal(grad, 0)
        
        # Update
        X = X + lr * grad
        
        # Project onto {X PSD, X[i,i] = 1}
        # First, fix diagonal to 1
        np.fill_diagonal(X, 1.0)
        
        # Project onto PSD: eigen-decompose, clip negative eigenvalues
        try:
            eigvals, eigvecs = np.linalg.eigh(X)
            eigvals_clipped = np.maximum(eigvals, 0)
            X = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        except:
            pass
        
        # Re-fix diagonal after PSD projection
        np.fill_diagonal(X, 1.0)
        
        # Symmetrize
        X = (X + X.T) / 2.0
        np.fill_diagonal(X, 1.0)
        
        # Evaluate
        val = -np.sum(adj * X)
        if val > best_val:
            best_val = val
            best_X = X.copy()
        
        # Adaptive learning rate
        if step % 100 == 99:
            lr *= 0.9
    
    return best_X


def _hyperplane_rounding(X, n, n_rounds=50):
    """
    Apply randomized hyperplane rounding to the SDP solution.
    Returns a list of label arrays.
    """
    labels_list = []
    # Get the matrix square root for vector representation
    try:
        eigvals, eigvecs = np.linalg.eigh(X)
        eigvals_clipped = np.maximum(eigvals, 0)
        V = eigvecs @ np.diag(np.sqrt(eigvals_clipped))
    except:
        V = np.random.randn(n, n)
        Q, R = np.linalg.qr(V)
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1
        V = Q * signs
    
    for _ in range(n_rounds):
        # Random unit vector
        z = np.random.randn(n)
        z /= np.linalg.norm(z)
        
        # Project vertices onto z
        projections = V.T @ z
        
        # Label by sign
        labels = (projections > 0).astype(int)
        
        # Ensure both sides non-empty
        if labels.sum() == 0 or labels.sum() == n:
            labels = np.random.randint(0, 2, size=n)
        
        labels_list.append(labels)
    
    return labels_list


def _pairwise_swap_search(adjacency, labels, deadline):
    """Try swapping pairs of vertices to improve the cut."""
    n = len(labels)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                return labels
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    continue
                # Check if swapping i and j improves
                # Current contribution of i and j to cut
                # After swap: i gets j's label, j gets i's label
                # Delta for i: (neighbors of i with label 1-labels[i]) - (neighbors of i with label labels[i])
                #   but excluding j since j also changes
                # This is complex; let's just compute directly for small n
                # For 100 nodes, O(n^2) per iteration is 10000, which is fine
                new_labels = labels.copy()
                new_labels[i], new_labels[j] = labels[j], labels[i]
                old_cut = _cut_size(adjacency, labels)
                new_cut = _cut_size(adjacency, new_labels)
                if new_cut > old_cut:
                    labels[i], labels[j] = labels[j], labels[i]
                    improved = True
                    break
            if improved:
                break
    return labels


def partition(adjacency):
    """
    Solve Maximum Cut using SDP relaxation, spectral init, greedy, and SA.
    """
    start_time = time.time()
    DEADLINE = start_time + 115.0
    
    n = len(adjacency)
    adjacency = np.array(adjacency, dtype=bool)
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Spectral initialization + greedy refinement
    spectral_labels = _spectral_init(adjacency)
    refined = _greedy_improve(adjacency, spectral_labels.copy(), DEADLINE)
    cut = _cut_size(adjacency, refined)
    best_labels = refined.copy()
    best_cut = cut
    
    # Phase 2: SDP relaxation + hyperplane rounding + greedy + SA
    remaining = DEADLINE - time.time()
    if remaining > 10.0:
        sdp_deadline = start_time + 30.0  # Give SDP up to 30s
        X = _sdp_relaxation(adjacency, n, sdp_deadline, n_steps=3000)
        
        # Multiple hyperplane roundings
        rounded_labels_list = _hyperplane_rounding(X, n, n_rounds=20)
        
        for rl in rounded_labels_list:
            if time.time() >= DEADLINE:
                break
            refined = _greedy_improve(adjacency, rl.copy(), DEADLINE)
            cut = _cut_size(adjacency, refined)
            if cut > best_cut:
                best_cut = cut
                best_labels = refined.copy()
    
    # Phase 3: Simulated annealing from best solution
    remaining = DEADLINE - time.time()
    if remaining > 5.0:
        sa_deadline = start_time + min(remaining - 2.0, 40.0)
        sa_labels = _simulated_annealing(adjacency, best_labels.copy(), sa_deadline, 
                                         start_temp=5.0, cooling=0.9995)
        sa_cut = _cut_size(adjacency, sa_labels)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    # Phase 4: Multiple random restarts with greedy + short SA
    num_restarts = 0
    while time.time() < DEADLINE:
        num_restarts += 1
        
        labels = np.random.randint(0, 2, size=n)
        refined = _greedy_improve(adjacency, labels.copy(), DEADLINE)
        cut = _cut_size(adjacency, refined)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = refined.copy()
        
        remaining = DEADLINE - time.time()
        if remaining > 5.0:
            sa_labels = _simulated_annealing(adjacency, refined.copy(), DEADLINE - 2.0,
                                             start_temp=3.0, cooling=0.999)
            sa_cut = _cut_size(adjacency, sa_labels)
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()
        
        if num_restarts >= 30:
            break
    
    # Final polish: one more greedy pass
    if time.time() < DEADLINE - 1.0:
        final = _greedy_improve(adjacency, best_labels.copy(), DEADLINE - 0.5)
        final_cut = _cut_size(adjacency, final)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
