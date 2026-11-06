# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def partition(adjacency):
    """
    Maximum Cut via Spectral Initialization + Multi-Start Local Search with Goemans-Williamson style rounding.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)

    # Convert to numpy array for fast operations
    adj = np.asarray(adjacency, dtype=np.float64)
    E = 0.5 * adj.sum()  # total number of edges

    # Deadline guard: 140 seconds (10s safety margin from 150s limit)
    deadline = time.time() + 140.0

    def get_cut_size(labels):
        """Compute the number of edges crossing the partition."""
        Z = 2.0 * labels - 1.0
        temp = adj @ Z
        zadjz = Z @ temp
        cut = 0.25 * (2.0 * E - zadjz)
        return cut

    def local_search(labels, max_iterations=100):
        """Greedy local search: flip a vertex if it improves the cut."""
        labels = labels.copy()
        current_cut = get_cut_size(labels)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 2.0:
                break
            improved = False
            for v in range(n):
                same_label = (labels == labels[v])
                diff_label = (labels != labels[v])
                
                same_count = adj[v] @ same_label
                diff_count = adj[v] @ diff_label
                
                gain = same_count - diff_count
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    current_cut += gain
                    improved = True
            
            if not improved:
                break
        
        return labels, current_cut

    def goemans_maxwell_rounding(fiedler):
        """
        Goemans-Williamson style rounding: use a random hyperplane to round the Fiedler vector.
        For each vertex, assign label based on sign of dot product with random vector.
        We try multiple random directions and keep the best.
        """
        best_labels = None
        best_cut = -1
        rng = np.random.default_rng(123)
        
        # Try several random directions
        for trial in range(10):
            if time.time() > deadline - 5.0:
                break
            # Random direction
            direction = rng.standard_normal(n)
            # Project fiedler onto direction and use sign
            scores = fiedler * direction
            # Use median as threshold for balanced partition
            median_val = np.median(scores)
            labels = (scores > median_val).astype(int)
            
            # Refine with local search
            labels, cut = local_search(labels, max_iterations=30)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
        
        return best_labels, best_cut

    def spectral_init():
        """Create initial partition using the Fiedler vector."""
        degrees = adj.sum(axis=1)
        D = sp.diags(degrees, format='csr')
        A = sp.csr_matrix(adj)
        L = D - A
        
        try:
            evals, evecs = spla.eigsh(L, k=2, sigma=0, which='LM', tol=1e-8)
            idx = np.argsort(evals)
            fiedler = evecs[:, idx[1]]
        except Exception:
            fiedler = np.random.randn(n)
        
        return fiedler

    # Multi-start local search
    best_labels = None
    best_cut = -1

    # Start with spectral initialization + Goemans-Williamson rounding
    fiedler = spectral_init()
    gm_labels, gm_cut = goemans_maxwell_rounding(fiedler)
    if gm_labels is not None and gm_cut > best_cut:
        best_cut = gm_cut
        best_labels = gm_labels.copy()

    # Also try simple thresholding of Fiedler vector
    median_val = np.median(fiedler)
    labels = (fiedler > median_val).astype(int)
    labels, cut = local_search(labels)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels.copy()

    # Try random restarts if time permits
    rng = np.random.default_rng(42)
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        labels = rng.integers(0, 2, size=n).astype(int)
        labels, cut = local_search(labels, max_iterations=50)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Final check
    if best_labels is None:
        best_labels = np.arange(n) % 2

    return best_labels.astype(int)
# EVOLVE-BLOCK-END
