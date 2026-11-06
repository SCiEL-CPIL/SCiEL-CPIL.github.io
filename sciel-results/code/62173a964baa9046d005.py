import numpy as np
import time

def priority(adjacency):
    """
    Computes vertex scores for the greedy MIS decoder.
    Uses a damped fixed-point iteration with a smooth log-sum-exp penalty
    for neighbor competition, plus a spectral (Fiedler) component.
    """
    adjacency = np.asarray(adjacency, dtype=float)
    n = adjacency.shape[0]
    
    # Degree vector
    degrees = adjacency.sum(axis=1)
    
    # Initial scores: negative degree (low degree = high score)
    scores = -degrees.copy()
    
    # Parameters
    alpha = 0.4  # Damping factor (slightly higher damping for stability)
    beta = 1.5   # Competition strength (increased to better penalize neighbors)
    gamma = 0.1  # Spectral component weight
    iterations = 80
    start_time = time.time()
    deadline = 150.0 - 5.0  # Safety margin
    
    # Compute spectral component (Fiedler vector) if n is small enough
    # For n=200, this is fast
    spectral_component = np.zeros(n)
    try:
        if n > 1:
            # Use scipy if available for faster eigenvalue computation
            try:
                from scipy.sparse.linalg import eigsh
                from scipy.sparse import csr_matrix
                A_csr = csr_matrix(adjacency)
                # Fiedler vector: second smallest eigenvalue
                # For sparse matrices, use eigsh
                # We want the eigenvector corresponding to the second smallest eigenvalue
                # eigsh finds eigenvalues near 'which'
                # For Fiedler, we want sigma=0 (smallest) but skip the constant vector
                # Use shift-invert mode for smallest eigenvalues
                try:
                    # Get two smallest eigenvalues
                    vals, vecs = eigsh(A_csr, k=2, sigma=0, which='LM', maxiter=1000)
                    # Sort by eigenvalue
                    idx = np.argsort(vals)
                    vecs = vecs[:, idx]
                    # The first is the constant vector (eigenvalue 0 for connected graphs)
                    # The second is the Fiedler vector
                    if vecs.shape[1] >= 2:
                        fiedler = vecs[:, 1]
                        # Normalize
                        fiedler = fiedler / (np.linalg.norm(fiedler) + 1e-10)
                        spectral_component = fiedler
                except Exception:
                    # Fallback to dense if sparse fails
                    from numpy.linalg import eigh
                    vals, vecs = eigh(adjacency)
                    idx = np.argsort(vals)
                    vecs = vecs[:, idx]
                    if vecs.shape[1] >= 2:
                        fiedler = vecs[:, 1]
                        fiedler = fiedler / (np.linalg.norm(fiedler) + 1e-10)
                        spectral_component = fiedler
            except ImportError:
                # No scipy, use numpy
                from numpy.linalg import eigh
                vals, vecs = eigh(adjacency)
                idx = np.argsort(vals)
                vecs = vecs[:, idx]
                if vecs.shape[1] >= 2:
                    fiedler = vecs[:, 1]
                    fiedler = fiedler / (np.linalg.norm(fiedler) + 1e-10)
                    spectral_component = fiedler
    except Exception:
        pass
    
    # Helper for stable LogSumExp
    def log_sum_exp(x):
        if len(x) == 0:
            return 0.0
        m = np.max(x)
        if m == -np.inf:
            return 0.0
        return m + np.log(np.sum(np.exp(x - m)))
    
    for _ in range(iterations):
        if time.time() > deadline:
            break
        
        new_scores = np.zeros(n)
        
        # Iterate over vertices
        for v in range(n):
            # Get neighbors
            neighbors = np.where(adjacency[v] > 0)[0]
            
            if len(neighbors) == 0:
                # Isolated vertex: high priority
                base = 0.0
                penalty = 0.0
            else:
                neighbor_scores = scores[neighbors]
                lse_val = log_sum_exp(neighbor_scores)
                # Penalty: if neighbors have high scores, lse is higher (less negative)
                # We want to LOWER the score of v if neighbors are good (high score)
                # So penalty = beta * lse_val
                # If lse is -5 (good neighbors), penalty = -5*beta (negative, lowers score)
                # If lse is -20 (bad neighbors), penalty = -20*beta (more negative, lowers score more)
                # Wait, that's the opposite. Let me reconsider.
                # scores are negative degrees initially, so low degree = high score (less negative)
                # If neighbors have high scores (less negative, e.g., -2), lse is higher (e.g., -2)
                # If neighbors have low scores (more negative, e.g., -10), lse is lower (e.g., -10)
                # We want v to have LOWER score if neighbors are GOOD (high score)
                # So if lse is -2 (good), penalty should be positive (to subtract)
                # If lse is -10 (bad), penalty should be less positive or negative
                # penalty = beta * lse_val: if lse=-2, penalty=-2*beta (negative)
                # This ADDS to the score (since we subtract penalty)
                # Let's use: target = base - beta * lse_val
                # base = -degrees[v]
                # If lse = -2 (good neighbors), target = -deg - beta*(-2) = -deg + 2*beta (higher score)
                # If lse = -10 (bad neighbors), target = -deg - beta*(-10) = -deg + 10*beta (even higher score)
                # This makes v have HIGHER score if neighbors are BAD. That's what we want!
                # So v is prioritized when its neighbors are not good candidates.
                base = -degrees[v]
                penalty = beta * lse_val
                target = base - penalty
            
            # Add spectral component
            target += gamma * spectral_component[v]
            
            new_scores[v] = alpha * scores[v] + (1 - alpha) * target
        
        scores = new_scores
        
        # Center to prevent drift
        scores -= np.mean(scores)
    
    # Final: boost isolated vertices slightly
    isolated = degrees == 0
    if np.any(isolated):
        scores[isolated] += 1.0
    
    return scores
