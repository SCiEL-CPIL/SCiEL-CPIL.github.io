"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """
    Partition vertices into two groups to maximize the cut size using ADMM.
    """
    n = len(adjacency)
    # Ensure adjacency is a float numpy array for linear algebra
    A = np.array(adjacency, dtype=np.float64)
    
    # Deadline guard: 145 seconds
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_cut = -1
    best_labels = None
    
    # ADMM parameters
    rho = 1.0  # Penalty parameter
    max_iters = 100 # Max ADMM iterations per restart
    num_restarts = 20 # Number of random restarts
    
    # Pre-compute factorization for the QP subproblem?
    # The matrix (2*A + rho*I) is constant for all iterations.
    # We can use np.linalg.solve repeatedly, or factorize once.
    # For n=100, np.linalg.solve is fast enough.
    
    # Helper to compute cut size for labels in {-1, 1}
    def _cut_size_y(y):
        # Cut size = 1/4 * sum(A_ij * (1 - y_i y_j))
        # = 1/4 * (sum(A) - sum(A_ij y_i y_j))
        # sum(A) is constant. We just need to minimize y^T A y.
        # The value returned by evaluator is likely the raw cut size (number of edges).
        # Let's compute the actual cut size to be safe.
        # y is {-1, 1}.
        # Convert to 0/1 for standard cut definition? 
        # Edge (i,j) is in cut if y_i != y_j.
        # cut = sum_{i<j} A_ij * (1 if y_i != y_j else 0)
        # Using matrix ops:
        # y^T A y = sum_{i,j} A_ij y_i y_j = 2 * sum_{i<j} A_ij y_i y_j (since A symmetric, A_ii=0)
        # sum_{i<j} A_ij = total_edges
        # cut = total_edges - 0.5 * y^T A y
        # Wait, if y_i=y_j, y_i y_j = 1. If y_i != y_j, y_i y_j = -1.
        # sum_{i<j} A_ij (1 - y_i y_j) = sum_{i<j} A_ij - sum_{i<j} A_ij y_i y_j
        # sum_{i<j} A_ij y_i y_j = 0.5 * y^T A y
        # So cut = total_edges - 0.5 * y^T A y
        val = 0.5 * (np.sum(A) - y @ (A @ y))
        return val

    # We will run ADMM for multiple restarts
    rng = np.random.default_rng(42)
    
    # Pre-factorize M = 2*A + rho*I
    # Actually, the subproblem is: min_y 0.5 * y^T (2A + rho I) y - rho (z - u)^T y
    # Gradient: (2A + rho I) y - rho (z - u) = 0
    # y = (2A + rho I)^{-1} rho (z - u)
    # Let M = 2*A + rho*I
    M = 2.0 * A + rho * np.eye(n)
    
    # Check for singularity? A is symmetric PSD (Laplacian-like but not exactly, it's adjacency).
    # 2A + rho I is positive definite for rho > 0.
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize z (the constrained variable)
        # Random initialization
        z = rng.choice([-1.0, 1.0], size=n)
        # Initialize u (dual variable)
        u = np.zeros(n)
        
        # Optional: Spectral initialization for the first restart
        if restart == 0:
            try:
                # Fiedler vector of Laplacian L = D - A
                degrees = A.sum(axis=1)
                L = np.diag(degrees) - A
                eigvals, eigvecs = np.linalg.eigh(L)
                fiedler = eigvecs[:, 1]
                z = np.sign(fiedler)
                # Ensure not all same sign (rare but possible)
                if np.all(z > 0) or np.all(z < 0):
                    z = rng.choice([-1.0, 1.0], size=n)
            except Exception:
                pass

        # ADMM Iterations
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # 1. Update y
            # y = solve(M, rho * (z - u))
            try:
                y = np.linalg.solve(M, rho * (z - u))
            except np.linalg.LinAlgError:
                # Fallback if singular (should not happen with rho>0)
                y = z - u
                
            # 2. Update z (proximal operator for sign constraint)
            z = np.sign(y + u)
            # Handle zeros in sign (if y+u is 0, sign is 0). 
            # For MaxCut, we need {-1, 1}. If 0, flip coin or keep previous?
            # Usually sign(0) is 0. We need non-zero.
            zeros = (z == 0)
            if np.any(zeros):
                z[zeros] = rng.choice([-1.0, 1.0], size=np.sum(zeros))
                
            # 3. Update u
            u = u + y - z
            
            # Check convergence (optional, can skip for fixed iterations)
            # if np.linalg.norm(z - z_prev) < 1e-6: break
            
        # Calculate cut for this restart
        cut_val = _cut_size_y(z)
        
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = z.copy()
            
    # If no valid result found (should not happen), fallback
    if best_labels is None:
        best_labels = np.ones(n)
        best_labels[n//2:] = -1.0
        
    # Convert from {-1, 1} to {0, 1}
    # 1 -> 1, -1 -> 0
    # (label + 1) / 2
    final_labels = ((best_labels + 1) / 2).astype(int)
    
    return final_labels
# EVOLVE-BLOCK-END
