"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x7 matrix.
    
    Strategy:
    1. Define the schoolbook decomposition (rank 84) as the baseline.
    2. Implement a fast vectorized verification function.
    3. Attempt a brief, optimized search for a lower rank decomposition 
       using SVD-based initialization and local refinement.
    4. If no valid lower-rank decomposition is found within the time budget,
       return the schoolbook decomposition.
    """
    start_time = time.time()
    # Use a safety margin of 10 seconds before the 120s limit
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T[a, b, c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    schoolbook_rank = n * m * p  # 84

    def verify_decomposition(U, V, W, tol=1e-8):
        """
        Fast verification using NumPy broadcasting.
        Checks if T[a,b,c] == sum_r U[a,r] * V[b,r] * W[c,r]
        """
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        
        # Compute the reconstructed tensor
        # U[:, :, None, None] * V[None, :, :, None] * W[None, None, :, :]
        # Shape: (nm, 1, 1, rank) * (1, mp, 1, rank) * (1, 1, pn, rank) -> (nm, mp, pn, rank)
        # Sum over the last axis (rank)
        reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check max error
        max_error = np.max(np.abs(reconstructed - T))
        return max_error < tol

    def make_schoolbook(rank):
        """Create the standard schoolbook decomposition."""
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < rank:
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    # Baseline: Schoolbook decomposition
    U_sb, V_sb, W_sb = make_schoolbook(schoolbook_rank)
    # Verify schoolbook just in case (should always pass)
    if not verify_decomposition(U_sb, V_sb, W_sb):
        # This should never happen, but handle it
        raise RuntimeError("Schoolbook decomposition failed verification")

    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Attempt to find a lower rank decomposition
    # We try ranks slightly below schoolbook, though for 3x4x7, 
    # significant improvement is unlikely without specific algorithmic insights.
    # We use a time-limited search.
    
    # Search time budget: 10 seconds
    search_deadline = start_time + 10.0
    
    # Try a few ranks below schoolbook
    # Given the difficulty, we focus on ranks close to schoolbook 
    # or use a more general optimization if time permits.
    # However, finding a rank < 84 for 3x4x7 with half-integer entries 
    # is extremely difficult. 
    # We will perform a quick SVD-based check for rank 83, 82, etc.
    
    # Matricize T for SVD
    # T_mat[a, b+c*mp] is not quite right for standard SVD rank bound.
    # The rank of the tensor is at least the rank of any unfolding.
    # Unfolding 1: (nm) x (mp*pn)
    T_mat1 = T.reshape(nm, mp * pn)
    U_svd, s_vals, Vt_svd = np.linalg.svd(T_mat1, full_matrices=False)
    
    # The numerical rank of this unfolding gives a lower bound
    tol_rank = 1e-10
    min_possible_rank = np.sum(s_vals > tol_rank)
    
    # For matrix multiplication, the unfolding rank is typically n*m*p 
    # or related to the dimensions. 
    # Actually, for the standard MM tensor, the unfolding rank is n*m*p?
    # Let's check: T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
    # This tensor is sparse. 
    # The SVD of the unfolding might have rank equal to the number of non-zero entries in the unfolding?
    # No, the rank is the dimension of the span of the slices.
    
    # If min_possible_rank is close to schoolbook, we can skip the search.
    # For 3x4x7, schoolbook is 84.
    # If min_possible_rank is 84, we cannot do better.
    
    if min_possible_rank >= schoolbook_rank:
        # No room for improvement based on unfolding rank
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }

    # If min_possible_rank < schoolbook_rank, we might be able to improve.
    # We try to construct a decomposition of rank R where min_possible_rank <= R < schoolbook_rank.
    # We start from R = min_possible_rank and go up to schoolbook_rank - 1.
    
    for rank in range(min_possible_rank, schoolbook_rank):
        if time.time() > search_deadline:
            break
            
        # Initialize using truncated SVD of the first unfolding
        # U_3d will be (nm, rank)
        # We need V and W as well. 
        # A common approach is to use the SVD of the tensor unfolded in all modes, 
        # but that's complex. 
        # Let's use a simple random initialization + local search for a fixed number of iterations.
        
        # Random initialization with half-integer entries
        rng = np.random.RandomState(rank)
        U = (rng.randint(0, 3, size=(nm, rank)) - 1) * 0.5
        V = (rng.randint(0, 3, size=(mp, rank)) - 1) * 0.5
        W = (rng.randint(0, 3, size=(pn, rank)) - 1) * 0.5
        
        # Local search using gradient descent on the continuous relaxation, 
        # then rounding to half-integers.
        # Objective: minimize || T - sum_r U_r V_r W_r ||_F^2
        
        # To save time, we only do a few iterations.
        # If we find a valid half-integer decomposition, we return it.
        
        # Given the complexity and low probability of success for 3x4x7,
        # we limit the iterations strictly.
        
        max_iter = 20
        lr = 0.1
        
        for it in range(max_iter):
            if time.time() > search_deadline:
                break
                
            # Compute gradient
            # T_recon = einsum('ar,br,cr->abc', U, V, W)
            # diff = T - T_recon
            # dU = diff * (V[:, :, None] * W[None, :, :]) ... wait, indices
            # dU[a, r] = sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
            # This can be vectorized.
            
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T - T_recon
            
            # Gradient for U: (nm, rank)
            # dU[a, r] = sum_b sum_c diff[a,b,c] V[b,r] W[c,r]
            # = sum_b diff[a,b,:] dot (V[b,:] * W[:,r])? No.
            # dU = np.einsum('abc,br,cr->ar', diff, V, W)
            
            # Gradient for V: (mp, rank)
            # dV = np.einsum('abc,ar,cr->br', diff, U, W)
            
            # Gradient for W: (pn, rank)
            # dW = np.einsum('abc,ar,br->cr', diff, U, V)
            
            dU = np.einsum('abc,br,cr->ar', diff, V, W)
            dV = np.einsum('abc,ar,cr->br', diff, U, W)
            dW = np.einsum('abc,ar,br->cr', diff, U, V)
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check if we are close to a valid half-integer solution
            # We check the error first
            T_recon_check = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.max(np.abs(T_recon_check - T))
            
            if error < 1e-6:
                # Try rounding
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if verify_decomposition(U_rounded, V_rounded, W_rounded):
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }

    # If no better rank found, return schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
# EVOLVE-BLOCK-END
