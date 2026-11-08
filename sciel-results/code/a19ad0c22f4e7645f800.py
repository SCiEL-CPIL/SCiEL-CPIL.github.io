"""Matrix multiplication 2x3x5 tensor decomposition - rank minimization."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 5]

def _strassen_2x2x2():
    """Return Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W each of shape (4, 7) where:
    - U indexed by (i,j) for input A[i,j]
    - V indexed by (j,k) for input B[j,k]
    - W indexed by (k,i) for output C[k,i] (trace convention)
    """
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # U: A[i,j] contributions (i=0,1; j=0,1) -> indices 0,1,2,3
    # V: B[j,k] contributions (j=0,1; k=0,1) -> indices 0,1,2,3
    # W: C[k,i] contributions (k=0,1; i=0,1) -> indices 0,1,2,3 (trace: k*n+i)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    # C00 += M1, C11 += M1
    W[0, 0] = 1  # C00 (k=0,i=0)
    W[3, 0] = 1  # C11 (k=1,i=1)
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    # C10 += M2
    W[2, 1] = 1  # C10 (k=1,i=0)
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1  # B11
    # C01 += M3
    W[1, 2] = 1  # C01 (k=0,i=1)
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1  # B00
    # C00 += M4, C10 += M4
    W[0, 3] = 1  # C00
    W[2, 3] = 1  # C10
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    # C00 -= M5, C01 += M5
    W[0, 4] = -1  # C00
    W[1, 4] = 1   # C01
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1  # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    # C11 += M6
    W[3, 5] = 1  # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1  # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    # C00 += M7
    W[0, 6] = 1  # C00
    
    return U, V, W

def _build_block_decomposition(n, m, p):
    """Build a block-based decomposition for n x m x p matrix multiplication.
    
    Strategy: Decompose the multiplication into blocks and use known
    low-rank decompositions for each block.
    
    For 2x3x5: We can view this as multiplying blocks.
    A is 2x3, B is 3x5, C is 2x5.
    
    We'll use a Kronecker-like structure. The key insight is that we can
    decompose the tensor using the structure of the matrix dimensions.
    
    One approach: treat the 2x3 multiplication as combining a 2x2 and 1x1 part,
    and the 3x5 as combining parts.
    
    Actually, let's use a more direct approach:
    - Use Strassen for the 2x2x2 sub-problem (rank 7)
    - Handle the remaining dimensions with rank-1 terms
    """
    # For 2x3x5, one strategy:
    # Split the 3 columns of A (and 3 rows of B) into groups.
    # 
    # Alternative: Use the fact that we can decompose as:
    # C = A * B where A is 2x3, B is 3x5
    # 
    # We can write this as a sum of rank-1 tensors, but we want to minimize rank.
    # 
    # Let's try a hybrid approach:
    # 1. Use Strassen for a 2x2x2 core (rank 7)
    # 2. Add rank-1 terms for the remaining interactions
    #
    # The 2x3x5 tensor has 30 entries. We want to express it as sum of rank-r tensors.
    #
    # A structured approach:
    # Think of A as [A_left (2x2), A_right (2x1)]
    # Think of B as [B_top (2x5); B_bottom (1x5)]
    # Then C = A_left * B_top + A_right * B_bottom
    #
    # A_left * B_top is 2x2 * 2x5, which can be decomposed using Strassen-like methods
    # A_right * B_bottom is 2x1 * 1x5, which is rank 2 (or less with structure)
    
    # Let me try a different decomposition strategy:
    # Use the "border rank" approach or explicit construction.
    
    # Actually, for this specific case, let's try to construct an explicit
    # decomposition with rank < 30 using the block structure.
    
    # Simple approach: Group the 3 middle dimension into 2+1
    # A = [A1 (2x2) | A2 (2x1)]
    # B = [B1 (2x5); B2 (1x5)]
    # C = A1*B1 + A2*B2
    
    # A1*B1: 2x2 * 2x5. We can use Strassen for the 2x2 part and extend.
    # For each column of B1 (2x5), we multiply A1 (2x2) by that column (2x1).
    # This is 5 separate 2x2*2x1 multiplications.
    # Each 2x2*2x1 has rank 4 (schoolbook) or can be done with rank 3 (Strassen-like).
    # Actually, 2x2*2x1: C = A*b where A is 2x2, b is 2x1, C is 2x1.
    # C0 = A00*b0 + A01*b1
    # C1 = A10*b0 + A11*b1
    # This is rank 4 schoolbook. Can we do better? 
    # Using Strassen ideas: 
    # M1 = (A00+A11)*b0
    # M2 = (A10+A11)*b0  
    # M3 = A00*(b0-b1)
    # M4 = A11*(b0-b1)
    # C0 = M1 - M4 + M3
    # C1 = M2 - M4
    # That's rank 4 still. Hmm.
    
    # Let me just try the direct numerical optimization but with a better
    # initialization based on the block structure.
    
    return None  # We'll use the numerical approach but with better init

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] as a 3D array."""
    T = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _block_init(n, m, p, rank, rng=None):
    """Initialize U, V, W using block structure to give a better starting point.
    
    Strategy: Initialize based on the natural block structure of the matrix
    multiplication, then perturb to allow optimization to find lower-rank solutions.
    """
    if rng is None:
        rng = np.random
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Start with a schoolbook-like initialization but with some structure
    # This gives a better starting point than pure random
    
    # First, fill with schoolbook terms up to min(rank, n*m*p)
    num_school = min(rank, n*m*p)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col < num_school:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    # If rank < n*m*p, the remaining columns are zero initially
    # Add small random perturbations to all entries to break symmetry
    if rank > 0:
        U += rng.randn(n*m, rank) * 0.1
        V += rng.randn(m*p, rank) * 0.1
        W += rng.randn(p*n, rank) * 0.1
    
    return U, V, W

def _strassen_init(n, m, p, rank, rng=None):
    """Initialize using Strassen decomposition for the 2x2x2 core.
    
    For 2x3x5, we can embed the Strassen 2x2x2 decomposition and add
    additional terms for the remaining dimensions.
    """
    if rng is None:
        rng = np.random
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Embed Strassen for the 2x2x2 sub-problem (first 2 cols of A, first 2 rows of B)
    S_U, S_V, S_W = _strassen_2x2x2()
    
    # Map Strassen indices to full problem indices
    # Strassen U is (4,7) indexed by (i,j) for i=0,1; j=0,1
    # Full U is (n*m, rank) indexed by i*m+j
    # For the 2x2 core: i in {0,1}, j in {0,1}
    for i in range(2):
        for j in range(2):
            U_idx = i*m + j
            for c in range(7):
                U[U_idx, c] = S_U[i*2+j, c]
    
    # Strassen V is (4,7) indexed by (j,k) for j=0,1; k=0,1
    # Full V is (m*p, rank) indexed by j*p+k
    for j in range(2):
        for k in range(2):
            V_idx = j*p + k
            for c in range(7):
                V[V_idx, c] = S_V[j*2+k, c]
    
    # Strassen W is (4,7) indexed by (k,i) for k=0,1; i=0,1
    # Full W is (p*n, rank) indexed by k*n+i (trace convention)
    for k in range(2):
        for i in range(2):
            W_idx = k*n + i
            for c in range(7):
                W[W_idx, c] = S_W[k*2+i, c]
    
    # Now we have used 7 columns for the 2x2x2 core.
    # We need to handle the remaining interactions:
    # - A column 2 (j=2) with all B rows
    # - B rows 2,3,4 (k=2,3,4) with A columns 0,1
    # 
    # Add rank-1 terms for these, or use additional structure.
    
    # For the remaining terms, add random initialization to the remaining columns
    remaining_cols = rank - 7
    if remaining_cols > 0:
        U[:, 7:] = rng.randn(n*m, remaining_cols) * 0.5
        V[:, 7:] = rng.randn(m*p, remaining_cols) * 0.5
        W[:, 7:] = rng.randn(p*n, remaining_cols) * 0.5
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] == T."""
    rank = U.shape[1]
    T_recon = np.zeros((n*m, m*p, n*p))
    for c in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
    T_true = _build_tensor(n, m, p)
    return np.max(np.abs(T_recon - T_true)) < tol

def _verify_fast(U, V, W, T, tol=1e-9):
    """Fast verification against precomputed tensor."""
    rank = U.shape[1]
    T_recon = np.zeros(T.shape)
    for c in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
    return np.max(np.abs(T_recon - T)) < tol

def _polish(U, V, W, T, n, m, p, max_iters=500, tol=1e-12):
    """High-precision polish pass with adaptive step size."""
    rank = U.shape[1]
    lr = 0.01
    
    for it in range(max_iters):
        # Compute residual
        recon = np.zeros((n*m, m*p, n*p))
        for c in range(rank):
            recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
        residual = recon - T
        err = np.sum(residual**2)
        
        if err < tol:
            break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        for c in range(rank):
            vw = np.einsum('b,c->bc', V[:,c], W[:,c])
            uw = np.einsum('a,c->ac', U[:,c], W[:,c])
            uv = np.einsum('a,b->ab', U[:,c], V[:,c])
            dU[:,c] = np.einsum('abc,bc->a', residual, vw)
            dV[:,c] = np.einsum('abc,ac->b', residual, uw)
            dW[:,c] = np.einsum('abc,ab->c', residual, uv)
        
        # Adaptive learning rate based on gradient magnitude
        grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
        if grad_norm > 1e-15:
            step_size = min(0.1 / grad_norm, lr)
        else:
            step_size = lr
        
        U -= step_size * dU
        V -= step_size * dV
        W -= step_size * dW
        
        # Scale invariance fix
        for c in range(rank):
            norm_u = np.linalg.norm(U[:,c])
            norm_v = np.linalg.norm(V[:,c])
            norm_w = np.linalg.norm(W[:,c])
            if norm_u > 1e-15 and norm_v > 1e-15 and norm_w > 1e-15:
                target = (norm_u * norm_v * norm_w) ** (1/3)
                U[:,c] *= target / norm_u
                V[:,c] *= target / norm_v
                W[:,c] *= target / norm_w
        
        # Adaptive learning rate adjustment
        if it % 20 == 0:
            if err > 1.0:
                lr *= 0.8
            elif err < 0.01:
                lr = min(lr * 1.2, 0.05)
    
    return U, V, W, err

def _polish_newton(U, V, W, T, n, m, p, max_iters=100, tol=1e-14):
    """Newton-type polish using Gauss-Newton for faster convergence near solution."""
    rank = U.shape[1]
    
    for it in range(max_iters):
        # Compute residual
        recon = np.zeros((n*m, m*p, n*p))
        for c in range(rank):
            recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
        residual = recon - T
        err = np.sum(residual**2)
        
        if err < tol:
            break
        
        # Compute Gauss-Newton step
        # J is the Jacobian of the residual with respect to all parameters
        # For efficiency, we'll do a damped Gauss-Newton
        
        # Compute gradients (same as before)
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        for c in range(rank):
            vw = np.einsum('b,c->bc', V[:,c], W[:,c])
            uw = np.einsum('a,c->ac', U[:,c], W[:,c])
            uv = np.einsum('a,b->ab', U[:,c], V[:,c])
            dU[:,c] = np.einsum('abc,bc->a', residual, vw)
            dV[:,c] = np.einsum('abc,ac->b', residual, uw)
            dW[:,c] = np.einsum('abc,ab->c', residual, uv)
        
        # Compute approximate Hessian diagonal for adaptive step
        # This is a simplified Gauss-Newton: use gradient norm to determine step
        grad_norm_sq = np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2)
        
        if grad_norm_sq > 1e-20:
            # Damped step
            damping = 1.0
            for damp_iter in range(10):
                U_new = U - damping * dU
                V_new = V - damping * dV
                W_new = W - damping * dW
                
                # Check if this step reduces error
                recon_new = np.zeros((n*m, m*p, n*p))
                for c in range(rank):
                    recon_new += np.einsum('a,b,c->abc', U_new[:,c], V_new[:,c], W_new[:,c])
                err_new = np.sum((recon_new - T)**2)
                
                if err_new < err:
                    U, V, W = U_new, V_new, W_new
                    break
                else:
                    damping *= 0.5
            else:
                break  # No improvement possible
        
        # Scale invariance fix
        for c in range(rank):
            norm_u = np.linalg.norm(U[:,c])
            norm_v = np.linalg.norm(V[:,c])
            norm_w = np.linalg.norm(W[:,c])
            if norm_u > 1e-15 and norm_v > 1e-15 and norm_w > 1e-15:
                target = (norm_u * norm_v * norm_w) ** (1/3)
                U[:,c] *= target / norm_u
                V[:,c] *= target / norm_v
                W[:,c] *= target / norm_w
    
    # Final error
    recon = np.zeros((n*m, m*p, n*p))
    for c in range(rank):
        recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
    final_err = np.sum((recon - T)**2)
    
    return U, V, W, final_err

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_school = n * m * p
    U_s = np.zeros((n*m, rank_school))
    V_s = np.zeros((m*p, rank_school))
    W_s = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1
    best_result = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    best_rank = rank_school
    
    T = _build_tensor(n, m, p)
    
    def _try_rank(rank, max_iters=2000, n_restarts=5, use_strassen=False):
        nonlocal best_result, best_rank
        for restart in range(n_restarts):
            if time.time() > deadline:
                return False
            
            # Choose initialization strategy
            if use_strassen and rank >= 7:
                U, V, W = _strassen_init(n, m, p, rank)
            else:
                U, V, W = _block_init(n, m, p, rank)
            
            # Gradient descent on F = 0.5 * ||sum_c u_c ⊗ v_c ⊗ w_c - T||^2
            lr = 0.01
            prev_err = float('inf')
            stagnant = 0
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return False
                
                # Compute residual
                recon = np.zeros((n*m, m*p, n*p))
                for c in range(rank):
                    recon += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
                residual = recon - T
                err = np.sum(residual**2)
                
                if err < 1e-10:
                    # Found near-exact decomposition, polish it
                    U, V, W, final_err = _polish(U, V, W, T, n, m, p, max_iters=300, tol=1e-13)
                    if final_err < 1e-10:
                        if rank < best_rank:
                            best_rank = rank
                            best_result = {
                                "U": U.tolist(),
                                "V": V.tolist(),
                                "W": W.tolist()
                            }
                        return True
                    # Polish didn't converge, try Newton polish
                    U, V, W, final_err = _polish_newton(U, V, W, T, n, m, p, max_iters=50, tol=1e-14)
                    if final_err < 1e-10:
                        if rank < best_rank:
                            best_rank = rank
                            best_result = {
                                "U": U.tolist(),
                                "V": V.tolist(),
                                "W": W.tolist()
                            }
                        return True
                    # Still not converged, continue
                    err = final_err
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                for c in range(rank):
                    vw = np.einsum('b,c->bc', V[:,c], W[:,c])
                    uw = np.einsum('a,c->ac', U[:,c], W[:,c])
                    uv = np.einsum('a,b->ab', U[:,c], V[:,c])
                    dU[:,c] = np.einsum('abc,bc->a', residual, vw)
                    dV[:,c] = np.einsum('abc,ac->b', residual, uw)
                    dW[:,c] = np.einsum('abc,ab->c', residual, uv)
                
                # Adaptive learning rate
                grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                if grad_norm > 1e-15:
                    step_size = min(0.05 / grad_norm, lr)
                else:
                    step_size = lr
                
                U -= step_size * dU
                V -= step_size * dV
                W -= step_size * dW
                
                # Scale invariance fix: normalize columns
                for c in range(rank):
                    norm_u = np.linalg.norm(U[:,c])
                    norm_v = np.linalg.norm(V[:,c])
                    norm_w = np.linalg.norm(W[:,c])
                    if norm_u > 1e-15 and norm_v > 1e-15 and norm_w > 1e-15:
                        target = (norm_u * norm_v * norm_w) ** (1/3)
                        U[:,c] *= target / norm_u
                        V[:,c] *= target / norm_v
                        W[:,c] *= target / norm_w
                
                # Stagnation detection
                if err > prev_err * 0.999:
                    stagnant += 1
                else:
                    stagnant = 0
                prev_err = err
                
                if stagnant > 50:
                    # Restart with perturbation
                    U += np.random.randn(n*m, rank) * 0.01
                    V += np.random.randn(m*p, rank) * 0.01
                    W += np.random.randn(p*n, rank) * 0.01
                    stagnant = 0
                
                if it % 50 == 0 and err > 1.0:
                    lr *= 0.9
                elif it % 50 == 0 and err < 0.1:
                    lr = min(lr * 1.1, 0.05)
        
        return False
    
    # Try ranks from lower bound upward
    lower_bound = n*m + m*p + n*p - n - m - p + 1  # = 22
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        # Use Strassen-based init for higher ranks, block init for lower
        use_strassen = rank >= 15
        success = _try_rank(rank, max_iters=1500, n_restarts=4, use_strassen=use_strassen)
        if success:
            break
    
    # Convert to lists
    return best_result

# EVOLVE-BLOCK-END
