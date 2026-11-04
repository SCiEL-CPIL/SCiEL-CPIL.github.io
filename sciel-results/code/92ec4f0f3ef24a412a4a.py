"""Matrix multiplication tensor decomposition for 3x3 * 3x7."""
import time
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for matrix multiplication 3x3 * 3x7.
    
    Returns dict with 'U', 'V', 'W' as lists of lists (integer entries).
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin from 150s limit
    
    def time_left():
        return deadline - time.time()
    
    # Build the multiplication tensor T[i,j,k] = e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # We want to decompose: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Schoolbook baseline: rank = n*m*p = 63
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    def verify_decomposition(U, V, W, num_tests=10):
        """Verify that the decomposition exactly reproduces matrix multiplication."""
        rng = np.random.RandomState(12345)
        for _ in range(num_tests):
            A = rng.randint(-10, 11, size=(n, m)).astype(float)
            B = rng.randint(-10, 11, size=(m, p)).astype(float)
            C_expected = A @ B
            
            # Reconstruct C[i,k] = sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
            # More efficiently: C = sum_r (A * U[:,r].T) @ (V[:,r] * B.T) * W[:,r].T
            # Actually: C[i,k] = sum_j sum_r U[i*m+j,r] * A[i,j] * V[j*p+k,r] * B[j,k] * W[k*n+i,r]
            # = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] * A[i,j] * V[j*p+k,r] * B[j,k])
            
            C_recon = np.zeros((n, p), dtype=float)
            for r in range(U.shape[1]):
                # For each r, compute contribution
                # U[:,r] has entries at positions i*m+j
                # V[:,r] has entries at positions j*p+k
                # W[:,r] has entries at positions k*n+i
                u_r = U[:, r].reshape(n, m)
                v_r = V[:, r].reshape(m, p)
                w_r = W[:, r].reshape(p, n)
                
                # Contribution: C[i,k] += sum_j u_r[i,j] * A[i,j] * v_r[j,k] * B[j,k] * w_r[k,i]
                # = w_r[k,i] * (A * u_r) @ (v_r * B) ... not quite
                # Let's do it directly for small matrices
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            C_recon[i, k] += u_r[i, j] * A[i, j] * v_r[j, k] * B[j, k] * w_r[k, i]
            
            if not np.allclose(C_recon, C_expected, atol=1e-9):
                return False
        return True
    
    # First, verify schoolbook
    if not verify_decomposition(U_sb, V_sb, W_sb):
        # Should never happen
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank decompositions
    # Strategy: For each candidate rank r, try to find U, V, W of that rank
    # that satisfy the tensor equation.
    
    # The tensor equation can be written as:
    # For all i,j,k: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 (it's the identity multiplication tensor)
    
    # This is a non-convex problem. We can try various approaches:
    # 1. Fix W to be the identity-like structure and solve for U, V
    # 2. Use alternating minimization
    # 3. Try structured decompositions
    
    # Approach: Since the 7 columns are somewhat independent, try to find
    # decompositions where W has a block-diagonal structure for the k-index.
    
    # Let's try a more systematic search using the fact that for small matrices,
    # we can set up the equations and solve.
    
    # The key insight: T[i,j,k] = delta_{i,j,k} in the sense that the multiplication
    # tensor has T[i,j,k] = 1 for all (i,j,k) triples that correspond to valid
    # matrix multiplication entries.
    
    # Actually, the multiplication tensor for A(i,j) * B(j,k) -> C(i,k) is:
    # T[i,j,k] = 1 for all i,j,k (since each term A[i,j]*B[j,k] contributes to C[i,k])
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a system of n*m*p = 63 equations in rank * (n*m + m*p + p*n) = rank * (9+21+21) = rank * 51 variables.
    
    # For rank < 63, this is underdetermined, so solutions may exist.
    
    # Let's try rank values from 62 down and see if we can find solutions.
    # We'll use a randomized approach with alternating least squares.
    
    def try_rank(r, max_iters=200, seed=42):
        """Try to find a rank-r decomposition using alternating minimization."""
        if r >= rank_schoolbook:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with random values
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        # Scale to avoid numerical issues
        U /= np.linalg.norm(U) * np.sqrt(r)
        V /= np.linalg.norm(V) * np.sqrt(r)
        W /= np.linalg.norm(W) * np.sqrt(r)
        
        for it in range(max_iters):
            if time_left() < 5:
                return None
            
            # Update W: For each (k,n) pair, W[k*n+i, :] should satisfy
            # sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1 for all i,j
            # For fixed i,k: sum_r W[k*n+i, r] * sum_j U[i*m+j, r] * V[j*p+k, r] = 1
            # Let B_ik[r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            # Then W[k*n+i, :] * B_ik = 1
            # This is a linear system: W[k*n+i, :] @ B_ik = 1
            
            # Similarly for U and V.
            
            # Let's do alternating updates:
            # 1. Fix V, W, update U
            # 2. Fix U, W, update V  
            # 3. Fix U, V, update W
            
            # Update U: For each (i,j), we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            # For fixed i,j: sum_r U[i*m+j, r] * sum_k V[j*p+k, r] * W[k*n+i, r] = 1
            # Let A_ij[r] = sum_k V[j*p+k, r] * W[k*n+i, r]
            # Then U[i*m+j, :] @ A_ij = 1
            
            # Update V: For each (j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
            # For fixed j,k: sum_r V[j*p+k, r] * sum_i U[i*m+j, r] * W[k*n+i, r] = 1
            # Let B_jk[r] = sum_i U[i*m+j, r] * W[k*n+i, r]
            # Then V[j*p+k, :] @ B_jk = 1
            
            # Update W: For each (k,i), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
            # For fixed k,i: sum_r W[k*n+i, r] * sum_j U[i*m+j, r] * V[j*p+k, r] = 1
            # Let C_ki[r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            # Then W[k*n+i, :] @ C_ki = 1
            
            for _ in range(3):  # 3 inner iterations per outer
                # Update U
                for i in range(n):
                    for j in range(m):
                        A_ij = np.zeros(r)
                        for k in range(p):
                            A_ij += V[j * p + k, :] * W[k * n + i, :]
                        # Solve U[i*m+j, :] @ A_ij = 1
                        # This is a linear system: A_ij @ U[i*m+j, :].T = 1
                        # Use lstsq
                        if np.linalg.norm(A_ij) > 1e-12:
                            U[i * m + j, :] = np.linalg.lstsq(A_ij, np.ones(1), rcond=None)[0]
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        B_jk = np.zeros(r)
                        for i in range(n):
                            B_jk += U[i * m + j, :] * W[k * n + i, :]
                        if np.linalg.norm(B_jk) > 1e-12:
                            V[j * p + k, :] = np.linalg.lstsq(B_jk, np.ones(1), rcond=None)[0]
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        C_ki = np.zeros(r)
                        for j in range(m):
                            C_ki += U[i * m + j, :] * V[j * p + k, :]
                        if np.linalg.norm(C_ki) > 1e-12:
                            W[k * n + i, :] = np.linalg.lstsq(C_ki, np.ones(1), rcond=None)[0]
            
            # Check convergence
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        error += abs(val - 1.0)
            
            if error < 1e-8:
                # Verify
                if verify_decomposition(U, V, W):
                    return (U, V, W)
        
        return None
    
    # Try ranks from 62 down to some minimum
    # For 3x3 * 3x7, the theoretical lower bound is max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But in practice, it's likely higher. Let's try from 62 down.
    
    for r in range(rank_schoolbook - 1, 20, -1):
        if time_left() < 10:
            break
        
        result = try_rank(r, max_iters=100, seed=r)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_U, best_V, best_W = U, V, W
            # Continue trying lower ranks
            continue
    
    # If we found a better rank, use it; otherwise use schoolbook
    if best_rank < rank_schoolbook:
        U_final, V_final, W_final = best_U, best_V, best_W
    else:
        U_final, V_final, W_final = U_sb, V_sb, W_sb
    
    # Convert to lists
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
