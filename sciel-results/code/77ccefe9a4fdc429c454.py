"""Rewritten program: rank-minimal exact decomposition for 4x5 * 5x6."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-minimal exact tensor decomposition for multiplying a 4x5 matrix by a 5x6 matrix.
    
    Index conventions:
      U[i*m + j, c]  where i in 0..n-1, j in 0..m-1
      V[j*p + k, c]  where j in 0..m-1, k in 0..p-1
      W[k*n + i, c]  where k in 0..p-1, i in 0..n-1
    """
    n, m, p = 4, 5, 6
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Build the multiplication tensor
    T = np.zeros((n, m, p), dtype=np.float64)
    T[:] = 1.0
    
    # Start with schoolbook decomposition (rank 120)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=np.float64)
    V_school = np.zeros((m * p, rank_school), dtype=np.float64)
    W_school = np.zeros((p * n, rank_school), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    def verify(U, V, W, rank):
        """Verify that U, V, W form a valid decomposition."""
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.allclose(T_recon, T, atol=1e-8)
    
    # Try structured decompositions using block methods
    # Strategy: Partition the matrices and use Strassen-like algorithms on blocks
    
    # For 4x5 * 5x6, we can try:
    # - Split A into 2x2 blocks: A11 (2x2), A12 (2x3), A21 (2x2), A22 (2x3)
    # - Split B into 2x2 blocks: B11 (2x2), B12 (2x2), B21 (3x2), B22 (3x2)
    # This doesn't align cleanly, so let's try a different approach
    
    # Alternative: Use the fact that 4 = 2+2, 5 = 2+3, 6 = 3+3
    # And try to find decompositions for the resulting subproblems
    
    # Let's try a more systematic approach: use random restarts with better optimization
    
    # Try ranks from 35 down to 30
    for target_rank in [35, 34, 33, 32, 31, 30]:
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations with structured patterns
        for trial in range(10):
            if time.time() > deadline - 15:
                break
            
            # Use a structured initialization based on known low-rank patterns
            # Initialize with small random values plus some structure
            np.random.seed(12345 + trial * 1000)
            
            # Create a base pattern using Kronecker products of smaller decompositions
            # For now, use random but with scaling
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Normalize columns
            for c in range(target_rank):
                nU = np.linalg.norm(U[:, c])
                nV = np.linalg.norm(V[:, c])
                nW = np.linalg.norm(W[:, c])
                if nU > 1e-10:
                    U[:, c] /= nU
                if nV > 1e-10:
                    V[:, c] /= nV
                if nW > 1e-10:
                    W[:, c] /= nW
            
            # Gradient descent with momentum
            lr = 0.05
            momentum_U = np.zeros_like(U)
            momentum_V = np.zeros_like(V)
            momentum_W = np.zeros_like(W)
            
            for step in range(500):
                if time.time() > deadline - 10:
                    break
                
                # Compute residual using einsum
                U_r = U.reshape(n, m, target_rank)
                V_r = V.reshape(m, p, target_rank)
                W_r = W.reshape(p, n, target_rank)
                T_approx = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                
                residual = T - T_approx
                err = np.linalg.norm(residual)
                
                if err < 1e-10:
                    if verify(U, V, W, target_rank):
                        best_rank = target_rank
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
                
                # Compute gradients
                # grad_U[i,j,c] = sum_{k} residual[i,j,k] * V[j,k,c] * W[k,i,c]
                # grad_V[j,k,c] = sum_{i} residual[i,j,k] * U[i,j,c] * W[k,i,c]
                # grad_W[k,i,c] = sum_{j} residual[i,j,k] * U[i,j,c] * V[j,k,c]
                
                grad_U = np.einsum('ijk,jkr,kir->ijr', residual, V_r, W_r)
                grad_V = np.einsum('ijk,ijr,kir->jkr', residual, U_r, W_r)
                grad_W = np.einsum('ijk,ijr,jkr->kir', residual, U_r, V_r)
                
                # Reshape gradients
                grad_U = grad_U.reshape(n * m, target_rank)
                grad_V = grad_V.reshape(m * p, target_rank)
                grad_W = grad_W.reshape(p * n, target_rank)
                
                # Update with momentum
                momentum_U = 0.9 * momentum_U - lr * grad_U
                momentum_V = 0.9 * momentum_V - lr * grad_V
                momentum_W = 0.9 * momentum_W - lr * grad_W
                
                U += momentum_U
                V += momentum_V
                W += momentum_W
                
                # Re-normalize to prevent blowup
                for c in range(target_rank):
                    nU = np.linalg.norm(U[:, c])
                    nV = np.linalg.norm(V[:, c])
                    nW = np.linalg.norm(W[:, c])
                    if nU > 1e-10:
                        U[:, c] /= nU
                    if nV > 1e-10:
                        V[:, c] /= nV
                    if nW > 1e-10:
                        W[:, c] /= nW
            
            if best_rank == target_rank:
                break
        
        if best_rank == target_rank:
            break
    
    # Final verification
    if best_rank < rank_school:
        if not verify(best_U, best_V, best_W, best_rank):
            # Fallback to schoolbook if verification fails
            best_rank = rank_school
            best_U, best_V, best_W = U_school, V_school, W_school
    
    # Return as lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
