"""Matrix multiplication 2,3,6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    
    The schoolbook rank is 2*3*6 = 36.
    We try to find a lower rank decomposition.
    
    For small matrix multiplications, we can use known results:
    - 2x3 * 3x6: The tensor is T ∈ R^{(2*3) × (3*6) × (6*2)} = R^{6 × 18 × 12}
    
    We'll attempt to find a decomposition of rank r by:
    1. First try the Kronecker structure: 6 = 3*2, so we can think of this as
       (2x3) * (3x(2*3)) which relates to (2x3)*(3x3) tensor (1x2)
    
    2. Actually, let me think more carefully. The multiplication (2x3)*(3x6) -> (2x6)
       can be viewed as: for each pair (i,k) with i in {0,1}, k in {0,...,5},
       C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor has entries T[i*m+j, j*p+k, k*n+i] = 1 (using the trace convention W[k*n+i, r]).
    
    Let me try a structured approach. Since 6 = 3*2, we can write k = k1*2 + k2 where k1 in {0,1}, k2 in {0,1}.
    
    Actually, let me just try to find a low-rank decomposition using numerical optimization.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # The tensor T is of size (n*m) x (m*p) x (p*n) = 6 x 18 x 12
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # Otherwise T[a, b, c] = 0
    
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try ranks from lower bound to schoolbook
    # Lower bound: max(n*m, m*p, p*n) / min(n,m,p) ... actually the lower bound is max(n, m, p) * min(n, m, p)
    # For (2,3,6): lower bound is at least max(6, 18, 12) = 18? No.
    # The rank must be at least the maximum of the "slices" in any mode.
    # Mode-0 slice rank: for each a, the matrix T[a,:,:] is 18x12. 
    # Actually, a better lower bound: the tensor rank r must satisfy that the matrix flattening
    # in any mode has rank <= r.
    
    # Let's compute the rank of each matricization
    # Mode-0: T reshaped to (n*m) x (m*p * p*n) = 6 x 216
    T0 = T.reshape(n*m, m*p * p*n)
    T1 = T.reshape(m*p, n*m * p*n)
    T2 = T.reshape(p*n, n*m * m*p)
    
    r0 = np.linalg.matrix_rank(T0, tol=1e-10)
    r1 = np.linalg.matrix_rank(T1, tol=1e-10)
    r2 = np.linalg.matrix_rank(T2, tol=1e-10)
    
    lower_bound = max(r0, r1, r2)
    
    # Try to find decomposition starting from lower_bound
    best_rank = n * m * p  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Try increasing ranks
    for rank in range(lower_bound, n * m * p + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Scale to avoid issues
            U /= np.linalg.norm(U) + 1e-10
            V /= np.linalg.norm(V) + 1e-10
            W /= np.linalg.norm(W) + 1e-10
            
            # Alternating least squares / gradient descent
            # We want to minimize ||T - sum_r U[:,r] V[:,r]^T W[:,r]^T||_F
            # i.e., minimize ||T - (U ⊗ V ⊗ W)^T vec(T)||^2 where ⊗ is Khatri-Rao product
            
            # Using ALS: fix two, solve for the third
            converged = False
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                # Solve for U: for each column r of U, we have
                # U[:,r] = argmin ||T - sum_s U_s V_s^T W_s^T||
                # This is equivalent to: U[:,r] * (V[:,r] ⊙ W[:,r]) should approximate T
                # where ⊙ is Khatri-Rao product
                # 
                # More precisely, if we fix V and W, then for each r:
                # U[:,r] = argmin_u ||vec(T) - sum_s u_s * (V[:,s] ⊙ W[:,s])||^2
                # So we solve: (V_kr_W)^T (V_kr_W) U = (V_kr_W)^T vec(T)
                # where V_kr_W is the Khatri-Rao product of V and W (m*p*p*n x rank)
                
                VkrW = np.kron(V, W)  # This gives (m*p * p*n) x rank
                # Actually, Khatri-Rao product: column r is V[:,r] ⊗ W[:,r]
                VkrW = np.zeros((m*p * p*n, rank))
                for r in range(rank):
                    VkrW[:, r] = np.kron(V[:, r], W[:, r])
                
                # Solve least squares for U
                try:
                    U_new, _, _, _ = np.linalg.lstsq(VkrW, T.reshape(-1), rcond=None)
                except:
                    break
                
                U_new = U_new.reshape(n*m, rank)
                
                # Solve for V
                WkrU = np.zeros((p*n * n*m, rank))
                for r in range(rank):
                    WkrU[:, r] = np.kron(W[:, r], U[:, r])
                try:
                    V_new, _, _, _ = np.linalg.lstsq(WkrU, T.reshape(-1), rcond=None)
                except:
                    break
                V_new = V_new.reshape(m*p, rank)
                
                # Solve for W
                UkrV = np.zeros((n*m * m*p, rank))
                for r in range(rank):
                    UkrV[:, r] = np.kron(U[:, r], V[:, r])
                try:
                    W_new, _, _, _ = np.linalg.lstsq(UkrV, T.reshape(-1), rcond=None)
                except:
                    break
                W_new = W_new.reshape(p*n, rank)
                
                # Check convergence
                U, V, W = U_new, V_new, W_new
                
                # Check if we've achieved exact decomposition
                # Reconstruct and check
                recon = np.zeros((n*m, m*p, p*n))
                for r in range(rank):
                    recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                
                error = np.max(np.abs(recon - T))
                
                if error < 1e-8:
                    converged = True
                    break
            
            if converged:
                best_rank = rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                break
        
        if best_rank == rank:
            break
    
    # If no better decomposition found, use schoolbook
    if best_U is None:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Round to nearest integers or half-integers if possible
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        # Verify the rounded version
        recon = np.zeros((n*m, m*p, p*n))
        for r in range(best_rank):
            recon += np.einsum('a,b,c->abc', U_rounded[:, r], V_rounded[:, r], W_rounded[:, r])
        
        if np.max(np.abs(recon - T)) < 1e-8:
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        else:
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
