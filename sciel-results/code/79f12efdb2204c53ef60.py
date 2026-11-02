"""Matrix multiplication 2x3x10 - rank-minimal tensor decomposition search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin within 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # The multiplication tensor T[i,j,k] = 1 if the product involves A[i,j]*B[j,k]
    # For 2x3 * 3x10 -> 2x10 output
    # T has indices (i, j, k) where i in [0,1], j in [0,2], k in [0,9]
    # T[i,j,k] = 1 (the tensor is the identity-like multiplication tensor)
    
    # Schoolbook: rank 60
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Strategy: Use the fact that for 2x3x10, we can potentially use
    # constructions from small matrix multiplication algorithms
    
    # One approach: treat the 3x10 part as 3x(3+3+4) and use blocking
    # Another: use known results for 2xm x mxp
    
    # Let's try a direct algebraic search for low-rank decompositions
    # using random initialization and optimization
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly represents matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # The tensor equation: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For all valid (i,j,k), T[i,j,k] = 1
        # For all other combinations, the product should give 0 (but actually the tensor
        # is only defined on valid indices)
        
        # Actually, the multiplication tensor for A(nxm) * B(mxq) -> C(nxq)
        # has T[i,j,k] = 1 for all i,j,k in their ranges, and the decomposition
        # must satisfy: for each output position (i,k), 
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) * (sum_k' V[j*p+k',c] B[j,k'])
        # Wait, let me re-derive.
        
        # The standard formulation: We want to compute C = A*B where A is n x m, B is m x p
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # 
        # A rank-r decomposition gives:
        # C[i,k] = sum_{c=0}^{r-1} (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
        # 
        # For this to equal sum_j A[i,j]*B[j,k] for ALL A and B, we need:
        # sum_c U[i*m+j1, c] * V[j2*p+k, c] = delta[j1,j2] * (coefficient for output (i,k))
        # 
        # Actually the correct condition is:
        # For each (i, k): sum_j A[i,j] B[j,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) (sum_j' V[j'*p+k,c] B[j',k])
        # 
        # Expanding the RHS: sum_c sum_j sum_j' U[i*m+j,c] V[j'*p+k,c] A[i,j] B[j',k]
        # 
        # For this to equal sum_j A[i,j] B[j,k] for all A, B:
        # sum_c U[i*m+j,c] V[j'*p+k,c] = delta_{j,j'} for each (i,k)
        # 
        # Wait, that's not quite right either. Let me think again.
        # 
        # The W matrix: W[k*n+i, c] is the coefficient. The standard formulation is:
        # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
        # 
        # For this to work: sum_c W[k*n+i,c] * U[i*m+j,c] * V[j'*p+k,c] = delta_{j,j'} for all i,k,j,j'
        
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        expected = 1.0 if j1 == j2 else 0.0
                        actual = 0.0
                        for c in range(rank):
                            actual += W[k * n + i, c] * U[i * m + j1, c] * V[j2 * p + k, c]
                        if abs(actual - expected) > 1e-9:
                            return False
        return True
    
    def try_rank(r, max_iters=500, seed=42):
        """Try to find a rank-r decomposition using random search."""
        if time.time() > deadline - 10:
            return None
        
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Scale to reasonable magnitudes
        U /= np.sqrt(r)
        V /= np.sqrt(r)
        W /= np.sqrt(r)
        
        best_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Compute current error
            err = 0.0
            for i in range(n):
                for k in range(p):
                    for j1 in range(m):
                        for j2 in range(m):
                            expected = 1.0 if j1 == j2 else 0.0
                            actual = 0.0
                            for c in range(r):
                                actual += W[k * n + i, c] * U[i * m + j1, c] * V[j2 * p + k, c]
                            err += (actual - expected) ** 2
            
            if err < best_err:
                best_err = err
            
            if best_err < 1e-12:
                # Check if entries are close to integers/half-integers
                break
            
            # Gradient-free update: perturb randomly
            lr = 0.1 / (1 + it * 0.01)
            U += lr * np.random.randn(n * m, r)
            V += lr * np.random.randn(m * p, r)
            W += lr * np.random.randn(p * n, r)
        
        if best_err < 1e-12:
            # Snap to nearest half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify the snapped version
            U_list = U_rounded.tolist()
            V_list = V_rounded.tolist()
            W_list = W_rounded.tolist()
            if verify_decomposition(U_list, V_list, W_list, r):
                return {'U': U_list, 'V': V_list, 'W': W_list}
        
        return None
    
    # Try ranks from low to high, starting with known bounds
    # Known: R(2,2,2)=7, so R(2,3,3) <= some value
    # For 2x3x10, let's try a range of ranks
    
    # Try some specific ranks that might work based on known constructions
    ranks_to_try = []
    
    # Based on known results, try ranks below 60
    # Start with a few promising ones
    for r in [20, 24, 28, 30, 32, 35, 40, 45, 50, 55]:
        if time.time() > deadline - 20:
            break
        
        result = try_rank(r, max_iters=200, seed=r)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    # If nothing found, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
