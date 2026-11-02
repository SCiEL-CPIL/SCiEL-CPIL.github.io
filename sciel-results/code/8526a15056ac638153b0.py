"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the multiplication tensor T[i,j,k] = delta (for MM: C[i,k] = sum_j A[i,j]*B[j,k])
    # The tensor decomposition condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta_{i,i'}*delta_{j,j'}*delta_{k,k'}
    # Actually for MM(A,B) where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i,j,k] = 1 if this is the "identity" operation
    # Wait, let me think about this more carefully.
    
    # The matrix multiplication tensor M_{i,j,k} where we compute C = A * B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The tensor T[i,j,k] = delta (it's 1 for all i,j,k in the "trivial" sense)
    # No wait. The decomposition is:
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal the multiplication tensor
    # 
    # For standard MM: T[i,j,k] = 1 for all valid i,j,k (since every product term contributes)
    # Actually no. Let me reconsider.
    #
    # The multiplication tensor for MM(n,m,p) has entries:
    # T[i,j,k] = 1 if and only if... hmm.
    #
    # Actually, the way tensor decomposition works for MM:
    # We want: for all A (n x m) and B (m x p), C = A*B
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # In tensor form, this is a bilinear map. The tensor T in {0,1}^{n x m x p} 
    # where T[i,j,k] = 1 for all i,j,k? No...
    #
    # Let me look at the schoolbook: U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1
    # for each (i,j,k). So for a given (i,j,k), the contribution is U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1*1*1 = 1.
    # 
    # The reconstructed tensor at position (i,j,k) is sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # In schoolbook, each (i,j,k) has exactly one c where all three are 1, so the sum is 1.
    # For other (i',j',k'), if (i',j',k') != (i,j,k), at most one of the three factors is nonzero... 
    # Actually no, in schoolbook each c corresponds to exactly one (i,j,k) triple.
    # So for a fixed (i',j',k'), sum_c U[i'*m+j',c]*V[j'*p+k',c]*W[k'*n+i',c] = 1 if there exists c with 
    # U[i'*m+j',c]=V[j'*p+k',c]=W[k'*n+i',c]=1, which is exactly when (i',j',k') matches the triple for c.
    # So the reconstructed tensor is T[i,j,k] = 1 for all i,j,k.
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _try_random_search(n, m, p, max_rank, time_limit, rng):
    """Try to find a low-rank decomposition via random search with refinement."""
    start = time.time()
    
    # Start with a reasonable rank and try to reduce
    for rank in range(max_rank, 15, -1):
        if time.time() - start > time_limit:
            break
            
        for attempt in range(20):
            if time.time() - start > time_limit:
                break
                
            # Random initialization
            U = rng.standard_normal((n*m, rank))
            V = rng.standard_normal((m*p, rank))
            W = rng.standard_normal((p*n, rank))
            
            # Simple gradient-free search: randomly perturb and keep if error decreases
            # First compute initial error
            def compute_error(U, V, W):
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(rank):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            err = abs(s - 1.0)
                            if err > max_err:
                                max_err = err
                return max_err
            
            err = compute_error(U, V, W)
            
            # Try to reduce error
            for step in range(100):
                if time.time() - start > time_limit:
                    break
                if err < 1e-9:
                    break
                    
                # Random perturbation
                factor = 0.1 if err > 0.1 else 0.01
                dU = rng.standard_normal((n*m, rank)) * factor
                dV = rng.standard_normal((m*p, rank)) * factor
                dW = rng.standard_normal((p*n, rank)) * factor
                
                new_U = U + dU
                new_V = V + dV
                new_W = W + dW
                
                new_err = compute_error(new_U, new_V, new_W)
                
                if new_err < err:
                    U, V, W = new_U, new_V, new_W
                    err = new_err
            
            if err < 1e-9:
                # Snap to nearest integers/half-integers
                U_int = np.round(2*U)/2
                V_int = np.round(2*V)/2
                W_int = np.round(2*W)/2
                
                if _verify(U_int, V_int, W_int, n, m, p):
                    return U_int.tolist(), V_int.tolist(), W_int.tolist(), rank
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Schoolbook gives rank 54
    schoolbook_rank = n * m * p
    
    # Try to find better decomposition
    # Known results: MM(3,3,3) has rank 23 (Strassen-like), MM(3,3,6) likely has lower rank than 54
    
    # For now, let's try a structured approach
    # The multiplication tensor for 3x3 * 3x6
    
    # Try random search for ranks below schoolbook
    rng = np.random.default_rng(42)
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Try structured initializations based on block structure
    # 3x3 * 3x6 can be seen as three 3x3 * 3x2 blocks
    
    # Let's try a systematic approach: enumerate small-rank candidates
    # This is a hard problem, so let's try a few strategies
    
    # Strategy 1: Try rank 30-50 with random search
    for target_rank in [45, 42, 40, 38, 36, 34, 32, 30]:
        if time.time() > deadline:
            break
            
        result = _try_random_search(n, m, p, target_rank, 20, rng)
        if result is not None:
            U, V, W, rank = result
            best_result = (U, V, W)
            best_rank = rank
            break
    
    if best_result is not None:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    # Fall back to schoolbook
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
