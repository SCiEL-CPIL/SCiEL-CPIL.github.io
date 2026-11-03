"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
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


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # should equal 1 if i,j,k form a valid multiplication term, else 0.
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
    # in the standard formulation where C[i,k] = sum_j A[i,j]*B[j,k].
    # The decomposition is: C[i,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_k V[j*p+k,c]*B[j,k])... 
    # Wait, let me re-read the convention.
    
    # The tensor for matrix multiplication (n x m) * (m x p) -> (n x p) is:
    # T[i, j, k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # In the standard rank decomposition:
    # C[i,k] = sum_c U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] ... this doesn't look right.
    
    # Let me think about this more carefully.
    # The matrix multiplication tensor is a 3-tensor T with indices (i, j, k) where:
    # i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 (it's the identity tensor for the multiplication)
    
    # A rank-r decomposition writes T as sum of r rank-1 tensors:
    # T[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # But the interface says:
    # U has shape (n*m, rank), indexed as U[i*m+j, c]
    # V has shape (m*p, rank), indexed as V[j*p+k, c]  
    # W has shape (p*n, rank), indexed as W[k*n+i, c]
    
    # So the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And T[i,j,k] = 1 for all valid i,j,k.
    
    # Let's verify:
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol


def _try_rank_reduction(U, V, W, n, m, p, max_time=10):
    """Try to reduce the rank of a decomposition by merging/removing terms."""
    start_time = time.time()
    rank = len(U[0])
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    best_rank = rank
    
    # Try to find pairs of columns that can be merged
    for attempt in range(50):
        if time.time() - start_time > max_time:
            break
        if best_rank <= 12:  # We want to get below schoolbook rank of 48
            break
            
        c1 = random.randint(0, best_rank - 1)
        c2 = random.randint(0, best_rank - 1)
        if c1 == c2:
            continue
            
        # Try to merge columns c1 and c2 into one
        # We need: U[:,c1]*V[:,c1]*W[:,c1] + U[:,c2]*V[:,c2]*W[:,c2] ≈ U_new*V_new*W_new
        # This is generally not possible, but let's try a simple approach:
        # Set one to zero and see if the other can be scaled to compensate
        
        # Actually, a better approach: try to find a linear combination
        # Let's try: new_col = U[:,c1] * a + U[:,c2] * b, etc.
        # But this is underdetermined. Let's just try removing one column and
        # scaling the other to see if it helps.
        
        # Try removing column c2
        new_U = np.delete(U_arr, c2, axis=1)
        new_V = np.delete(V_arr, c2, axis=1)
        new_W = np.delete(W_arr, c2, axis=1)
        
        # Verify
        if _verify(new_U.tolist(), new_V.tolist(), new_W.tolist(), n, m, p, tol=1e-7):
            U_arr, V_arr, W_arr = new_U, new_V, new_W
            best_rank -= 1
    
    if best_rank < rank:
        return U_arr.tolist(), V_arr.tolist(), W_arr.tolist(), best_rank
    return None


def _random_search(n, m, p, target_rank, max_time=30, seed=42):
    """Random search for a low-rank decomposition."""
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    best_rank = None
    best_decomp = None
    
    for r in range(target_rank, 12):  # Try ranks from target_rank up to 11
        if time.time() - start_time > max_time:
            break
            
        for trial in range(100):
            if time.time() - start_time > max_time:
                break
                
            # Initialize with random small values
            U = rng.randn(n*m, r) * 0.1
            V = rng.randn(m*p, r) * 0.1
            W = rng.randn(p*n, r) * 0.1
            
            # Use gradient-free optimization (Nelder-Mead style or just random perturbation)
            # For simplicity, let's use a simple iterative refinement
            for iteration in range(200):
                if time.time() - start_time > max_time:
                    break
                    
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err += (val - 1.0)**2
                
                if err < 1e-10:
                    if best_rank is None or r < best_rank:
                        best_rank = r
                        best_decomp = (U.copy(), V.copy(), W.copy())
                    break
                    
                # Perturb
                lr = 0.01 * (1 - iteration/200)
                U += rng.randn(n*m, r) * lr
                V += rng.randn(m*p, r) * lr
                W += rng.randn(p*n, r) * lr
    
    if best_decomp is not None:
        U, V, W = best_decomp
        # Convert to fractions for exactness
        U_frac = [[Fraction(x).limit_denominator(1000) for x in row] for row in U.tolist()]
        V_frac = [[Fraction(x).limit_denominator(1000) for x in row] for row in V.tolist()]
        W_frac = [[Fraction(x).limit_denominator(1000) for x in row] for row in W.tolist()]
        
        # Verify with fractions
        if _verify(U_frac, V_frac, W_frac, n, m, p, tol=0):
            return U_frac, V_frac, W_frac, best_rank
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    # Start with schoolbook
    result = _schoolbook()
    best_rank = n * m * p
    
    # Try to improve
    while time.time() < deadline - 10:
        # Try random search with decreasing target rank
        for target in range(13, best_rank):
            if time.time() > deadline - 10:
                break
            res = _random_search(n, m, p, target, max_time=min(30, deadline - time.time() - 10))
            if res is not None:
                U, V, W, r = res
                if r < best_rank:
                    best_rank = r
                    result = {"U": U, "V": V, "W": W}
    
    return result

# EVOLVE-BLOCK-END
