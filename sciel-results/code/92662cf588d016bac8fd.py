"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W form a valid decomposition.
    For each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1.
    For all other index combos, the sum should be 0.
    """
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Efficient: for each c, outer product of U[:,c], V[:,c], W[:,c]
    # But that's rank * (n*m * m*p * p*n) which is 32 * 256 = 8192 ops, fine.
    # Actually let's just do it directly for the small sizes.
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        uc = Ua[:, c]  # n*m
        vc = Va[:, c]  # m*p
        wc = Wa[:, c]  # p*n
        # Reshape
        u = uc.reshape(n, m)
        v = vc.reshape(m, p)
        w = wc.reshape(p, n)
        # T += sum over i,j,k of u[i,j]*v[j,k]*w[k,i]
        # This is the trace of u @ v @ w (but w is p*n, need to be careful)
        # Actually: sum_i,j,k u[i,j] v[j,k] w[k,i] = trace(u @ v @ w)
        # u is n x m, v is m x p, w is p x n -> u@v is n x p, (u@v)@w is n x n, trace is sum_i (u@v)@w[i,i]
        # But we need the full tensor, not just the trace.
        # Let me just compute T directly for small sizes.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    
    # For a valid decomposition, T[i,j,k] should be 1 for all i,j,k
    # Wait, that's not right. The matrix multiplication tensor is:
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor T[i,j,k] = 1 if this is the identity contribution... 
    # Actually no. The matrix multiplication tensor M[i,j,k] = 1 for all i,j,k
    # because (A*B)[i,k] = sum_j A[i,j] * B[j,k], and each term A[i,j]*B[j,k]
    # contributes to output (i,k). The tensor is the trilinear form where
    # sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[i,k] = sum_{i,k} C[i,k] * (A*B)[i,k]
    # So T[i,j,k] = 1 for all i,j,k.
    
    # The decomposition says: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # So we need T[i,j,k] = 1 for all i,j,k.
    if np.max(np.abs(T - 1.0)) > tol:
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120.0
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions using structured approaches
    # For 2x4x4, the schoolbook rank is 32.
    # Known lower bounds: max(n*m, m*p, p*n) = max(8, 16, 8) = 16
    # Strassen-style or other algorithms might do better.
    
    # Let's try a systematic search for rank r decompositions with rational entries
    # For small r, we can try random initialization and gradient-free search.
    
    # Actually, let's think about this differently. The matrix multiplication tensor
    # for (n,m,p) = (2,4,4) has a specific structure. Let's try to exploit it.
    
    # One approach: use the fact that we're multiplying a 2x4 by a 4x4.
    # (A*B)[0,0] = sum_j A[0,j]*B[j,0]
    # (A*B)[0,1] = sum_j A[0,j]*B[j,1]
    # (A*B)[0,2] = sum_j A[0,j]*B[j,2]
    # (A*B)[0,3] = sum_j A[0,j]*B[j,3]
    # (A*B)[1,0] = sum_j A[1,j]*B[j,0]
    # etc.
    
    # A key insight: if we can share computations between rows of A or columns of B,
    # we can reduce the rank.
    
    # Let's try a simple structured decomposition:
    # For each output element (i,k), we need sum_j A[i,j]*B[j,k]
    # We can write this as a sum of rank-1 terms in the (i,j,k) space.
    
    # Try rank = 16 first (the lower bound)
    # Then try 17, 18, ... up to 31
    
    # For each rank r, try random initializations and see if we can find a solution.
    # Use a simple iterative method: fix two matrices, solve for the third.
    
    np.random.seed(42)
    
    for r in range(16, min(best_rank, 32)):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations for this rank
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small integers
            U = np.random.randint(-2, 3, (n*m, r), dtype=np.float64)
            V = np.random.randint(-2, 3, (m*p, r), dtype=np.float64)
            W = np.random.randint(-2, 3, (p*n, r), dtype=np.float64)
            
            # Check if this random init works (very unlikely but cheap)
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                break
            
            # Try alternating least squares style: fix U,V, solve for W
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # For fixed U,V, this is linear in W.
            # Build the system: for each (i,j,k), sum_c (U[i*m+j,c]*V[j*p+k,c]) * W[k*n+i,c] = 1
            # The unknowns are W[k*n+i,c] for all k,i,c.
            # There are p*n*r unknowns.
            # There are n*m*p equations.
            
            # Let's build the linear system.
            # Variable index: (k,i,c) -> k*n + i, then c -> (k*n+i)*r + c
            # Equation for (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[(k*n+i)*r + c] = 1
            
            num_eq = n * m * p
            num_var = p * n * r
            
            if num_var >= num_eq:
                # Overdetermined or square system
                A_mat = np.zeros((num_eq, num_var), dtype=np.float64)
                b_vec = np.ones(num_eq, dtype=np.float64)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(r):
                                val = U[i*m+j, c] * V[j*p+k, c]
                                if val != 0:
                                    var_idx = (k*n+i)*r + c
                                    A_mat[eq_idx, var_idx] = val
                            eq_idx += 1
                
                # Solve least squares
                try:
                    W_sol, residuals, rank_sys, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W_sol = W_sol.reshape(p*n, r)
                    
                    # Check solution
                    if _verify(U.tolist(), V.tolist(), W_sol.tolist(), n, m, p, tol=1e-6):
                        best = {"U": U.tolist(), "V": V.tolist(), "W": W_sol.tolist()}
                        best_rank = r
                        break
                except:
                    pass
        
        if best_rank == r:
            break
    
    return best
# EVOLVE-BLOCK-END
