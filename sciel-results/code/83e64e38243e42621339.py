"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    The schoolbook rank is 4*4*6 = 96.
    We will attempt to find a lower rank decomposition.
    
    Known results for small matrix multiplication:
    - 2x2 * 2x2: rank 7 (Strassen)
    - 2x2 * 2x3: rank ?
    - 4x4 * 4x4: rank ? (best known is 49)
    
    For 4x4 * 4x6, we can try to construct a decomposition using Kronecker products
    or other structured approaches.
    
    Strategy:
    1. Start with the schoolbook decomposition (rank 96)
    2. Try to find a lower rank decomposition using random search or structured methods
    3. Use the fact that 4 = 2*2 and 6 = 2*3 to potentially use Kronecker products
    
    The tensor for matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p
    can be written as:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    In tensor form: T[i,j,k] = e_i * e_j * e_k for each (i,j,k)
    
    A rank-1 decomposition term is: U[:,c] * V[:,c] * W[:,c]
    where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    The convention is W[k*n + i, c] (note: p*n, not n*p)
    
    For a rank-1 term to contribute to C[i,k] from A[i,j]*B[j,k]:
    U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for the specific (i,j,k)
    and 0 otherwise (or sum over c gives the correct result)
    
    Actually, the tensor identity is:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    This is a system of n*m*p equations with rank * (n*m + m*p + p*n) unknowns.
    
    For n=4, m=4, p=6: 4*4*6 = 96 equations
    Unknowns: rank * (16 + 24 + 24) = rank * 64
    
    For rank 96: 96 * 64 = 6144 unknowns for 96 equations - very underdetermined
    For rank 48: 48 * 64 = 3072 unknowns for 96 equations - still underdetermined
    
    Let's try to find a low-rank decomposition.
    """
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Use random initialization and iterative refinement
    
    def try_rank(rank):
        """Try to find a rank-r decomposition."""
        if rank >= best_rank:
            return None
        
        # Random initialization
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Iterative refinement: fix two matrices, solve for the third
        for iteration in range(50):
            # Fix V, W, solve for U
            # For each (i,j): sum_k V[j*p+k, c] * W[k*n+i, c] * U[i*m+j, c] = 1
            # This is: (V_j * W_i^T) dot U_i_j = 1 where V_j is row j of V reshaped
            # More precisely: sum_c U[i*m+j, c] * (sum_k V[j*p+k, c] * W[k*n+i, c]) = 1
            
            # Build the linear system for U
            # For each (i,j), we have one equation
            # Let A[i,j,c] = sum_k V[j*p+k, c] * W[k*n+i, c]
            # Then sum_c A[i,j,c] * U[i*m+j, c] = 1
            
            A = np.zeros((n, m, rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        A[i, j, :] += V[j * p + k, :] * W[k * n + i, :]
            
            # Solve for U
            for i in range(n):
                for j in range(m):
                    # sum_c A[i,j,c] * U[i*m+j, c] = 1
                    # This is a single equation with rank unknowns
                    # Underdetermined if rank > 1
                    # Use minimum norm solution
                    if np.any(A[i, j, :] != 0):
                        U[i * m + j, :] = A[i, j, :] / np.dot(A[i, j, :], A[i, j, :])
                    else:
                        return None  # Cannot solve
            
            # Fix U, W, solve for V
            A = np.zeros((m, p, rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        A[j, k, :] += U[i * m + j, :] * W[k * n + i, :]
            
            for j in range(m):
                for k in range(p):
                    if np.any(A[j, k, :] != 0):
                        V[j * p + k, :] = A[j, k, :] / np.dot(A[j, k, :], A[j, k, :])
                    else:
                        return None
            
            # Fix U, V, solve for W
            A = np.zeros((p, n, rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        A[k, i, :] += U[i * m + j, :] * V[j * p + k, :]
            
            for k in range(p):
                for i in range(n):
                    if np.any(A[k, i, :] != 0):
                        W[k * n + i, :] = A[k, i, :] / np.dot(A[k, i, :], A[k, i, :])
                    else:
                        return None
        
        # Verify the solution
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    max_err = max(max_err, abs(val - 1))
        
        if max_err < 1e-6:
            # Try to round to integers or half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            max_err_rounded = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_rounded[i * m + j, :], V_rounded[j * p + k, :] * W_rounded[k * n + i, :])
                        max_err_rounded = max(max_err_rounded, abs(val - 1))
            
            if max_err_rounded < 1e-6:
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            else:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Try ranks from low to high
    for rank in range(24, best_rank):
        for attempt in range(10):
            np.random.seed(attempt * 1000 + rank)
            result = try_rank(rank)
            if result is not None:
                return result
    
    return best
# EVOLVE-BLOCK-END
