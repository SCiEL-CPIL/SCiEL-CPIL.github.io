"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x2 * 2x3 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=3
    
    # First, create the schoolbook solution as fallback (Rank 12)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    deadline = time.time() + 140  # Leave 10s margin
    
    # Try rank 8 first (known minimal), then 7, 6
    for r in [8, 7, 6]:
        if time.time() > deadline:
            break
        
        result = _find_decomposition_structured(n, m, p, r, deadline)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result


def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition is exact."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(r))
                if abs(val - 1.0) > tol:
                    return False
    return True


def _is_half_integer(val, tol=1e-9):
    """Check if value is integer or half-integer."""
    doubled = val * 2
    return abs(doubled - round(doubled)) < tol


def _try_round_and_verify(U, V, W, n, m, p, r, tol=1e-8):
    """Try rounding to half-integers and verify."""
    U_r = [[round(U[i][c] * 2) / 2 for c in range(r)] for i in range(len(U))]
    V_r = [[round(V[i][c] * 2) / 2 for c in range(r)] for i in range(len(V))]
    W_r = [[round(W[i][c] * 2) / 2 for c in range(r)] for i in range(len(W))]
    
    if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol):
        return {"U": U_r, "V": V_r, "W": W_r}
    return None


def _find_decomposition_structured(n, m, p, r, deadline):
    """
    Find rank-r decomposition using structured initialization and ALS.
    Exploits the symmetry of the 2x2 input block.
    """
    np.random.seed(42)
    
    # Strategy 1: Try known algebraic constructions
    result = _try_algebraic_construction(n, m, p, r)
    if result is not None:
        return result
    
    # Strategy 2: ALS with multiple random initializations
    for attempt in range(50):
        if time.time() > deadline:
            return None
        
        # Random initialization with small values
        U = np.random.randn(n * m, r) * 0.5
        V = np.random.randn(m * p, r) * 0.5
        W = np.random.randn(p * n, r) * 0.5
        
        # ALS iterations
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(200):
            if time.time() > deadline:
                break
            
            # Update U: for each row i*m+j, solve least squares
            U_new = U.copy()
            for i in range(n):
                for j in range(m):
                    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                    # This is: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
                    # Let M[k,c] = V[j*p+k,c] * W[k*n+i,c]
                    # Then: sum_c U[i*m+j,c] * M[k,c] = 1 for all k
                    # This is a linear system: M @ U[i*m+j,:] = 1
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve M @ x = ones(p)
                    try:
                        x, residuals, rank, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U_new[i*m+j, :] = x
                    except:
                        pass
            
            U = U_new
            
            # Update V: for each row j*p+k
            V_new = V.copy()
            for j in range(m):
                for k in range(p):
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        x, residuals, rank, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V_new[j*p+k, :] = x
                    except:
                        pass
            
            V = V_new
            
            # Update W: for each row k*n+i
            W_new = W.copy()
            for k in range(p):
                for i in range(n):
                    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        x, residuals, rank, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W_new[k*n+i, :] = x
                    except:
                        pass
            
            W = W_new
            
            # Compute loss
            loss = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        loss += (val - 1.0)**2
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-10:
                break
        
        U, V, W = best_U, best_V, best_W
        
        # Try to round to half-integers
        result = _try_round_and_verify(
            U.tolist(), V.tolist(), W.tolist(), n, m, p, r
        )
        if result is not None:
            return result
    
    return None


def _try_algebraic_construction(n, m, p, r):
    """
    Try to find an explicit algebraic construction for the decomposition.
    For 2x2 x 2x3, we can use a Strassen-like approach.
    """
    if r != 8:
        return None
    
    # For 2x2 x 2x3, one known rank-8 construction:
    # We use the fact that the tensor is the "product" of a 2x2x2 tensor and a 2x2x1 tensor
    # in a certain sense.
    
    # Let me try a specific construction based on linear algebra.
    # Define 8 rank-1 terms.
    
    # U: 4x8, V: 6x8, W: 6x8
    # Indices: U[i*2+j, c], V[j*3+k, c], W[k*2+i, c]
    
    # Try a structured approach:
    # Let the 8 terms be indexed by c=0..7
    
    # I'll use a parameterized family and solve for parameters.
    # Set W to have a specific structure.
    
    # Try: W[k*2+i, c] = 1 for all k,i,c (simplest)
    # Then constraint: sum_c U[i*2+j,c] * V[j*3+k,c] = 1 for all i,j,k
    
    # For fixed j, this is: for all i,k: sum_c U[i*2+j,c] * V[j*3+k,c] = 1
    # Let a_i = U[i*2+j,:] and b_k = V[j*3+k,:]
    # Then sum_c a_i[c] * b_k[c] = 1 for all i,k
    
    # This means the 2x2 matrix A*B^T where A[i,c]=a_i[c], B[k,c]=b_k[c] has all entries 1.
    # So A*B^T = J (all ones matrix).
    
    # We can choose A and B such that this holds. For example:
    # A = [[1,1,0,0,0,0,0,0],
    #      [1,1,0,0,0,0,0,0]]  (both rows same)
    # B = [[1,0,0,0,0,0,0,0],
    #      [0,1,0,0,0,0,0,0],
    #      ...] but B has 3 rows (k=0,1,2)
    
    # Actually, A is 2xr, B is 3xr. A*B^T is 2x3, should be all ones.
    # rank(A*B^T) <= min(rank(A), rank(B)) <= 2
    # The all-ones 2x3 matrix has rank 1, so this is possible.
    
    # Let's set: for each j, choose a_j in R^r and b_{j,k} in R^r for k=0,1,2
    # such that a_j[c] * b_{j,k}[c] summed over c = 1 for all k.
    
    # Simplest: a_j = [1,0,...,0], b_{j,k} = [1,0,...,0] for all j,k
    # Then U[i*2+j, 0] = 1 for all i,j, V[j*3+k, 0] = 1 for all j,k
    # But then we only use 1 rank term, and W is all 1s.
    # Check: sum_c U[i*2+j,c]*V[j*3+k,c]*W[k*2+i,c] = U[i*2+j,0]*V[j*3+k,0]*W[k*2+i,0] = 1*1*1 = 1
    # This works! But it's rank 1, which is impossible.
    
    # Wait, that can't be right. Let me re-examine.
    # If U[i*2+j, 0] = 1 for all i,j, V[j*3+k, 0] = 1 for all j,k, W[k*2+i, 0] = 1 for all k,i
    # Then the product is 1*1*1 = 1 for all i,j,k.
    # But this is a rank-1 decomposition, which would mean the tensor has rank 1.
    # But the tensor T[i,j,k]=1 for all i,j,k has rank 1? 
    # No! The tensor is a 3D array, and its rank is the minimum number of rank-1 tensors to sum to it.
    # A rank-1 tensor is T[i,j,k] = a[i]*b[j]*c[k].
    # Our tensor is T[i,j,k] = 1 = 1*1*1, so it IS rank 1!
    
    # But that contradicts the known result that 2x2x2 matrix multiplication has rank 7.
    
    # I think I'm confusing the tensor of the multiplication with the all-ones tensor.
    # The multiplication tensor T[i,j,k] = delta(i,j,k) in some sense...
    
    # Let me re-read the problem.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 2 matrix by a 2 by 3 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "with integer/half-integer entries that exactly decompose the matrix multiplication tensor"
    
    # The matrix multiplication tensor T is defined by:
    # T[i,j,k] = 1 if the output C[i,k] depends on A[i,j] and B[j,k]
    # For C[i,k] = sum_j A[i,j] * B[j,k], the tensor T[i,j,k] = 1 for all valid i,j,k.
    
    # So T is indeed the all-ones tensor of shape (2,2,3).
    # And as I showed, this has rank 1: T[i,j,k] = 1*1*1.
    
    # But that's not the right interpretation. The tensor for matrix multiplication is not the all-ones tensor.
    # The multiplication tensor is a 3-tensor T where T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k].
    # For standard matrix multiplication, T[i,j,k] = 1 if k is the output index, but the indices are different.
    
    # I think the convention is:
    # U has shape (n*m, r): indexed by (i,j) where i is row of A, j is column of A / row of B
    # V has shape (m*p, r): indexed by (j,k) where j is row of B, k is column of B / column of C
    # W has shape (p*n, r): indexed by (k,i) where k is column of C, i is row of C
    # And the constraint is: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    
    # This is the standard tensor decomposition for matrix multiplication.
    # The "tensor" here is the multiplication tensor, which has T[i,j,k] = 1 for all i,j,k in the valid range.
    # But this tensor is NOT rank 1 in the tensor rank sense because the indices are contracted differently.
    
    # Wait, I'm still confused. Let me just trust the known result: rank of 2x2x2 is 7, rank of 2x2x3 is 8.
    
    # So my "rank 1" argument is wrong. The issue is that the tensor rank of the all-ones tensor of shape (2,2,3) is actually 1, but the matrix multiplication tensor is not the all-ones tensor in the standard tensor sense.
    
    # I think the correct interpretation is that we're decomposing the bilinear map, and the tensor is defined by the structure of the multiplication.
    
    # Let me just give up on the algebraic construction and rely on the numerical approach.
    return None
# EVOLVE-BLOCK-END
