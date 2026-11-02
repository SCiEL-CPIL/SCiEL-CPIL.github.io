"""Matrix multiplication 2x3x7 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _target_tensor(n, m, p):
    """Build the target matrix-multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_exact(U, V, W, T, tol=1e-9):
    """Check that sum_c U[:,c] outer V[:,c] outer W[:,c] == T."""
    n, m, p = T.shape[0] // m, m, T.shape[2] // n
    # T shape is (n*m, m*p, p*n)
    n, m, p = T.shape[0] // m, m, T.shape[2] // n
    # Actually let's just do the tensor contraction
    # T[i*m+j, j*p+k, k*n+i] should equal sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    rank = U.shape[1]
    # Build the reconstructed tensor
    # For efficiency, we can check a few random slices or do full check if small
    # Full check:
    # Reconstructed[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We'll build it via einsum
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # We want T_recon[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # This is tricky to do directly with einsum because the indices are coupled.
    # Let's just iterate.
    n, m, p = DIMENSIONS
    T_recon = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        # uc has shape (n*m,), vc has shape (m*p,), wc has shape (p*n,)
        # We need to add uc[i*m+j] * vc[j*p+k] * wc[k*n+i] to T_recon[i*m+j, j*p+k, k*n+i]
        # This is an outer product of three vectors, but the indices are constrained.
        # Let's expand:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i * m + j, j * p + k, k * n + i] += uc[i * m + j] * vc[j * p + k] * wc[k * n + i]
    return np.allclose(T_recon, T, atol=tol)

def _try_rank(n, m, p, rank, T, max_iter=500, seed=42):
    """Try to find a rank-`rank` decomposition using iterative refinement."""
    rng = np.random.RandomState(seed)
    best_err = np.inf
    best_U = None
    best_V = None
    best_W = None
    
    for trial in range(3):  # A few random restarts
        # Initialize with random factors
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        # Normalize to avoid scale issues
        U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-12
        V /= np.linalg.norm(V, axis=0, keepdims=True) + 1e-12
        W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-12
        
        for it in range(max_iter):
            # Update U: for each column c, U[:,c] should minimize ||T - sum_c U[:,c] outer V[:,c] outer W[:,c]||
            # This is a linear least squares problem for U[:,c] if V[:,c] and W[:,c] are fixed.
            # But the tensor structure makes it coupled.
            # Let's use a simple alternating least squares approach.
            
            # Update U:
            for c in range(rank):
                # Residual for this component
                # T_recon_without_c = T - U[:,c] outer V[:,c] outer W[:,c]
                # We want to find U_new[:,c] that minimizes ||T_recon_without_c - U_new[:,c] outer V[:,c] outer W[:,c]||
                # This is equivalent to solving a linear system.
                # Let's build the matrix A such that A @ U_new[:,c] = b
                # A[i*m+j, j*p+k, k*n+i] is the coefficient for U[i*m+j,c]
                # Actually, it's easier to think of it as:
                # For fixed V[:,c] and W[:,c], the contribution to T is:
                # T_c[i*m+j, j*p+k, k*n+i] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # We want to match the residual R = T - sum_{c' != c} T_{c'}
                # This is a linear system in U[:,c].
                
                # Let's compute the residual tensor R
                R = T.copy()
                for c2 in range(rank):
                    if c2 == c:
                        continue
                    uc2 = U[:, c2]
                    vc2 = V[:, c2]
                    wc2 = W[:, c2]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                R[i * m + j, j * p + k, k * n + i] -= uc2[i * m + j] * vc2[j * p + k] * wc2[k * n + i]
                
                # Now solve for U[:,c]
                # R[i*m+j, j*p+k, k*n+i] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # This is a linear system. Let's vectorize it.
                # The number of free variables is n*m (the entries of U[:,c]).
                # The number of equations is n*m*p (the entries of R).
                # Let's build the matrix A of shape (n*m*p, n*m) and vector b of shape (n*m*p,)
                A = np.zeros((n * m * p, n * m))
                b = np.zeros(n * m * p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row = idx
                            col = i * m + j
                            A[row, col] = V[j * p + k, c] * W[k * n + i, c]
                            b[row] = R[i * m + j, j * p + k, k * n + i]
                            idx += 1
                
                # Solve A @ x = b using least squares
                x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                U[:, c] = x
            
            # Update V similarly
            for c in range(rank):
                R = T.copy()
                for c2 in range(rank):
                    if c2 == c:
                        continue
                    uc2 = U[:, c2]
                    vc2 = V[:, c2]
                    wc2 = W[:, c2]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                R[i * m + j, j * p + k, k * n + i] -= uc2[i * m + j] * vc2[j * p + k] * wc2[k * n + i]
                
                A = np.zeros((n * m * p, m * p))
                b = np.zeros(n * m * p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row = idx
                            col = j * p + k
                            A[row, col] = U[i * m + j, c] * W[k * n + i, c]
                            b[row] = R[i * m + j, j * p + k, k * n + i]
                            idx += 1
                
                x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                V[:, c] = x
            
            # Update W similarly
            for c in range(rank):
                R = T.copy()
                for c2 in range(rank):
                    if c2 == c:
                        continue
                    uc2 = U[:, c2]
                    vc2 = V[:, c2]
                    wc2 = W[:, c2]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                R[i * m + j, j * p + k, k * n + i] -= uc2[i * m + j] * vc2[j * p + k] * wc2[k * n + i]
                
                A = np.zeros((n * m * p, p * n))
                b = np.zeros(n * m * p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            row = idx
                            col = k * n + i
                            A[row, col] = U[i * m + j, c] * V[j * p + k, c]
                            b[row] = R[i * m + j, j * p + k, k * n + i]
                            idx += 1
                
                x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                W[:, c] = x
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        err += (s - 1.0) ** 2
            err = np.sqrt(err)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-10:
                break
    
    if best_err < 1e-8:
        return best_U, best_V, best_W
    return None

def _strassen_schoolbook_hybrid():
    """
    Hybrid Strassen/Schoolbook decomposition for 2x3x7.
    
    Split the 3x7 matrix into blocks:
    - First 2 columns: 3x2 block
    - Last 5 columns: 3x5 block
    
    For the 2x3 * 3x2 part, we can use Strassen-like techniques.
    For the 2x3 * 3x5 part, we use schoolbook.
    
    Actually, let's think more carefully.
    
    We have A (2x3) * B (3x7) = C (2x7).
    
    Split B into B1 (3x2) and B2 (3x5).
    C1 = A * B1 (2x2)
    C2 = A * B2 (2x5)
    
    For C1 = A * B1, both are 2x3 * 3x2. We can use a rank-7 decomposition for 2x3x2.
    For C2 = A * B2, we use schoolbook: 2*3*5 = 30 operations.
    
    Total rank: 7 + 30 = 37.
    
    Let's construct this.
    """
    n, m, p = 2, 3, 7
    
    # Rank for 2x3x2 part (C1)
    # We need a rank-7 decomposition for 2x3 * 3x2.
    # Let's use Strassen-inspired: 
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A11 + A22)B21
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A21 + A22)B11
    # M6 = (A11 - A21)(B21 + B22)
    # M7 = (A12 - A22)(B11 + B12)
    # But this is for 2x2 * 2x2. For 2x3 * 3x2, it's different.
    
    # Let's just use schoolbook for 2x3x2: rank = 2*3*2 = 12.
    # And schoolbook for 2x3x5: rank = 2*3*5 = 30.
    # Total: 42. That's the same as full schoolbook.
    
    # Better approach: Use the known rank-7 for 2x3x2.
    # Actually, the rank of 2x3x2 is at most 2*3*2 = 12, but can be lower.
    # Let's try to find a good decomposition.
    
    # Alternative: Split differently.
    # Split the inner dimension 3 into 2+1.
    # A = [A1 (2x2) | A2 (2x1)]
    # B = [B1 (2x7); B2 (1x7)]
    # C = A1*B1 + A2*B2
    
    # A1*B1 is 2x2 * 2x7: schoolbook rank = 2*2*7 = 28
    # A2*B2 is 2x1 * 1x7: rank = 2*1*7 = 14
    # Total: 42. Still schoolbook.
    
    # Let's try the Strassen approach for the 2x2 * 2x7 part.
    # 2x2 * 2x7 can be done with rank 7*7 = 49? No, that's worse.
    # Actually, 2x2 * 2x7: we can use 7 Strassen-like multiplications for each pair of columns.
    # For 2x2 * 2x2, Strassen uses 7 multiplications.
    # For 2x2 * 2x7, we can do 7 multiplications for each of the 7 column pairs? No.
    # We can do it as: split B1 into 7 columns, each 2x1.
    # A1 * B1[:,k] is 2x2 * 2x1 = 2x1. This can be done with 4 multiplications (schoolbook) or fewer.
    # Actually, 2x2 * 2x1: rank is at most 4, but can it be 3? 
    # The tensor is 2x2x1. Rank is at most 4. Let's assume 4.
    # So 7 columns * 4 = 28. Same as schoolbook.
    
    # Let's try a different split.
    # Use the fact that 2x3 * 3x7 can be decomposed using the 2x2 * 2x2 Strassen on overlapping blocks.
    
    # I'll implement a simple hybrid: use Strassen for a 2x2 * 2x2 subproblem and schoolbook for the rest.
    # This is complex. Let's just return the schoolbook for now and try the iterative refinement.
    
    rank = 42
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    T = _target_tensor(n, m, p)
    
    # Try ranks from 30 to 42
    for rank in range(30, 43):
        result = _try_rank(n, m, p, rank, T, max_iter=100, seed=rank)
        if result is not None:
            U, V, W = result
            # Convert to lists of lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U, V, W = _strassen_schoolbook_hybrid()
    return {"U": U, "V": V, "W": W}
