import time
import random
import numpy as np
import sys
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

def verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid tensor decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    W index is k*n + i where k is col (0..p-1), i is row (0..n-1)
    U index is i*m + j where i is row (0..n-1), j is col (0..m-1)
    V index is j*p + k where j is row (0..m-1), k is col (0..p-1)
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i*m + j]
                v_val = V[j*p + k]
                w_val = W[k*n + i]
                s = 0.0
                for c in range(rank):
                    s += u_val[c] * v_val[c] * w_val[c]
                if abs(s - 1.0) > 1e-9:
                    return False, f"Triple ({i},{j},{k}) gives {s}, expected 1"
    return True, "ok"

def schoolbook_decomp(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def winograd_decomp(n, m, p):
    """
    Winograd-style decomposition for matrix multiplication.
    We partition the computation into blocks and use Winograd's algorithm
    for each block to reduce the number of multiplications.
    
    For multiplying (n x m) by (m x p), we can use Winograd's method
    which reduces multiplications by using the identity:
    (a+b)(c+d) = ac + ad + bc + bd - (a-b)(c-d)
    
    More generally, for a block of size k x k, Winograd's algorithm
    uses a specific number of multiplications depending on k.
    
    For our case n=3, m=4, p=8, we can try different block sizes.
    """
    # For simplicity, let's try to use Winograd's F_2^2 or F_2^1 methods
    # F_2^1: 2x1 matrix mult uses 2 mults instead of 4
    # F_2^2: 2x2 matrix mult uses 5 mults instead of 8
    
    # Let's implement a general approach: partition into 2x2 blocks where possible
    # and use Winograd's 5-mult method for each 2x2 block.
    
    # Actually, for a more systematic approach, let's use the following:
    # The matmul tensor T[i,j,k] = 1 for all i,j,k
    # We want to find a low-rank decomposition.
    
    # Winograd's key insight: for computing a 2x2 product, instead of 8 mults:
    # m0 = (a00+a01)*(b00+b10)
    # m1 = (a00-a01)*(b00-b10) 
    # m2 = (a00+a10)*(b00+b01)
    # m3 = (a00-a10)*(b00-b01)
    # m4 = (a01+a11)*(b10+b11)
    # Then:
    # c00 = m0 + m1 + m2 - m3 + m4  (wait, this isn't quite right)
    
    # Let me use the correct Winograd F_2^2 formula:
    # For C = A*B where A, B are 2x2:
    # m0 = (a00 + a01) * (b00 + b10)
    # m1 = (a00 - a01) * (b00 - b10)
    # m2 = (a00 + a10) * (b00 + b01)
    # m3 = (a00 - a10) * (b00 - b01)
    # m4 = (a10 + a11) * (b01 + b11)
    # c00 = m0 + m2 - m3
    # c01 = m2 + m4 - m3
    # c10 = m0 + m3 - m1
    # c11 = m1 + m4 + m3
    
    # Wait, I need to double-check this. Let me derive it properly.
    # Actually, the standard Winograd F_2^2 uses 5 multiplications:
    # m1 = (a00+a01)*(b00+b10)
    # m2 = (a00-a01)*(b00-b10)
    # m3 = (a00+a10)*(b00+b01)
    # m4 = (a00-a10)*(b00-b01)
    # m5 = (a10+a11)*(b01+b11)
    # c00 = m1 + m3 - m4
    # c01 = m3 + m5 - m4
    # c10 = m1 + m4 - m2
    # c11 = m2 + m5 + m4
    
    # Let me verify: 
    # c00 should = a00*b00 + a01*b10
    # m1 + m3 - m4 = (a00+a01)(b00+b10) + (a00+a10)(b00+b01) - (a00-a10)(b00-b01)
    # = a00*b00 + a00*b10 + a01*b00 + a01*b10 + a00*b00 + a00*b01 + a10*b00 + a10*b01
    #   - a00*b00 + a00*b01 + a10*b00 - a10*b01
    # = a00*b00 + a00*b10 + a01*b00 + a01*b10 + a00*b01 + a10*b00
    # Hmm, that doesn't look right. Let me look up the correct formula.
    
    # Actually, I think I'm overcomplicating this. Let me use a simpler approach:
    # For the specific dimensions 3x4 x 4x8, I'll try to find a decomposition
    # by exploiting the structure.
    
    # One approach: use the fact that we can group columns of B.
    # B is 4x8, so we can view it as 2 blocks of 4x4.
    # Then C = A * B = [A*B1, A*B2] where B1, B2 are 4x4.
    # Each 3x4 * 4x4 multiplication has schoolbook rank 3*4*4 = 48.
    # So total would be 96, same as before.
    
    # To actually reduce rank, we need Winograd or similar.
    # Let me try a different approach: use the Strassen-like recursion.
    
    # For now, let's just return schoolbook and let the search find improvements.
    return schoolbook_decomp(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 140  # safety margin
    
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    def check_time():
        return time.time() < deadline
    
    # Start with schoolbook decomposition
    best_U, best_V, best_W = schoolbook_decomp(n, m, p)
    best_rank = n * m * p
    
    # Strategy: Use ALS (Alternating Least Squares) to find a lower-rank decomposition
    # We want to minimize the error ||T - sum_c U_c ⊗ V_c ⊗ W_c||_F
    # where T[i,j,k] = 1 for all i,j,k in the matmul tensor.
    # 
    # Actually, for exact decomposition, we need T[i,j,k] = 1 for all (i,j,k)
    # where the tensor is defined. For matmul, the tensor T has entries:
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 elsewhere.
    # 
    # Wait, I need to be more careful. The matmul tensor is:
    # T[a,b,c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a,b,c] = 1 if i=i', j=j', k=k' (i.e., the indices match), else 0.
    # 
    # More precisely: T[i*m+j, j*p+k, k*n+i] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And T[a,b,c] = 0 for all other (a,b,c).
    # 
    # For a rank-r decomposition: T = sum_{c=0}^{r-1} U_c ⊗ V_c ⊗ W_c
    # where U_c has length n*m, V_c has length m*p, W_c has length p*n.
    # 
    # The constraint is: for all (i,j,k):
    #   sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 1
    # And for all other (a,b,c) not of the form (i*m+j, j*p+k, k*n+i):
    #   sum_c U_c[a] * V_c[b] * W_c[c] = 0
    # 
    # This is a system of equations. We can use ALS to find a solution.
    
    def tensor_entry(a, b, c, n, m, p):
        """Return T[a,b,c] for the matmul tensor."""
        # a = i*m + j, b = j'*p + k, c = k'*n + i'
        i, j = divmod(a, m)
        j2, k = divmod(b, p)
        k2, i2 = divmod(c, n)
        # T = 1 if i==i2 and j==j2 and k==k2, else 0
        if i == i2 and j == j2 and k == k2:
            return 1.0
        return 0.0
    
    def build_tensor(n, m, p):
        """Build the full matmul tensor as a 3D array."""
        T = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        return T
    
    T = build_tensor(n, m, p)
    
    # ALS approach:
    # Fix V and W, solve for U (linear least squares)
    # Fix U and W, solve for V
    # Fix U and V, solve for W
    # Iterate until convergence.
    
    # For exact decomposition, we need the residual to be 0.
    # Let's try with various ranks and see if we can get exact solutions.
    
    def als_decompose(T, rank, n, m, p, max_iter=100, tol=1e-10):
        """ALS tensor decomposition. Returns (U, V, W) or None if failed."""
        n_m, m_p, p_n = T.shape
        random.seed(42)
        
        # Initialize with random values
        U = np.random.randn(n_m, rank) * 0.1
        V = np.random.randn(m_p, rank) * 0.1
        W = np.random.randn(p_n, rank) * 0.1
        
        for it in range(max_iter):
            if not check_time():
                return None
            
            # Update U: for each a, solve min_U sum_{b,c} (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
            # This is a linear system: for each a, U[a,:] = (V^T W ⊙ ...) 
            # Actually, for fixed a: T[a,:,:] = sum_r U[a,r] * V[:,r] * W[:,r] (element-wise)
            # Let M_a = V * W (element-wise, shape m_p x p_n)
            # Then T[a,:,:] = U[a,:] @ diag(M_a) ... no, this isn't quite right.
            # 
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed a: T[a,:] = sum_r U[a,r] * (V[:,r] * W[:,r]) (flattened)
            # Let v_w[r] = V[:,r] * W[:,r] (flattened, length m_p*p_n)
            # Then T[a,:] (flattened) = sum_r U[a,r] * v_w[r]
            # This is a linear system: T_flat[a] = U[a,:] @ V_W_matrix
            # where V_W_matrix[r, (b,c)] = V[b,r] * W[c,r]
            
            # Build the design matrix for U update
            # For each r, the vector v_w[r] has entries v_w[r][b*p_n + c] = V[b,r] * W[c,r]
            V_W = np.zeros((rank, m_p * p_n))
            for r in range(rank):
                V_W[r, :] = (V[:, r].reshape(-1, 1) * W[:, r].reshape(1, -1)).flatten()
            
            # T_flat[a] = sum_r U[a,r] * V_W[r, :]
            # So U = T_flat @ V_W^T @ (V_W @ V_W^T)^{-1}
            # Or solve the linear system: V_W @ U^T = T_flat^T
            
            T_flat = T.reshape(n_m, -1)
            
            # Solve: V_W @ U^T = T_flat^T
            # U^T = V_W^+ @ T_flat^T
            try:
                U_new, residuals, rank_u, sv = np.linalg.lstsq(V_W, T_flat.T, rcond=None)
                U = U_new.T
            except:
                return None
            
            # Update V similarly
            U_W = np.zeros((rank, n_m * p_n))
            for r in range(rank):
                U_W[r, :] = (U[:, r].reshape(-1, 1) * W[:, r].reshape(1, -1)).flatten()
            
            T_flat_v = T.reshape(m_p, -1)
            # T[b,a,c] = sum_r V[b,r] * U[a,r] * W[c,r]
            # For fixed b: T[:,b,:] (reshaped) = sum_r V[b,r] * (U[:,r] * W[:,r])
            T_flat_v = np.zeros((m_p, n_m * p_n))
            for b in range(m_p):
                T_flat_v[b, :] = T[:, b, :].reshape(-1)
            
            try:
                V_new, residuals, rank_v, sv = np.linalg.lstsq(U_W, T_flat_v.T, rcond=None)
                V = V_new.T
            except:
                return None
            
            # Update W
            U_V = np.zeros((rank, n_m * m_p))
            for r in range(rank):
                U_V[r, :] = (U[:, r].reshape(-1, 1) * V[:, r].reshape(1, -1)).flatten()
            
            T_flat_w = np.zeros((p_n, n_m * m_p))
            for c in range(p_n):
                T_flat_w[c, :] = T[:, :, c].reshape(-1)
            
            try:
                W_new, residuals, rank_w, sv = np.linalg.lstsq(U_V, T_flat_w.T, rcond=None)
                W = W_new.T
            except:
                return None
            
            # Check convergence
            # Compute residual
            T_recon = np.zeros_like(T)
            for r in range(rank):
                T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            residual = np.linalg.norm(T - T_recon)
            
            if residual < tol:
                return U, V, W
        
        return None
    
    # Try ALS with various ranks, starting from schoolbook rank and going down
    # For 3x4x8, schoolbook is 96. Let's try ranks from 96 down to some lower bound.
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # But for matmul, the lower bound is actually higher.
    
    # Let's try a few ranks with ALS
    ranks_to_try = [best_rank]
    # Try some lower ranks
    for r in range(max(32, best_rank - 20), best_rank):
        if r not in ranks_to_try:
            ranks_to_try.append(r)
    ranks_to_try.sort()
    
    found = False
    for rank in ranks_to_try:
        if not check_time():
            break
        
        # Try multiple random initializations
        for trial in range(5):
            if not check_time():
                break
            
            result = als_decompose(T, rank, n, m, p, max_iter=50)
            if result is not None:
                U_als, V_als, W_als = result
                # Verify
                valid, msg = verify(U_als, V_als, W_als, n, m, p)
                if valid:
                    # Convert to lists
                    best_U = U_als.tolist()
                    best_V = V_als.tolist()
                    best_W = W_als.tolist()
                    best_rank = rank
                    found = True
                    print(f"Found exact decomposition with rank {rank}", file=sys.stderr)
                    break
        
        if found:
            # Try to go lower
            continue
    
    # If we found a lower rank, try to go even lower
    if found and best_rank > 32:
        for rank in range(max(32, best_rank - 10), best_rank):
            if not check_time():
                break
            for trial in range(3):
                if not check_time():
                    break
                result = als_decompose(T, rank, n, m, p, max_iter=50)
                if result is not None:
                    U_als, V_als, W_als = result
                    valid, msg = verify(U_als, V_als, W_als, n, m, p)
                    if valid:
                        best_U = U_als.tolist()
                        best_V = V_als.tolist()
                        best_W = W_als.tolist()
                        best_rank = rank
                        print(f"Found exact decomposition with rank {rank}", file=sys.stderr)
    
    return {"U": best_U, "V": best_V, "W": best_W}



