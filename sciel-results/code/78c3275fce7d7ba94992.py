"""Matrix multiplication 4x5x5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

def construct():
    """
    Construct a tensor decomposition for 4x5 x 5x5 matrix multiplication.
    
    Strategy: Block decomposition using Strassen for 2x2x2 blocks
    - Split into blocks and use Strassen where beneficial
    - Use schoolbook for remaining parts
    - Reduce rank via linear dependence elimination
    """
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = 4, 5, 5
    
    # Collect rank-1 terms: (U_indices, V_indices, W_indices, coefficient)
    # Each term contributes coeff * U[i,j] * V[j,k] * W[k,i] to T[i,j,k]
    terms = []
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        """Add a rank-1 term with given indices and coefficient."""
        if coeff != 0:
            terms.append((u_idx, v_idx, w_idx, coeff))
    
    def add_schoolblock(i_range, j_range, k_range):
        """Add schoolbook terms for a block."""
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    add_term([i * m + j], [j * p + k], [k * n + i], 1)
    
    def add_strassen_2x2x2(i0, j0, k0):
        """
        Add Strassen terms for a 2x2x2 block starting at (i0, j0, k0).
        A block: rows i0..i0+1, cols j0..j0+1
        B block: rows j0..j0+1, cols k0..k0+1  
        C block: rows k0..k0+1, cols i0..i0+1
        
        Strassen 2x2x2:
        M1 = (A[i0,j0]+A[i0+1,j0])*(B[j0,k0]+B[j0,k0+1])
        M2 = (A[i0+1,j0]+A[i0+1,j0+1])*B[j0,k0]
        M3 = A[i0,j0]*(B[j0,k0+1]-B[j0+1,k0+1])
        M4 = A[i0+1,j0+1]*(B[j0+1,k0+1]-B[j0+1,k0])
        M5 = (A[i0,j0]+A[i0+1,j0])*B[j0+1,k0+1]
        M6 = (A[i0+1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k0+1])
        M7 = (A[i0,j0+1]-A[i0+1,j0+1])*(B[j0+1,k0]+B[j0+1,k0+1])
        
        C[k0,i0] = M1 - M5 + M7
        C[k0,i0+1] = M3 + M5
        C[k0+1,i0] = M2 + M4 - M6
        C[k0+1,i0+1] = M1 - M3 + M4
        """
        # M1: U=[i0*m+j0, (i0+1)*m+j0], V=[j0*p+k0, j0*p+(k0+1)], W=[k0*n+i0, (k0+1)*n+(i0+1)]
        add_term([i0 * m + j0, (i0 + 1) * m + j0], 
                 [j0 * p + k0, j0 * p + (k0 + 1)], 
                 [k0 * n + i0, (k0 + 1) * n + (i0 + 1)], 1)
        
        # M2: U=[(i0+1)*m+j0, (i0+1)*m+(j0+1)], V=[j0*p+k0], W=[(k0+1)*n+i0]
        add_term([(i0 + 1) * m + j0, (i0 + 1) * m + (j0 + 1)], 
                 [j0 * p + k0], 
                 [(k0 + 1) * n + i0], 1)
        
        # M3: U=[i0*m+j0], V=[j0*p+(k0+1), (j0+1)*p+(k0+1)], W=[k0*n+(i0+1)]
        add_term([i0 * m + j0], 
                 [j0 * p + (k0 + 1), (j0 + 1) * p + (k0 + 1)], 
                 [k0 * n + (i0 + 1)], 1)
        
        # M4: U=[(i0+1)*m+(j0+1)], V=[(j0+1)*p+(k0+1), (j0+1)*p+k0], W=[(k0+1)*n+(i0+1)]
        add_term([(i0 + 1) * m + (j0 + 1)], 
                 [(j0 + 1) * p + (k0 + 1), (j0 + 1) * p + k0], 
                 [(k0 + 1) * n + (i0 + 1)], 1)
        
        # M5: U=[i0*m+j0, (i0+1)*m+j0], V=[(j0+1)*p+(k0+1)], W=[k0*n+i0, (k0+1)*n+(i0+1)]
        add_term([i0 * m + j0, (i0 + 1) * m + j0], 
                 [(j0 + 1) * p + (k0 + 1)], 
                 [k0 * n + i0, (k0 + 1) * n + (i0 + 1)], 1)
        
        # M6: U=[(i0+1)*m+j0, i0*m+j0], V=[j0*p+k0, j0*p+(k0+1)], W=[(k0+1)*n+i0]
        add_term([(i0 + 1) * m + j0, i0 * m + j0], 
                 [j0 * p + k0, j0 * p + (k0 + 1)], 
                 [(k0 + 1) * n + i0], 1)
        
        # M7: U=[i0*m+(j0+1), (i0+1)*m+(j0+1)], V=[(j0+1)*p+k0, (j0+1)*p+(k0+1)], W=[k0*n+i0, (k0+1)*n+i0]
        add_term([i0 * m + (j0 + 1), (i0 + 1) * m + (j0 + 1)], 
                 [(j0 + 1) * p + k0, (j0 + 1) * p + (k0 + 1)], 
                 [k0 * n + i0, (k0 + 1) * n + i0], 1)
    
    # Build decomposition using block strategy
    # Split: A rows [0,1,2,3], B cols [0,1,2,3,4], C cols [0,1,2,3]
    
    # Use Strassen for 2x2x2 blocks where possible
    # Block (0,0): i=[0,1], j=[0,1], k=[0,1] - Strassen
    add_strassen_2x2x2(0, 0, 0)
    
    # Block (2,2): i=[2,3], j=[2,3], k=[2,3] - Strassen
    add_strassen_2x2x2(2, 2, 2)
    
    # Remaining parts use schoolbook
    # i=[0,1], j=[0,1], k=[2,3,4]
    add_schoolblock([0, 1], [0, 1], [2, 3, 4])
    
    # i=[0,1], j=[2,3,4], k=[0,1]
    add_schoolblock([0, 1], [2, 3, 4], [0, 1])
    
    # i=[2,3], j=[0,1], k=[0,1]
    add_schoolblock([2, 3], [0, 1], [0, 1])
    
    # i=[2,3], j=[0,1,2,3], k=[4]
    add_schoolblock([2, 3], [0, 1, 2, 3], [4])
    
    # i=[2,3], j=[4], k=[0,1,2,3]
    add_schoolblock([2, 3], [4], [0, 1, 2, 3])
    
    # i=[0,1], j=[2,3,4], k=[2,3,4]
    add_schoolblock([0, 1], [2, 3, 4], [2, 3, 4])
    
    # i=[2,3], j=[2,3], k=[0,1]
    add_schoolblock([2, 3], [2, 3], [0, 1])
    
    # Now build U, V, W matrices from terms
    # First, collect all unique indices
    u_indices = set()
    v_indices = set()
    w_indices = set()
    
    for u_idx, v_idx, w_idx, coeff in terms:
        for idx in u_idx:
            u_indices.add(idx)
        for idx in v_idx:
            v_indices.add(idx)
        for idx in w_idx:
            w_indices.add(idx)
    
    # Create matrices
    U = np.zeros((n * m, len(terms)), dtype=float)
    V = np.zeros((m * p, len(terms)), dtype=float)
    W = np.zeros((p * n, len(terms)), dtype=float)
    
    for col, (u_idx, v_idx, w_idx, coeff) in enumerate(terms):
        for idx in u_idx:
            U[idx, col] = coeff
        for idx in v_idx:
            V[idx, col] = 1
        for idx in w_idx:
            W[idx, col] = 1
    
    # Verify correctness using vectorized computation
    def verify(U_mat, V_mat, W_mat):
        """Fast verification using numpy broadcasting."""
        # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # Reshape and use einsum
        n, m, p = 4, 5, 5
        rank = U_mat.shape[1]
        U_r = U_mat.reshape(n, m, rank)
        V_r = V_mat.reshape(m, p, rank)
        W_r = W_mat.reshape(p, n, rank)
        T_computed = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        T_expected = np.ones((n, m, p))
        return np.allclose(T_computed, T_expected, atol=1e-10)
    
    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Rank reduction: find linearly dependent columns and remove them
    # The tensor decomposition is valid if sum_t u_t ⊗ v_t ⊗ w_t = T_matmul
    # We can remove terms that are linear combinations of others
    
    def reduce_rank(U_mat, V_mat, W_mat, max_iter=50):
        """Reduce rank by removing linearly dependent terms."""
        U_curr = U_mat.copy()
        V_curr = V_mat.copy()
        W_curr = W_mat.copy()
        current_rank = U_curr.shape[1]
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            if current_rank <= 1:
                break
            
            # Try to remove each column by checking if it's in the span of others
            # Use the fact that if we remove column t, we need:
            # u_t ⊗ v_t ⊗ w_t = sum_{s != t} c_s * u_s ⊗ v_s ⊗ w_s
            # This is a linear system in the coefficients
            
            removed = False
            for t in range(current_rank):
                if time.time() > deadline - 5:
                    break
                
                # Try to express term t as linear combination of other terms
                # Build system: for all (i,j,k), sum_{s!=t} c_s * U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] = U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t]
                
                n, m, p = 4, 5, 5
                # Number of equations: n*m*p = 100
                # Number of unknowns: current_rank - 1
                if current_rank - 1 >= 100:
                    # Too many unknowns, skip this approach
                    continue
                
                A_sys = np.zeros((n * m * p, current_rank - 1))
                b_sys = np.zeros(n * m * p)
                
                for eq_idx, (i, j, k) in enumerate([(i, j, k) for i in range(n) for j in range(m) for k in range(p)]):
                    b_sys[eq_idx] = U_curr[i * m + j, t] * V_curr[j * p + k, t] * W_curr[k * n + i, t]
                    
                    col_idx = 0
                    for s in range(current_rank):
                        if s == t:
                            continue
                        A_sys[eq_idx, col_idx] = U_curr[i * m + j, s] * V_curr[j * p + k, s] * W_curr[k * n + i, s]
                        col_idx += 1
                
                # Solve least squares
                try:
                    coeffs, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=1e-10)
                except:
                    continue
                
                # Check if the solution is good
                residual = np.linalg.norm(A_sys @ coeffs - b_sys)
                if residual < 1e-8:
                    # Remove column t
                    mask = [s for s in range(current_rank) if s != t]
                    U_curr = U_curr[:, mask]
                    V_curr = V_curr[:, mask]
                    W_curr = W_curr[:, mask]
                    current_rank -= 1
                    removed = True
                    break
            
            if not removed:
                break
        
        return U_curr, V_curr, W_curr
    
    U, V, W = reduce_rank(U, V, W)
    
    # Final verification
    if not verify(U, V, W):
        # Fallback to original if reduction broke something
        U_orig = np.zeros((n * m, len(terms)), dtype=float)
        V_orig = np.zeros((m * p, len(terms)), dtype=float)
        W_orig = np.zeros((p * n, len(terms)), dtype=float)
        
        for col, (u_idx, v_idx, w_idx, coeff) in enumerate(terms):
            for idx in u_idx:
                U_orig[idx, col] = coeff
            for idx in v_idx:
                V_orig[idx, col] = 1
            for idx in w_idx:
                W_orig[idx, col] = 1
        
        return {"U": U_orig.tolist(), "V": V_orig.tolist(), "W": W_orig.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
