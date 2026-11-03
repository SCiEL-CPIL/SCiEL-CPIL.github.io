import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 7]
DEADLINE = 120.0

def _schoolbook(n, m, p):
    """Return schoolbook decomposition for n x m x p multiplication.
    Returns U (n*m, n*m*p), V (m*p, n*m*p), W (p*n, n*m*p) as numpy arrays.
    """
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
    return U, V, W

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # Strassen's 7 products
    # M1 = (A11+A22)(B12+B21)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # A indices: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    # B indices: B11=(0,0), B12=(0,1), B21=(1,0), B22=(1,1)
    # C indices: C11=(0,0), C12=(0,1), C21=(1,0), C22=(1,1)
    
    # M1: (A11+A22)(B12+B21)
    U[0, 0] = 1; U[3, 0] = 1
    V[1, 0] = 1; V[2, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C11 and C22
    
    # M2: (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # C21 and C22
    
    # M3: A11(B21-B22)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C12 and C22
    
    # M4: A22(B11-B22)
    U[3, 3] = 1
    V[0, 3] = 1; V[3, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # C11 and C21
    
    # M5: (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # C11 and C12
    
    # M6: (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C22
    
    # M7: (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C11
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify decomposition by testing with random matrices."""
    np.random.seed(12345)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    R = U.shape[1]
    U_r = U.reshape(n, m, R)
    V_r = V.reshape(m, p, R)
    W_r = W.reshape(p, n, R)
    
    S_A = np.einsum('ijr,ij->ir', U_r, A)
    S_B = np.einsum('jkr,jk->kr', V_r, B)
    C_comp = np.einsum('kir,ir,kr->ik', W_r, S_A, S_B)
    
    return np.allclose(C_true, C_comp, atol=tol)

def _verify_tensor(U, V, W, n, m, p, tol=1e-8):
    """Verify tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k."""
    R = U.shape[1]
    U_r = U.reshape(n, m, R)
    V_r = V.reshape(m, p, R)
    W_r = W.reshape(p, n, R)
    
    T_comp = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    T_target = np.ones((n, m, p))
    
    return np.allclose(T_comp, T_target, atol=tol)

def _build_78_rank_decomposition():
    """Build the 78-rank decomposition using Strassen on 2x2 blocks.
    
    Split n=3 into 2+1.
    For the 2x4x7 part: split m=4 into 2+2.
    - Block (0:2, 0:2): 2x2x7 -> use Strassen-like for 2x2x2 three times? No.
    
    Actually, let me think more carefully.
    
    n=3, m=4, p=7.
    
    Split n = 2 + 1.
    
    Part 1: 2 x 4 x 7 (rows 0,1 of A)
    Part 2: 1 x 4 x 7 (row 2 of A)
    
    For Part 2 (1x4x7): schoolbook rank = 1*4*7 = 28.
    
    For Part 1 (2x4x7): split m = 2 + 2.
    - Sub-part 1a: 2 x 2 x 7
    - Sub-part 1b: 2 x 2 x 7
    
    For each 2x2x7: split p = 2+2+2+1.
    - 2x2x2 blocks: 3 of them, each uses Strassen rank 7. Total: 3*7 = 21.
    - 2x2x1 block: schoolbook rank = 2*2*1 = 4.
    Total for one 2x2x7: 21 + 4 = 25.
    
    So Part 1: 2 * 25 = 50.
    Part 2: 28.
    Total: 50 + 28 = 78.
    """
    n, m, p = 3, 4, 7
    total_rank = 78
    
    U = np.zeros((n * m, total_rank))  # (12, 78)
    V = np.zeros((m * p, total_rank))  # (28, 78)
    W = np.zeros((p * n, total_rank))  # (21, 78)
    
    col = 0
    
    # Part 1: rows i=0,1 of A (n_part=2), all m=4, all p=7
    # Split m into [0:2] and [2:4]
    for m_offset in [0, 2]:
        # 2 x 2 x 7 subproblem
        # Split p into [0:2], [2:4], [4:6], [6:7]
        for p_start, p_len in [(0, 2), (2, 2), (4, 2), (6, 1)]:
            if p_len == 2:
                # 2x2x2: use Strassen
                U_s, V_s, W_s = _strassen_2x2x2()
                for r in range(7):
                    # U: rows i=0,1 (global i = 0,1), cols j=m_offset, m_offset+1
                    for i_local in range(2):
                        for j_local in range(2):
                            U[(i_local) * m + (m_offset + j_local), col + r] += U_s[i_local * 2 + j_local, r]
                    # V: rows j=m_offset, m_offset+1, cols k=p_start, p_start+1
                    for j_local in range(2):
                        for k_local in range(2):
                            V[(m_offset + j_local) * p + (p_start + k_local), col + r] += V_s[j_local * 2 + k_local, r]
                    # W: rows k=p_start, p_start+1, cols i=0,1
                    for k_local in range(2):
                        for i_local in range(2):
                            W[(p_start + k_local) * n + i_local, col + r] += W_s[k_local * 2 + i_local, r]
                col += 7
            else:
                # 2x2x1: schoolbook
                for i_local in range(2):
                    for j_local in range(2):
                        for k_local in range(1):
                            U[i_local * m + (m_offset + j_local), col] = 1
                            V[(m_offset + j_local) * p + (p_start + k_local), col] = 1
                            W[(p_start + k_local) * n + i_local, col] = 1
                            col += 1
    
    # Part 2: row i=2 of A, all m=4, all p=7: schoolbook 1x4x7 = 28
    for j in range(4):
        for k in range(7):
            U[2 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 2, col] = 1
            col += 1
    
    assert col == total_rank, f"Expected {total_rank}, got {col}"
    
    return U, V, W

def _try_local_refinement(U, V, W, time_limit=30.0):
    """Try to reduce rank by finding linear dependencies among columns.
    
    If two columns are linearly dependent, we can merge them.
    More generally, if a column is a linear combination of others, we can eliminate it.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    R = U.shape[1]
    
    # Check for zero columns first
    for c in range(R):
        if time.time() - start_time > time_limit:
            break
        if np.allclose(U[:, c], 0) and np.allclose(V[:, c], 0) and np.allclose(W[:, c], 0):
            # Remove this column
            U = np.delete(U, c, axis=1)
            V = np.delete(V, c, axis=1)
            W = np.delete(W, c, axis=1)
            R -= 1
            c -= 1
    
    # Try to find columns that can be eliminated
    # For each column c, check if it can be expressed as a linear combination of others
    # This is expensive, so we'll do a simplified version
    
    # Check for duplicate columns
    for c in range(R):
        if time.time() - start_time > time_limit:
            break
        for d in range(c + 1, R):
            if time.time() - start_time > time_limit:
                break
            # Check if column c is a scalar multiple of column d
            if np.allclose(U[:, c], U[:, d]) and np.allclose(V[:, c], V[:, d]) and np.allclose(W[:, c], W[:, d]):
                # They're identical, remove d
                U = np.delete(U, d, axis=1)
                V = np.delete(V, d, axis=1)
                W = np.delete(W, d, axis=1)
                R -= 1
                break
    
    return U, V, W

def _try_eliminate_columns(U, V, W, time_limit=40.0):
    """More aggressive column elimination using linear algebra.
    
    For each column c, we try to find coefficients alpha_d such that:
    U[:,c] = sum_d alpha_d * U[:,d]
    V[:,c] = sum_d alpha_d * V[:,d]  
    W[:,c] = sum_d alpha_d * W[:,d]
    for d != c.
    
    If such coefficients exist, we can eliminate column c.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    R = U.shape[1]
    
    # Build a combined matrix where each column represents (U_col, V_col, W_col)
    # We want to find if any column is in the span of the others
    
    # For efficiency, work with the combined vector
    combined = np.vstack([U.flatten(), V.flatten(), W.flatten()])  # (n*m*R + m*p*R + p*n*R, R)
    
    # Try to eliminate columns from the end
    for c in range(R - 1, -1, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Check if column c is in the span of the other columns
        others = list(range(R))
        others.remove(c)
        
        if len(others) == 0:
            continue
        
        # Solve: combined[:, c] = combined[:, others] @ alpha
        A_mat = combined[:, others]
        b_vec = combined[:, c]
        
        try:
            alpha, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Check if the solution is exact
            reconstructed = A_mat @ alpha
            if np.allclose(reconstructed, b_vec, atol=1e-8):
                # Column c is in the span of others, eliminate it
                # But we need to verify the decomposition still works
                U_new = np.delete(U, c, axis=1)
                V_new = np.delete(V, c, axis=1)
                W_new = np.delete(W, c, axis=1)
                
                if _verify(U_new, V_new, W_new, n, m, p, tol=1e-6):
                    U, V, W = U_new, V_new, W_new
                    R -= 1
                    combined = np.vstack([U.flatten(), V.flatten(), W.flatten()])
                    continue
        except Exception:
            pass
    
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build the 78-rank decomposition
    U, V, W = _build_78_rank_decomposition()
    
    # Verify it
    if not _verify(U, V, W, n, m, p, tol=1e-8):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {
            "U": [[float(x) for x in row] for row in U.tolist()],
            "V": [[float(x) for x in row] for row in V.tolist()],
            "W": [[float(x) for x in row] for row in W.tolist()]
        }
    
    # Try local refinement
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10:
        U, V, W = _try_local_refinement(U, V, W, time_limit=min(20, DEADLINE - 10 - elapsed))
    
    # Try more aggressive elimination
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        U, V, W = _try_eliminate_columns(U, V, W, time_limit=min(30, DEADLINE - 5 - elapsed))
    
    # Final verification
    if not _verify(U, V, W, n, m, p, tol=1e-6):
        # Fallback to the 78-rank version
        U, V, W = _build_78_rank_decomposition()
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 2:
        # Safety: return what we have
        pass
    
    return {
        "U": [[float(x) for x in row] for row in U.tolist()],
        "V": [[float(x) for x in row] for row in V.tolist()],
        "W": [[float(x) for x in row] for row in W.tolist()]
    }
