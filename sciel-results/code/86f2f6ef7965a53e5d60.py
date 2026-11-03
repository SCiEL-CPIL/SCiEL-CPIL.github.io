import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def _strassen_222():
    """Return Strassen 2x2x2 decomposition as (U, V, W) each of shape (4, 7) or (4, 7) or (4, 7).
    U rows: (i, j) for i in 0..1, j in 0..1 -> row index i*2+j
    V rows: (j, k) for j in 0..1, k in 0..1 -> row index j*2+k
    W rows: (k, i) for k in 0..1, i in 0..1 -> row index k*2+i
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A00)(B00+B10)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # U rows: A00(row0), A01(row1), A10(row2), A11(row3)
    # V rows: B00(row0), B01(row1), B10(row2), B11(row3)
    # W rows: C00(row0), C01(row1), C10(row2), C11(row3)
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (A00+A10)B11
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6: (A11-A00)(B00+B10)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_2x5x6_strassen():
    """Build rank-54 decomposition for 2x5x6 matrix multiplication.
    
    Partition:
    - A[:, 0:2] (2x2) * B[0:2, :] (2x6): split B into three 2x2 blocks, use Strassen for each -> 3*7=21
    - A[:, 2:4] (2x2) * B[2:4, :] (2x6): same -> 3*7=21
    - A[:, 4:5] (2x1) * B[4:5, :] (1x6): schoolbook -> 2*1*6=12
    Total: 54
    """
    n, m, p = 2, 5, 6
    rank = 54
    
    U = np.zeros((n * m, rank))  # (10, 54)
    V = np.zeros((m * p, rank))  # (30, 54)
    W = np.zeros((p * n, rank))  # (12, 54)
    
    col = 0
    StrU, StrV, StrW = _strassen_222()
    
    # Block 1: A[:, 0:2] * B[0:2, :] -> 2x2x6
    # Split into three 2x2x2: B cols 0:2, 2:4, 4:6
    for k_offset in [0, 2, 4]:
        for t in range(7):
            # U: A rows 0,1 cols 0,1 -> U[i*5 + j, col] where i in 0..1, j in 0..1
            for i in range(2):
                for j in range(2):
                    U[i * m + j, col] = StrU[i * 2 + j, t]
            
            # V: B rows 0,1 cols k_offset..k_offset+1 -> V[j*p + k, col] where j in 0..1, k in k_offset..k_offset+1
            for j in range(2):
                for k in range(2):
                    V[j * p + (k_offset + k), col] = StrV[j * 2 + k, t]
            
            # W: C cols k_offset..k_offset+1, rows 0,1 -> W[k*n + i, col] where k in k_offset..k_offset+1, i in 0..1
            for k in range(2):
                for i in range(2):
                    W[(k_offset + k) * n + i, col] = StrW[k * 2 + i, t]
            
            col += 1
    
    # Block 2: A[:, 2:4] * B[2:4, :] -> 2x2x6
    # Split into three 2x2x2: B cols 0:2, 2:4, 4:6
    for k_offset in [0, 2, 4]:
        for t in range(7):
            # U: A rows 0,1 cols 2,3 -> U[i*5 + (2+j), col] where i in 0..1, j in 0..1
            for i in range(2):
                for j in range(2):
                    U[i * m + (2 + j), col] = StrU[i * 2 + j, t]
            
            # V: B rows 2,3 cols k_offset..k_offset+1 -> V[(2+j)*p + k, col] where j in 0..1, k in k_offset..k_offset+1
            for j in range(2):
                for k in range(2):
                    V[(2 + j) * p + (k_offset + k), col] = StrV[j * 2 + k, t]
            
            # W: C cols k_offset..k_offset+1, rows 0,1 -> W[(k_offset+k)*n + i, col]
            for k in range(2):
                for i in range(2):
                    W[(k_offset + k) * n + i, col] = StrW[k * 2 + i, t]
            
            col += 1
    
    # Block 3: A[:, 4:5] * B[4:5, :] -> 2x1x6, schoolbook
    for i in range(2):
        for k in range(6):
            U[i * m + 4, col] = 1
            V[4 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=5, p=6, num_trials=10):
    """Verify that the decomposition exactly decomposes matrix multiplication."""
    for _ in range(num_trials):
        A = np.random.randint(0, 10, size=(n, m)).astype(float)
        B = np.random.randint(0, 10, size=(m, p)).astype(float)
        C_true = A @ B
        
        C_calc = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for t in range(U.shape[1]):
                    u_val = 0.0
                    for j in range(m):
                        u_val += U[i * m + j, t] * A[i, j]
                    v_val = 0.0
                    for j in range(m):
                        v_val += V[j * p + k, t] * B[j, k]
                    C_calc[i, k] += W[k * n + i, t] * u_val * v_val
        
        if not np.allclose(C_true, C_calc, atol=1e-8):
            return False
    return True

def _term_to_vector(U, V, W, t, n, m, p):
    """Convert rank-1 term t to a vector in the tensor space (n*m*p dimensional).
    The tensor entry T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    """
    vec = np.zeros(n * m * p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                vec[i * m * p + j * p + k] = U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
    return vec

def _find_minimal_rank(U, V, W, n, m, p, deadline):
    """Try to find a rank-reduced decomposition using exact linear algebra.
    
    Approach: 
    1. Compute the tensor vector for each term
    2. Use Gaussian elimination with Fraction to find independent terms
    3. If we find dependencies, try to express dependent terms as combinations
    4. Reconstruct a minimal decomposition
    
    Returns (U', V', W') with rank < original if possible, else None
    """
    rank_orig = U.shape[1]
    tensor_dim = n * m * p
    
    # Compute all term vectors as Fractions for exact arithmetic
    term_vectors = []
    for t in range(rank_orig):
        vec = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(int(U[i * m + j, t]), 1) * Fraction(int(V[j * p + k, t]), 1) * Fraction(int(W[k * n + i, t]), 1)
                    vec.append(val)
        term_vectors.append(vec)
    
    # Gaussian elimination to find pivot terms
    # Work with the matrix where rows are tensor dimensions, columns are terms
    # We want to find which terms are linearly independent
    
    # Build matrix: M[d, t] = term_vectors[t][d]
    M = [[term_vectors[t][d] for t in range(rank_orig)] for d in range(tensor_dim)]
    
    # Transpose for elimination: we want to find independent columns
    # Use column-oriented Gaussian elimination
    pivot_cols = []
    pivot_rows = []
    rank_found = 0
    
    # Work on a copy
    M_work = [row[:] for row in M]
    
    for col in range(rank_orig):
        if time.time() > deadline:
            return None
        
        # Find pivot
        pivot_row = -1
        for r in range(rank_found, tensor_dim):
            if M_work[r][col] != 0:
                pivot_row = r
                break
        
        if pivot_row == -1:
            continue
        
        # Swap rows
        if pivot_row != rank_found:
            M_work[pivot_row], M_work[rank_found] = M_work[rank_found], M_work[pivot_row]
        
        # Normalize pivot row
        pivot_val = M_work[rank_found][col]
        for c in range(col, rank_orig):
            M_work[rank_found][c] = M_work[rank_found][c] / pivot_val
        
        # Eliminate below
        for r in range(rank_found + 1, tensor_dim):
            factor = M_work[r][col]
            if factor != 0:
                for c in range(col, rank_orig):
                    M_work[r][c] -= factor * M_work[rank_found][c]
        
        pivot_cols.append(col)
        pivot_rows.append(pivot_row)
        rank_found += 1
    
    if rank_found >= rank_orig:
        # All terms are independent, can't reduce
        return None
    
    # We found that only rank_found terms are independent
    # Now we need to find coefficients to express dependent terms as combinations of pivot terms
    # But this is complex. For now, just return the pivot terms if they form a valid decomposition
    
    # Actually, the tensor is fixed. If we remove a term, the tensor changes.
    # We can only remove terms if their tensor vector is a linear combination of others,
    # AND the remaining terms still decompose the same tensor.
    
    # The correct approach: the tensor T = sum_t v_t where v_t is the term vector.
    # If v_t = sum_{s in S} c_s * v_s for some subset S, then we can't simply remove t.
    # We need the sum to remain T.
    
    # So the rank is the minimum number of rank-1 tensors that sum to T.
    # This is the tensor rank, which is NP-hard to compute in general.
    
    # For this specific problem, let's try a different approach:
    # Check if any term's vector is zero (it contributes nothing)
    for t in range(rank_orig):
        if all(v == 0 for v in term_vectors[t]):
            # This term is zero, can be removed
            U_new = np.delete(U, t, axis=1)
            V_new = np.delete(V, t, axis=1)
            W_new = np.delete(W, t, axis=1)
            return U_new, V_new, W_new
    
    # Check for duplicate terms
    seen = {}
    for t in range(rank_orig):
        key = tuple(term_vectors[t])
        if key in seen:
            # Duplicate found - but we can't just remove one since the sum would change
            # Unless the coefficient is 1 and we have two identical terms, then we can't merge
            pass
        else:
            seen[key] = t
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Try Strassen-based construction first
    try:
        U, V, W = _build_2x5x6_strassen()
        if _verify_decomposition(U, V, W, n, m, p):
            # Try to reduce rank
            reduced = _find_minimal_rank(U, V, W, n, m, p, deadline)
            if reduced is not None:
                U_r, V_r, W_r = reduced
                if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    except Exception:
        pass
    
    # Fallback: schoolbook decomposition
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
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
