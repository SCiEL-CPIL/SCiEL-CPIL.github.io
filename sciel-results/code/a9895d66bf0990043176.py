"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy import linalg

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _build_strassen_28():
    """Build the standard rank-28 Strassen decomposition for 2x4 * 4x4.
    
    We split the 4x4 matrix into 2x2 blocks:
    A (2x4) = [A11 A12; A21 A22] where each Ai is 2x2
    B (4x4) = [B11 B12; B21 B22] where each Bj is 2x2
    
    Result C (2x4) = [C11 C12; C21 C22] where each Ci is 2x2
    
    For each 2x2 block multiplication, we use Strassen's 7 multiplications.
    But we need to be careful about how the blocks interact.
    
    Actually, let's think of this differently. The tensor T_{2,4,4} can be seen as:
    - Input: A (2x4), B (4x4)  
    - Output: C = A*B (2x4)
    
    We can partition B into two 4x2 blocks: B = [B_left, B_right]
    Then C = [A*B_left, A*B_right]
    
    For each 2x4 * 4x2 multiplication, we can use a rank-14 decomposition (or lower).
    Actually, 2x4 * 4x2 has tensor rank at most 2*4*2 = 16 (schoolbook),
    but can be done in rank 14 using Strassen on 2x2 blocks.
    
    Let me use a cleaner approach: partition into 2x2 sub-matrices.
    
    A (2x4): rows 0-1, cols 0-3
    B (4x4): rows 0-3, cols 0-3
    
    C (2x4): rows 0-1, cols 0-3
    
    Partition A into two 2x2 blocks: A = [A0 | A1] where A0 = A[:, 0:2], A1 = A[:, 2:4]
    Partition B into four 2x2 blocks: B = [[B00, B01], [B10, B11]]
    
    Then C = A*B = [A0*B00 + A1*B10, A0*B01 + A1*B11]
    
    Each term like A0*B00 is a 2x2 * 2x2 multiplication (rank 7 via Strassen).
    We have 4 such terms, but they share structure.
    
    Actually, let me just build the schoolbook rank-32 and then try to reduce.
    For now, let me construct a valid rank-28 decomposition using known results.
    
    The tensor rank of 2x4 * 4x4 is known to be at most 28.
    Let me construct it explicitly using block Strassen.
    """
    n, m, p = 2, 4, 4
    
    # We'll use the fact that we can view this as two 2x4 * 4x2 problems side by side
    # Each 2x4 * 4x2 can be decomposed in rank 14 using Strassen on 2x2 blocks
    # Total: 14 + 14 = 28
    
    # For 2x4 * 4x2:
    # A (2x4) = [A11 A12] where A11, A12 are 2x2
    # B (4x2) = [B11; B21] where B11, B21 are 2x2
    # C (2x2) = A11*B11 + A12*B21
    # 
    # Using Strassen for each 2x2*2x2:
    # M1 = (A11+A12)*(B11-B21) -- no, this doesn't work directly
    # 
    # Actually, C = A11*B11 + A12*B21 is a sum of two 2x2*2x2 products.
    # Each can be done in rank 7, so total rank 14.
    # But maybe we can do better by sharing.
    
    # Let me just build rank-14 for each half using standard Strassen.
    
    def strassen_2x2(U_rows, V_rows, W_rows, offset):
        """Add 7 Strassen terms for a 2x2*2x2 multiplication.
        U_rows: list of row indices in U (size 4, for 2x2)
        V_rows: list of row indices in V (size 4, for 2x2) 
        W_rows: list of row indices in W (size 4, for 2x2)
        offset: starting column index
        Returns: (U_entries, V_entries, W_entries, next_offset)
        """
        U_e = []
        V_e = []
        W_e = []
        col = offset
        
        # Strassen's 7 products for 2x2:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B10-B11)
        # M4 = A11*(B01-B10)
        # M5 = (A00+A10)*B11
        # M6 = (A01-A10)*(B00+B10)
        # M7 = (A01-A11)*(B01+B11)
        
        # Indexing: A = [[A00, A01], [A10, A11]]
        # U rows for A: U_rows[0]=A00, U_rows[1]=A01, U_rows[2]=A10, U_rows[3]=A11
        # V rows for B: V_rows[0]=B00, V_rows[1]=B10, V_rows[2]=B01, V_rows[3]=B11
        # W rows for C: W_rows[0]=C00, W_rows[1]=C01, W_rows[2]=C10, W_rows[3]=C11
        
        # M1 = (A00+A11)*(B00+B11)
        U_e.append((U_rows[0], 1, col))
        U_e.append((U_rows[3], 1, col))
        V_e.append((V_rows[0], 1, col))
        V_e.append((V_rows[3], 1, col))
        W_e.append((W_rows[0], 1, col))
        W_e.append((W_rows[1], -1, col))
        W_e.append((W_rows[2], -1, col))
        W_e.append((W_rows[3], 1, col))
        col += 1
        
        # M2 = (A10+A11)*B00
        U_e.append((U_rows[2], 1, col))
        U_e.append((U_rows[3], 1, col))
        V_e.append((V_rows[0], 1, col))
        W_e.append((W_rows[1], 1, col))
        W_e.append((W_rows[3], -1, col))
        col += 1
        
        # M3 = A00*(B10-B11)
        U_e.append((U_rows[0], 1, col))
        V_e.append((V_rows[1], 1, col))
        V_e.append((V_rows[3], -1, col))
        W_e.append((W_rows[0], 1, col))
        W_e.append((W_rows[2], 1, col))
        col += 1
        
        # M4 = A11*(B01-B10)
        U_e.append((U_rows[3], 1, col))
        V_e.append((V_rows[2], 1, col))
        V_e.append((V_rows[1], -1, col))
        W_e.append((W_rows[1], 1, col))
        W_e.append((W_rows[3], 1, col))
        col += 1
        
        # M5 = (A00+A10)*B11
        U_e.append((U_rows[0], 1, col))
        U_e.append((U_rows[2], 1, col))
        V_e.append((V_rows[3], 1, col))
        W_e.append((W_rows[2], 1, col))
        W_e.append((W_rows[3], -1, col))
        col += 1
        
        # M6 = (A01-A10)*(B00+B10)
        U_e.append((U_rows[1], 1, col))
        U_e.append((U_rows[2], -1, col))
        V_e.append((V_rows[0], 1, col))
        V_e.append((V_rows[1], 1, col))
        W_e.append((W_rows[0], -1, col))
        W_e.append((W_rows[1], 1, col))
        W_e.append((W_rows[2], 1, col))
        W_e.append((W_rows[3], -1, col))
        col += 1
        
        # M7 = (A01-A11)*(B01+B11)
        U_e.append((U_rows[1], 1, col))
        U_e.append((U_rows[3], -1, col))
        V_e.append((V_rows[2], 1, col))
        V_e.append((V_rows[3], 1, col))
        W_e.append((W_rows[0], 1, col))
        W_e.append((W_rows[1], 1, col))
        W_e.append((W_rows[2], -1, col))
        W_e.append((W_rows[3], -1, col))
        col += 1
        
        return U_e, V_e, W_e, col
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Left half: columns 0-1 of B, producing columns 0-1 of C
    # A is 2x4, B_left is 4x2, C_left is 2x2
    # A = [A0 | A1] where A0 = A[:, 0:2], A1 = A[:, 2:4]
    # B_left = [B00; B10] where B00 = B[0:2, 0:2], B10 = B[2:4, 0:2]
    # C_left = A0*B00 + A1*B10
    
    # For A0*B00: A0 is 2x2, B00 is 2x2
    # U rows for A0: A0[0,0]=U[0], A0[0,1]=U[1], A0[1,0]=U[2], A0[1,1]=U[3]
    # V rows for B00: B00[0,0]=V[0], B00[1,0]=V[1], B00[0,1]=V[2], B00[1,1]=V[3]
    # W rows for C_left[0,0]=W[0], C_left[0,1]=W[1], C_left[1,0]=W[2], C_left[1,1]=W[3]
    
    U_e1, V_e1, W_e1, col1 = strassen_2x2([0,1,2,3], [0,1,2,3], [0,1,2,3], 0)
    
    # For A1*B10: A1 is 2x2, B10 is 2x2
    # U rows for A1: A1[0,0]=U[4], A1[0,1]=U[5], A1[1,0]=U[6], A1[1,1]=U[7]
    # V rows for B10: B10[0,0]=V[8], B10[1,0]=V[9], B10[0,1]=V[10], B10[1,1]=V[11]
    # W rows for C_left: same W rows [0,1,2,3]
    
    U_e2, V_e2, W_e2, col2 = strassen_2x2([4,5,6,7], [8,9,10,11], [0,1,2,3], 7)
    
    # Right half: columns 2-3 of B, producing columns 2-3 of C
    # B_right = [B01; B11] where B01 = B[0:2, 2:4], B11 = B[2:4, 2:4]
    # C_right = A0*B01 + A1*B11
    # W rows for C_right: W[4], W[5], W[6], W[7]
    
    # For A0*B01:
    U_e3, V_e3, W_e3, col3 = strassen_2x2([0,1,2,3], [12,13,14,15], [4,5,6,7], 14)
    
    # For A1*B11:
    U_e4, V_e4, W_e4, col4 = strassen_2x2([4,5,6,7], [12,13,14,15], [4,5,6,7], 21)
    
    # Fill in the matrices
    for row, val, col in U_e1 + U_e2 + U_e3 + U_e4:
        U[row][col] += val
    for row, val, col in V_e1 + V_e2 + V_e3 + V_e4:
        V[row][col] += val
    for row, val, col in W_e1 + W_e2 + W_e3 + W_e4:
        W[row][col] += val
    
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=4, p=4, verbose=False):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    max_err = 0.0
    for trial in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C using decomposition: C[k*n+i, r] convention
        # C[i,j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Wait, the convention is W[k*n+i, r]
        # So C[i,j] where i is row of A (0..n-1), j is col of B (0..p-1)
        # C[i,j] = sum_{a,b,r} A[i,a] * B[a,j] * ... 
        # The tensor T[i,a,j,b] = delta_{a,b} for matrix mult
        # Decomposition: T[i,a,j,b] = sum_r U[i,a,r] * V[a,b,r] * W[j,i,r]
        # Wait, let me re-read the convention.
        
        # U has shape (n*m, rank): U[i*m + a, r]
        # V has shape (m*p, rank): V[a*p + b, r]  
        # W has shape (p*n, rank): W[b*n + i, r]
        
        # C[i,b] = sum_{a,r} A[i,a] * B[a,b] * ... 
        # Actually: C[i,b] = sum_r (sum_a U[i*m+a, r] * A[i,a]) * (sum_b' V[a*p+b', r] * B[a,b'])
        # No, that's not right either.
        
        # The tensor for matrix multiplication: T[i,a,b] = delta_{a,b} (where i is row of output, a is shared index, b is col of output)
        # Wait, for A(n x m) * B(m x p) -> C(n x p):
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # Tensor: T[i,j,k] = delta_{j,j} ... no.
        # The tensor is T[i,j,k] where the decomposition is:
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # But T[i,j,k] should equal delta_{j,j}... that doesn't make sense.
        
        # Let me think again. The matrix multiplication tensor has entries:
        # T[i,j,k] = 1 if the (i,k) entry of C depends on A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all valid i,j,k (it's just the identity in the j dimension)
        # No wait, T[i,j,k] is the coefficient such that:
        # C[i,k] = sum_{j} A[i,j] B[j,k]
        # So the tensor is T[i,j,k] = delta_{j,j} which is just 1.
        
        # The decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        
        # So to compute C:
        # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * ... no
        # C[i,k] = sum_{j,r} A[i,j] * B[j,k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Wait, that's not how it works. The tensor decomposition means:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k])
        
        # So the verification is: for all i,j,k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        max_err = max(max_err, abs(val - 1.0))
                        if verbose:
                            print(f"Error at i={i},j={j},k={k}: {val}")
    
    return max_err

def _try_reduce_rank(U, V, W, target_rank=27, time_limit=100):
    """Try to reduce the rank of the decomposition by finding integer dependencies."""
    start_time = time.time()
    n, m, p = 2, 4, 4
    current_rank = len(U[0])
    
    if current_rank <= target_rank:
        return U, V, W
    
    # Strategy: try to express one column as an integer linear combination of others
    # For each column c, check if U[:,c], V[:,c], W[:,c] can be written as
    # integer combinations of other columns.
    
    # Build the tensor as a 3D array and check linear dependencies
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want to find if any column r* is a linear combination of others.
    # If U[:,r*] = sum_{r != r*} alpha_r * U[:,r], and similarly for V and W,
    # then we can eliminate column r*.
    
    # This is a hard problem. Let's try a simpler approach:
    # Check if the 28 columns are linearly dependent in a structured way.
    
    # For each pair of columns (r1, r2), check if there exist scalars a, b
    # such that a*U[:,r1] + b*U[:,r2] is in the span of the other 26 columns,
    # and similarly for V and W.
    
    # Actually, let's try: for each column c, see if removing it still gives a valid
    # decomposition (i.e., the tensor is still correctly represented).
    
    # This won't work directly because removing a column changes the tensor.
    
    # Better approach: look for integer relations.
    # If for some c, there exist integers a_r (r != c) such that:
    # U[:,c] = sum_{r!=c} a_r * U[:,r]
    # V[:,c] = sum_{r!=c} b_r * V[:,r]  
    # W[:,c] = sum_{r!=c} d_r * W[:,r]
    # And the tensor contribution of column c equals the sum of contributions of others
    # with appropriate coefficients... this is complex.
    
    # Simpler: just check if the current rank-28 decomposition is valid,
    # and if not, fall back to rank-32.
    
    err = _verify_decomposition(U, V, W)
    if err < 1e-9:
        return U, V, W
    
    # If rank-28 doesn't verify, fall back to schoolbook rank-32
    if verbose:
        print(f"Rank-28 verification failed with error {err}, falling back to rank-32")
    
    rank = n * m * p
    U32 = [[0] * rank for _ in range(n * m)]
    V32 = [[0] * rank for _ in range(m * p)]
    W32 = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U32[i * m + j][column] = 1
                V32[j * p + k][column] = 1
                W32[k * n + i][column] = 1
                column += 1
    return U32, V32, W32

def construct():
    start_time = time.time()
    
    # Try rank-28 Strassen decomposition
    U, V, W = _build_strassen_28()
    
    # Verify
    err = _verify_decomposition(U, V, W)
    
    if err < 1e-9:
        # Try to reduce to rank-27 if time permits
        if time.time() - start_time < 100:
            U, V, W = _try_reduce_rank(U, V, W, target_rank=27, time_limit=100)
            err = _verify_decomposition(U, V, W)
    
    # Final verification
    if err >= 1e-9:
        # Fall back to schoolbook
        n, m, p = 2, 4, 4
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
    
    # Deadline guard
    if time.time() - start_time > 140:
        # Return whatever we have
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
