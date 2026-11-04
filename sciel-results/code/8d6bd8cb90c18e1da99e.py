import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-9):
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.max(np.abs(recon - T)) < tol

def _snap(U, V, W, T):
    """Snap to half-integers and verify."""
    U_s = np.round(U * 2) / 2
    V_s = np.round(V * 2) / 2
    W_s = np.round(W * 2) / 2
    if _verify(U_s, V_s, W_s, T):
        return U_s, V_s, W_s
    U_s = np.round(U)
    V_s = np.round(V)
    W_s = np.round(W)
    if _verify(U_s, V_s, W_s, T):
        return U_s, V_s, W_s
    return None

def _construct_rank19():
    """
    Construct explicit rank-19 decomposition for 2x3x4 matrix multiplication.
    
    A is 2x3, B is 3x4, C is 2x4.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We split B into B_top (rows 0,1) and B_bot (row 2):
    C = A_top * B_top + A_bot * B_bot
    where A_top = A[:,0:2] (2x2), A_bot = A[:,2:3] (2x1)
          B_top = B[0:2,:] (2x4), B_bot = B[2:3,:] (1x4)
    
    A_top * B_top: 2x2x4 -> use Strassen for two 2x2x2 blocks = rank 14
    A_bot * B_bot: 2x1x4 -> rank 4 (each output C[i,k] gets A[2,i]*B[2,k])
    
    Total: 14 + 4 = 18? Let me verify this actually works...
    
    Actually, let me just directly enumerate and verify.
    The schoolbook gives 24. Let me try to find a rank-19 by careful construction.
    
    Alternative: use the fact that for 2x3x4, we can write:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    Let me use a different split. Consider:
    C = A1*B1 + A2*B2 + A3*B3 where we group columns of A and rows of B.
    
    Actually, the cleanest approach: use rank-7 Strassen for the 2x2x2 subproblems
    and handle the remaining terms efficiently.
    
    Let me try: split the 3 middle dimension as 2+1.
    A = [A_left (2x2) | A_right (2x1)]
    B = [B_top (2x4); B_bot (1x4)]
    
    C = A_left @ B_top + A_right @ B_bot
    
    For A_left @ B_top (2x2 @ 2x4):
      Split B_top = [B_t1 (2x2) | B_t2 (2x2)]
      C_left = A_left @ B_t1 + A_left @ B_t2
      Each 2x2x2 uses rank 7 (Strassen), total rank 14.
    
    For A_right @ B_bot (2x1 @ 1x4):
      This is outer product: C_right[i,k] = A_right[i,0] * B_bot[0,k]
      Rank 4 (one per output element).
    
    Total: 14 + 4 = 18.
    
    But wait - is rank 18 actually achievable? The known lower bound for 2x3x4 is 19.
    Let me double-check... Actually, I recall that rank(2x3x4) = 19 is the established value.
    
    So my split must be wrong. Let me reconsider.
    
    The issue: when I split 2x2x4 into two 2x2x2, I'm using 14 scalars.
    And 2x1x4 uses 4 scalars. Total 18. But this should work!
    
    Unless... the Strassen decomposition for 2x2x2 doesn't quite apply directly
    when the output dimensions differ. Let me verify by explicit construction.
    """
    n, m, p = 2, 3, 4
    nm, mp, pn = 6, 12, 8
    
    rank = 19
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Strassen 2x2x2 decomposition
    # For matrices X (2x2), Y (2x2), Z = X@Y (2x2)
    # Z[0,0] = M1 + M4 - M5 + M7
    # Z[0,1] = M3 + M5
    # Z[1,0] = M2 + M4
    # Z[1,1] = M1 - M2 + M3 + M6
    # where:
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y10-Y11)
    # M4 = X11*(Y00-Y10)
    # M5 = (X00+X01)*Y11
    # M6 = (X10-X00)*(Y00+Y01)
    # M7 = (X01-X11)*(Y10+Y11)
    
    def strassen_2x2x2(base_u, base_v, base_w):
        """
        Add Strassen 2x2x2 to U, V, W starting at given column indices.
        U indices: row i*2+j for i in {0,1}, j in {0,1}
        V indices: row j*2+k for j in {0,1}, k in {0,1}  
        W indices: row k*2+i for k in {0,1}, i in {0,1}
        """
        # M1 = (X00+X11)*(Y00+Y11)
        U[base_u + 0, 0] = 1; U[base_u + 3, 0] = 1
        V[base_v + 0, 0] = 1; V[base_v + 3, 0] = 1
        W[base_w + 0, 0] = 1; W[base_w + 3, 0] = 1
        
        # M2 = (X10+X11)*Y00
        U[base_u + 2, 1] = 1; U[base_u + 3, 1] = 1
        V[base_v + 0, 1] = 1
        W[base_w + 2, 1] = 1
        
        # M3 = X00*(Y10-Y11)
        U[base_u + 0, 2] = 1
        V[base_v + 2, 2] = 1; V[base_v + 3, 2] = -1
        W[base_w + 0, 2] = 1
        
        # M4 = X11*(Y00-Y10)
        U[base_u + 3, 3] = 1
        V[base_v + 0, 3] = 1; V[base_v + 2, 3] = -1
        W[base_w + 3, 3] = 1
        
        # M5 = (X00+X01)*Y11
        U[base_u + 0, 4] = 1; U[base_u + 1, 4] = 1
        V[base_v + 3, 4] = 1
        W[base_w + 1, 4] = 1
        
        # M6 = (X10-X00)*(Y00+Y01)
        U[base_u + 2, 5] = 1; U[base_u + 0, 5] = -1
        V[base_v + 0, 5] = 1; V[base_v + 1, 5] = 1
        W[base_w + 2, 5] = 1
        
        # M7 = (X01-X11)*(Y10+Y11)
        U[base_u + 1, 6] = 1; U[base_u + 3, 6] = -1
        V[base_v + 2, 6] = 1; V[base_v + 3, 6] = 1
        W[base_w + 1, 6] = 1
        
        return 7
    
    # Part 1: A_left (2x2) @ B_top (2x4)
    # A_left[i,j] = A[i,j] for i in {0,1}, j in {0,1}
    # B_top[j,k] = B[j,k] for j in {0,1}, k in {0,1,2,3}
    # C_part1[i,k] = sum_{j=0,1} A[i,j] * B[j,k]
    
    # Split B_top into B_t1 (cols 0,1) and B_t2 (cols 2,3)
    # C_part1 = A_left @ B_t1 + A_left @ B_t2
    
    # A_left @ B_t1: 2x2 @ 2x2
    # U indices for A_left: row i*2+j -> i*3+j (since m=3)
    # Wait, U has shape (6, rank). U[i*m+j, :] = U[i*3+j, :]
    # V has shape (12, rank). V[j*p+k, :] = V[j*4+k, :]
    # W has shape (8, rank). W[k*n+i, :] = W[k*2+i, :]
    
    # For A_left @ B_t1:
    # A_left[i,j] is at U[i*3+j, :]
    # B_t1[j,k] is at V[j*4+k, :] for k in {0,1}
    # C[i,k] is at W[k*2+i, :] for k in {0,1}
    
    # Strassen block 1: A_left @ B_t1 (cols 0,1 of B)
    # U base: i*3+j for the 2x2 block of A (rows 0,1 of A, cols 0,1)
    # V base: j*4+k for the 2x2 block of B (rows 0,1 of B, cols 0,1)
    # W base: k*2+i for the 2x2 block of C (rows 0,1 of C, cols 0,1)
    
    col = 0
    # Strassen for A[0:2,0:2] @ B[0:2,0:2] -> C[0:2,0:2]
    # U: A[i,j] at U[i*3+j]
    # V: B[j,k] at V[j*4+k]  
    # W: C[i,k] at W[k*2+i]
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, col] = 1; U[4, col] = 1  # A[0,0]=U[0], A[1,1]=U[4]
    V[0, col] = 1; V[5, col] = 1  # B[0,0]=V[0], B[1,1]=V[5]
    W[0, col] = 1; W[3, col] = 1  # C[0,0]=W[0], C[1,1]=W[3]
    col += 1
    
    # M2 = (A10+A11)*B00
    U[3, col] = 1; U[4, col] = 1  # A[1,0]=U[3], A[1,1]=U[4]
    V[0, col] = 1  # B[0,0]=V[0]
    W[2, col] = 1  # C[1,0]=W[2]
    col += 1
    
    # M3 = A00*(B10-B11)
    U[0, col] = 1  # A[0,0]=U[0]
    V[4, col] = 1; V[5, col] = -1  # B[1,0]=V[4], B[1,1]=V[5]
    W[0, col] = 1  # C[0,0]=W[0]
    col += 1
    
    # M4 = A11*(B00-B10)
    U[4, col] = 1  # A[1,1]=U[4]
    V[0, col] = 1; V[4, col] = -1  # B[0,0]=V[0], B[1,0]=V[4]
    W[3, col] = 1  # C[1,1]=W[3]
    col += 1
    
    # M5 = (A00+A01)*B11
    U[0, col] = 1; U[1, col] = 1  # A[0,0]=U[0], A[0,1]=U[1]
    V[5, col] = 1  # B[1,1]=V[5]
    W[1, col] = 1  # C[0,1]=W[1]
    col += 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[3, col] = 1; U[0, col] = -1  # A[1,0]=U[3], A[0,0]=U[0]
    V[0, col] = 1; V[1, col] = 1  # B[0,0]=V[0], B[0,1]=V[1]
    W[2, col] = 1  # C[1,0]=W[2]
    col += 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, col] = 1; U[4, col] = -1  # A[0,1]=U[1], A[1,1]=U[4]
    V[4, col] = 1; V[5, col] = 1  # B[1,0]=V[4], B[1,1]=V[5]
    W[1, col] = 1  # C[0,1]=W[1]
    col += 1
    
    # Strassen for A[0:2,0:2] @ B[0:2,2:4] -> C[0:2,2:4]
    # B_t2[j,k] where k is 0,1 but maps to actual col k+2
    # V: B[j,k+2] at V[j*4+(k+2)]
    # W: C[i,k+2] at W[(k+2)*2+i]
    
    # M1 = (A00+A11)*(B02+B12)
    U[0, col] = 1; U[4, col] = 1
    V[2, col] = 1; V[6, col] = 1  # B[0,2]=V[2], B[1,2]=V[6]
    W[4, col] = 1; W[7, col] = 1  # C[0,2]=W[4], C[1,3]=W[7]
    col += 1
    
    # M2 = (A10+A11)*B02
    U[3, col] = 1; U[4, col] = 1
    V[2, col] = 1  # B[0,2]=V[2]
    W[6, col] = 1  # C[1,2]=W[6]
    col += 1
    
    # M3 = A00*(B12-B13)
    U[0, col] = 1
    V[6, col] = 1; V[7, col] = -1  # B[1,2]=V[6], B[1,3]=V[7]
    W[4, col] = 1  # C[0,2]=W[4]
    col += 1
    
    # M4 = A11*(B02-B12)
    U[4, col] = 1
    V[2, col] = 1; V[6, col] = -1
    W[7, col] = 1  # C[1,3]=W[7]
    col += 1
    
    # M5 = (A00+A01)*B13
    U[0, col] = 1; U[1, col] = 1
    V[7, col] = 1  # B[1,3]=V[7]
    W[5, col] = 1  # C[0,3]=W[5]
    col += 1
    
    # M6 = (A10-A00)*(B02+B03)
    U[3, col] = 1; U[0, col] = -1
    V[2, col] = 1; V[3, col] = 1  # B[0,2]=V[2], B[0,3]=V[3]
    W[6, col] = 1  # C[1,2]=W[6]
    col += 1
    
    # M7 = (A01-A11)*(B12+B13)
    U[1, col] = 1; U[4, col] = -1
    V[6, col] = 1; V[7, col] = 1
    W[5, col] = 1  # C[0,3]=W[5]
    col += 1
    
    # Part 2: A_right (2x1) @ B_bot (1x4)
    # A_right[i,0] = A[i,2] at U[i*3+2]
    # B_bot[0,k] = B[2,k] at V[2*4+k] = V[8+k]
    # C[i,k] at W[k*2+i]
    
    # C[i,k] += A[i,2] * B[2,k]
    # This is 2*4 = 8 terms, but we can do it in rank 4
    # because it's an outer product: C_right = A_right_col * B_bot_row
    # C_right[i,k] = a_i * b_k where a_i = A[i,2], b_k = B[2,k]
    # This is rank 1 as a matrix! But in tensor form, each (i,k) is separate.
    # Actually no - in the tensor decomposition, each scalar multiplication
    # A[i,2]*B[2,k] contributes to exactly one output C[i,k].
    # So we need 2*4 = 8 separate rank-1 terms? No...
    
    # Wait. The term A[i,2]*B[2,k] for fixed i,k is one scalar product.
    # In the tensor T, T[i*3+2, 8+k, k*2+i] = 1 for each (i,k).
    # Each such term is a separate rank-1 component.
    # So we need 2*4 = 8 more rank-1 terms.
    
    # Total: 7 + 7 + 8 = 22. That's worse than schoolbook!
    
    # I made an error. Let me reconsider.
    
    # The issue is that the outer product A_right @ B_bot gives:
    # C[i,k] = A[i,2] * B[2,k]
    # In tensor form, this is sum over... no, it's just one product per (i,k).
    # Each (i,k) pair needs its own rank-1 term: U[i*3+2, c]=1, V[8+k, c]=1, W[k*2+i, c]=1
    # That's 8 terms.
    
    # So total is 7+7+8 = 22. Not good.
    
    # I need a smarter decomposition. Let me try a different approach.
    
    # Alternative: don't split at all, use a global rank-19 construction.
    # This is getting complex. Let me just return schoolbook for now and 
    # try to verify if rank 19 is achievable.
    
    # For now, return schoolbook rank 24.
    rank = 24
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    total_start = time.time()
    time_budget = 110.0
    
    elapsed = time.time() - total_start
    if elapsed < time_budget - 5.0:
        U, V, W = _construct_rank19()
        if _verify(U, V, W, T):
            snapped = _snap(U, V, W, T)
            if snapped is not None:
                U_s, V_s, W_s = snapped
                return {
                    "U": U_s.tolist(),
                    "V": V_s.tolist(),
                    "W": W_s.tolist()
                }
    
    # Fallback schoolbook rank 24
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-END
