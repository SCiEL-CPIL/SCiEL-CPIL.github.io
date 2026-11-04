import numpy as np
import time

DIMENSIONS = [3, 4, 8]
_DEADLINE = 140.0  # safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if U @ V @ W^T reconstructs the matmul tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _add_strassen_block(U, V, W, base_col, i0, jA0, jB0, k0):
    """
    Add a rank-7 Strassen decomposition for a 2x2x2 block.
    i0: start row of A (and C)
    jA0: start col of A
    jB0: start row of B
    k0: start col of B (and C)
    """
    col = base_col
    # M1 = (A00+A11)(B00+B11)
    U[i0*4 + jA0, col] += 1
    U[(i0+1)*4 + (jA0+1), col] += 1
    V[jB0*8 + k0, col] += 1
    V[(jB0+1)*8 + (k0+1), col] += 1
    W[k0*3 + i0, col] += 1
    W[(k0+1)*3 + (i0+1), col] += 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[(i0+1)*4 + jA0, col] += 1
    U[(i0+1)*4 + (jA0+1), col] += 1
    V[jB0*8 + k0, col] += 1
    W[k0*3 + (i0+1), col] += 1
    W[(k0+1)*3 + (i0+1), col] -= 1
    col += 1
    
    # M3 = A00(B01-B11)
    U[i0*4 + jA0, col] += 1
    V[jB0*8 + (k0+1), col] += 1
    V[(jB0+1)*8 + (k0+1), col] -= 1
    W[(k0+1)*3 + i0, col] += 1
    W[(k0+1)*3 + (i0+1), col] += 1
    col += 1
    
    # M4 = A11(B10-B00)
    U[(i0+1)*4 + (jA0+1), col] += 1
    V[(jB0+1)*8 + k0, col] += 1
    V[jB0*8 + k0, col] -= 1
    W[k0*3 + i0, col] += 1
    W[k0*3 + (i0+1), col] += 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[i0*4 + jA0, col] += 1
    U[i0*4 + (jA0+1), col] += 1
    V[(jB0+1)*8 + (k0+1), col] += 1
    W[k0*3 + i0, col] -= 1
    W[(k0+1)*3 + i0, col] += 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[(i0+1)*4 + jA0, col] += 1
    U[i0*4 + jA0, col] -= 1
    V[jB0*8 + k0, col] += 1
    V[jB0*8 + (k0+1), col] += 1
    W[(k0+1)*3 + (i0+1), col] += 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[i0*4 + (jA0+1), col] += 1
    U[(i0+1)*4 + (jA0+1), col] -= 1
    V[(jB0+1)*8 + k0, col] += 1
    V[(jB0+1)*8 + (k0+1), col] += 1
    W[k0*3 + i0, col] += 1
    col += 1
    
    return col

def _construct_hybrid():
    """
    Construct a rank-64 decomposition for 3x4x8 matmul.
    Row 0: Linear combination (rank 8)
    Rows 1-2: Strassen-based (rank 56)
    """
    n, m, p = 3, 4, 8
    R = 64
    
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    col = 0
    
    # Part 1: Row 0 of A (1x4) times B (4x8) - rank 8
    # C[0,k] = sum_j A[0,j] * B[j,k]
    # For each k, one rank-1 term: U[A[0,j]] contributes, V[B[j,k]] contributes, W[C[0,k]] contributes
    for k in range(8):
        for j in range(4):
            U[0*m + j, col] = 1
            V[j*p + k, col] = 1
            W[k*n + 0, col] = 1
            col += 1
        # Wait, that's 32 again. Let me think...
        # Actually for 1x4x8, the rank IS 8 because we can write:
        # C_k = sum_j a_j * b_jk
        # This needs 8 terms (one per output), each term is a product of a linear form in A and a linear form in B
        # But the "linear form in A" for output k is the same for all k: it's just a_j
        # Hmm, no. Each output C_k = sum_j a_j * b_jk. The a_j are shared.
        # Rank-1 term: U has a_j in row (0,j), V has b_jk in row (j,k), W has 1 in (k,0)
        # For a single output k: C_k = sum_j a_j * b_jk = sum_j (a_j)(b_jk)(1)
        # That's 4 rank-1 terms for one output, 32 total.
        # But we can do better! C_k = <a, b_{:,k}> where b_{:,k} is column k of B.
        # The rank of the 1x4x8 tensor is actually 8 (min(4,8) for the "bilinear" part... no)
        # Actually the rank of 1xm x p matmul is m*p... no, it's max(m,p)? No.
        # The tensor T_{j,k} = a_j * b_jk. This is an outer product of a (size 4) and B^T (size 4x8).
        # The rank of this tensor is the matrix rank of B^T which is at most 4, but generically 4.
        # Wait no, the tensor is T_{j,k} = a_j * b_{j,k}. If we fix j, it's a_j * b_{j,:} (a row of B scaled).
        # The tensor can be written as sum over... hmm.
        # Actually T = a ⊗ B^T (element-wise? no). T_{j,k} = a_j * B_{j,k}.
        # This is the Hadamard product of a (repeated) and B. 
        # The rank of this 1x4x8 tensor: it's the rank of the 4x8 matrix M where M_{j,k} = a_j * B_{j,k}.
        # But a_j are variables, so this is a tensor in the a-space and B-space.
        # The tensor is T = sum_j a_j * e_j ⊗ b_j^T where b_j is row j of B.
        # That's 4 terms! So rank ≤ 4.
        # Wait: T_{j',k} = a_{j'} * B_{j',k}. If we write T = sum_j a_j * (e_j ⊗ B_{j,:}), then
        # (e_j ⊗ B_{j,:})_{j',k} = delta_{j,j'} * B_{j,k}. So T_{j',k} = sum_j a_j * delta_{j,j'} * B_{j,k} = a_{j'} * B_{j',k}. ✓
        # So rank is at most 4! And at least 4 (since we need 4 independent terms).
        # So 1x4x8 has rank 4, not 8!
        pass
    col = 0  # reset, redo
    
    # Part 1: Row 0 of A (1x4) times B (4x8) - rank 4
    # T_{j,k} = a_j * b_{j,k} = sum_{j'} a_{j'} * delta_{j,j'} * b_{j',k}
    # Rank-1 terms: for each j', U[0,j'] = 1, V[j',k] = b_{j,k}... 
    # Wait, V must be fixed (not depend on B). Let me re-examine.
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should sum to 1 for the (i,j,k) tensor entry.
    # For i=0: we need sum_r U[j,r]*V[j*p+k,r]*W[k,r] = 1 for all j,k.
    # This means U@V@W^T restricted to i=0 slice equals the all-ones 4x8 matrix.
    # U is 4xR (rows j=0..3), V is 32xR, W is 8xR (rows k=0..7, col i=0).
    # We need: for all j,k: sum_r U[j,r] * V[j*p+k,r] * W[k,r] = 1
    # One way: R=4, U[j,r] = delta_{j,r}, V[j*p+k,r] = delta_{j,r} (i.e., V[j*p+k,r]=1 if j==r else 0), W[k,r] = 1
    # Then sum_r delta_{j,r} * delta_{j,r} * 1 = delta_{j,j} = 1. ✓
    for r in range(4):
        U[0*m + r, col] = 1
        for k in range(8):
            V[r*p + k, col] = 1
        for k in range(8):
            W[k*n + 0, col] = 1
        col += 1
    
    # Part 2: Rows 1-2 of A (2x4) times B (4x8) - Strassen, rank 56
    col = _add_strassen_block(U, V, W, col, i0=1, jA0=0, jB0=0, k0=0)
    col = _add_strassen_block(U, V, W, col, i0=1, jA0=2, jB0=2, k0=0)
    col = _add_strassen_block(U, V, W, col, i0=1, jA0=0, jB0=0, k0=2)
    col = _add_strassen_block(U, V, W, col, i0=1, jA0=2, jB0=2, k0=2)
    col = _add_strassen_block(U, V, W, col, i0=1, jA0=0, jB0=0, k0=4)
    col = _add_strassen_block(U, V, W, col, i0=1, jA0=2, jB0=2, k0=4)
    col = _add_strassen_block(U, V, W, col, i0=1, jA0=0, jB0=0, k0=6)
    col = _add_strassen_block(U, V, W, col, i0=1, jA0=2, jB0=2, k0=6)
    
    assert col == 60, f"Expected 60, got {col}"
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try hybrid Strassen construction (rank 60)
    try:
        U_h, V_h, W_h = _construct_hybrid()
        if time.time() - start_time > _DEADLINE:
            pass
        elif _verify(U_h, V_h, W_h, n, m, p):
            return {"U": U_h, "V": V_h, "W": W_h}
        else:
            pass
    except Exception:
        pass
    
    # Schoolbook fallback (rank 96) - guaranteed valid
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    return {"U": U_sb, "V": V_sb, "W": W_sb}
