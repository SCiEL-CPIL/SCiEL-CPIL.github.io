"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_6x6_block_strassen():
    """
    Construct a rank-189 decomposition for 6x6 matrix multiplication
    by using a 3x3 block structure with each 2x2 block multiplication
    done via Strassen's 7-multiplication algorithm.
    """
    n, m, p = 6, 6, 6
    rank = 189
    U = [[0.0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0.0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0.0] * rank for _ in range(p * n)]  # 36 x 189
    
    # Strassen coefficients for 2x2 matrix multiply:
    # P0 = (A00+A11)(B00+B11)
    # P1 = (A10+A11)*B00
    # P2 = A00*(B10-B11)
    # P3 = A11*(B01-B00)
    # P4 = (A00+A01)*B11
    # P5 = (A10-A00)*(B00+B10)
    # P6 = (A01-A11)*(B01+B11)
    #
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    
    col = 0
    for i in range(3):  # row block of A / row block of C
        for k in range(3):  # col block of A / row block of B
            for j in range(3):  # col block of B / col block of C
                # Flat index helpers
                # A is 6x6, row-major. Block (i, k) contains rows 2i, 2i+1 and cols 2k, 2k+1.
                # A_ik[a, b] is at (2i + a) * 6 + (2k + b)
                def aA(a, b): return (2*i + a) * 6 + (2*k + b)
                
                # B is 6x6, row-major. Block (k, j) contains rows 2k, 2k+1 and cols 2j, 2j+1.
                # B_kj[a, b] is at (2k + a) * 6 + (2j + b)
                def bB(a, b): return (2*k + a) * 6 + (2*j + b)
                
                # W row index is k*n + i. 
                # C element (r, c) where r=2i+a, c=2j+b.
                # W index for C[r, c] is c * n + r = (2j + b) * 6 + (2i + a).
                def cW(a, b): return (2*j + b) * 6 + (2*i + a)
                
                # P0 = (A00+A11)(B00+B11)
                U[aA(0,0)][col] += 1
                U[aA(1,1)][col] += 1
                V[bB(0,0)][col] += 1
                V[bB(1,1)][col] += 1
                W[cW(0,0)][col] += 1
                col += 1
                
                # P1 = (A10+A11)*B00
                U[aA(1,0)][col] += 1
                U[aA(1,1)][col] += 1
                V[bB(0,0)][col] += 1
                W[cW(1,0)][col] += 1
                col += 1
                
                # P2 = A00*(B10-B11)
                U[aA(0,0)][col] += 1
                V[bB(1,0)][col] += 1
                V[bB(1,1)][col] += -1
                W[cW(0,1)][col] += 1
                col += 1
                
                # P3 = A11*(B01-B00)
                U[aA(1,1)][col] += 1
                V[bB(0,1)][col] += 1
                V[bB(0,0)][col] += -1
                W[cW(0,0)][col] += 1
                W[cW(1,0)][col] += 1
                col += 1
                
                # P4 = (A00+A01)*B11
                U[aA(0,0)][col] += 1
                U[aA(0,1)][col] += 1
                V[bB(1,1)][col] += 1
                W[cW(0,0)][col] += -1
                W[cW(0,1)][col] += 1
                col += 1
                
                # P5 = (A10-A00)*(B00+B10)
                U[aA(1,0)][col] += 1
                U[aA(0,0)][col] += -1
                V[bB(0,0)][col] += 1
                V[bB(1,0)][col] += 1
                W[cW(1,1)][col] += 1
                col += 1
                
                # P6 = (A01-A11)*(B01+B11)
                U[aA(0,1)][col] += 1
                U[aA(1,1)][col] += -1
                V[bB(0,1)][col] += 1
                V[bB(1,1)][col] += 1
                W[cW(0,0)][col] += 1
                col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}

def _build_schoolbook(n=6, m=6, p=6):
    """Standard O(nmp) decomposition with rank = n*m*p = 216."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _validate_decomposition(U, V, W, n=6, m=6, p=6):
    """
    Validate by checking random inputs.
    Correctly handles the index mapping:
    C[i, k] corresponds to W row index k*n + i.
    """
    try:
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rank = Ua.shape[1]
    except Exception:
        return False

    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    A_flat = A.reshape(n * m)
    B_flat = B.reshape(m * p)

    # Compute C via decomposition
    # C_decomp_flat has index k*n + i corresponding to C[i, k]
    C_decomp_flat = np.zeros(p * n)
    
    # Efficient computation:
    # C[i, k] = sum_j sum_c U[i*m+j, c] A[i, j] V[j*p+k, c] B[j, k] W[k*n+i, c]
    # Let's iterate over c for speed if rank is small, or use matrix ops.
    # Rank 189 is small enough for a loop in Python if optimized, but NumPy is better.
    
    # We can reshape U, V, W to 3D tensors?
    # U is (n*m, rank). Reshape to (n, m, rank).
    # V is (m*p, rank). Reshape to (m, p, rank).
    # W is (p*n, rank). Reshape to (p, n, rank).
    
    U_3d = Ua.reshape(n, m, rank)
    V_3d = Va.reshape(m, p, rank)
    W_3d = Wa.reshape(p, n, rank)
    
    # A is (n, m), B is (m, p).
    # We want C[i, k].
    # Term for c: 
    #   (U_3d[:, :, c] * A) @ (V_3d[:, :, c] * B) * W_3d[:, :, c].T ?
    # Let's check dimensions.
    # U_3d[:, :, c] is (n, m). Multiply by A (n, m) element-wise -> (n, m).
    # V_3d[:, :, c] is (m, p). Multiply by B (m, p) element-wise -> (m, p).
    # Matrix product: (n, m) @ (m, p) -> (n, p).
    # This gives a matrix M_c where M_c[i, k] = sum_j U[i, j, c] A[i, j] V[j, k, c] B[j, k].
    # Then multiply by W[k, i, c] (from W_3d[k, i, c]).
    # So C[i, k] += M_c[i, k] * W_3d[k, i, c].
    
    C_total = np.zeros((n, p))
    for c in range(rank):
        uc = U_3d[:, :, c] * A
        vc = V_3d[:, :, c] * B
        mc = uc @ vc
        # mc is (n, p). W_3d[:, :, c] is (p, n).
        # We need to multiply mc[i, k] by W_3d[k, i, c].
        # W_3d.T is (n, p) with W_3d.T[i, k] = W_3d[k, i, c].
        C_total += mc * W_3d[:, :, c].T
        
    # C_total[i, k] is the computed C[i, k].
    # We need to compare with A @ B.
    C_direct = A @ B
    
    # Check if C_total matches C_direct
    if np.max(np.abs(C_total - C_direct)) > 1e-9:
        return False
        
    return True

def construct():
    t0 = time.time()
    
    # Build the block-Strassen decomposition (rank 189)
    result = _build_6x6_block_strassen()
    
    # Validate
    if time.time() - t0 < 110:
        try:
            if not _validate_decomposition(result["U"], result["V"], result["W"]):
                # Fallback to schoolbook
                result = _build_schoolbook()
        except Exception:
            # If validation throws an exception, fallback to be safe
            result = _build_schoolbook()
    
    # Safety margin
    if time.time() - t0 > 115:
        result = _build_schoolbook()
        
    return result

# EVOLVE-BLOCK-END
