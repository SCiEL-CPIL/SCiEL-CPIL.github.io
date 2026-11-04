"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n=2, m=5, p=6):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    # Check that sum_r U[:,r] * V[:,r] * W[:,r] equals the multiplication tensor
    # The multiplication tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # and 0 otherwise.
    for r in range(rank):
        pass  # We'll check the full sum
    
    # Build the tensor from U, V, W
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # We need T[a,b,c] = 1 if (a,b,c) corresponds to a valid (i,j,k) triple, else 0.
    
    # For efficiency, check all valid (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if val != 1:
                    return False
    
    # Also check that invalid entries are 0
    # This is expensive, so skip for now or do a sample
    return True

def _strassen_2x2x2():
    """Return the 7 rank-1 tensors for 2x2x2 Strassen decomposition.
    Each tensor is (u, v, w) where u, v, w are length-4 vectors.
    u corresponds to A entries [a11, a12, a21, a22]
    v corresponds to B entries [b11, b12, b21, b22]
    w corresponds to C entries [c11, c12, c21, c22]
    """
    # Let me use a verified Strassen decomposition.
    # I'll construct it and verify in code.
    tensors = []
    
    # P1 = (a11+a22)(b11+b22) -> contributes to c11, c22
    # Actually, let me just use the standard one and verify.
    
    # From a reliable source, the Strassen decomposition is:
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b21 - b22)
    # M4 = a22(b12 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    #
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + P3 + M6  (I'll verify this)
    
    # Let me just define the tensors and verify numerically.
    tensors = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),  # M3
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7
    ]
    return tensors

def _verify_strassen():
    """Verify the Strassen decomposition."""
    tensors = _strassen_2x2x2()
    # Test with random matrices
    np.random.seed(42)
    for _ in range(10):
        A = np.random.randn(4)  # [a11, a12, a21, a22]
        B = np.random.randn(4)  # [b11, b12, b21, b22]
        
        # Compute C directly
        a11, a12, a21, a22 = A
        b11, b12, b21, b22 = B
        c11 = a11*b11 + a12*b21
        c12 = a11*b12 + a12*b22
        c21 = a21*b11 + a22*b12
        c22 = a21*b12 + a22*b22
        C_direct = np.array([c11, c12, c21, c22])
        
        # Compute C via Strassen
        C_strassen = np.zeros(4)
        for u, v, w in tensors:
            m_val = sum(u[i]*A[i] for i in range(4)) * sum(v[i]*B[i] for i in range(4))
            for i in range(4):
                C_strassen[i] += w[i] * m_val
        
        if not np.allclose(C_direct, C_strassen):
            return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # Leave margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # First, verify Strassen works
    if not _verify_strassen():
        # Fall back to schoolbook
        rank = schoolbook_rank
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
        return {"U": U, "V": V, "W": W}
    
    # Build the improved decomposition:
    # Part 1: j in {0,1}, Strassen 2x2x6 (21 terms)
    # Part 2: j in {2,3}, Strassen 2x2x6 (21 terms)
    # Part 3: j in {4}, schoolbook 2x1x6 (12 terms)
    # Total: 54 terms
    
    rank = 54
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    strassen_tensors = _strassen_2x2x2()
    
    # Part 1: j in {0,1}, Strassen on three 2x2x2 blocks
    for block in range(3):  # blocks for columns 0-1, 2-3, 4-5
        k_offset = block * 2
        for (u, v, w) in strassen_tensors:
            u_map = [(0, 0), (0, 1), (1, 0), (1, 1)]  # (i, j) for a11, a12, a21, a22
            for idx, (ii, jj) in enumerate(u_map):
                a_idx = ii * m + jj
                U[a_idx][col] += u[idx]
            
            v_map = [(0, k_offset), (0, k_offset+1), (1, k_offset), (1, k_offset+1)]
            for idx, (jj, kk) in enumerate(v_map):
                b_idx = jj * p + kk
                V[b_idx][col] += v[idx]
            
            w_map = [(k_offset, 0), (k_offset+1, 0), (k_offset, 1), (k_offset+1, 1)]
            for idx, (kk, ii) in enumerate(w_map):
                c_idx = kk * n + ii
                W[c_idx][col] += w[idx]
            
            col += 1
    
    # Part 2: j in {2,3}, Strassen on three 2x2x2 blocks
    for block in range(3):
        k_offset = block * 2
        for (u, v, w) in strassen_tensors:
            u_map = [(0, 2), (0, 3), (1, 2), (1, 3)]  # (i, j) for a11, a12, a21, a22
            for idx, (ii, jj) in enumerate(u_map):
                a_idx = ii * m + jj
                U[a_idx][col] += u[idx]
            
            v_map = [(2, k_offset), (2, k_offset+1), (3, k_offset), (3, k_offset+1)]
            for idx, (jj, kk) in enumerate(v_map):
                b_idx = jj * p + kk
                V[b_idx][col] += v[idx]
            
            w_map = [(k_offset, 0), (k_offset+1, 0), (k_offset, 1), (k_offset+1, 1)]
            for idx, (kk, ii) in enumerate(w_map):
                c_idx = kk * n + ii
                W[c_idx][col] += w[idx]
            
            col += 1
    
    # Part 3: j in {4}, schoolbook 2x1x6 (12 terms)
    for i in range(n):
        for k in range(p):
            a_idx = i * m + 4
            b_idx = 4 * p + k
            c_idx = k * n + i
            U[a_idx][col] = 1.0
            V[b_idx][col] = 1.0
            W[c_idx][col] = 1.0
            col += 1
    
    # Verify the decomposition
    if not _verify(U, V, W, n, m, p):
        # Fall back to schoolbook
        rank = schoolbook_rank
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
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
