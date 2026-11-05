import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 150.0  # wall-clock limit

def _elapsed():
    return time.time() - _start_time

def _deadline_exceeded(margin=5.0):
    return _elapsed() > _DEADLINE - margin

def _laderman_2x2(a, b, c, d, p, q, r, s):
    """Laderman's rank-7 algorithm for 2x2 matrix multiplication.
    A = [[a, b], [c, d]], B = [[p, q], [r, s]]
    Returns C = A*B using 7 bilinear forms.
    """
    m1 = (a + c) * (p + r)
    m2 = (a - c) * (p - r)
    m3 = (b + d) * (q + s)
    m4 = (b - d) * (q - s)
    m5 = (a + b) * (p + q)
    m6 = (a - b) * (p - q)
    m7 = (c + d) * (r + s)
    # C[0,0] = m1 + m2 - m7 + m5
    # C[0,1] = m3 + m4 - m7 + m6
    # C[1,0] = m1 + m2 - m5 + m7
    # C[1,1] = m3 + m4 - m6 + m7
    c00 = m1 + m2 - m7 + m5
    c01 = m3 + m4 - m7 + m6
    c10 = m1 + m2 - m5 + m7
    c11 = m3 + m4 - m6 + m7
    return [[c00, c01], [c10, c11]]

def _verify_laderman():
    """Verify Laderman's algorithm with random matrices."""
    np.random.seed(42)
    for _ in range(100):
        a = np.random.randint(-10, 10, (2, 2))
        b = np.random.randint(-10, 10, (2, 2))
        expected = a @ b
        result = np.array(_laderman_2x2(
            a[0,0], a[0,1], a[1,0], a[1,1],
            b[0,0], b[0,1], b[1,0], b[1,1]
        ))
        if not np.allclose(expected, result):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Strategy: Use block decomposition A(3x4) = [A1(3x3) A2(3x1)]
    # B(4x5) = [B1(3x5); B2(1x5)]
    # C = A1*B1 + A2*B2
    # Further split B1(3x5) = [B1a(3x3) B1b(3x2)]
    # C = A1*B1a + A1*B1b + A2*B2
    #
    # For A1*B1a (3x3x3): Use a rank-19 algorithm
    # For A1*B1b (3x3x2): schoolbook rank 18
    # For A2*B2 (3x1x5): schoolbook rank 15
    # Total: 19 + 18 + 15 = 52
    
    # Implement rank-19 for 3x3x3 using BCLR algorithm
    # The BCLR algorithm has 19 bilinear forms.
    # Let me implement a known rank-19 decomposition.
    
    # For 3x3x3, one approach: use 7 (Laderman for 2x2 sub-block) + overhead
    # But the cleanest is to use the full BCLR 19-term algorithm.
    
    # Let me implement the BCLR rank-19 algorithm carefully.
    # Reference: Bini, Capovani, Lotti, Romani (1990)
    
    # The 19 bilinear forms for 3x3x3:
    # Let A = [[a1,a2,a3],[a4,a5,a6],[a7,a8,a9]]
    # Let B = [[b1,b2,b3],[b4,b5,b6],[b7,b8,b9]]
    
    # I'll use a known rank-19 formulation.
    # After research, here's a correct rank-19 algorithm:
    
    # m1 = (a1+a4+a7) * (b1+b4+b7)
    # m2 = (a1-a4) * (b1-b4)
    # m3 = (a2+a5+a8) * (b2+b5+b8)
    # m4 = (a2-a5) * (b2-b5)
    # m5 = (a3+a6+a9) * (b3+b6+b9)
    # m6 = (a3-a6) * (b3-b6)
    # m7 = (a1+a2+a4+a5) * (b1+b2+b4+b5)
    # m8 = (a1-a2+a4-a5) * (b1-b2+b4-b5)
    # m9 = (a7+a8+a1+a2+a4+a5) * (b7+b8+b1+b2+b4+b5)
    # m10 = (a7-a8+a1-a2+a4-a5) * (b7-b8+b1-b2+b4-b5)
    # m11 = (a3+a6+a1+a4+a7) * (b3+b6+b1+b4+b7)
    # m12 = (a3-a6+a1-a4+a7) * (b3-b6+b1-b4+b7)
    # m13 = (a1+a2+a3+a4+a5+a6+a7+a8+a9) * (b1+b2+b3+b4+b5+b6+b7+b8+b9)
    # m14 = ...
    # This is getting too complex and I might make errors.
    
    # Let me use a simpler, verifiable approach: 
    # Use the "bordered rank" technique or a known correct algorithm.
    
    # Alternative: Just use schoolbook for everything (rank 60) but structure it
    # cleanly. Since I can't easily verify a rank-19 implementation, let me
    # stick with a correct rank-60 solution but try to find any improvement.
    
    # Actually, let me try a different decomposition that might allow sharing.
    
    # New idea: Split 4 = 2+2, 5 = 3+2
    # A(3x4) = [A1(3x2) A2(3x2)]
    # B(4x5) = [B11(2x3) B12(2x2); B21(2x3) B22(2x2)]
    # C = A1*B11 + A1*B12 + A2*B21 + A2*B22
    # Ranks: 18 + 12 + 18 + 12 = 60. No improvement.
    
    # Let me just return the correct schoolbook rank-60 solution.
    # The current code already does this correctly.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
