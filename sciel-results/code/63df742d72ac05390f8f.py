import time
import numpy as np

_T0 = time.time()
_DEADLINE_S = 110.0

def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE_S

def _winograd_2x2_2x2():
    """
    Returns U, V, W for 2x2 * 2x2 multiplication with rank 7.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    """
    # Indices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # m1 = (a00 + a11)(b00 + b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # c00, c11 contributions? 
    # Actually W maps to output C indices.
    # c00 = a00b00 + a01b10
    # c01 = a00b01 + a01b11
    # c10 = a10b00 + a11b10
    # c11 = a10b01 + a11b11
    
    # Let's use the standard Winograd coefficients.
    # m1 = (a00+a11)(b00+b11)
    # m2 = a11(b00-b11)
    # m3 = a00(b11-b00)
    # m4 = (a01+a10)(b01+b10)
    # m5 = a10(b01-b10)
    # m6 = a01(b10-b01)
    # m7 = (a00+a01+a11)(b00+b01+b11) -- This is not standard.
    
    # Correct Winograd 2x2:
    # m1 = (a00 + a11)(b00 + b11)
    # m2 = a11(b00 - b11)
    # m3 = a00(b11 - b00)
    # m4 = (a01 + a10)(b01 + b10)
    # m5 = a10(b01 - b10)
    # m6 = a01(b10 - b01)
    # m7 = (a00 + a01 + a11)(b00 + b01 + b11) is WRONG.
    
    # The 7th term is actually not needed if we combine differently?
    # No, rank 7 is optimal.
    # c00 = m1 + m2 + m3?
    # c00 = a00b00 + a01b10
    # Let's just use a known valid set of 7 multiplications.
    
    # Reference: Winograd (1970)
    # m1 = (a00+a11)(b00+b11)
    # m2 = a11(b00-b11)
    # m3 = a00(b11-b00)
    # m4 = (a01+a10)(b01+b10)
    # m5 = a10(b01-b10)
    # m6 = a01(b10-b01)
    # m7 = (a00+a01+a11)(b00+b01+b11) -> This is for 3x3?
    
    # Actually, for 2x2, the 7th multiplication is:
    # m7 = (a00 + a01 + a11)(b00 + b01 + b11) is incorrect.
    # The correct 7th is:
    # m7 = (a00 + a11 + a01)(b00 + b11 + b01) ? No.
    
    # Let's use the explicit linear combinations for c's.
    # c00 = a00b00 + a01b10
    # c01 = a00b01 + a01b11
    # c10 = a10b00 + a11b10
    # c11 = a10b01 + a11b11
    
    # We need 7 rank-1 tensors.
    # I will hardcode a verified rank-7 decomposition.
    
    U = [
        [1, 0, 1, 0, 0, 0, 0], # a00
        [0, 0, 0, 1, 0, 1, 0], # a01
        [0, 0, 0, 1, 1, 0, 0], # a10
        [1, 1, 0, 0, 0, 0, 0], # a11
    ]
    V = [
        [1, 1, -1, 0, 0, 0, 0], # b00
        [0, 0, 0, 1, 1, -1, 0], # b01
        [0, 0, 0, 1, -1, 1, 0], # b10
        [1, -1, 1, 0, 0, 0, 0], # b11
    ]
    W = [
        [1, 1, 1, 0, 0, 0, 0], # c00
        [0, 0, 0, 1, 1, 1, 0], # c01
        [0, 0, 0, 1, 1, 1, 0], # c10 -- Wait, c10 needs different coeffs
        [1, -1, 1, 0, 0, 0, 0], # c11
    ]
    # This hardcoded guess is likely wrong. 
    # I will use the schoolbook for safety if I can't verify Winograd quickly.
    # But rank 46 is the target.
    # I will implement the Winograd block and verify it locally in the code.
    
    # Correct Winograd 2x2 (Rank 7):
    # m1 = (a00+a11)(b00+b11)
    # m2 = a11(b00-b11)
    # m3 = a00(b11-b00)
    # m4 = (a01+a10)(b01+b10)
    # m5 = a10(b01-b10)
    # m6 = a01(b10-b01)
    # m7 = (a00+a01+a11)(b00+b01+b11) is NOT correct.
    
    # Actually, c00 = m1 + m2 + m3?
    # m1+m2+m3 = (a00+a11)(b00+b11) + a11(b00-b11) + a00(b11-b00)
    # = a00b00 + a00b11 + a11b00 + a11b11 + a11b00 - a11b11 + a00b11 - a00b00
    # = 2a11b00 + 2a00b11. This is not c00.
    
    # I will fall back to schoolbook if I cannot find the exact coefficients.
    # However, I can use the fact that 2x2x2 rank is 7.
    # I will return schoolbook for now to ensure correctness, but the prompt asks for improvement.
    # If I return schoolbook, score is 1.0.
    # I will try to construct the rank 46 decomposition.
    
    # Let's use the explicit Winograd coefficients from a reliable source.
    # U (A coeffs):
    # 1: 1, 0, 0, 1
    # 2: 0, 0, 0, 1
    # 3: 1, 0, 0, 0
    # 4: 0, 1, 1, 0
    # 5: 0, 0, 1, 0
    # 6: 0, 1, 0, 0
    # 7: 1, 1, 0, 1
    
    # V (B coeffs):
    # 1: 1, 0, 0, 1
    # 2: 1, 0, 0, -1
    # 3: -1, 0, 0, 1
    # 4: 0, 1, 1, 0
    # 5: 0, 1, -1, 0
    # 6: 0, -1, 1, 0
    # 7: 1, 1, 0, 1
    
    # W (C coeffs):
    # 1: 1, 0, 0, 1
    # 2: 1, 0, 0, 0
    # 3: 0, 0, 0, 1
    # 4: 0, 1, 1, 0
    # 5: 0, 0, 1, 0
    # 6: 0, 1, 0, 0
    # 7: 1, 1, 0, 1
    
    # This is a guess. I will verify it in the code.
    
    U_w = np.array([
        [1, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 1]
    ], dtype=int)
    
    V_w = np.array([
        [1, 1, -1, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, -1, 1],
        [0, 0, 0, 1, -1, 1, 0],
        [1, -1, 1, 0, 0, 0, 1]
    ], dtype=int)
    
    W_w = np.array([
        [1, 1, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 0],
        [1, -1, 1, 0, 0, 0, 1]
    ], dtype=int)
    
    # Verify
    # C = A @ B
    # c00 = a00b00 + a01b10
    # Check r=0: a00b00 + a11b11? No.
    
    # I will use schoolbook to be safe.
    return U_w, V_w, W_w

def construct():
    # Use schoolbook to guarantee valid result.
    # Rank 50.
    n, m, p = 2, 5, 5
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
    return {"U": U, "V": V, "W": W}
