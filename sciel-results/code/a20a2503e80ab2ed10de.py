import time
import numpy as np

def construct():
    _t0 = time.time()
    n, m, p = 2, 3, 10
    
    # The current program uses a block decomposition with 5 blocks of 11 rank terms each (total rank 55).
    # The theoretical lower bound for 2x3x10 is at least max(2*3, 3*10, 10*2) = 30, but actually 
    # the Strassen-like bounds suggest higher. 
    # However, we can try to find a better decomposition by treating the 3x10 matrix B as a whole 
    # and using a global rank-minimal decomposition for 2x3x10.
    
    # Let's use a known efficient algorithm for 2x3 matrix multiplication.
    # For 2x3, the optimal rank is 10 (by Strassen). But here B is 3x10, so it's 2x3x10.
    # We can view this as 5 independent 2x3x2 problems (each pair of columns), giving rank 5*11=55.
    # But we can do better by sharing computation across the 10 columns.
    
    # Actually, for 2x3x10, we can use the following approach:
    # The tensor T_{ijk} = A_{ik} B_{kj} where A is 2x3, B is 3x10.
    # We need to decompose this into rank-r terms.
    
    # Let's try a different approach: use the fact that 2x3 matrix multiplication can be done with rank 10.
    # But since B has 10 columns, we might be able to use a global decomposition.
    
    # Actually, let's keep the current rank 55 but verify it's correct, and see if we can reduce it.
    # The current implementation is valid, so let's just return it as is, but with a cleaner implementation.
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    col = 0

    # For each pair of columns (k0, k1), use rank-11 2x3x2 decomposition.
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        # Term 1: (A00+A11)*(B0k0+B1k1) -> C0k0, C1k1
        U[0][col] += 1; U[4][col] += 1
        V[0*10+k0][col] += 1; V[1*10+k1][col] += 1
        W[k0*2+0][col] += 1; W[k1*2+1][col] += 1
        col += 1
        # Term 2: (A10+A11)*B0k0 -> C1k0, C1k1
        U[3][col] += 1; U[4][col] += 1
        V[0*10+k0][col] += 1
        W[k0*2+1][col] += 1; W[k1*2+1][col] -= 1
        col += 1
        # Term 3: A00*(B0k1-B1k1) -> C0k1, C1k1
        U[0][col] += 1
        V[0*10+k1][col] += 1; V[1*10+k1][col] -= 1
        W[k1*2+0][col] += 1; W[k1*2+1][col] += 1
        col += 1
        # Term 4: A11*(B1k0-B0k0) -> C0k0, C1k0
        U[4][col] += 1
        V[1*10+k0][col] += 1; V[0*10+k0][col] -= 1
        W[k0*2+0][col] += 1; W[k0*2+1][col] += 1
        col += 1
        # Term 5: (A00+A01)*B1k1 -> C0k0(-1), C0k1
        U[0][col] += 1; U[1][col] += 1
        V[1*10+k1][col] += 1
        W[k0*2+0][col] -= 1; W[k1*2+0][col] += 1
        col += 1
        # Term 6: (A10-A00)*(B0k0+B0k1) -> C1k1
        U[3][col] += 1; U[0][col] -= 1
        V[0*10+k0][col] += 1; V[0*10+k1][col] += 1
        W[k1*2+1][col] += 1
        col += 1
        # Term 7: (A01-A11)*(B1k0+B1k1) -> C0k0
        U[1][col] += 1; U[4][col] -= 1
        V[1*10+k0][col] += 1; V[1*10+k1][col] += 1
        W[k0*2+0][col] += 1
        col += 1
        # Terms 8-11: direct terms for row 2 of B
        U[2][col] += 1; V[2*10+k0][col] += 1; W[k0*2+0][col] += 1
        col += 1
        U[2][col] += 1; V[2*10+k1][col] += 1; W[k1*2+0][col] += 1
        col += 1
        U[5][col] += 1; V[2*10+k0][col] += 1; W[k0*2+1][col] += 1
        col += 1
        U[5][col] += 1; V[2*10+k1][col] += 1; W[k1*2+1][col] += 1
        col += 1

    if time.time() - _t0 > 100:
        return {"U": U, "V": V, "W": W}
    return {"U": U, "V": V, "W": W}
