import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    
    # Rank-7 decomposition of 2x2 matrix multiplication (Strassen)
    # T_{i,j,k} = A_{i,l} * B_{l,k} for i,j,k in {0,1}
    # We represent U, V, W as 4x4 matrices (row-major: A[i*m+j], B[j*p+k], C[k*n+i])
    # For a 2x2 block: A indices 0..3, B indices 0..3, C indices 0..3
    
    # The 7 rank-1 terms for 2x2 matrix multiplication:
    # Term 1: u=[1,0,0,0], v=[0,0,0,1], w=[1,0,0,1]  -> a00*b11 -> c00+c11
    # Term 2: u=[0,0,0,1], v=[1,0,0,0], w=[0,0,1,0]  -> a11*b00 -> c10
    # Term 3: u=[0,1,0,0], v=[0,1,0,0], w=[0,1,0,0]  -> a01*b01 -> c01
    # Term 4: u=[1,0,1,0], v=[0,1,0,1], w=[1,0,1,0]  -> (a00+a10)*(b01+b11) -> c00+c10
    # Term 5: u=[0,1,1,0], v=[1,0,1,0], w=[0,1,1,0]  -> (a01+a11)*(b00+b10) -> c01+c11
    # Term 6: u=[1,1,0,0], v=[0,0,1,1], w=[1,1,0,0]  -> (a00+a01)*(b10+b11) -> c00+c01
    # Term 7: u=[1,0,0,0], v=[0,0,1,1], w=[0,0,1,1]  -> a00*(b10+b11) -> c10+c11
    
    # Let me verify this decomposition:
    # c00 = term1(1) + term4(1) + term6(1) - term7(1) = 1+1+1-1 = 2? That's wrong.
    # 
    # Actually, the standard Strassen decomposition has signs. Let me use the correct one.
    # 
    # Standard Strassen for 2x2:
    # M1 = a00*(b11-b01)
    # M2 = (a00+a01)*b10
    # M3 = (a10+a11)*b01
    # M4 = a11*(b00-b10)
    # M5 = (a00+a10)*(b00+b11)
    # M6 = (a01-a11)*(b10+b11)
    # M7 = (a10-a00)*(b00+b01)
    # 
    # c00 = M1 + M4 + M5 - M7
    # c01 = M2 + M5
    # c10 = M3 + M7
    # c11 = M1 - M2 + M3 + M6
    # 
    # Each M_i is a product of two linear forms, so each contributes 2 rank-1 terms when expanded.
    # But the total rank is 7, not 14, because the linear combinations can be absorbed.
    # 
    # The actual rank-7 decomposition uses 7 vectors u_i, v_i, w_i such that:
    # T = sum_{i=1}^7 u_i ⊗ v_i ⊗ w_i
    # 
    # Let me look up the correct decomposition. One standard form is:
    # u1 = [1,0,0,0], v1 = [0,0,0,1], w1 = [1,0,0,1]
    # u2 = [0,0,0,1], v2 = [1,0,0,0], w2 = [0,0,1,0]
    # u3 = [0,1,0,0], v3 = [0,1,0,0], w3 = [0,1,0,0]
    # u4 = [1,0,1,0], v4 = [0,1,0,1], w4 = [1,0,1,0]
    # u5 = [0,1,1,0], v5 = [1,0,1,0], w5 = [0,1,1,0]
    # u6 = [1,1,0,0], v6 = [0,0,1,1], w6 = [1,1,0,0]
    # u7 = [1,0,0,0], v7 = [0,0,1,1], w7 = [0,0,1,1]
    # 
    # Let me verify c00:
    # From u1: a00*b11 -> c00: +1
    # From u4: (a00+a10)*(b01+b11) -> c00: +1
    # From u6: (a00+a01)*(b10+b11) -> c00: +1
    # From u7: a00*(b10+b11) -> c00: 0 (w7 has 0 at c00)
    # Wait, w7 = [0,0,1,1], so c00 gets 0 from u7.
    # Total for c00: 1+1+1 = 3? That's not right.
    # 
    # I think I need to use the actual Strassen signs. Let me just use the schoolbook for now
    # but try to get a lower rank by using Strassen where possible.
    
    # For simplicity and correctness, let me use a hybrid approach:
    # - Use Strassen for 2x2 blocks where possible
    # - Use schoolbook for the rest
    
    # Actually, let me just implement the correct Strassen decomposition.
    # The key insight is that each "multiplication" in Strassen is a product of two linear forms,
    # which can be represented as a single rank-1 term in the tensor decomposition if we allow
    # the vectors to have non-zero entries in multiple positions.
    
    # Let me define the 7 terms properly:
    strassen_terms = [
        # (u, v, w) where each is a 4-element vector
        ([1,0,0,0], [0,0,0,1], [1,0,0,1]),   # a00 * b11 -> c00, c11
        ([0,0,0,1], [1,0,0,0], [0,0,1,0]),   # a11 * b00 -> c10
        ([0,1,0,0], [0,1,0,0], [0,1,0,0]),   # a01 * b01 -> c01
        ([1,0,1,0], [0,1,0,1], [1,0,1,0]),   # (a00+a10)*(b01+b11) -> c00, c10
        ([0,1,1,0], [1,0,1,0], [0,1,1,0]),   # (a01+a11)*(b00+b10) -> c01, c11
        ([1,1,0,0], [0,0,1,1], [1,1,0,0]),   # (a00+a01)*(b10+b11) -> c00, c01
        ([1,0,0,0], [0,0,1,1], [0,0,1,1]),   # a00*(b10+b11) -> c10, c11
    ]
    
    # But this doesn't give the right signs. Let me just use schoolbook for correctness.
    
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
# EVOLVE-BLOCK-END
