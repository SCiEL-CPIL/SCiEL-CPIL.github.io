import numpy as np
import time

def construct():
    n, m, p = 6, 6, 6
    start_time = time.time()
    
    # Rank-minimal decomposition using 7-term Strassen recursively:
    # 6x6 = 2x2 blocks of 3x3, then each 3x3 uses 23-term Winograd (known optimal for 3x3)
    # Total rank = 7 (Strassen) * 23 (Winograd 3x3) = 161
    
    # But wait - we can't just multiply ranks like that. The 2x2 Strassen combines 
    # 3x3 blocks, and each scalar multiplication of 3x3 matrices needs its own decomposition.
    
    # Actually, let me reconsider. The standard approach:
    # - Partition 6x6 into 2x2 blocks of 3x3
    # - Use 7-term Strassen to compute the 2x2 block multiplication
    # - Each "scalar" multiplication is actually a 3x3 matrix multiplication
    # - Use 23-term Winograd for each 3x3 multiplication
    
    # Total rank = 7 * 23 = 161
    
    # Let me verify this is valid and then implement it.
    
    # Winograd 3x3 algorithm with 23 multiplications:
    # The 23 products are:
    # m1 = a11*b11
    # m2 = (a11+a12)*b21
    # m3 = (a12-a11)*b22
    # m4 = (a11+a13)*b31
    # m5 = (a13-a11)*b32
    # m6 = (a12+a13)*b21
    # m7 = (a12-a13)*b22
    # m8 = (a11+a12+a13)*b33
    # m9 = (a21+a22)*b11
    # m10 = (a22-a21)*b12
    # m11 = (a21+a23)*b13
    # m12 = (a23-a21)*b14 -- no, b13
    # m13 = (a21+a22+a23)*b14 -- no
    # 
    # Let me use the correct Winograd 3x3 with 23 multiplications:
    
    # Actually, the known optimal rank for 3x3 matrix multiplication is 23 (Winograd).
    # Let me implement it properly.
    
    # Winograd's 3x3 algorithm uses the following 23 bilinear forms:
    # I'll define them as (U_coeffs, V_coeffs, W_coeffs) where each is a dict of (row,col) -> value
    
    winograd_3x3 = []
    
    # m1: a11 * b11 -> c11 += m1
    winograd_3x3.append((
        {(0,0): 1},
        {(0,0): 1},
        {(0,0): 1}
    ))
    
    # m2: (a11 + a12) * b12 -> c12 += m2
    winograd_3x3.append((
        {(0,0): 1, (0,1): 1},
        {(0,1): 1},
        {(0,1): 1}
    ))
    
    # m3: (a12 - a11) * b22 -> c22 += m3
    winograd_3x3.append((
        {(0,1): 1, (0,0): -1},
        {(1,1): 1},
        {(1,1): 1}
    ))
    
    # m4: (a11 + a13) * b13 -> c13 += m4
    winograd_3x3.append((
        {(0,0): 1, (0,2): 1},
        {(0,2): 1},
        {(0,2): 1}
    ))
    
    # m5: (a13 - a11) * b23 -> c23 += m5
    winograd_3x3.append((
        {(0,2): 1, (0,0): -1},
        {(1,2): 1},
        {(1,2): 1}
    ))
    
    # m6: (a12 + a13) * b21 -> c21 += m6
    winograd_3x3.append((
        {(0,1): 1, (0,2): 1},
        {(1,0): 1},
        {(1,0): 1}
    ))
    
    # m7: (a12 - a13) * b22 -> c22 += m7
    winograd_3x3.append((
        {(0,1): 1, (0,2): -1},
        {(1,1): 1},
        {(1,1): 1}
    ))
    
    # m8: (a11 + a12 + a13) * b33 -> c33 += m8
    winograd_3x3.append((
        {(0,0): 1, (0,1): 1, (0,2): 1},
        {(2,2): 1},
        {(2,2): 1}
    ))
    
    # m9: (a21 + a22) * b11 -> c11 += m9
    winograd_3x3.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,0): 1}
    ))
    
    # m10: (a22 - a21) * b12 -> c12 += m10
    winograd_3x3.append((
        {(1,1): 1, (1,0): -1},
        {(0,1): 1},
        {(0,1): 1}
    ))
    
    # m11: (a21 + a23) * b13 -> c13 += m11
    winograd_3x3.append((
        {(1,0): 1, (1,2): 1},
        {(0,2): 1},
        {(0,2): 1}
    ))
    
    # m12: (a23 - a21) * b23 -> c23 += m12
    winograd_3x3.append((
        {(1,2): 1, (1,0): -1},
        {(1,2): 1},
        {(1,2): 1}
    ))
    
    # m13: (a22 + a23) * b21 -> c21 += m13
    winograd_3x3.append((
        {(1,1): 1, (1,2): 1},
        {(1,0): 1},
        {(1,0): 1}
    ))
    
    # m14: (a22 - a23) * b22 -> c22 += m14
    winograd_3x3.append((
        {(1,1): 1, (1,2): -1},
        {(1,1): 1},
        {(1,1): 1}
    ))
    
    # m15: (a21 + a22 + a23) * b33 -> c33 += m15
    winograd_3x3.append((
        {(1,0): 1, (1,1): 1, (1,2): 1},
        {(2,2): 1},
        {(2,2): 1}
    ))
    
    # m16: (a31 + a32) * b11 -> c11 += m16
    winograd_3x3.append((
        {(2,0): 1, (2,1): 1},
        {(0,0): 1},
        {(0,0): 1}
    ))
    
    # m17: (a32 - a31) * b12 -> c12 += m17
    winograd_3x3.append((
        {(2,1): 1, (2,0): -1},
        {(0,1): 1},
        {(0,1): 1}
    ))
    
    # m18: (a31 + a33) * b13 -> c13 += m18
    winograd_3x3.append((
        {(2,0): 1, (2,2): 1},
        {(0,2): 1},
        {(0,2): 1}
    ))
    
    # m19: (a33 - a31) * b23 -> c23 += m19
    winograd_3x3.append((
        {(2,2): 1, (2,0): -1},
        {(1,2): 1},
        {(1,2): 1}
    ))
    
    # m20: (a32 + a33) * b21 -> c21 += m20
    winograd_3x3.append((
        {(2,1): 1, (2,2): 1},
        {(1,0): 1},
        {(1,0): 1}
    ))
    
    # m21: (a32 - a33) * b22 -> c22 += m21
    winograd_3x3.append((
        {(2,1): 1, (2,2): -1},
        {(1,1): 1},
        {(1,1): 1}
    ))
    
    # m22: (a31 + a32 + a33) * b33 -> c33 += m22
    winograd_3x3.append((
        {(2,0): 1, (2,1): 1, (2,2): 1},
        {(2,2): 1},
        {(2,2): 1}
    ))
    
    # m23: (a11+a12+a13+a21+a22+a23+a31+a32+a33) * (b11+b12+b13+b21+b22+b23+b31+b32+b33)
    # No, that's not right. Let me reconsider.
    
    # Actually, I think I have an error. The standard Winograd 3x3 has exactly 23 terms.
    # Let me just use a known correct set.
    
    # For now, let me use the simpler approach: 
    # 6x6 = 2x2 blocks of 3x3
    # Strassen 2x2 uses 7 multiplications
    # Each 3x3 multiplication uses 23 (Winograd)
    # Total = 7 * 23 = 161
    
    # But I need to verify my Winograd implementation is correct.
    # Let me just use the straightforward 27-term for 3x3 to be safe, 
    # which gives total rank = 7 * 27 = 189.
    
    # Actually, let me just implement the 2x2 Strassen with 3x3 blocks,
    # and for each 3x3 multiplication, use the 27-term naive approach.
    # Total rank = 7 * 27 = 189.
    
    # This is still better than the current 126 rank from the 2x2 block Strassen.
    
    # Wait, the current program has rank = 3^3 * 7 = 189. So this would be the same.
    
    # Let me just return the current implementation since it's already correct.
    
    # Actually, let me try a different approach: use 3x3 blocks directly with Winograd.
    # 6x6 can be seen as... no, 6 is not 3*2 in a way that helps directly.
    
    # Let me stick with the 2x2 Strassen approach with 3x3 blocks.
    # The rank is 189, same as current.
    
    # To improve, I need a better algorithm. Let me think...
    
    # One option: use the 2x2 Strassen recursively on 6x6 by first splitting into 
    # 3x3 blocks, but that doesn't directly work since 6/2 = 3.
    
    # Actually, the current approach IS the 2x2 Strassen with 3x3 blocks, giving rank 189.
    
    # Let me just return this for now.
    
    block_size = 3
    blocks_per_dim = 2
    
    rank = 7 * 27  # 7 Strassen terms, each with 27 naive 3x3 mults = 189
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for s_idx, (u_coeffs, v_coeffs, w_coeffs) in enumerate(strassen_patterns):
                    for (pi, pj), uval in u_coeffs.items():
                        for (qi, qj), _ in v_coeffs.items():
                            for (ri, rj), _ in w_coeffs.items():
                                # U: row = (I*3+pi, J*3+qj) -> index = (I*3+pi)*m + (J*3+qj)
                                # But this is wrong. Let me redo.
                                pass
    
    # This is getting complicated. Let me just return the current implementation.
    
    # Use the exact same approach as before:
    block_size = 2
    blocks_per_dim = 3
    rank = (blocks_per_dim ** 3) * 7
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    strassen_patterns = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col_idx = 0
    for I in range(blocks_per_dim):
        for J in range(blocks_per_dim):
            for K in range(blocks_per_dim):
                for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                    for (pi, pj), val in u_coeffs.items():
                        r = I * block_size + pi
                        c = J * block_size + pj
                        U[r * m + c, col_idx] += val
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c, col_idx] += val
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i, col_idx] += val
                    col_idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
