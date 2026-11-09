"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split m into two groups of 2
    # Split p into four groups of 2
    # For each group pair, use Strassen's algorithm for 2x2x2
    
    # Strassen coefficients for 2x2x2
    # Indices i, j, k in {0, 1}
    # T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}
    # Strassen decomposition has 7 terms
    
    # We will build U, V, W for the whole tensor
    # Total rank = 2 * 4 * 7 = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2 matrices
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b01 - b11)
    # M4 = a11(b10 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # We need to map these to U, V, W entries.
    # Each Mx corresponds to a rank-1 term in the decomposition of the 2x2x2 tensor.
    # The term is u_x \otimes v_x \otimes w_x.
    # u_x is a vector in R^{2x2}, v_x in R^{2x2}, w_x in R^{2x2}.
    # The contribution to C is sum_x (u_x \odot A) (v_x \odot B) w_x^T?
    # No, the decomposition is T = sum_x u_x \otimes v_x \otimes w_x.
    # This means C_{i,k} = sum_j A_{i,j} B_{j,k} is computed by:
    # C_{i,k} = sum_x (sum_j A_{i,j} u_{x, i,j}) (sum_j v_{x, j,k} B_{j,k}) w_{x, k,i}
    # Wait, the standard form is:
    # C = sum_x (A * u_x) (B * v_x) w_x^T?
    # No, the bilinear map is:
    # C_{i,k} = sum_x L_x(A) R_x(B) W_{x, k,i}
    # where L_x(A) = sum_{i,j} A_{i,j} u_{x, i,j}
    # R_x(B) = sum_{j,k} B_{j,k} v_{x, j,k}
    
    # For Strassen, the terms are linear combinations of A and B entries.
    # M1 = (a00 + a11)(b00 + b11)
    # This corresponds to u1 having 1 at (0,0) and (1,1), v1 having 1 at (0,0) and (1,1).
    # w1 has 1 at (0,0) and (1,1)?
    # Let's check c00 = M1 + ...
    # The output C is a 2x2 matrix.
    # The term M1 contributes to c00 and c11.
    # Specifically, M1 is added to c00 and c11.
    # So w1 should have 1 at (0,0) and (1,1).
    
    # Let's define the 7 vectors for u, v, w for 2x2 case.
    # u1 = [1, 0, 0, 1] (a00 + a11)
    # v1 = [1, 0, 0, 1] (b00 + b11)
    # w1 = [1, 0, 0, 1] (c00 + c11)
    
    # u2 = [0, 0, 1, 1] (a10 + a11)
    # v2 = [1, 0, 0, 0] (b00)
    # w2 = [0, 0, 1, 0] (c10)
    
    # u3 = [1, 0, 0, 0] (a00)
    # v3 = [0, 1, 0, -1] (b01 - b11)
    # w3 = [0, 1, 0, 0] (c01)
    
    # u4 = [0, 0, 0, 1] (a11)
    # v4 = [0, 0, 1, -1] (b10 - b00)
    # w4 = [0, 0, 1, 0] (c10)
    
    # u5 = [1, 1, 0, 0] (a00 + a01)
    # v5 = [0, 0, 0, 1] (b11)
    # w5 = [0, 1, 0, 0] (c01)
    
    # u6 = [0, 0, 1, -1] (a10 - a00)
    # v6 = [1, 1, 0, 0] (b00 + b01)
    # w6 = [0, 0, 0, 1] (c11)
    
    # u7 = [0, 1, 0, -1] (a01 - a11)
    # v7 = [0, 0, 1, 1] (b10 + b11)
    # w7 = [1, 0, 0, 0] (c00)
    
    # Let's verify c00 = M1 + M4 - M5 + M7
    # w1[0,0] = 1, w4[0,0] = 0, w5[0,0] = 0, w7[0,0] = 1.
    # Sum = 1 + 0 - 0 + 1 = 2?
    # Wait, M4 is a11(b10-b00).
    # w4 should contribute to c10?
    # c10 = M2 + M4.
    # w2[1,0] = 1, w4[1,0] = 1. Sum = 2?
    # No, the coefficients in w are not just 1.
    # The decomposition is T = sum u_x \otimes v_x \otimes w_x.
    # This means C_{i,k} = sum_x (u_x \cdot A) (v_x \cdot B) w_{x, k,i}.
    # For M1: u1 \cdot A = a00+a11, v1 \cdot B = b00+b11.
    # Contribution to C is (a00+a11)(b00+b11) * w1.
    # We want this to contribute to c00 and c11.
    # So w1 should be [1, 0, 0, 1] (flattened).
    
    # Let's check the sum for c00.
    # Terms: M1, M4, M5, M7.
    # M1: w1[0,0] = 1.
    # M4: w4[0,0] = 0.
    # M5: w5[0,0] = 0.
    # M7: w7[0,0] = 1.
    # Sum = 1 + 0 - 0 + 1 = 2.
    # But c00 should be a00b00 + a01b10.
    # M1 = (a00+a11)(b00+b11) = a00b00 + a00b11 + a11b00 + a11b11.
    # M7 = (a01-a11)(b10+b11) = a01b10 + a01b11 - a11b10 - a11b11.
    # M4 = a11(b10-b00) = a11b10 - a11b00.
    # M5 = (a00+a01)b11 = a00b11 + a01b11.
    
    # c00 = M1 + M4 - M5 + M7
    # = (a00b00 + a00b11 + a11b00 + a11b11) + (a11b10 - a11b00) - (a00b11 + a01b11) + (a01b10 + a01b11 - a11b10 - a11b11)
    # = a00b00 + (a00b11 - a00b11) + (a11b00 - a11b00) + (a11b11 - a11b11) + (a11b10 - a11b10) + (-a01b11 + a01b11) + a01b10
    # = a00b00 + a01b10.
    # Correct.
    
    # So the w vectors are:
    # w1 = [1, 0, 0, 1]
    # w2 = [0, 0, 1, 0]
    # w3 = [0, 1, 0, 0]
    # w4 = [0, 0, 1, 0]
    # w5 = [0, 1, 0, 0]
    # w6 = [0, 0, 0, 1]
    # w7 = [1, 0, 0, 0]
    
    # Note that w2 and w4 are the same. w3 and w5 are the same.
    # This is fine.
    
    # Now we map this to the global indices.
    # Block (gj, gk) where gj in {0,1}, gk in {0,1,2,3}.
    # Local i in {0,1}, j' in {0,1}, k' in {0,1}.
    # Global i = i.
    # Global j = 2*gj + j'.
    # Global k = 2*gk + k'.
    
    # U index: i * m + j = i * 4 + (2*gj + j').
    # V index: j * p + k = (2*gj + j') * 8 + (2*gk + k').
    # W index: k * n + i = (2*gk + k') * 2 + i.
    
    # For each block, we add the 7 terms.
    # Total 56 terms.
    
    # Predefine the 7 vectors for u, v, w for 2x2 case.
    # u_vecs, v_vecs, w_vecs are lists of 7 vectors of length 4.
    
    u_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1]
    ]
    
    v_strassen = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    w_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    col = 0
    for gj in range(2):
        for gk in range(4):
            for r in range(7):
                # u_strassen[r] is [u00, u01, u10, u11]
                # v_strassen[r] is [v00, v01, v10, v11]
                # w_strassen[r] is [w00, w01, w10, w11]
                
                for li in range(2):
                    for lj in range(2):
                        # Local index lj corresponds to j'
                        # Global j = 2*gj + lj
                        # U index = li * 4 + (2*gj + lj)
                        u_idx = li * 4 + (2 * gj + lj)
                        U[u_idx][col] += u_strassen[r][li * 2 + lj]
                        
                    for lk in range(2):
                        # Local index lk corresponds to k'
                        # Global k = 2*gk + lk
                        # V index = (2*gj + lj) * 8 + (2*gk + lk)
                        # Wait, v_strassen[r] is indexed by [lj, lk]
                        # v_strassen[r][lj * 2 + lk]
                        # We need to add to V for all lj, lk
                        pass
                
                # Correct way:
                # Iterate over local j', k'
                for lj in range(2):
                    for lk in range(2):
                        # V index
                        v_idx = (2 * gj + lj) * 8 + (2 * gk + lk)
                        V[v_idx][col] += v_strassen[r][lj * 2 + lk]
                        
                # W index
                for lk in range(2):
                    for li in range(2):
                        # W index = (2*gk + lk) * 2 + li
                        w_idx = (2 * gk + lk) * 2 + li
                        W[w_idx][col] += w_strassen[r][lk * 2 + li]
                        
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
