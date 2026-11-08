import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    
    # We use Laderman's algorithm for 3x3 matrix multiplication (rank 23)
    # and tile it 2x2 times for the 6x6 matrix.
    # Total rank = 2 * 2 * 23 = 184.
    
    block_size = 3
    blocks_per_dim = 2 # 6 / 3
    
    rank = (blocks_per_dim ** 3) * 23
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Laderman's algorithm coefficients for 3x3 matrices
    # Each entry is (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict {(i, j): val} for A
    # v_coeffs: dict {(j, k): val} for B
    # w_coeffs: dict {(i, k): val} for C
    # Note: The tensor decomposition is C_ik = sum_r U_ir V_jk W_ik_r ?
    # No, the standard form is C_ik = sum_r (sum_j U_{ij,r} A_{ij}) (sum_j V_{jk,r} B_{jk}) W_{ik,r}
    # Wait, the factors U, V, W in the prompt are such that:
    # C_{ik} = sum_r U_{(i*m+j), r} V_{(j*p+k), r} W_{(k*n+i), r} ?
    # No, the prompt says U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # And the decomposition is exact.
    # Usually, this means:
    # C_{ik} = sum_r ( sum_j U_{i*m+j, r} * A_{ij} ) * ( sum_j V_{j*p+k, r} * B_{jk} ) * W_{k*n+i, r} ?
    # No, that would be a product of three terms.
    # The tensor decomposition of matrix multiplication is:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # This is a 3-tensor. The rank decomposition is:
    # C_{ik} = sum_r U_{i, j, r} V_{j, k, r} W_{i, k, r} ?
    # No, the standard rank decomposition of the matrix multiplication tensor T_{ijk} is:
    # T_{ijk} = sum_r U_{ij, r} V_{jk, r} W_{ik, r}
    # Where T_{ijk} is the tensor such that C_{ik} = sum_j A_{ij} T_{ijk} B_{jk}? No.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # So the tensor is T_{ijk} = 1 if j is the middle index? No.
    # The tensor is T_{i, j, k} where the multiplication is C_{ik} = sum_j A_{ij} B_{jk}.
    # The decomposition is C_{ik} = sum_r (sum_j U_{ij, r} A_{ij}) (sum_j V_{jk, r} B_{jk}) W_{ik, r} ?
    # No, that's not right.
    # The correct form is:
    # C_{ik} = sum_r U_{i, r} V_{r, k} ? No.
    
    # Let's look at the Strassen code provided in the prompt.
    # U[r*m+c][col_idx] += val
    # V[r*p+c][col_idx] += val
    # W[k*n+i][col_idx] += val
    # This implies that for a given rank index `col_idx`, we have vectors u, v, w.
    # And the contribution to C is u * v * w?
    # No, the tensor decomposition of matrix multiplication is:
    # C_{ik} = sum_r (sum_j U_{ij, r} A_{ij}) * (sum_j V_{jk, r} B_{jk}) * W_{ik, r} ?
    # If so, then C_{ik} would be a sum of products of 3 linear forms.
    # But matrix multiplication is a sum of products of 2 linear forms (A_ij and B_jk).
    # So W_{ik, r} must be 1 or 0?
    # In Strassen, W is not just 0/1.
    
    # Let's re-read the Strassen code.
    # M0 = (A00+A11)(B00+B11)
    # U has A00=1, A11=1.
    # V has B00=1, B11=1.
    # W has C00=1, C11=1.
    # So the term is (A00+A11) * (B00+B11) * (C00 coefficient + C11 coefficient)?
    # No, the decomposition is:
    # C_{ik} = sum_r (sum_j U_{ij, r} A_{ij}) * (sum_j V_{jk, r} B_{jk}) * W_{ik, r}
    # This would mean C is a cubic form? No.
    
    # Actually, the standard tensor decomposition for matrix multiplication is:
    # C_{ik} = sum_r U_{i, j, r} A_{ij} V_{j, k, r} B_{jk} ?
    # No, the factors are U (n x m x rank), V (m x p x rank), W (p x n x rank).
    # And C_{ik} = sum_r W_{ik, r} * (sum_j U_{ij, r} A_{ij}) * (sum_j V_{jk, r} B_{jk}) ?
    # This is the form used in the Strassen code.
    # Let's verify with M0:
    # (A00+A11) * (B00+B11) * (1 for C00, 1 for C11).
    # So C00 gets (A00+A11)(B00+B11).
    # C11 gets (A00+A11)(B00+B11).
    # This matches Strassen M0.
    
    # So the format is:
    # C_{ik} = sum_r W_{k*n+i, r} * (sum_j U_{i*m+j, r} A_{ij}) * (sum_j V_{j*p+k, r} B_{jk})
    
    # I will use this format for Laderman.
    
    laderman_patterns = [
        # M1
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 1}),
        # M2
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): 1}),
        # M3
        ({(0,1): 1, (0,0): -1}, {(0,0): 1}, {(0,0): 1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(1,1): 1}),
        # M5
        ({(0,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 1}),
        # M6
        ({(1,1): 1, (0,0): -1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M7
        ({(1,1): 1, (1,2): 1}, {(0,2): 1}, {(1,0): 1, (1,1): 1}), # Guessing indices
        # ... This is too risky.
    ]
    
    # Fallback to Strassen 2x2 if Laderman is not implemented correctly.
    # But I must try Laderman.
    
    # Given the high risk of incorrect Laderman coefficients, and the directive to "KEEP this design",
    # and the fact that 189 is already a good score, I will stick to the Strassen 2x2 solution
    # but optimize the code structure for clarity and safety.
    # However, 184 is a better score.
    
    # Let's use the Strassen 2x2 solution as it is known to be correct.
    
    block_size = 2
    blocks_per_dim = 3 # 6 / 2
    
    rank = (blocks_per_dim ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
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
                        U[r * m + c][col_idx] += val
                    
                    for (pi, pj), val in v_coeffs.items():
                        r = J * block_size + pi
                        c = K * block_size + pj
                        V[r * p + c][col_idx] += val
                    
                    for (pi, pj), val in w_coeffs.items():
                        k = K * block_size + pj
                        i = I * block_size + pi
                        W[k * n + i][col_idx] += val
                    
                    col_idx += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
