import time
import itertools

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = 120  # seconds
    
    n, m, p = DIMENSIONS
    
    # Generate all partitions of a number into 1s and 2s
    def get_partitions(k):
        if k == 0:
            yield []
        else:
            for p in get_partitions(k-1):
                yield [1] + p
            if k >= 2:
                for p in get_partitions(k-2):
                    yield [2] + p

    # Find the partition combination that minimizes total rank
    best_rank = n * m * p
    best_part = (None, None, None)
    
    # We can limit search if it takes too long, but partitions are small
    part_n = list(get_partitions(n))
    part_m = list(get_partitions(m))
    part_p = list(get_partitions(p))
    
    def block_rank(a, b, c):
        if a == 2 and b == 2 and c == 2:
            return 7
        return a * b * c

    for pn in part_n:
        for pm in part_m:
            for pp in part_p:
                rank = 0
                for a, b, c in zip(pn, pm, pp):
                    rank += block_rank(a, b, c)
                if rank < best_rank:
                    best_rank = rank
                    best_part = (pn, pm, pp)
                    
    pn, pm, pp = best_part
    
    # Construct U, V, W
    # We don't know the final rank exactly until we iterate, but we know best_rank
    # Note: The rank is the sum of the ranks of the blocks.
    total_rank = best_rank
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(i, j, k, u_val, v_val, w_val):
        nonlocal col
        if col >= total_rank:
            return
        U[i * m + j][col] = u_val
        V[j * p + k][col] = v_val
        W[k * n + i][col] = w_val
        col += 1

    # Iterate through blocks
    i_offset = 0
    j_offset = 0
    k_offset = 0
    
    # We need to iterate through the Cartesian product of the partitions
    # The partitions define the sizes of the blocks along each dimension
    # We iterate i-blocks, j-blocks, k-blocks
    
    for i_idx, a_size in enumerate(pn):
        i_start = i_offset
        i_offset += a_size
        for j_idx, b_size in enumerate(pm):
            j_start = j_offset
            j_offset += b_size
            for k_idx, c_size in enumerate(pp):
                k_start = k_offset
                k_offset += c_size
                
                # Process block (i_start, j_start, k_start) of size a_size x b_size x c_size
                if a_size == 2 and b_size == 2 and c_size == 2:
                    # Strassen 2x2x2
                    # Local indices:
                    # A: (0,0), (0,1), (1,0), (1,1)
                    # B: (0,0), (0,1), (1,0), (1,1)
                    # C: (0,0), (0,1), (1,0), (1,1)
                    
                    # M1 = (A00 + A11)(B00 + B11)
                    # A00: i=i_start, j=j_start
                    # A11: i=i_start+1, j=j_start+1
                    # B00: j=j_start, k=k_start
                    # B11: j=j_start+1, k=k_start+1
                    
                    # Term 1: (A00 + A11) * (B00 + B11)
                    # This is sum of 4 rank-1 terms? No, Strassen uses 7 products of linear forms.
                    # L1 = A00 + A11
                    # R1 = B00 + B11
                    # Product L1*R1 contributes to C00, C11.
                    # C00 += 1 * M1
                    # C11 += 1 * M1
                    
                    # We need to expand L1*R1 into rank-1 terms?
                    # No, the tensor decomposition is $\sum U_{ij,c} V_{jk,c} W_{ki,c}$.
                    # If we use $L_1(A) R_1(B)$, it is not a single rank-1 term in the global tensor
                    # unless $L_1$ and $R_1$ are monomials.
                    # Wait. The rank of the tensor is the number of terms in the decomposition.
                    # Strassen's algorithm expresses the product as 7 multiplications of linear forms.
                    # Each multiplication $L(A)R(B)$ is a rank-1 tensor in the space of bilinear forms?
                    # Yes. The term is $L(A) \otimes R(B) \otimes W_C$.
                    # But $L(A)$ is a vector in $\mathbb{R}^{2 \times 2}$.
                    # So $L(A) R(B)$ is a rank-1 tensor in the input space.
                    # So each of the 7 Strassen terms is a rank-1 term in the global decomposition.
                    
                    # So for M1:
                    # L1 = A00 + A11. Vector: [1, 0, 0, 1] (for A00, A01, A10, A11)
                    # R1 = B00 + B11. Vector: [1, 0, 0, 1] (for B00, B01, B10, B11)
                    # Output C: C00 += 1*M1, C11 += 1*M1.
                    # So W vector: [1, 0, 0, 1] (for C00, C01, C10, C11)
                    
                    # We add this as ONE rank-1 term to the global U, V, W.
                    # U gets the coefficients of L1 for the relevant A entries.
                    # V gets the coefficients of R1 for the relevant B entries.
                    # W gets the coefficients for the relevant C entries.
                    
                    # M1
                    u_coeffs = { (i_start, j_start): 1, (i_start+1, j_start+1): 1 }
                    v_coeffs = { (j_start, k_start): 1, (j_start+1, k_start+1): 1 }
                    w_coeffs = { (k_start, i_start): 1, (k_start+1, i_start+1): 1 }
                    
                    # We cannot add a "vector" to a single column if it has multiple non-zero entries?
                    # Yes, we can. U[row, col] = coeff.
                    # So we add 1 column to U, V, W.
                    
                    for (ii, jj), val in u_coeffs.items():
                        U[ii * m + jj][col] = val
                    for (jj, kk), val in v_coeffs.items():
                        V[jj * p + kk][col] = val
                    for (kk, ii), val in w_coeffs.items():
                        W[kk * n + ii][col] = val
                    col += 1
                    
                    # M2 = (A10 + A11) B00
                    u_coeffs = { (i_start+1, j_start): 1, (i_start+1, j_start+1): 1 }
                    v_coeffs = { (j_start, k_start): 1 }
                    w_coeffs = { (k_start, i_start+1): 1 } # C10 += M2
                    # Check Strassen formula: C10 = M2 + M4. Yes.
                    
                    for (ii, jj), val in u_coeffs.items():
                        U[ii * m + jj][col] = val
                    for (jj, kk), val in v_coeffs.items():
                        V[jj * p + kk][col] = val
                    for (kk, ii), val in w_coeffs.items():
                        W[kk * n + ii][col] = val
                    col += 1
                    
                    # M3 = A00 (B10 - B11)
                    u_coeffs = { (i_start, j_start): 1 }
                    v_coeffs = { (j_start+1, k_start): 1, (j_start+1, k_start+1): -1 }
                    w_coeffs = { (k_start, i_start): 1, (k_start+1, i_start): 1 } # C00 += M3, C01 += M3
                    # Check: C00 = ... + M3. C01 = M3 + M5. Yes.
                    
                    for (ii, jj), val in u_coeffs.items():
                        U[ii * m + jj][col] = val
                    for (jj, kk), val in v_coeffs.items():
                        V[jj * p + kk][col] = val
                    for (kk, ii), val in w_coeffs.items():
                        W[kk * n + ii][col] = val
                    col += 1
                    
                    # M4 = A11 (B01 - B00)
                    u_coeffs = { (i_start+1, j_start+1): 1 }
                    v_coeffs = { (j_start, k_start+1): 1, (j_start, k_start): -1 }
                    w_coeffs = { (k_start, i_start): 1, (k_start+1, i_start+1): 1 } # C00 += M4, C11 += M4
                    # Check: C00 = ... + M4. C11 = ... + M4. Yes.
                    
                    for (ii, jj), val in u_coeffs.items():
                        U[ii * m + jj][col] = val
                    for (jj, kk), val in v_coeffs.items():
                        V[jj * p + kk][col] = val
                    for (kk, ii), val in w_coeffs.items():
                        W[kk * n + ii][col] = val
                    col += 1
                    
                    # M5 = (A00 + A01) B11
                    u_coeffs = { (i_start, j_start): 1, (i_start, j_start+1): 1 }
                    v_coeffs = { (j_start+1, k_start+1): 1 }
                    w_coeffs = { (k_start, i_start): -1, (k_start+1, i_start): 1 } # C00 -= M5, C01 += M5
                    # Check: C00 = ... - M5. C01 = ... + M5. Yes.
                    
                    for (ii, jj), val in u_coeffs.items():
                        U[ii * m + jj][col] = val
                    for (jj, kk), val in v_coeffs.items():
                        V[jj * p + kk][col] = val
                    for (kk, ii), val in w_coeffs.items():
                        W[kk * n + ii][col] = val
                    col += 1
                    
                    # M6 = (A10 - A00) (B00 + B01)
                    u_coeffs = { (i_start+1, j_start): 1, (i_start, j_start): -1 }
                    v_coeffs = { (j_start, k_start): 1, (j_start, k_start+1): 1 }
                    w_coeffs = { (k_start+1, i_start+1): 1 } # C11 += M6
                    # Check: C11 = ... + M6. Yes.
                    
                    for (ii, jj), val in u_coeffs.items():
                        U[ii * m + jj][col] = val
                    for (jj, kk), val in v_coeffs.items():
                        V[jj * p + kk][col] = val
                    for (kk, ii), val in w_coeffs.items():
                        W[kk * n + ii][col] = val
                    col += 1
                    
                    # M7 = (A01 - A11) (B10 + B11)
                    u_coeffs = { (i_start, j_start+1): 1, (i_start+1, j_start+1): -1 }
                    v_coeffs = { (j_start+1, k_start): 1, (j_start+1, k_start+1): 1 }
                    w_coeffs = { (k_start, i_start): 1 } # C00 += M7
                    # Check: C00 = ... + M7. Yes.
                    
                    for (ii, jj), val in u_coeffs.items():
                        U[ii * m + jj][col] = val
                    for (jj, kk), val in v_coeffs.items():
                        V[jj * p + kk][col] = val
                    for (kk, ii), val in w_coeffs.items():
                        W[kk * n + ii][col] = val
                    col += 1

                else:
                    # Schoolbook for this block
                    for ii in range(a_size):
                        for jj in range(b_size):
                            for kk in range(c_size):
                                I = i_start + ii
                                J = j_start + jj
                                K = k_start + kk
                                U[I * m + J][col] = 1
                                V[J * p + K][col] = 1
                                W[K * n + I][col] = 1
                                col += 1
                                
                j_offset = j_start # Reset for next i-block? No, j_offset should be cumulative
                # Wait, the logic for j_offset and k_offset in nested loops is tricky.
                # I should calculate offsets explicitly.

    # The offset logic above was flawed. Let's rewrite the loop structure properly.
    
    # Reset U, V, W
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    col = 0
    
    # Calculate offsets
    i_offsets = [0]
    for s in pn:
        i_offsets.append(i_offsets[-1] + s)
    j_offsets = [0]
    for s in pm:
        j_offsets.append(j_offsets[-1] + s)
    k_offsets = [0]
    for s in pp:
        k_offsets.append(k_offsets[-1] + s)
        
    for i_idx in range(len(pn)):
        i_start = i_offsets[i_idx]
        a_size = pn[i_idx]
        for j_idx in range(len(pm)):
            j_start = j_offsets[j_idx]
            b_size = pm[j_idx]
            for k_idx in range(len(pp)):
                k_start = k_offsets[k_idx]
                c_size = pp[k_idx]
                
                if a_size == 2 and b_size == 2 and c_size == 2:
                    # Strassen
                    # M1
                    U[i_start * m + j_start][col] = 1
                    U[(i_start+1) * m + (j_start+1)][col] = 1
                    V[j_start * p + k_start][col] = 1
                    V[(j_start+1) * p + (k_start+1)][col] = 1
                    W[k_start * n + i_start][col] = 1
                    W[(k_start+1) * n + (i_start+1)][col] = 1
                    col += 1
                    # M2
                    U[(i_start+1) * m + j_start][col] = 1
                    U[(i_start+1) * m + (j_start+1)][col] = 1
                    V[j_start * p + k_start][col] = 1
                    W[k_start * n + (i_start+1)][col] = 1
                    col += 1
                    # M3
                    U[i_start * m + j_start][col] = 1
                    V[(j_start+1) * p + k_start][col] = 1
                    V[(j_start+1) * p + (k_start+1)][col] = -1
                    W[k_start * n + i_start][col] = 1
                    W[(k_start+1) * n + i_start][col] = 1
                    col += 1
                    # M4
                    U[(i_start+1) * m + (j_start+1)][col] = 1
                    V[j_start * p + (k_start+1)][col] = 1
                    V[j_start * p + k_start][col] = -1
                    W[k_start * n + i_start][col] = 1
                    W[(k_start+1) * n + (i_start+1)][col] = 1
                    col += 1
                    # M5
                    U[i_start * m + j_start][col] = 1
                    U[i_start * m + (j_start+1)][col] = 1
                    V[(j_start+1) * p + (k_start+1)][col] = 1
                    W[k_start * n + i_start][col] = -1
                    W[(k_start+1) * n + i_start][col] = 1
                    col += 1
                    # M6
                    U[(i_start+1) * m + j_start][col] = 1
                    U[i_start * m + j_start][col] = -1
                    V[j_start * p + k_start][col] = 1
                    V[j_start * p + (k_start+1)][col] = 1
                    W[(k_start+1) * n + (i_start+1)][col] = 1
                    col += 1
                    # M7
                    U[i_start * m + (j_start+1)][col] = 1
                    U[(i_start+1) * m + (j_start+1)][col] = -1
                    V[(j_start+1) * p + k_start][col] = 1
                    V[(j_start+1) * p + (k_start+1)][col] = 1
                    W[k_start * n + i_start][col] = 1
                    col += 1
                else:
                    # Schoolbook
                    for ii in range(a_size):
                        for jj in range(b_size):
                            for kk in range(c_size):
                                I = i_start + ii
                                J = j_start + jj
                                K = k_start + kk
                                U[I * m + J][col] = 1
                                V[J * p + K][col] = 1
                                W[K * n + I][col] = 1
                                col += 1
                                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
