"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strassen decomposition for 2x2 matrix multiplication
    # For A = [[a,b],[c,d]], B = [[e,f],[g,h]]:
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Each Pi gives a rank-1 term in the 2x2 multiplication tensor
    # U factors (for A): which elements of A participate in each Pi
    # V factors (for B): which elements of B participate in each Pi
    # W factors (for C): which elements of C are affected by each Pi
    
    # 2x2 Strassen rank-1 terms: (U_2x2, V_2x2, W_2x2)
    # Indexing: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[i,k] -> i*2+k
    strassen_2x2 = [
        # Pi: (u_coeffs for A, v_coeffs for B, w_coeffs for C)
        # P1 = (A00+A11)(B00+B11)
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # P2 = (A10+A11)*B00
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        # P3 = A00*(B01-B11)
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
        # P4 = A11*(B10-B00)
        ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),
        # P5 = (A00+A01)*B11
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        # P6 = (A10-A00)*(B00+B01)
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        # P7 = (A01-A11)*(B10+B11)
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    
    # For 6x6 = 3x3 blocks of 2x2
    # Block (bi,bj) of output gets contributions from block (bi,bk) of A and (bk,bj) of B
    # for bk = 0,1,2
    # Within each 2x2 block, use Strassen
    
    rank = 9 * 7  # 9 block pairs * 7 Strassen terms = 63
    U = [[0] * rank for _ in range(n * m)]  # 36 x 63
    V = [[0] * rank for _ in range(m * p)]  # 36 x 63
    W = [[0] * rank for _ in range(p * n)]  # 36 x 63
    
    col = 0
    for bi in range(3):  # block row of A / block row of C
        for bk in range(3):  # block col of A / block row of B
            for bj in range(3):  # block col of B / block col of C
                for s in range(7):  # Strassen term
                    u_coeffs, v_coeffs, w_coeffs = strassen_2x2[s]
                    
                    # A block (bi, bk): elements at rows bi*2+li, cols bk*2+lk
                    # A[bi*2+li, bk*2+lk] for li,lk in {0,1}
                    # Index in A (36 elements): (bi*2+li) * 6 + (bk*2+lk)
                    for li in range(2):
                        for lk in range(2):
                            idx_a = li * 2 + lk  # local 2x2 index
                            global_row = bi * 2 + li
                            global_col = bk * 2 + lk
                            U[global_row * m + global_col][col] = u_coeffs[idx_a]
                    
                    # B block (bk, bj): elements at rows bk*2+lk, cols bj*2+lj
                    # B[bk*2+lk, bj*2+lj] for lk,lj in {0,1}
                    # Index in B (36 elements): (bk*2+lk) * 6 + (bj*2+lj)
                    for lk in range(2):
                        for lj in range(2):
                            idx_b = lk * 2 + lj  # local 2x2 index
                            global_row = bk * 2 + lk
                            global_col = bj * 2 + lj
                            V[global_row * p + global_col][col] = v_coeffs[idx_b]
                    
                    # C block (bi, bj): elements at rows bi*2+li, cols bj*2+lj
                    # C[bi*2+li, bj*2+lj] for li,lj in {0,1}
                    # Index in C (36 elements): (bi*2+li) * 6 + (bj*2+lj)
                    # But W uses trace convention: W[k*n+i, r] where C[i,k] = sum_j A[i,j]*B[j,k]
                    # So W index = k*n + i where i is row, k is col
                    for li in range(2):
                        for lj in range(2):
                            idx_c = li * 2 + lj  # local 2x2 index
                            global_row = bi * 2 + li
                            global_col = bj * 2 + lj
                            # W[k*n + i] where i=global_row, k=global_col
                            W[global_col * n + global_row][col] = w_coeffs[idx_c]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
