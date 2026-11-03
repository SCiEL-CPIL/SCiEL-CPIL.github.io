"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 2 blocks of j (size 2)
    # Each block: 2 Strassen 2x2x2 (rank 7 each) + 1 schoolbook 2x2x3 (rank 12)
    # Rank per block = 7 + 7 + 12 = 26
    # Total rank = 2 * 26 = 52
    rank = 52
    
    U = [[0] * rank for _ in range(n * m)] # 8 x 52
    V = [[0] * rank for _ in range(m * p)] # 28 x 52
    W = [[0] * rank for _ in range(p * n)] # 14 x 52
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs):
        nonlocal col
        for u_idx, u_c in zip(u_indices, u_coeffs):
            U[u_idx][col] += u_c
        for v_idx, v_c in zip(v_indices, v_coeffs):
            V[v_idx][col] += v_c
        for w_idx, w_c in zip(w_indices, w_coeffs):
            W[w_idx][col] += w_c
        col += 1

    for b in range(2):
        # b is the block index for j (0 or 1)
        # j_local is 0 or 1
        # j_global = 2*b + j_local
        
        # Part 1: 2x2x2 Strassen for k in 0..1 (if b=0? No, k is global)
        # The split of k is independent of b.
        # k in 0..1 is handled by Strassen block r=0
        # k in 2..3 is handled by Strassen block r=1
        # k in 4..6 is handled by Schoolbook block r=2
        
        # Strassen Block r=0 (k=0,1)
        k_start = 0
        for i in range(2):
            for j_loc in range(2):
                j_global = 2 * b + j_loc
                u_idx = i * m + j_global
                v_idx_0 = j_global * p + k_start
                v_idx_1 = j_global * p + k_start + 1
                w_idx_00 = k_start * n + i
                w_idx_01 = (k_start + 1) * n + i
                w_idx_10 = k_start * n + (1 - i) # Wait, W index is k*n+i.
                # W index for C[i][k] is k*n + i.
                # So for C[0][0] -> 0*2+0=0. C[1][0] -> 0*2+1=1.
                # C[0][1] -> 1*2+0=2. C[1][1] -> 1*2+1=3.
                
                # We need to map local (i, j_loc) to global indices.
                # U row: i * 4 + (2*b + j_loc)
                # V row: (2*b + j_loc) * 7 + k
                # W row: k * 2 + i
                
                # Let's define local to global index maps
                def u_idx(i, j_loc): return i * m + (2 * b + j_loc)
                def v_idx(j_loc, k): return (2 * b + j_loc) * p + k
                def w_idx(k, i): return k * n + i

                # M1 = (A00 + A11)(B00 + B11)
                add_term(
                    [u_idx(0, 0), u_idx(1, 1)], [1, 1],
                    [v_idx(0, k_start), v_idx(1, k_start+1)], [1, 1],
                    [w_idx(k_start, 0), w_idx(k_start+1, 1)], [1, 1]
                )
                
                # M2 = (A10 + A11)B00
                add_term(
                    [u_idx(1, 0), u_idx(1, 1)], [1, 1],
                    [v_idx(0, k_start)], [1],
                    [w_idx(k_start, 1), w_idx(k_start+1, 1)], [1, -1]
                )
                
                # M3 = A00(B10 - B11)
                add_term(
                    [u_idx(0, 0)], [1],
                    [v_idx(1, k_start), v_idx(1, k_start+1)], [1, -1],
                    [w_idx(k_start+1, 0), w_idx(k_start+1, 1)], [1, 1]
                )
                
                # M4 = A11(B01 - B00)
                add_term(
                    [u_idx(1, 1)], [1],
                    [v_idx(0, k_start+1), v_idx(0, k_start)], [1, -1],
                    [w_idx(k_start, 0), w_idx(k_start, 1)], [-1, 1]
                )
                
                # M5 = (A00 + A01)B11
                add_term(
                    [u_idx(0, 0), u_idx(0, 1)], [1, 1],
                    [v_idx(1, k_start+1)], [1],
                    [w_idx(k_start, 0)], [1]
                )
                
                # M6 = (A10 - A00)(B00 + B01)
                add_term(
                    [u_idx(1, 0), u_idx(0, 0)], [1, -1],
                    [v_idx(0, k_start), v_idx(0, k_start+1)], [1, 1],
                    [w_idx(k_start+1, 1)], [1]
                )
                
                # M7 = (A01 - A11)(B10 + B11)
                add_term(
                    [u_idx(0, 1), u_idx(1, 1)], [1, -1],
                    [v_idx(1, k_start), v_idx(1, k_start+1)], [1, 1],
                    [w_idx(k_start+1, 0), w_idx(k_start, 0)], [1, -1]
                )

        # Strassen Block r=1 (k=2,3)
        k_start = 2
        for i in range(2):
            for j_loc in range(2):
                # Same structure, just different k_start
                def u_idx(i, j_loc): return i * m + (2 * b + j_loc)
                def v_idx(j_loc, k): return (2 * b + j_loc) * p + k
                def w_idx(k, i): return k * n + i

                # M1
                add_term(
                    [u_idx(0, 0), u_idx(1, 1)], [1, 1],
                    [v_idx(0, k_start), v_idx(1, k_start+1)], [1, 1],
                    [w_idx(k_start, 0), w_idx(k_start+1, 1)], [1, 1]
                )
                # M2
                add_term(
                    [u_idx(1, 0), u_idx(1, 1)], [1, 1],
                    [v_idx(0, k_start)], [1],
                    [w_idx(k_start, 1), w_idx(k_start+1, 1)], [1, -1]
                )
                # M3
                add_term(
                    [u_idx(0, 0)], [1],
                    [v_idx(1, k_start), v_idx(1, k_start+1)], [1, -1],
                    [w_idx(k_start+1, 0), w_idx(k_start+1, 1)], [1, 1]
                )
                # M4
                add_term(
                    [u_idx(1, 1)], [1],
                    [v_idx(0, k_start+1), v_idx(0, k_start)], [1, -1],
                    [w_idx(k_start, 0), w_idx(k_start, 1)], [-1, 1]
                )
                # M5
                add_term(
                    [u_idx(0, 0), u_idx(0, 1)], [1, 1],
                    [v_idx(1, k_start+1)], [1],
                    [w_idx(k_start, 0)], [1]
                )
                # M6
                add_term(
                    [u_idx(1, 0), u_idx(0, 0)], [1, -1],
                    [v_idx(0, k_start), v_idx(0, k_start+1)], [1, 1],
                    [w_idx(k_start+1, 1)], [1]
                )
                # M7
                add_term(
                    [u_idx(0, 1), u_idx(1, 1)], [1, -1],
                    [v_idx(1, k_start), v_idx(1, k_start+1)], [1, 1],
                    [w_idx(k_start+1, 0), w_idx(k_start, 0)], [1, -1]
                )

        # Schoolbook Block r=2 (k=4,5,6)
        for k in range(4, 7):
            for i in range(2):
                for j_loc in range(2):
                    # Term 1: A[i][j_loc] * B[j_loc][k] -> C[i][k]
                    # This is just 4 terms for each k.
                    # i=0, j_loc=0 -> A00 B0k -> C0k
                    # i=0, j_loc=1 -> A01 B1k -> C0k
                    # i=1, j_loc=0 -> A10 B0k -> C1k
                    # i=1, j_loc=1 -> A11 B1k -> C1k
                    
                    # We iterate i, j_loc.
                    # u_idx(i, j_loc)
                    # v_idx(j_loc, k)
                    # w_idx(k, i)
                    
                    add_term(
                        [i * m + (2 * b + j_loc)], [1],
                        [(2 * b + j_loc) * p + k], [1],
                        [k * n + i], [1]
                    )

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
