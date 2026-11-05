"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 68 decomposition
    rank = 68
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, w_vals):
        nonlocal col
        for idx in u_indices:
            U[idx][col] = 1
        for idx in v_indices:
            V[idx][col] = 1
        for idx, val in zip(w_indices, w_vals):
            W[idx][col] = val
        col += 1

    # 1. A44 B44 (Rank 7, Strassen)
    # A44 indices: i in 0..3, j in 0..3 -> i*5+j
    # B44 indices: j in 0..3, k in 0..3 -> j*5+k
    # C44 indices: k in 0..3, i in 0..3 -> k*5+i
    
    # M1 = (A11+A22)(B11+B22)
    # A11: 0,1,5,6. A22: 12,13,17,18
    u_m1 = [0, 1, 5, 6, 12, 13, 17, 18]
    # B11: 0,1,5,6. B22: 12,13,17,18
    v_m1 = [0, 1, 5, 6, 12, 13, 17, 18]
    # C11 (0,1,5,6) + C22 (12,13,17,18)
    w_m1 = [0, 1, 5, 6, 12, 13, 17, 18]
    w_vals_m1 = [1, 1, 1, 1, 1, 1, 1, 1]
    add_term(u_m1, v_m1, w_m1, w_vals_m1)
    
    # M2 = (A21+A22)B11
    # A21: 10,11,15,16. A22: 12,13,17,18
    u_m2 = [10, 11, 15, 16, 12, 13, 17, 18]
    # B11: 0,1,5,6
    v_m2 = [0, 1, 5, 6]
    # C21 (10,11,15,16)
    w_m2 = [10, 11, 15, 16]
    w_vals_m2 = [1, 1, 1, 1]
    add_term(u_m2, v_m2, w_m2, w_vals_m2)
    
    # M3 = A11(B12-B22)
    # A11: 0,1,5,6
    u_m3 = [0, 1, 5, 6]
    # B12: 2,3,7,8. B22: 12,13,17,18
    v_m3 = [2, 3, 7, 8, 12, 13, 17, 18]
    v_vals_m3 = [1, 1, 1, 1, -1, -1, -1, -1]
    # Wait, add_term assumes V is 1. I need to handle values in V too?
    # The interface says "integer/half-integer entries".
    # I should modify add_term to accept values for U and V as well, or just set them directly.
    # Let's refactor add_term to accept lists of (index, value).
    
    # Re-define add_term
    def add_term_full(u_pairs, v_pairs, w_pairs):
        nonlocal col
        for idx, val in u_pairs:
            U[idx][col] = val
        for idx, val in v_pairs:
            V[idx][col] = val
        for idx, val in w_pairs:
            W[idx][col] = val
        col += 1

    # Reset col and U,V,W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # 1. A44 B44 (Rank 7)
    # M1
    u = [(0,1),(1,1),(5,1),(6,1),(12,1),(13,1),(17,1),(18,1)]
    v = [(0,1),(1,1),(5,1),(6,1),(12,1),(13,1),(17,1),(18,1)]
    w = [(0,1),(1,1),(5,1),(6,1),(12,1),(13,1),(17,1),(18,1)]
    add_term_full(u, v, w)
    
    # M2
    u = [(10,1),(11,1),(15,1),(16,1),(12,1),(13,1),(17,1),(18,1)]
    v = [(0,1),(1,1),(5,1),(6,1)]
    w = [(10,1),(11,1),(15,1),(16,1)]
    add_term_full(u, v, w)
    
    # M3
    u = [(0,1),(1,1),(5,1),(6,1)]
    v = [(2,1),(3,1),(7,1),(8,1),(12,-1),(13,-1),(17,-1),(18,-1)]
    w = [(2,1),(3,1),(7,1),(8,1)]
    add_term_full(u, v, w)
    
    # M4
    u = [(12,1),(13,1),(17,1),(18,1)]
    v = [(10,1),(11,1),(15,1),(16,1),(0,-1),(1,-1),(5,-1),(6,-1)]
    w = [(10,1),(11,1),(15,1),(16,1)]
    add_term_full(u, v, w)
    
    # M5
    u = [(0,1),(1,1),(5,1),(6,1),(2,1),(3,1),(7,1),(8,1)]
    v = [(12,1),(13,1),(17,1),(18,1)]
    w = [(12,-1),(13,-1),(17,-1),(18,-1)]
    add_term_full(u, v, w)
    
    # M6
    u = [(10,1),(11,1),(15,1),(16,1),(0,-1),(1,-1),(5,-1),(6,-1)]
    v = [(0,1),(1,1),(5,1),(6,1),(2,1),(3,1),(7,1),(8,1)]
    w = [(12,1),(13,1),(17,1),(18,1)]
    add_term_full(u, v, w)
    
    # M7
    u = [(2,1),(3,1),(7,1),(8,1),(12,-1),(13,-1),(17,-1),(18,-1)]
    v = [(10,1),(11,1),(15,1),(16,1),(12,1),(13,1),(17,1),(18,1)]
    w = [(0,1),(1,1),(5,1),(6,1)]
    add_term_full(u, v, w)

    # 2. A41 B14 (Rank 16)
    # A41: col 4 of A44. Indices: 4, 9, 14, 19
    # B14: row 4 of B44. Indices: 20, 21, 22, 23
    # C44: indices 0..18 (except those used?)
    # C_{ik} = A_{i,4} B_{4,k}
    for i in range(4):
        for k in range(4):
            u = [(i*5+4, 1)]
            v = [(4*5+k, 1)]
            w = [(k*5+i, 1)]
            add_term_full(u, v, w)

    # 3. A44 B41 (Rank 16)
    # A44: 0..18
    # B41: col 4 of B44. Indices: 4, 9, 14, 19
    # C41: col 4 of C44. Indices: 4, 9, 14, 19?
    # C_{i,4} = sum_j A_{i,j} B_{j,4}
    for i in range(4):
        for j in range(4):
            u = [(i*5+j, 1)]
            v = [(j*5+4, 1)]
            w = [(4*5+i, 1)]
            add_term_full(u, v, w)

    # 4. A41 B11 (Rank 4)
    # A41: 4, 9, 14, 19
    # B11: 24
    # C41: 4, 9, 14, 19
    for i in range(4):
        u = [(i*5+4, 1)]
        v = [(24, 1)]
        w = [(4*5+i, 1)]
        add_term_full(u, v, w)

    # 5. A14 B44 (Rank 16)
    # A14: row 4 of A44. Indices: 20, 21, 22, 23
    # B44: 0..18
    # C14: row 4 of C44. Indices: 20, 21, 22, 23
    for j in range(4):
        for k in range(4):
            u = [(4*5+j, 1)]
            v = [(j*5+k, 1)]
            w = [(k*5+4, 1)]
            add_term_full(u, v, w)

    # 6. A11 B14 (Rank 4)
    # A11: 24
    # B14: 20, 21, 22, 23
    # C14: 20, 21, 22, 23
    for k in range(4):
        u = [(24, 1)]
        v = [(4*5+k, 1)]
        w = [(k*5+4, 1)]
        add_term_full(u, v, w)

    # 7. A14 B41 (Rank 4)
    # A14: 20, 21, 22, 23
    # B41: 4, 9, 14, 19
    # C11: 24
    for j in range(4):
        u = [(4*5+j, 1)]
        v = [(j*5+4, 1)]
        w = [(24, 1)]
        add_term_full(u, v, w)

    # 8. A11 B11 (Rank 1)
    u = [(24, 1)]
    v = [(24, 1)]
    w = [(24, 1)]
    add_term_full(u, v, w)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
