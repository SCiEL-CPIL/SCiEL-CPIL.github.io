import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    deadline = time.time() + 140  # Safety margin before 150s kill

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(U_entry, V_entry, W_entry):
        U_col = [0] * (n * m)
        for idx, val in U_entry.items():
            U_col[idx] = val
        V_col = [0] * (m * p)
        for idx, val in V_entry.items():
            V_col[idx] = val
        W_col = [0] * (p * n)
        for idx, val in W_entry.items():
            W_col[idx] = val
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # Use rank-6 decomposition for 2x3x3, applied to each of 3 column groups
    # For 2x3x3: A is 2x3, B is 3x3, C is 2x3
    # Rank-6 decomposition:
    # We use the fact that 2x3x3 can be done in 6 multiplications
    
    # Known rank-6 for 2x3x3:
    # Let A = [[a00,a01,a02],[a10,a11,a12]], B = [[b00,b01,b02],[b10,b11,b12],[b20,b21,b22]]
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    # C[0,1] = a00*b01 + a01*b11 + a02*b21
    # C[0,2] = a00*b02 + a01*b12 + a02*b22
    # C[1,0] = a10*b00 + a11*b10 + a12*b20
    # C[1,1] = a10*b01 + a11*b11 + a12*b21
    # C[1,2] = a10*b02 + a11*b12 + a12*b22
    
    # Rank-6 construction using Strassen-like ideas:
    # P1 = a00 * b00
    # P2 = a01 * b10  
    # P3 = a02 * b20
    # P4 = (a10+a00) * (b01-b00)  -> gives a10*b01 - a10*b00 + a00*b01 - a00*b00
    # P5 = (a11+a01) * (b11-b10)  -> gives a11*b11 - a11*b10 + a01*b11 - a01*b10
    # P6 = (a12+a02) * (b21-b20)  -> gives a12*b21 - a12*b20 + a02*b21 - a02*b20
    
    # Hmm, this doesn't quite work. Let me use a verified rank-6 decomposition.
    
    # Actually, for 2x3x3, the tensor rank is 6. One explicit decomposition:
    # Using the structure that 2x3x3 = (2x3) x 3, we can think of it as
    # computing 2 vectors of length 3 (rows of A) dotted with 3 columns of B.
    
    # Let me use a different approach: the 2x3x3 tensor has 18 non-zero entries
    # (all entries are 1). We need 6 rank-1 tensors.
    
    # A valid rank-6 decomposition for 2x3x3:
    # Term 1: U=[1,0,0,0,0,0], V=[1,0,0,0,0,0], W=[1,0,0,0,0,0]  -> (i=0,k=0,j=0)
    # This is getting complex. Let me just use schoolbook (rank 9) for 2x3x3 
    # and see if I can reduce.
    
    # Schoolbook for 2x3x3: 9 terms
    # For each column group g (0,1,2), columns j=3g, 3g+1, 3g+2
    for g in range(3):
        for i in range(n):
            for k in range(m):
                for j in range(3):
                    jj = g * 3 + j
                    add_term(
                        {i * m + k: 1},
                        {k * p + jj: 1},
                        {jj * n + i: 1}
                    )
    
    rank = len(U_cols)  # Should be 27

    # Verify
    def verify_decomposition(U_c, V_c, W_c):
        for i in range(n):
            for k2 in range(m):
                for j in range(p):
                    lhs = 0
                    for r in range(len(U_c)):
                        lhs += U_c[r][i * m + k2] * V_c[r][k2 * p + j] * W_c[r][j * n + i]
                    if abs(lhs - 1) > 1e-9:
                        return False
        return True

    if not verify_decomposition(U_cols, V_cols, W_cols):
        U = [[U_cols[r][i] for r in range(rank)] for i in range(n * m)]
        V = [[V_cols[r][j] for r in range(rank)] for j in range(m * p)]
        W = [[W_cols[r][k] for r in range(rank)] for k in range(p * n)]
        return {"U": U, "V": V, "W": W}

    # Now try to reduce rank using the linear algebra approach
    U_mat = np.array(U_cols, dtype=float).T
    V_mat = np.array(V_cols, dtype=float).T
    W_mat = np.array(W_cols, dtype=float).T

    max_removal_attempts = 30
    for attempt in range(max_removal_attempts):
        if time.time() > deadline - 10:
            break
        removed_any = False
        current_rank = len(U_cols)
        for r_remove in range(current_rank):
            if time.time() > deadline - 10:
                break
            equations_A = []
            equations_b = []
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        a = i * m + k
                        b = k * p + j
                        c = j * n + i
                        lhs = U_mat[a, r_remove] * V_mat[b, r_remove] * W_mat[c, r_remove]
                        if abs(lhs) > 1e-12:
                            row = []
                            for s in range(current_rank):
                                if s == r_remove:
                                    continue
                                val = U_mat[a, s] * V_mat[b, s] * W_mat[c, s]
                                row.append(val)
                            equations_A.append(row)
                            equations_b.append(lhs)
            if not equations_A:
                continue
            A_sys = np.array(equations_A)
            b_sys = np.array(equations_b)
            try:
                d, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                if len(residuals) > 0 and residuals[0] > 1e-8:
                    continue
                valid = True
                for i in range(n):
                    for k in range(m):
                        for j in range(p):
                            a = i * m + k
                            b = k * p + j
                            c = j * n + i
                            lhs = U_mat[a, r_remove] * V_mat[b, r_remove] * W_mat[c, r_remove]
                            rhs = 0
                            idx = 0
                            for s in range(current_rank):
                                if s == r_remove:
                                    continue
                                rhs += d[idx] * U_mat[a, s] * V_mat[b, s] * W_mat[c, s]
                                idx += 1
                            if abs(lhs - rhs) > 1e-7:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    U_cols.pop(r_remove)
                    V_cols.pop(r_remove)
                    W_cols.pop(r_remove)
                    U_mat = np.array(U_cols, dtype=float).T
                    V_mat = np.array(V_cols, dtype=float).T
                    W_mat = np.array(W_cols, dtype=float).T
                    removed_any = True
                    break
            except:
                continue
        if not removed_any:
            break

    rank = len(U_cols)
    U = [[U_cols[r][i] for r in range(rank)] for i in range(n * m)]
    V = [[V_cols[r][j] for r in range(rank)] for j in range(m * p)]
    W = [[W_cols[r][k] for r in range(rank)] for k in range(p * n)]
    return {"U": U, "V": V, "W": W}
