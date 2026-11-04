import numpy as np
import time

DEADLINE = 115.0
_start = time.time()

def construct():
    global _start, DEADLINE
    n, m, p = 3, 5, 7
    rank = 99
    
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    
    # Strassen block: I=[1,2], J=[1,2], K=[1,2] (indices 1,2,3)
    # Local dims 2x2x2
    # Strassen factors for 2x2x2:
    # M0 = (A11+A22)(B11+B22)
    # M1 = (A21+A22)(B11)
    # M2 = (A11)(B22-B12)
    # M3 = (A22)(B21-B11)
    # M4 = (A11+A12)(B22)
    # M5 = (A21-A11)(B11+B12)
    # M6 = (A12-A22)(B21+B22)
    # C11 = M0 - M3 + M5
    # C12 = M2 + M4
    # C21 = M1 + M3
    # C22 = M0 - M2 + M6
    
    # Global indices:
    # I block: 1,2 -> global i = 1, 2 (0-indexed)
    # J block: 1,2 -> global j = 1, 2 (0-indexed)
    # K block: 1,2 -> global k = 1, 2 (0-indexed)
    
    # Strassen factors mapping to local (i, j) for U, (j, k) for V, (k, i) for W
    # Local i, j, k are 0, 1. Global = local + 1.
    
    # M0: U: (0,0),(1,1); V: (0,0),(1,1); W: (0,0),(1,1)
    # M1: U: (1,0),(1,1); V: (0,0); W: (1,0)
    # M2: U: (0,0); V: (0,1),(1,1); W: (0,1),(1,1)
    # M3: U: (1,1); V: (1,0),(0,0); W: (1,0),(0,0)
    # M4: U: (0,0),(0,1); V: (1,1); W: (0,1)
    # M5: U: (1,0),(0,0); V: (0,0),(0,1); W: (1,0)
    # M6: U: (0,1),(1,1); V: (1,0),(1,1); W: (1,1)
    
    strassen_factors = [
        # (u_indices, v_indices, w_indices)
        ([(0,0), (1,1)], [(0,0), (1,1)], [(0,0), (1,1)]),
        ([(1,0), (1,1)], [(0,0)], [(1,0)]),
        ([(0,0)], [(0,1), (1,1)], [(0,1), (1,1)]),
        ([(1,1)], [(1,0), (0,0)], [(1,0), (0,0)]),
        ([(0,0), (0,1)], [(1,1)], [(0,1)]),
        ([(1,0), (0,0)], [(0,0), (0,1)], [(1,0)]),
        ([(0,1), (1,1)], [(1,0), (1,1)], [(1,1)]),
    ]
    
    # We need to assign signs to W to get the correct C.
    # C11 (local i=0, k=0) = M0 - M3 + M5
    # C12 (local i=0, k=1) = M2 + M4
    # C21 (local i=1, k=0) = M1 + M3
    # C22 (local i=1, k=1) = M0 - M2 + M6
    
    # W is indexed by [k*n + i].
    # For this block, local k is 0,1 -> global k=1,2.
    # Local i is 0,1 -> global i=1,2.
    
    # W contributions:
    # C11 (gi=1, gk=1): W[1*3+1]=W[4] gets M0(+), M3(-), M5(+)
    # C12 (gi=1, gk=2): W[2*3+1]=W[7] gets M2(+), M4(+)
    # C21 (gi=2, gk=1): W[1*3+2]=W[5] gets M1(+), M3(+)
    # C22 (gi=2, gk=2): W[2*3+2]=W[8] gets M0(+), M2(-), M6(+)
    
    for c, (ue, ve, we) in enumerate(strassen_factors):
        # U
        for (li, lj) in ue:
            gi = 1 + li
            gj = 1 + lj
            U[gi*m + gj][col + c] = 1
        # V
        for (lj, lk) in ve:
            gj = 1 + lj
            gk = 1 + lk
            V[gj*p + gk][col + c] = 1
    
    # W
    # M0
    W[4][col + 0] += 1  # C11
    W[8][col + 0] += 1  # C22
    # M1
    W[5][col + 1] += 1  # C21
    # M2
    W[7][col + 2] += 1  # C12
    W[8][col + 2] += -1 # C22
    # M3
    W[4][col + 3] += -1 # C11
    W[5][col + 3] += 1  # C21
    # M4
    W[7][col + 4] += 1  # C12
    # M5
    W[4][col + 5] += 1  # C11
    # M6
    W[8][col + 6] += 1  # C22
    
    col += 7
    
    # Schoolbook for remaining blocks:
    # Blocks are defined by partitions:
    # I: [0], [1,2]
    # J: [0], [1,2], [3,4]
    # K: [0], [1,2], [3,4,5,6]
    
    # Strassen handled: I[1], J[1], K[1] -> Global I=1,2; J=1,2; K=1,2
    
    # Remaining combinations:
    # 1. I[0], J[0], K[0]: 1x1x1
    # 2. I[0], J[0], K[1]: 1x1x2
    # 3. I[0], J[0], K[2]: 1x1x4
    # 4. I[0], J[1], K[0]: 1x2x1
    # 5. I[0], J[1], K[1]: 1x2x2
    # 6. I[0], J[1], K[2]: 1x2x4
    # 7. I[0], J[2], K[0]: 1x2x1
    # 8. I[0], J[2], K[1]: 1x2x2
    # 9. I[0], J[2], K[2]: 1x2x4
    # 10. I[1], J[0], K[0]: 2x1x1
    # 11. I[1], J[0], K[1]: 2x1x2
    # 12. I[1], J[0], K[2]: 2x1x4
    # 13. I[1], J[1], K[0]: 2x2x1
    # 14. I[1], J[2], K[0]: 2x2x1
    # 15. I[1], J[2], K[1]: 2x2x2 (Wait, Strassen was J[1], K[1] which is J=1,2 K=1,2. J[2] is 3,4. So this is new)
    # 16. I[1], J[2], K[2]: 2x2x4
    
    # Let's just iterate over all block combinations and skip the one we already did.
    # Or simpler: just loop over all i, j, k in the ranges, but that's not how block decomposition works.
    # Block decomposition implies we only use schoolbook for the non-Strassen parts.
    # The parts are:
    # I=0: J=0,1,2; K=0,1,2
    # I=1,2: J=0,2; K=0,1,2 (Since J=1, K=1 was Strassen)
    # I=1,2: J=1; K=0,2 (Since J=1, K=1 was Strassen)
    
    # Let's just iterate over the specific blocks.
    blocks = [
        (0, [0], [0], [0]), # I=0, J=0, K=0
        (0, [0], [0], [1,2]), # I=0, J=0, K=1,2
        (0, [0], [0], [3,4,5,6]), # I=0, J=0, K=3,4,5,6
        (0, [0], [1,2], [0]), # I=0, J=1,2, K=0
        (0, [0], [1,2], [1,2]), # I=0, J=1,2, K=1,2
        (0, [0], [1,2], [3,4,5,6]), # I=0, J=1,2, K=3,4,5,6
        (0, [0], [3,4], [0]), # I=0, J=3,4, K=0
        (0, [0], [3,4], [1,2]), # I=0, J=3,4, K=1,2
        (0, [0], [3,4], [3,4,5,6]), # I=0, J=3,4, K=3,4,5,6
        (1, [1,2], [0], [0]), # I=1,2, J=0, K=0
        (1, [1,2], [0], [1,2]), # I=1,2, J=0, K=1,2
        (1, [1,2], [0], [3,4,5,6]), # I=1,2, J=0, K=3,4,5,6
        (1, [1,2], [1,2], [0]), # I=1,2, J=1,2, K=0
        # (1, [1,2], [1,2], [1,2]) is Strassen
        (1, [1,2], [1,2], [3,4,5,6]), # I=1,2, J=1,2, K=3,4,5,6
        (1, [1,2], [3,4], [0]), # I=1,2, J=3,4, K=0
        (1, [1,2], [3,4], [1,2]), # I=1,2, J=3,4, K=1,2
        (1, [1,2], [3,4], [3,4,5,6]), # I=1,2, J=3,4, K=3,4,5,6
    ]
    
    for i_start, i_list, j_list, k_list in blocks:
        for gi in i_list:
            for gj in j_list:
                for gk in k_list:
                    U[gi*m + gj][col] = 1
                    V[gj*p + gk][col] = 1
                    W[gk*n + gi][col] = 1
                    col += 1
    
    # Verify
    U_np = U
    V_np = V
    W_np = W
    
    # Check rank
    if col != rank:
        # This should not happen, but if it does, fallback to schoolbook
        rank = col
        U = U[:, :rank]
        V = V[:, :rank]
        W = W[:, :rank]
    
    # Verification logic
    # sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] == (j == b)?
    # Wait, the tensor is T_{ijk} = A_{ij} B_{jk}.
    # Decomposition: T = sum_c U_c \otimes V_c \otimes W_c.
    # U_c is n x m. V_c is m x p. W_c is p x n.
    # U[i*m+j, c] is U_c[i,j].
    # V[j*p+k, c] is V_c[j,k].
    # W[k*n+i, c] is W_c[k,i].
    # We want sum_c W_c[k,i] * U_c[i,j] * V_c[j,k] = delta_{jk}? No.
    # The task says: "The output factor follows the official trace convention W[k*n+i,r]."
    # And "exactly decompose the matrix multiplication tensor".
    # The tensor for matrix multiplication C = A * B is T_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So T_{i,j,k} = A_{i,j} B_{j,k}.
    # The decomposition is T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
    # Note the indices on W are k, i.
    # So for each i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if j==j (always true for the term) ... wait.
    # The tensor value is 1 if the indices match the multiplication pattern.
    # Actually, the tensor is T_{i,j,k} which is 1 if we are computing C_{i,k} from A_{i,j} and B_{j,k}.
    # Since A_{i,j} and B_{j,k} are independent variables, the tensor entry T_{i,j,k} is 1 for all i,j,k?
    # No, the tensor is T_{i,j,k} = \delta_{?}
    # Let's look at the verification in the seed program.
    # `s += w[k * n + i][c] * u[i * m + a][c] * v[b * p + k][c]`
    # `if abs(s - (1.0 if a == b else 0.0)) > tol: return False`
    # Here `a` is the column of A (j) and `b` is the row of B (j').
    # So the condition is that the sum is 1 if a==b, 0 if a!=b.
    # This means U and V must be orthogonal in some sense?
    # Yes, this is the standard definition. The tensor is T_{i, j, j'} such that it picks out the contraction.
    # So my construction should satisfy this.
    
    for i in range(n):
        for a in range(m):
            for b in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += W[k*n + i][c] * U[i*m + a][c] * V[b*p + k][c]
                    if abs(s - (1.0 if a == b else 0.0)) > 1e-9:
                        # Fallback to schoolbook
                        rank = n*m*p
                        U = np.zeros((n*m, rank), dtype=int)
                        V = np.zeros((m*p, rank), dtype=int)
                        W = np.zeros((p*n, rank), dtype=int)
                        col = 0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    U[i*m+j][col] = 1
                                    V[j*p+k][col] = 1
                                    W[k*n+i][col] = 1
                                    col += 1
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
