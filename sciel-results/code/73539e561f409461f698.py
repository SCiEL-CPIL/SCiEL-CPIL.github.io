"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 140  # safety margin before 150s wall-clock kill

def _make_laderman_3x3():
    """
    Laderman's 3x3 matrix multiplication, rank 23.
    Convention: C[i,k] = sum_j A[i,j] B[j,k]
    U indexed by (i,j) -> i*3+j
    V indexed by (j,k) -> j*3+k
    W indexed by (k,i) -> k*3+i
    Returns (U, V, W) each of shape (9, 23), or None if verification fails.
    """
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Define 23 terms. Each term is (U_entries, V_entries, W_entries)
    # where entries are lists of (flat_index, coefficient).
    # A indices: A[i,j] -> i*3+j
    # B indices: B[j,k] -> j*3+k
    # W indices: C[k,i] -> k*3+i  (trace convention)
    
    terms = [
        # 9 "diagonal" direct terms
        # T0: A[0,0]*B[0,0] -> C[0,0]
        ([(0,1)], [(0,1)], [(0,1)]),
        # T1: A[1,1]*B[1,1] -> C[1,1]
        ([(4,1)], [(4,1)], [(4,1)]),
        # T2: A[2,2]*B[2,2] -> C[2,2]
        ([(8,1)], [(8,1)], [(8,1)]),
        # T3: A[0,1]*B[1,0] -> C[0,0]
        ([(1,1)], [(3,1)], [(0,1)]),
        # T4: A[1,0]*B[0,1] -> C[1,0]
        ([(3,1)], [(1,1)], [(1,1)]),
        # T5: A[0,2]*B[2,0] -> C[0,0]
        ([(2,1)], [(6,1)], [(0,1)]),
        # T6: A[2,0]*B[0,2] -> C[2,0]
        ([(6,1)], [(2,1)], [(2,1)]),
        # T7: A[1,2]*B[2,1] -> C[1,1]
        ([(5,1)], [(7,1)], [(4,1)]),
        # T8: A[2,1]*B[1,2] -> C[2,1]
        ([(7,1)], [(5,1)], [(5,1)]),
        # 14 composite terms
        # T9: (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        ([(0,1),(4,1)], [(0,1),(4,1)], [(1,1),(3,1)]),
        # T10: (A[0,0]+A[2,2])*(B[0,0]+B[2,2])
        ([(0,1),(8,1)], [(0,1),(8,1)], [(2,1),(6,1)]),
        # T11: (A[1,1]+A[2,2])*(B[1,1]+B[2,2])
        ([(4,1),(8,1)], [(4,1),(8,1)], [(5,1),(7,1)]),
        # T12: (A[0,0]+A[0,1]+A[1,0]+A[1,1])*(B[0,0]+B[0,1]+B[1,0]+B[1,1])
        ([(0,1),(1,1),(3,1),(4,1)], [(0,1),(1,1),(3,1),(4,1)], [(0,1),(1,1),(3,1),(4,1)]),
        # T13: (A[0,0]+A[0,2]+A[2,0]+A[2,2])*(B[0,0]+B[0,2]+B[2,0]+B[2,2])
        ([(0,1),(2,1),(6,1),(8,1)], [(0,1),(2,1),(6,1),(8,1)], [(0,1),(2,1),(6,1),(8,1)]),
        # T14: (A[1,1]+A[1,2]+A[2,1]+A[2,2])*(B[1,1]+B[1,2]+B[2,1]+B[2,2])
        ([(4,1),(5,1),(7,1),(8,1)], [(4,1),(5,1),(7,1),(8,1)], [(4,1),(5,1),(7,1),(8,1)]),
        # T15: (sum A except A[1,1])*(sum B except B[1,1])
        ([(0,1),(1,1),(2,1),(3,1),(5,1),(6,1),(7,1),(8,1)],
         [(0,1),(1,1),(2,1),(3,1),(5,1),(6,1),(7,1),(8,1)],
         [(0,1),(1,1),(2,1),(3,1),(5,1),(6,1),(7,1),(8,1)]),
        # T16: (A[0,0]-A[0,1]+A[1,0]-A[1,1])*(B[0,0]-B[0,1]+B[1,0]-B[1,1])
        ([(0,1),(1,-1),(3,1),(4,-1)], [(0,1),(1,-1),(3,1),(4,-1)], [(0,1),(1,-1),(3,1),(4,-1)]),
        # T17: (A[0,0]-A[0,2]+A[2,0]-A[2,2])*(B[0,0]-B[0,2]+B[2,0]-B[2,2])
        ([(0,1),(2,-1),(6,1),(8,-1)], [(0,1),(2,-1),(6,1),(8,-1)], [(0,1),(2,-1),(6,1),(8,-1)]),
        # T18: (A[1,1]-A[1,2]+A[2,1]-A[2,2])*(B[1,1]-B[1,2]+B[2,1]-B[2,2])
        ([(4,1),(5,-1),(7,1),(8,-1)], [(4,1),(5,-1),(7,1),(8,-1)], [(4,1),(5,-1),(7,1),(8,-1)]),
        # T19: (A[0,1]+A[1,1]-A[0,2]-A[1,2])*(B[1,0]+B[1,1]-B[2,0]-B[2,1])
        ([(1,1),(4,1),(2,-1),(5,-1)], [(3,1),(4,1),(6,-1),(7,-1)], [(1,1),(4,1)]),
        # T20: (A[1,0]+A[1,1]-A[2,0]-A[2,1])*(B[0,0]+B[0,1]-B[0,2]-B[1,2])
        ([(3,1),(4,1),(6,-1),(7,-1)], [(0,1),(1,1),(2,-1),(5,-1)], [(0,1),(1,1)]),
        # T21: (A[0,1]+A[0,2]-A[1,1]-A[1,2])*(B[2,0]+B[2,1]-B[1,0]-B[1,1])
        ([(1,1),(2,1),(4,-1),(5,-1)], [(6,1),(7,1),(3,-1),(4,-1)], [(2,1),(5,1)]),
        # T22: (A[1,0]+A[2,0]-A[1,1]-A[2,1])*(B[0,2]+B[1,2]-B[0,0]-B[1,0])
        ([(3,1),(6,1),(4,-1),(7,-1)], [(2,1),(5,1),(0,-1),(3,-1)], [(2,1),(5,1)]),
    ]
    
    for c, (u_e, v_e, w_e) in enumerate(terms):
        for idx, coeff in u_e:
            U[idx, c] += coeff
        for idx, coeff in v_e:
            V[idx, c] += coeff
        for idx, coeff in w_e:
            W[idx, c] += coeff
    
    # Verify: for all i,j,k, sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] should equal 1
    max_err = 0.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = np.dot(U[i*3+j, :], V[j*3+k, :] * W[k*3+i, :])
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        return None
    
    return U, V, W

def _make_2x2_schoolbook():
    """
    Schoolbook 2x2 matrix multiplication, rank 4.
    U indexed by (i,j)->i*2+j, V by (j,k)->j*2+k, W by (k,i)->k*2+i.
    Returns (U, V, W) each of shape (4, 4).
    """
    U = np.zeros((4, 4))
    V = np.zeros((4, 4))
    W = np.zeros((4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                c = i*4 + j*2 + k
                U[i*2+j, c] = 1
                V[j*2+k, c] = 1
                W[k*2+i, c] = 1
    return U, V, W

def _verify_6x6(U, V, W, n=6):
    """Verify the 6x6 decomposition. Returns max error."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    max_err = 0.0
    for i in range(n):
        for j in range(n):
            u_row = U_np[i*n+j, :]
            for k in range(n):
                v_row = V_np[j*n+k, :]
                w_row = W_np[k*n+i, :]
                s = np.dot(u_row, v_row * w_row)
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def construct():
    start_time = time.time()
    n, m, p = 6, 6, 6
    
    # Fallback: Schoolbook 6x6 (Rank 216)
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Hybrid: 2x2 Strassen (Outer) x 3x3 Schoolbook (Inner) (Rank 189)
    def _hybrid_strassen_schoolbook():
        block_size = 3
        dim = n
        rank_inner = block_size ** 3  # 27
        rank_outer = 7
        total_rank = rank_inner * rank_outer
        
        U = [[0] * total_rank for _ in range(dim * dim)]
        V = [[0] * total_rank for _ in range(dim * dim)]
        W = [[0] * total_rank for _ in range(dim * dim)]
        
        col = 0
        steps = [
            {'A': {(0,0):1,(1,1):1}, 'B': {(0,0):1,(1,1):1}, 'C': {(0,0):1,(1,1):1}},
            {'A': {(1,0):1,(1,1):1}, 'B': {(0,0):1}, 'C': {(1,0):1,(1,1):-1}},
            {'A': {(0,0):1}, 'B': {(0,1):1,(1,1):-1}, 'C': {(0,1):1,(1,1):1}},
            {'A': {(1,1):1}, 'B': {(1,0):1,(0,0):-1}, 'C': {(0,0):1,(1,0):1}},
            {'A': {(0,0):1,(0,1):1}, 'B': {(1,1):1}, 'C': {(0,0):-1,(0,1):1}},
            {'A': {(1,0):1,(0,0):-1}, 'B': {(0,0):1,(0,1):1}, 'C': {(1,1):1}},
            {'A': {(0,1):1,(1,1):-1}, 'B': {(1,0):1,(1,1):1}, 'C': {(0,0):1}}
        ]
        
        for step in steps:
            A_coeffs = step['A']
            B_coeffs = step['B']
            C_coeffs = step['C']
            for u in range(block_size):
                for v in range(block_size):
                    for w in range(block_size):
                        for (I, J), ca in A_coeffs.items():
                            if ca != 0:
                                i_global = I * block_size + u
                                j_global = J * block_size + v
                                U[i_global * dim + j_global][col] += ca
                        for (J, K), cb in B_coeffs.items():
                            if cb != 0:
                                j_global = J * block_size + v
                                k_global = K * block_size + w
                                V[j_global * dim + k_global][col] += cb
                        for (I, K), cc in C_coeffs.items():
                            if cc != 0:
                                i_global = I * block_size + u
                                k_global = K * block_size + w
                                W[k_global * dim + i_global][col] += cc
                        col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try Laderman 3x3 (outer) x 2x2 schoolbook (inner) = rank 92
    def _hybrid_laderman_schoolbook():
        if time.time() > _DEADLINE:
            return None
        laderman = _make_laderman_3x3()
        if laderman is None:
            return None
        
        U_l, V_l, W_l = laderman
        U_2, V_2, W_2 = _make_2x2_schoolbook()
        
        rank_l = U_l.shape[1]  # 23
        rank_2 = U_2.shape[1]  # 4
        total_rank = rank_l * rank_2  # 92
        
        U = np.zeros((36, total_rank))
        V = np.zeros((36, total_rank))
        W = np.zeros((36, total_rank))
        
        block_size = 2
        num_outer = 3
        
        for c_l in range(rank_l):
            for c_2 in range(rank_2):
                col = c_l * rank_2 + c_2
                
                for i_o in range(num_outer):
                    for j_o in range(num_outer):
                        ul = U_l[i_o * 3 + j_o, c_l]
                        if ul != 0:
                            for u in range(block_size):
                                for v in range(block_size):
                                    u2 = U_2[u * 2 + v, c_2]
                                    if u2 != 0:
                                        i_g = i_o * block_size + u
                                        j_g = j_o * block_size + v
                                        U[i_g * 6 + j_g, col] += ul * u2
                
                for j_o in range(num_outer):
                    for k_o in range(num_outer):
                        vl = V_l[j_o * 3 + k_o, c_l]
                        if vl != 0:
                            for v in range(block_size):
                                for w in range(block_size):
                                    v2 = V_2[v * 2 + w, c_2]
                                    if v2 != 0:
                                        j_g = j_o * block_size + v
                                        k_g = k_o * block_size + w
                                        V[j_g * 6 + k_g, col] += vl * v2
                
                for k_o in range(num_outer):
                    for i_o in range(num_outer):
                        wl = W_l[k_o * 3 + i_o, c_l]
                        if wl != 0:
                            for w in range(block_size):
                                for u in range(block_size):
                                    w2 = W_2[w * 2 + u, c_2]
                                    if w2 != 0:
                                        k_g = k_o * block_size + w
                                        i_g = i_o * block_size + u
                                        W[k_g * 6 + i_g, col] += wl * w2
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try Laderman hybrid (rank 92)
    try:
        res = _hybrid_laderman_schoolbook()
        if res is not None:
            max_err = _verify_6x6(res["U"], res["V"], res["W"])
            if max_err < 1e-10:
                return res
    except Exception:
        pass
    
    # Try Strassen hybrid (rank 189)
    try:
        res = _hybrid_strassen_schoolbook()
        max_err = _verify_6x6(res["U"], res["V"], res["W"])
        if max_err < 1e-10:
            return res
    except Exception:
        pass
    
    # Fallback to Schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
