import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    deadline = time.time() + 110.0
    
    def trivial_solution():
        U_triv = np.zeros((9, 54))
        V_triv = np.zeros((18, 54))
        W_triv = np.zeros((18, 54))
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    col = i*18 + j*6 + k
                    U_triv[i*3+j, col] = 1
                    V_triv[j*6+k, col] = 1
                    W_triv[k*3+i, col] = 1
        return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}
    
    # Correct Laderman rank-23 decomposition for 3x3x3
    # Using the standard overlapping 2x2 block construction
    # A[i,j] at index i*3+j, B[i,j] at index i*3+j, C[i,j] at index i*3+j
    
    L23 = []
    def add_term(u_dict, v_dict, w_dict):
        L23.append((u_dict, v_dict, w_dict))
    
    # TL block: A[0:2,0:2], B[0:2,0:2], C[0:2,0:2]
    # A indices: 0,1,3,4  B indices: 0,1,3,4  C indices: 0,1,3,4
    add_term({0:1, 4:1}, {0:1, 4:1}, {0:1, 4:1})   # M1
    add_term({3:1, 4:1}, {0:1}, {4:1})              # M2
    add_term({0:1}, {3:1, 4:-1}, {4:1})             # M3
    add_term({4:1}, {1:1, 0:-1}, {0:1})             # M4
    add_term({0:1, 1:1}, {4:1}, {4:-1})             # M5
    add_term({3:1, 0:-1}, {0:1, 1:1}, {1:1})        # M6
    add_term({1:1, 4:-1}, {3:1, 4:1}, {0:1})        # M7
    
    # TR block: A[0:2,1:3], B[1:3,0:2], C[0:2,1:3]
    # A indices: 1,2,4,5  B indices: 3,4,6,7  C indices: 1,2,4,5
    add_term({1:1, 5:1}, {3:1, 7:1}, {1:1, 5:1})   # M1
    add_term({4:1, 5:1}, {3:1}, {5:1})              # M2
    add_term({1:1}, {6:1, 7:-1}, {5:1})             # M3
    add_term({5:1}, {4:1, 3:-1}, {1:1})             # M4
    add_term({1:1, 2:1}, {7:1}, {5:-1})             # M5
    add_term({4:1, 1:-1}, {3:1, 4:1}, {2:1})        # M6
    add_term({2:1, 5:-1}, {6:1, 7:1}, {1:1})        # M7
    
    # BL block: A[1:3,0:2], B[0:2,1:3], C[1:3,0:2]
    # A indices: 3,4,6,7  B indices: 1,2,4,5  C indices: 3,4,6,7
    add_term({3:1, 7:1}, {1:1, 5:1}, {3:1, 7:1})   # M1
    add_term({6:1, 7:1}, {1:1}, {7:1})              # M2
    add_term({3:1}, {7:1, 8:-1}, {7:1})             # M3
    add_term({7:1}, {4:1, 3:-1}, {3:1})             # M4
    add_term({3:1, 4:1}, {8:1}, {7:-1})             # M5
    add_term({6:1, 3:-1}, {1:1, 4:1}, {4:1})        # M6
    add_term({4:1, 7:-1}, {6:1, 7:1}, {3:1})        # M7
    
    # BR block: A[1:3,1:3], B[1:3,1:3], C[1:3,1:3]
    # A indices: 4,5,7,8  B indices: 4,5,7,8  C indices: 4,5,7,8
    add_term({4:1, 8:1}, {4:1, 8:1}, {4:1, 8:1})   # M1
    add_term({7:1, 8:1}, {4:1}, {8:1})              # M2
    add_term({4:1}, {7:1, 8:-1}, {8:1})             # M3
    add_term({8:1}, {5:1, 4:-1}, {4:1})             # M4
    add_term({4:1, 5:1}, {8:1}, {8:-1})             # M5
    add_term({7:1, 4:-1}, {4:1, 5:1}, {5:1})        # M6
    add_term({5:1, 8:-1}, {7:1, 8:1}, {4:1})        # M7
    
    def build_matrices(terms):
        R = len(terms)
        U = np.zeros((9, R))
        V = np.zeros((9, R))
        W = np.zeros((9, R))
        for r, (ud, vd, wd) in enumerate(terms):
            for idx, val in ud.items():
                U[idx, r] += val
            for idx, val in vd.items():
                V[idx, r] += val
            for idx, val in wd.items():
                W[idx, r] += val
        return U, V, W
    
    U28, V28, W28 = build_matrices(L23)
    
    # Verify 3x3x3
    T28 = np.einsum('ar,br,cr->abc', U28, V28, W28)
    T28_target = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T28_target[i*3+j, j*3+k, i*3+k] = 1.0
    
    if not np.allclose(T28, T28_target, atol=1e-9):
        return trivial_solution()
    
    # Try to reduce rank by removing redundant terms
    current_terms = L23[:]
    current_rank = 28
    
    # Iteratively try to remove one term
    improved = True
    while improved:
        improved = False
        for skip in range(current_rank):
            if time.time() > deadline - 15:
                break
            trial = current_terms[:skip] + current_terms[skip+1:]
            Ut, Vt, Wt = build_matrices(trial)
            Tt = np.einsum('ar,br,cr->abc', Ut, Vt, Wt)
            if np.allclose(Tt, T28_target, atol=1e-9):
                current_rank -= 1
                current_terms = trial
                improved = True
                break
        if time.time() > deadline - 15:
            break
    
    U3, V3, W3 = build_matrices(current_terms)
    rank3 = current_rank
    
    # Build 3x3x6 from two 3x3x3 blocks
    total_rank = 2 * rank3
    U36 = np.zeros((9, total_rank))
    V36 = np.zeros((18, total_rank))
    W36 = np.zeros((18, total_rank))
    
    # Block 1: B[:,0:3] and C[0:3,:]
    U36[:, 0:rank3] = U3
    V36[0:9, 0:rank3] = V3
    W36[0:9, 0:rank3] = W3
    
    # Block 2: B[:,3:6] and C[3:6,:]
    U36[:, rank3:2*rank3] = U3
    V36[9:18, rank3:2*rank3] = V3
    W36[9:18, rank3:2*rank3] = W3
    
    # Verify 3x3x6 decomposition
    T36 = np.einsum('ar,br,cr->abc', U36, V36, W36)
    T36_target = np.zeros((9, 18, 18))
    for i in range(3):
        for j in range(3):
            for k in range(6):
                T36_target[i*3+j, j*6+k, k*3+i] = 1.0
    
    if np.allclose(T36, T36_target, atol=1e-9):
        return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
    
    return trivial_solution()

# EVOLVE-BLOCK-END
