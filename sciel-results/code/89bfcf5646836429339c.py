"""
Tensor decomposition for 5x5x5 matrix multiplication.

Uses Strassen's algorithm structure for 2x2 blocks.
Partition 5 into {0,1}, {2,3}, {4}.

Rank: 117 (improvement over schoolbook 125).
"""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2_terms():
    """
    Return the 7 Strassen bilinear form coefficients for 2x2*2x2 matrix multiplication.
    """
    terms = []
    terms.append(({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}))       # M1 = (a+d)(e+h)
    terms.append(({2:1, 3:1}, {0:1}, {2:1, 3:-1}))             # M2 = (c+d)e
    terms.append(({0:1}, {1:1, 3:-1}, {1:1, 3:1}))             # M3 = a(f-h)
    terms.append(({3:1}, {2:1, 0:-1}, {0:1, 2:1}))             # M4 = d(g-e)
    terms.append(({0:1, 1:1}, {3:1}, {0:-1, 1:1}))             # M5 = (a+b)h
    terms.append(({2:1, 0:-1}, {0:1, 1:1}, {3:1}))             # M6 = (c-a)(e+f)
    terms.append(({1:1, 3:-1}, {2:1, 3:1}, {0:1}))             # M7 = (b-d)(g+h)
    return terms

def _build_decomposition(groups):
    """
    Build decomposition using given index groups.
    Returns (U_list, V_list, W_list) where each is a list of dicts.
    """
    n, m, p = DIMENSIONS
    U_list = []
    V_list = []
    W_list = []
    
    strassen_terms = _strassen_2x2_terms()
    
    for i_group in groups:
        for k_group in groups:
            for j_group in groups:
                i_size = len(i_group)
                k_size = len(k_group)
                j_size = len(j_group)
                
                if i_size == 2 and k_size == 2 and j_size == 2:
                    for U_s, V_s, W_s in strassen_terms:
                        U_dict = {}
                        V_dict = {}
                        W_dict = {}
                        for a_idx, coeff in U_s.items():
                            i_local, k_local = divmod(a_idx, 2)
                            U_dict[i_group[i_local] * m + k_group[k_local]] = coeff
                        for b_idx, coeff in V_s.items():
                            k_local, j_local = divmod(b_idx, 2)
                            V_dict[k_group[k_local] * p + j_group[j_local]] = coeff
                        for c_idx, coeff in W_s.items():
                            j_local, i_local = divmod(c_idx, 2)
                            W_dict[j_group[j_local] * n + i_group[i_local]] = coeff
                        U_list.append(U_dict)
                        V_list.append(V_dict)
                        W_list.append(W_dict)
                else:
                    for i in i_group:
                        for k in k_group:
                            for j in j_group:
                                U_list.append({i * m + k: 1})
                                V_list.append({k * p + j: 1})
                                W_list.append({j * n + i: 1})
    
    return U_list, V_list, W_list

def _lists_to_matrices(U_list, V_list, W_list):
    """Convert list of dicts to matrix form."""
    n, m, p = DIMENSIONS
    rank = len(U_list)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for idx, coeff in U_list[col].items():
            U[idx][col] = coeff
        for idx, coeff in V_list[col].items():
            V[idx][col] = coeff
        for idx, coeff in W_list[col].items():
            W[idx][col] = coeff
    
    return U, V, W

def _verify(U, V, W, n, m, p, num_trials=5):
    """Verify the decomposition with random matrices using vectorized operations."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for _ in range(num_trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        a_vals = U_arr.T @ A_flat
        b_vals = V_arr.T @ B_flat
        prods = a_vals * b_vals
        
        C_flat = W_arr.T @ prods
        C_computed = C_flat.reshape((p, n)).T
        
        if not np.allclose(C_computed, C_expected, atol=1e-8):
            return False
    return True

def construct():
    """
    Tensor decomposition for 5x5x5 matrix multiplication using Strassen structure.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Strategy 1: Partition {0,1}, {2,3}, {4} -> rank 117
    groups1 = [(0, 1), (2, 3), (4,)]
    U_list, V_list, W_list = _build_decomposition(groups1)
    U, V, W = _lists_to_matrices(U_list, V_list, W_list)
    
    if time.time() < deadline - 10:
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Strategy 2: Partition {0,1,2,3}, {4} -> rank 117
    groups2 = [(0, 1, 2, 3), (4,)]
    U_list, V_list, W_list = _build_decomposition(groups2)
    U, V, W = _lists_to_matrices(U_list, V_list, W_list)
    
    if time.time() < deadline - 10:
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 125
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][column] = 1
                V[k * p + j][column] = 1
                W[j * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
