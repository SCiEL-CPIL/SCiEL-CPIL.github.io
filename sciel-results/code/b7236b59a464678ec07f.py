import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
    return np.allclose(T_comp, T_target, atol=tol)

def _build_strassen_222():
    """
    Build a rank-7 decomposition for 2x2 matrix multiplication.
    Returns U (4,7), V (4,7), W (4,7) such that
    T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] equals 1 iff the 2x2x2 matmul tensor.
    
    Tensor convention: a = i*m + j, b = j*p + k, c = k*n + i
    For 2x2x2: n=2, m=2, p=2
    a = i*2 + j, b = j*2 + k, c = k*2 + i
    Target: T[i*2+j, j*2+k, k*2+i] = 1
    """
    n, m, p = 2, 2, 2
    A_dim, B_dim, C_dim = 4, 4, 4
    
    U = np.zeros((A_dim, 7))
    V = np.zeros((B_dim, 7))
    W = np.zeros((C_dim, 7))
    
    # Strassen 7 terms for 2x2:
    # M1 = (A11 - A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # a indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # b indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # c indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # Term 1: (A11 - A22)(B11 + B22)
    U[0, 0] = 1;  U[3, 0] = -1
    V[0, 0] = 1;  V[3, 0] = 1
    W[0, 0] = 1;  W[3, 0] = 1
    
    # Term 2: (A21 + A22) B11
    U[2, 1] = 1;  U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # Term 3: A11 (B12 - B22)
    U[0, 2] = 1
    V[1, 2] = 1;  V[3, 2] = -1
    W[1, 2] = 1
    
    # Term 4: A22 (B21 - B11)
    U[3, 3] = 1
    V[2, 3] = 1;  V[0, 3] = -1
    W[2, 3] = 1;  W[3, 3] = 1  # Wait, C22 gets M4? No.
    # Let me recheck: C22 = M1 - M2 + M3 + M6. C21 = M2 + M4.
    # So M4 contributes to C21 only? No: C21 = M2 + M4, C22 = M1 - M2 + M3 + M6
    # M4 contributes to C21 only? Let me re-derive.
    # Actually standard Strassen:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So M4 appears in C11 and C21.
    # W indices: c=0 is (0,0)=C11, c=1 is (0,1)=C12, c=2 is (1,0)=C21, c=3 is (1,1)=C22
    W[0, 3] = 1;  W[2, 3] = 1
    
    # Term 5: (A11 + A12) B22
    U[0, 4] = 1;  U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1;  W[1, 4] = 1
    
    # Term 6: (A21 - A11)(B11 + B12)
    U[2, 5] = 1;  U[0, 5] = -1
    V[0, 5] = 1;  V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 7: (A12 - A22)(B21 + B22)
    U[1, 6] = 1;  U[3, 6] = -1
    V[2, 6] = 1;  V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_trivial_nm_p(n, m, p):
    """Trivial rank n*m*p decomposition."""
    A_dim = n * m
    B_dim = m * p
    C_dim = p * n
    rank = n * m * p
    
    U = np.zeros((A_dim, rank))
    V = np.zeros((B_dim, rank))
    W = np.zeros((C_dim, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, r] = 1
                V[b, r] = 1
                W[c, r] = 1
                r += 1
    
    return U, V, W

def _build_2x4x7_rank50():
    """
    Build a rank-50 decomposition for 2x4x7 matrix multiplication
    using block Strassen decomposition.
    
    Strategy: Split the 4-dimensional index space into 2x2 blocks.
    A is 2x4, B is 4x7, C is 7x2.
    
    Partition the "j" index (0..3) into two groups: j0={0,1}, j1={2,3}
    Partition the "k" index (0..6) into groups: k0={0,1}, k1={2,3}, k2={4,5}, k3={6}
    
    For each (j_group, k_group) pair, we have a sub-problem.
    When j_group has size 2 and k_group has size 2, it's a 2x2x2 problem -> Strassen (rank 7)
    When j_group has size 2 and k_group has size 1, it's a 2x2x1 problem -> schoolbook (rank 2)
    When j_group has size 1 and k_group has size 2, it's a 2x1x2 problem -> schoolbook (rank 2)
    When j_group has size 1 and k_group has size 1, it's a 2x1x1 problem -> schoolbook (rank 1)
    
    Let's count:
    j groups: 2 groups of size 2
    k groups: 3 groups of size 2, 1 group of size 1
    
    Pairs:
    - (j_size2, k_size2): 2 * 3 = 6 pairs, each rank 7 -> 42
    - (j_size2, k_size1): 2 * 1 = 2 pairs, each rank 2 -> 4
    Total: 46
    
    Hmm, that's 46. Let me verify.
    Actually wait, I need to be more careful. The "n" dimension is 2 (rows of A, cols of C).
    The sub-problem for fixed j_group and k_group is:
    For each i in {0,1}: C[i, k] = sum over j in j_group of A[i,j] * B[j,k]
    
    This is a 2 x |j_group| x |k_group| sub-problem.
    If |j_group|=2 and |k_group|=2: 2x2x2 -> rank 7 (Strassen)
    If |j_group|=2 and |k_group|=1: 2x2x1 -> rank 2 (each of the 2 outputs needs one product)
    If |j_group|=1 and |k_group|=2: 2x1x2 -> rank 2
    If |j_group|=1 and |k_group|=1: 2x1x1 -> rank 1 (two products, one for each i)
    
    Wait, 2x2x1: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i=0,1, k fixed.
    That's 2 output entries, each is a sum of 2 products. Rank 2? No, rank 4?
    Actually the tensor is T[i, j, k] where i in {0,1}, j in {0,1}, k fixed (single k).
    T[i,j,k] = 1 for all i,j. That's a 2x2x1 tensor. Rank = 2.
    Because T = u * v^T where u is length 2 (for i) and v is length 2 (for j), with k dimension 1.
    So rank 2. Correct.
    
    2x1x2: similarly rank 2.
    2x1x1: T[i,j,k] = 1, rank 1? No, T is 2x1x1. It's just two ones. Rank 1? 
    T[i,0,0] = 1 for i=0,1. This is vec([1,1]) * [1] * [1]. Rank 1.
    
    So total: 6*7 + 2*2 = 42 + 4 = 46.
    
    But wait, I should double-check the k grouping. k goes 0..6, so 7 values.
    k0={0,1}, k1={2,3}, k2={4,5}, k3={6}. Sizes: 2,2,2,1.
    
    j goes 0..3, so 4 values. j0={0,1}, j1={2,3}. Sizes: 2,2.
    
    Pairs (j_g, k_g):
    (j0,k0): 2x2x2 -> 7
    (j0,k1): 2x2x2 -> 7
    (j0,k2): 2x2x2 -> 7
    (j0,k3): 2x2x1 -> 2
    (j1,k0): 2x2x2 -> 7
    (j1,k1): 2x2x2 -> 7
    (j1,k2): 2x2x2 -> 7
    (j1,k3): 2x2x1 -> 2
    
    Total: 6*7 + 2*2 = 42 + 4 = 46.
    
    Let me implement this.
    """
    n, m, p = 2, 4, 7
    A_dim = n * m  # 8
    B_dim = m * p  # 28
    C_dim = p * n  # 14
    
    # j groups
    j_groups = [(0, 1), (2, 3)]  # each is a tuple of j indices
    # k groups
    k_groups = [(0, 1), (2, 3), (4, 5), (6)]
    
    U = []
    V = []
    W = []
    
    for jg in j_groups:
        for kg in k_groups:
            j_indices = list(jg)
            k_indices = list(kg)
            j_size = len(j_indices)
            k_size = len(k_indices)
            
            if j_size == 2 and k_size == 2:
                # 2x2x2 Strassen sub-problem
                S_U, S_V, S_W = _build_strassen_222()
                # S_U has shape (4,7), S_V (4,7), S_W (4,7)
                # Map sub-indices to global indices
                for r in range(7):
                    u_col = np.zeros(A_dim)
                    v_col = np.zeros(B_dim)
                    w_col = np.zeros(C_dim)
                    for si, i in enumerate(range(2)):  # i = 0, 1
                        for sj, j in enumerate(j_indices):  # j in j_indices
                            a = i * m + j
                            u_col[a] = S_U[i * 2 + sj, r]
                    for sj, j in enumerate(j_indices):
                        for sk, k in enumerate(k_indices):
                            b = j * p + k
                            v_col[b] = S_V[sj * 2 + sk, r]
                    for sk, k in enumerate(k_indices):
                        for si, i in enumerate(range(2)):
                            c = k * n + i
                            w_col[c] = S_W[sk * 2 + si, r]
                    U.append(u_col)
                    V.append(v_col)
                    W.append(w_col)
            elif j_size == 2 and k_size == 1:
                # 2x2x1 schoolbook: rank 2
                k0 = k_indices[0]
                for sj, j in enumerate(j_indices):
                    u_col = np.zeros(A_dim)
                    v_col = np.zeros(B_dim)
                    w_col = np.zeros(C_dim)
                    for si, i in enumerate(range(2)):
                        a = i * m + j
                        u_col[a] = 1
                    b = j * p + k0
                    v_col[b] = 1
                    for si, i in enumerate(range(2)):
                        c = k0 * n + i
                        w_col[c] = 1
                    U.append(u_col)
                    V.append(v_col)
                    W.append(w_col)
            elif j_size == 1 and k_size == 2:
                # 2x1x2 schoolbook: rank 2
                j0 = j_indices[0]
                for sk, k in enumerate(k_indices):
                    u_col = np.zeros(A_dim)
                    v_col = np.zeros(B_dim)
                    w_col = np.zeros(C_dim)
                    for si, i in enumerate(range(2)):
                        a = i * m + j0
                        u_col[a] = 1
                    b = j0 * p + k
                    v_col[b] = 1
                    for si, i in enumerate(range(2)):
                        c = k * n + i
                        w_col[c] = 1
                    U.append(u_col)
                    V.append(v_col)
                    W.append(w_col)
            elif j_size == 1 and k_size == 1:
                # 2x1x1 schoolbook: rank 1
                j0 = j_indices[0]
                k0 = k_indices[0]
                u_col = np.zeros(A_dim)
                v_col = np.zeros(B_dim)
                w_col = np.zeros(C_dim)
                for si, i in enumerate(range(2)):
                    a = i * m + j0
                    u_col[a] = 1
                b = j0 * p + k0
                v_col[b] = 1
                for si, i in enumerate(range(2)):
                    c = k0 * n + i
                    w_col[c] = 1
                U.append(u_col)
                V.append(v_col)
                W.append(w_col)
    
    U = np.column_stack(U)
    V = np.column_stack(V)
    W = np.column_stack(W)
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = 135  # safety margin
    
    n, m, p = DIMENSIONS
    A_dim = n * m
    B_dim = m * p
    C_dim = p * n
    
    # Build rank-46 decomposition
    U, V, W = _build_2x4x7_rank50()
    
    if time.time() < deadline:
        if _verify_decomposition(U, V, W, n, m, p):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fall back to trivial
    U_triv, V_triv, W_triv = _build_trivial_nm_p(n, m, p)
    
    return {
        "U": U_triv.tolist(),
        "V": V_triv.tolist(),
        "W": W_triv.tolist()
    }

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Expected[i*m + j, j*p + k, k*n + i] = 1
    
    if np.allclose(T, Expected, atol=1e-9):
        print(f"Valid decomposition with rank {U.shape[1]}")
    else:
        print("Invalid decomposition")
        diff = np.abs(T - Expected)
        print(f"Max diff: {diff.max()}")
