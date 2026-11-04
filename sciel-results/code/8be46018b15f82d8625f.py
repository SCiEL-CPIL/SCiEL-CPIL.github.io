import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def _strassen_2x2():
    """
    Strassen rank-7 decomposition for 2x2 x 2x2 -> 2x2.
    Returns (U, V, W) each of shape (4, 7).
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)*e
    # M3 = a*(f-g)
    # M4 = d*(h-g)
    # M5 = (a+b)(e+f)
    # M6 = (c-d)(g+h)
    # M7 = (b-d)(e+h)
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M4
    # C21 = M2 + M5
    # C22 = M1 - M2 + M3 + M6
    
    # U rows: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # V rows: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # W rows: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[2, 1] = 1  # C01 += M2, C10 += M2
    
    # M3 = A00*(B10-B01)
    U[0, 2] = 1
    V[2, 2] = 1; V[1, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C01 += M3, C11 += M3
    
    # M4 = A11*(B11-B01)
    U[3, 3] = 1
    V[3, 3] = 1; V[1, 3] = -1
    W[0, 3] = -1; W[3, 3] = 1  # C00 -= M4, C11 += M4
    
    # M5 = (A00+A01)*(B00+B10)
    U[0, 4] = 1; U[1, 4] = 1
    V[0, 4] = 1; V[2, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1  # C00 += M5, C10 += M5
    
    # M6 = (A10-A11)*(B01+B11)
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[3, 5] = 1
    W[3, 5] = 1  # C11 += M6
    
    # M7 = (A01-A11)*(B00+B10)
    U[1, 6] = 1; U[3, 6] = -1
    V[0, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1  # C00 += M7
    
    return U, V, W


def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(actual - 1.0) > tol:
                    return False
    return True


def _schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_block_decomposition(n, m, p):
    """
    Block-structured decomposition using Strassen for 2x2 blocks.
    For n=2, m=3, p=6: partition j into 3 blocks of size 2, k into 3 blocks of size 2.
    Each (i, j-block, k-block) interaction is a 2x2 x 2x2 -> 2x2 multiplication.
    Total rank = 3 * 7 = 21... wait, let me reconsider.
    
    Actually for n=2, m=3, p=6:
    A is 2x3, B is 3x6.
    Partition B's rows (j=0,1,2) into blocks: j=0, j=1, j=2 (each size 1)
    Or partition into: {j=0,1} and {j=2}? That gives 2x2 and 2x1.
    
    Better: partition j into 3 groups of 2? No, m=3.
    Let me think differently.
    
    Partition the 3x6 B matrix into 3 blocks of 1x2 (each row of B is 1x2).
    Then C[i,k] = sum_j A[i,j] * B[j,k].
    
    Alternative: partition j into {0,1} and {2}.
    Block 1: A[:, 0:2] (2x2) * B[0:2, :] (2x6) -> 2x6
    Block 2: A[:, 2:3] (2x1) * B[2:3, :] (1x6) -> 2x6
    
    For Block 1: 2x2 * 2x6. Partition k into 3 groups of 2.
    Each sub-block: 2x2 * 2x2 -> 2x2, rank 7 each. Total for Block 1: 3*7 = 21.
    
    For Block 2: 2x1 * 1x6. This is just outer product, rank 6.
    Actually 2x1 * 1x6: C[i,k] = A[i,2] * B[2,k]. Rank = 2*1*6 = 12? No.
    C is 2x6, C[i,k] = A[i,2]*B[2,k]. This is a rank-1 matrix (outer product of A[:,2] and B[2,:]).
    Tensor decomposition: for each (i,k) pair, T[i, j=2, k] = 1.
    U[i*3+2, col] = 1, V[2*6+k, col] = 1, W[k*2+i, col] = 1.
    That's 12 terms. But we can do better: it's an outer product so rank = 2*6 = 12? 
    No wait, the tensor is T[i, j, k] where j is fixed at 2.
    T[i, 2, k] = 1 for all i,k. This is a 2x1x6 tensor with all ones.
    Rank of all-ones tensor of size a x b x c is min(a,b,c)... no.
    Actually rank of the all-ones 2x1x6 tensor: T[i,0,k] = 1.
    This factors as: sum_c u_c[i] * v_c[0] * w_c[k].
    We need u_c[i]*v_c[0]*w_c[k] to sum to 1 for all i,k.
    Simplest: rank 2. u_1=[1,0], u_2=[0,1], v_1=v_2=1, w_1=[1,1,1,1,1,1], w_2=[0,0,0,0,0,0]? No.
    Actually: T[i,k] = 1 = 1*1. So rank 1: u=[1,1], v=[1], w=[1,1,1,1,1,1].
    Check: u[i]*v[0]*w[k] = 1*1*1 = 1. Yes! Rank 1.
    
    So total rank = 21 + 1 = 22? That seems too good. Let me verify.
    
    Wait, I need to be more careful. Let me re-examine.
    
    For the 2x2 * 2x2 block (Strassen), the tensor is:
    T_block[i, j, k] = 1 if (i,j,k) corresponds to C[i,k] += A[i,j]*B[j,k]
    
    In the full tensor T[i*3+j, j*6+k, k*2+i]:
    For block (j in {0,1}, k in {0,1}):
    T[i*3+j, j*6+k, k*2+i] = 1 for i in {0,1}, j in {0,1}, k in {0,1}
    
    This is a 2x2x2 tensor (with specific index offsets) that equals the standard matrix mult tensor.
    Strassen gives rank 7 for this.
    
    Similarly for (j in {0,1}, k in {2,3}), (j in {0,1}, k in {4,5}).
    Each is a 2x2x2 tensor, rank 7.
    
    For (j=2, k in {0,1}), (j=2, k in {2,3}), (j=2, k in {4,5}):
    T[i*3+2, 2*6+k, k*2+i] = 1 for i in {0,1}, k in respective range.
    Each is a 2x1x2 tensor with all ones. Rank 1 each (as shown above).
    Total for j=2 blocks: 3 * 1 = 3.
    
    Grand total: 3*7 + 3*1 = 21 + 3 = 24.
    
    Hmm, that's better than 36 but let me double-check the rank-1 claim for 2x1x2 all-ones.
    T[i, 0, k] = 1 for i in {0,1}, k in {0,1}.
    Decomposition: sum_c u_c[i] * v_c[0] * w_c[k] = 1 for all i,k.
    With rank 1: u[0]=1, u[1]=1, v[0]=1, w[0]=1, w[1]=1.
    Then u[i]*v[0]*w[k] = 1*1*1 = 1. Correct!
    
    So total rank = 24.
    """
    n, m, p = 2, 3, 6
    # Partition j: {0,1} (size 2) and {2} (size 1)
    # Partition k: {0,1}, {2,3}, {4,5} (each size 2)
    
    total_rank = 0
    U = []
    V = []
    W = []
    
    for _ in range(n * m):
        U.append([])
    for _ in range(m * p):
        V.append([])
    for _ in range(p * n):
        W.append([])
    
    def add_term(u_row, v_row, w_row, u_val, v_val, w_val):
        nonlocal total_rank
        col = total_rank
        for r in range(n * m):
            U[r].append(0)
        for r in range(m * p):
            V[r].append(0)
        for r in range(p * n):
            W[r].append(0)
        U[u_row][col] = u_val
        V[v_row][col] = v_val
        W[w_row][col] = w_val
        total_rank += 1
    
    # Strassen for 2x2 x 2x2
    S_U, S_V, S_W = _strassen_2x2()
    
    # For each (j_block, k_block) where j_block is {0,1} and k_block is one of {0,1},{2,3},{4,5}
    k_blocks = [(0, 1), (2, 3), (4, 5)]
    
    for kb in k_blocks:
        k_start = kb[0]
        for r in range(7):
            # For each Strassen term r, add contributions for all (i, j) in {0,1}x{0,1}
            for i in range(2):
                for j in range(2):
                    u_val = S_U[i * 2 + j, r]
                    if u_val == 0:
                        continue
                    v_val = S_V[j * 2 + k_start, r] if False else None
                    # Actually V index is j*2 + k_within_block
                    # j is 0 or 1, k_within is 0 or 1
                    # V row = j * 2 + k_within
                    # But we need to map to global V index = j_global * 6 + k_global
                    # j_global = j (since j_block starts at 0)
                    # k_global = k_start + k_within
                    
                    # The Strassen V matrix has rows indexed by j_local*2 + k_local
                    # where j_local in {0,1}, k_local in {0,1}
                    # Global: j_global = j_local, k_global = k_start + k_local
                    # So V_global_row = j_global * 6 + k_global = j_local * 6 + k_start + k_local
                    
                    # Similarly W: W_local row = k_local * 2 + i_local
                    # Global: W_global_row = k_global * 2 + i_global = (k_start + k_local) * 2 + i
                    
                    pass
        
        # Let me redo this more carefully
        for r in range(7):
            for i_local in range(2):
                for j_local in range(2):
                    u_val = S_U[i_local * 2 + j_local, r]
                    if u_val == 0:
                        continue
                    # V local row = j_local * 2 + k_local, but we sum over k_local
                    # Actually in Strassen, each term r has specific non-zero entries in U, V, W
                    # U[i_local*2+j_local, r] * V[j_local*2+k_local, r] * W[k_local*2+i_local, r]
                    # We need to add a term for each (i_local, j_local, k_local) combination
                    # But that's not how it works. In the tensor decomposition, each rank-1 term
                    # contributes to ALL (i,j,k) simultaneously.
                    
                    # Let me think again. The Strassen decomposition says:
                    # sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = delta(i,j,k) for the 2x2x2 tensor
                    # where delta = 1 iff (i,j,k) is a valid (row, col, out) triple.
                    
                    # For our block, the tensor entries are:
                    # T[i*3+j, j*6+k, k*2+i] = 1 for i in {0,1}, j in {0,1}, k in {0,1}
                    # (with k offset by k_start)
                    
                    # So for each Strassen rank-1 term r:
                    # U_contrib[i*3+j] += S_U[i*2+j, r]
                    # V_contrib[j*6+(k_start+k)] += S_V[j*2+k, r]  
                    # W_contrib[(k_start+k)*2+i] += S_W[k*2+i, r]
                    
                    # But this adds 8 terms per Strassen term (2*2*2), which is wrong.
                    # The correct way: each Strassen term r is ONE rank-1 term in the 2x2x2 tensor.
                    # It contributes to the full tensor as:
                    # For the specific (i,j,k) where S_U[i*2+j,r]*S_V[j*2+k,r]*S_W[k*2+i,r] != 0
                    # No wait, that's not right either.
                    
                    # A rank-1 tensor is U[i,j] * V[j,k] * W[k,i] where U is 4x1, V is 4x1, W is 4x1.
                    # In the full tensor, this becomes:
                    # U_full[i*3+j] = S_U[i*2+j, r]
                    # V_full[j*6+k_start+k] = S_V[j*2+k, r]
                    # W_full[(k_start+k)*2+i] = S_W[k*2+i, r]
                    # for ALL i,j,k in {0,1}.
                    
                    # So each Strassen term r gives ONE rank-1 term in the full tensor,
                    # but with non-zero entries in multiple rows of U_full, V_full, W_full.
                    
                    # Let's just build it directly.
                    pass
    
    # Let me restart with a cleaner approach
    U = np.zeros((n * m, 0))
    V = np.zeros((m * p, 0))
    W = np.zeros((p * n, 0))
    
    S_U, S_V, S_W = _strassen_2x2()
    
    # Block 1: j in {0,1}, k in {0,1}
    # Block 2: j in {0,1}, k in {2,3}  
    # Block 3: j in {0,1}, k in {4,5}
    # Block 4: j=2, k in {0,1}
    # Block 5: j=2, k in {2,3}
    # Block 6: j=2, k in {4,5}
    
    for kb_start in [0, 2, 4]:
        # Strassen block: 2x2 x 2x2
        for r in range(7):
            u_vec = S_U[:, r]  # shape (4,)
            v_vec = S_V[:, r]  # shape (4,)
            w_vec = S_W[:, r]  # shape (4,)
            
            U_new = np.zeros(n * m)
            V_new = np.zeros(m * p)
            W_new = np.zeros(p * n)
            
            for i in range(2):
                for j in range(2):
                    U_new[i * m + j] = u_vec[i * 2 + j]
            
            for j in range(2):
                for k in range(2):
                    V_new[j * p + kb_start + k] = v_vec[j * 2 + k]
            
            for k in range(2):
                for i in range(2):
                    W_new[(kb_start + k) * n + i] = w_vec[k * 2 + i]
            
            U = np.hstack([U, U_new.reshape(-1, 1)])
            V = np.hstack([V, V_new.reshape(-1, 1)])
            W = np.hstack([W, W_new.reshape(-1, 1)])
    
    # Blocks for j=2: rank 1 each
    for kb_start in [0, 2, 4]:
        U_new = np.zeros(n * m)
        V_new = np.zeros(m * p)
        W_new = np.zeros(p * n)
        
        for i in range(2):
            U_new[i * m + 2] = 1.0
        
        for k in range(2):
            V_new[2 * p + kb_start + k] = 1.0
            W_new[(kb_start + k) * n + 0] = 1.0
            W_new[(kb_start + k) * n + 1] = 1.0
        
        U = np.hstack([U, U_new.reshape(-1, 1)])
        V = np.hstack([V, V_new.reshape(-1, 1)])
        W = np.hstack([W, W_new.reshape(-1, 1)])
    
    U_list = [[float(round(x, 10)) for x in row] for row in U]
    V_list = [[float(round(x, 10)) for x in row] for row in V]
    W_list = [[float(round(x, 10)) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Strategy 1: Block-structured Strassen decomposition (rank 24)
    res = _strassen_block_decomposition(n, m, p)
    if _verify_decomposition(res['U'], res['V'], res['W'], n, m, p, tol=1e-9):
        return res
    
    # Fallback: Schoolbook (rank 36)
    return _schoolbook_decomposition(n, m, p)
