"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 5.0
import numpy as np
from itertools import product

def _check_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2(A, B):
    """
    Standard Strassen decomposition for 2x2 * 2x2.
    Returns list of 7 (U_row_indices, V_col_indices, W_col_indices, coeff) tuples
    where each represents a rank-1 contribution.
    
    Actually, let's return the three matrices directly for a 2x2x2 block.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices:
    U: (i, j) -> i*2 + j
    V: (j, k) -> j*2 + k
    W: (k, i) -> k*2 + i
    """
    # Strassen formulas:
    # M1 = (A11 - A22) * (B22 + B11)
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
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Column 0: M1
    # A11 - A22: U[0,0]=1, U[3,0]=-1
    # B22 + B11: V[3,0]=1, V[0,0]=1
    # C11: W[0,0]=1
    U[0, 0] = 1; U[3, 0] = -1
    V[3, 0] = 1; V[0, 0] = 1
    W[0, 0] = 1
    
    # Column 1: M2
    # A21 + A22: U[2,1]=1, U[3,1]=1
    # B11: V[0,1]=1
    # C21: W[2,1]=1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # Column 2: M3
    # A11: U[0,2]=1
    # B12 - B22: V[1,2]=1, V[3,2]=-1
    # C12: W[1,2]=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # Column 3: M4
    # A22: U[3,3]=1
    # B21 - B11: V[2,3]=1, V[0,3]=-1
    # C11: W[0,3]=1, C21: W[2,3]=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Column 4: M5
    # A11 + A12: U[0,4]=1, U[1,4]=1
    # B22: V[3,4]=1
    # C11: W[0,4]=-1, C12: W[1,4]=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Column 5: M6
    # A21 - A11: U[2,5]=1, U[0,5]=-1
    # B11 + B12: V[0,5]=1, V[1,5]=1
    # C22: W[3,5]=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Column 6: M7
    # A12 - A22: U[1,6]=1, U[3,6]=-1
    # B21 + B22: V[2,6]=1, V[3,6]=1
    # C11: W[0,6]=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_strassen_hybrid():
    """
    Construct a rank-18 decomposition using Strassen for the first 4 columns of B
    and schoolbook for the 5th column.
    
    n=2, m=2, p=5
    A is 2x2, B is 2x5, C is 2x5
    
    We split B into B[:, :4] and B[:, 4].
    C[:, :4] = A * B[:, :4]
    C[:, 4] = A * B[:, 4]
    
    For C[:, :4], we can view B[:, :4] as two 2x2 blocks: B[:, 0:2] and B[:, 2:4].
    Wait, Strassen multiplies 2x2 by 2x2.
    So C[:, 0:2] = A * B[:, 0:2] (2x2 * 2x2 -> 2x2)
    C[:, 2:4] = A * B[:, 2:4] (2x2 * 2x2 -> 2x2)
    C[:, 4] = A * B[:, 4] (2x2 * 2x1 -> 2x1)
    
    Rank for first part: 7 (Strassen)
    Rank for second part: 7 (Strassen)
    Rank for third part: 2 (Schoolbook for 2x2 * 2x1 is 2*2*1=4? No, 2x2 * 2x1 is 4 scalars, so rank 4 schoolbook. But wait, 2x2 * 2x1 can be done in rank 4. Is there a better way? No, rank of 2x2x1 tensor is 4. Actually, rank of matmul tensor [2,2,1] is 4. Let's verify. U(4,4), V(2,4), W(2,4). Yes, 4 is minimal for 2x2x1? No, rank of 2x2x2 is 7. Rank of 2x2x1 is 4.
    
    Total rank = 7 + 7 + 4 = 18.
    """
    n, m, p = 2, 2, 5
    rank = 18
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Part 1: C[:, 0:2] = A * B[:, 0:2]
    # A is rows 0,1. B[:, 0:2] is cols 0,1. C[:, 0:2] is cols 0,1.
    # Strassen U, V, W for 2x2 * 2x2
    U1, V1, W1 = _strassen_2x2(None, None)
    
    # Map indices for Part 1
    # U indices: i*2+j (0..3) -> same as global U
    # V indices: j*2+k (0..3) -> global V is j*5+k. So k=0,1.
    # W indices: k*2+i (0..3) -> global W is k*2+i. So k=0,1.
    
    for c in range(7):
        # U
        for r in range(4):
            U[r, c] = U1[r, c]
        
        # V
        # V1[r, c] where r = j*2+k. j in 0..1, k in 0..1.
        # Global V index = j*5+k.
        for r in range(4):
            j = r // 2
            k = r % 2
            global_v_idx = j * 5 + k
            V[global_v_idx, c] = V1[r, c]
        
        # W
        # W1[r, c] where r = k*2+i. k in 0..1, i in 0..1.
        # Global W index = k*2+i. Same as r.
        for r in range(4):
            W[r, c] = W1[r, c]
    
    # Part 2: C[:, 2:4] = A * B[:, 2:4]
    # B[:, 2:4] corresponds to k=2,3.
    # Strassen U, V, W for 2x2 * 2x2
    U2, V2, W2 = _strassen_2x2(None, None)
    
    offset = 7
    for c in range(7):
        # U
        for r in range(4):
            U[r, offset + c] = U2[r, c]
        
        # V
        # V2[r, c] where r = j*2+k_local. k_local in 0..1.
        # Global k = 2 + k_local.
        for r in range(4):
            j = r // 2
            k_local = r % 2
            global_k = 2 + k_local
            global_v_idx = j * 5 + global_k
            V[global_v_idx, offset + c] = V2[r, c]
        
        # W
        # W2[r, c] where r = k_local*2+i.
        # Global k = 2 + k_local.
        for r in range(4):
            k_local = r // 2
            i = r % 2
            global_k = 2 + k_local
            global_w_idx = global_k * 2 + i
            W[global_w_idx, offset + c] = W2[r, c]
    
    # Part 3: C[:, 4] = A * B[:, 4]
    # Schoolbook for 2x2 * 2x1.
    # i in 0..1, j in 0..1, k=4.
    # 4 terms.
    offset = 14
    for idx, (i, j) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
        c = offset + idx
        U[i*2+j, c] = 1
        V[j*5+4, c] = 1
        W[4*2+i, c] = 1
        
    return U.tolist(), V.tolist(), W.tolist()

def _try_improve_rank(n, m, p, rank, max_attempts=20000):
    """Try to find a rank-r decomposition by random search, starting from hybrid if possible."""
    U_shape = (n*m, rank)
    V_shape = (m*p, rank)
    W_shape = (p*n, rank)
    
    candidates = [-2, -1, 0, 1, 2]
    
    for attempt in range(max_attempts):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            return None
        
        # Sparse random initialization
        U = np.zeros(U_shape)
        V = np.zeros(V_shape)
        W = np.zeros(W_shape)
        
        for c in range(rank):
            n_u = np.random.randint(1, 4)
            n_v = np.random.randint(1, 4)
            n_w = np.random.randint(1, 4)
            
            u_idx = np.random.choice(n*m, size=n_u, replace=False)
            v_idx = np.random.choice(m*p, size=n_v, replace=False)
            w_idx = np.random.choice(p*n, size=n_w, replace=False)
            
            U[u_idx, c] = np.random.choice(candidates, size=n_u)
            V[v_idx, c] = np.random.choice(candidates, size=n_v)
            W[w_idx, c] = np.random.choice(candidates, size=n_w)
        
        if _check_decomposition(U, V, W, n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try to find a lower rank decomposition
    best_result = None
    best_rank = schoolbook_rank
    
    # First, try the Strassen hybrid (rank 18)
    U18, V18, W18 = _construct_strassen_hybrid()
    if _check_decomposition(U18, V18, W18, n, m, p):
        best_result = (U18, V18, W18)
        best_rank = 18
    else:
        # Fallback to schoolbook if hybrid fails (should not happen)
        rank = schoolbook_rank
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
    
    # Try to improve below 18
    for rank in range(13, best_rank):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        result = _try_improve_rank(n, m, p, rank, max_attempts=5000)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
