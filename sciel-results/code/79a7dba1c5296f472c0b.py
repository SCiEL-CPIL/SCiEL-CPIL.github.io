"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

_START_TIME = time.time()
_TIME_LIMIT = 130.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _elapsed()

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    r = U.shape[1]
    if r == 0:
        return False
    T = np.ones((n, m, p), dtype=np.float64)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T_computed = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T_computed, T, atol=tol)

def _strassen_28():
    """
    Build Strassen-based decomposition for C = A * B where A is 2x4, B is 4x4, C is 2x4.
    
    Split into 4 subproblems of size 2x2x2:
    Total rank: 4 * 7 = 28
    """
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    # Strassen 7 terms for 2x2x2:
    U_coeffs = [
        [(0,0,1), (1,1,1)],      # M0
        [(1,0,1), (1,1,1)],      # M1
        [(0,0,1)],               # M2
        [(1,1,1)],               # M3
        [(0,0,1), (0,1,1)],      # M4
        [(1,0,1), (0,0,-1)],     # M5
        [(0,1,1), (1,1,-1)],     # M6
    ]
    
    V_coeffs = [
        [(0,0,1), (1,1,1)],      # M0
        [(0,0,1)],               # M1
        [(0,1,1), (1,1,-1)],     # M2
        [(1,0,1), (0,0,-1)],     # M3
        [(1,1,1)],               # M4
        [(0,0,1), (0,1,1)],      # M5
        [(1,0,1), (1,1,1)],      # M6
    ]
    
    W_coeff_map = {
        0: [(0,0,1), (1,1,1)],    # M0: C00+=1, C11+=1
        1: [(1,0,1), (1,1,-1)],   # M1: C10+=1, C11+=-1
        2: [(0,1,1), (1,1,1)],    # M2: C01+=1, C11+=1
        3: [(0,0,1), (1,0,1)],    # M3: C00+=1, C10+=1
        4: [(0,0,-1), (0,1,1)],   # M4: C00+=-1, C01+=1
        5: [(1,1,1)],             # M5: C11+=1
        6: [(0,0,1)],             # M6: C00+=1
    }
    
    col = 0
    for jp in range(2):  # j_pair: 0={0,1}, 1={2,3}
        for kp in range(2):  # k_pair: 0={0,1}, 1={2,3}
            for t in range(7):
                # U coefficients
                for (i_l, j_l, coeff) in U_coeffs[t]:
                    i = i_l
                    j = jp * 2 + j_l
                    U[i * m + j, col] += coeff
                # V coefficients
                for (j_l, k_l, coeff) in V_coeffs[t]:
                    j = jp * 2 + j_l
                    k = kp * 2 + k_l
                    V[j * p + k, col] += coeff
                # W coefficients
                for (i_l, k_l, coeff) in W_coeff_map[t]:
                    i = i_l
                    k = kp * 2 + k_l
                    W[k * n + i, col] += coeff
                col += 1
    
    return U, V, W

def _kron_strassen():
    """
    Alternative construction using Kronecker product structure.
    The 2x4x4 tensor can be viewed as T_2x2x2 (Strassen) tensor product with identity-like structure.
    This may yield different linear dependencies in the factor matrices.
    """
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    # Strassen 2x2x2 factors
    U2 = np.array([
        [1, 0, 0, 1], # i=0, j=0: M0, M2, M4, M5(-1)
        [1, 1, 1, 0], # i=0, j=1: M0, M1, M2, M6
        [0, 1, 0, 1], # i=1, j=0: M1, M3, M5
        [1, 1, 0, -1] # i=1, j=1: M0, M1, M3, M6(-1)
    ])
    # Let's re-derive carefully.
    # M0: U(0,0)=1, U(1,1)=1
    # M1: U(1,0)=1, U(1,1)=1
    # M2: U(0,0)=1
    # M3: U(1,1)=1
    # M4: U(0,0)=1, U(0,1)=1
    # M5: U(1,0)=1, U(0,0)=-1
    # M6: U(0,1)=1, U(1,1)=-1
    
    U2 = np.zeros((4, 7))
    U2[0, 0] = 1; U2[3, 0] = 1 # M0
    U2[2, 1] = 1; U2[3, 1] = 1 # M1
    U2[0, 2] = 1 # M2
    U2[3, 3] = 1 # M3
    U2[0, 4] = 1; U2[1, 4] = 1 # M4
    U2[2, 5] = 1; U2[0, 5] = -1 # M5
    U2[1, 6] = 1; U2[3, 6] = -1 # M6
    
    V2 = np.zeros((4, 7))
    V2[0, 0] = 1; V2[3, 0] = 1 # M0
    V2[0, 1] = 1 # M1
    V2[1, 2] = 1; V2[3, 2] = -1 # M2
    V2[2, 3] = 1; V2[0, 3] = -1 # M3
    V2[3, 4] = 1 # M4
    V2[0, 5] = 1; V2[1, 5] = 1 # M5
    V2[2, 6] = 1; V2[3, 6] = 1 # M6
    
    W2 = np.zeros((4, 7))
    W2[0, 0] = 1; W2[3, 0] = 1 # M0
    W2[2, 1] = 1; W2[3, 1] = -1 # M1
    W2[1, 2] = 1; W2[3, 2] = 1 # M2
    W2[0, 3] = 1; W2[2, 3] = 1 # M3
    W2[0, 4] = -1; W2[1, 4] = 1 # M4
    W2[3, 5] = 1 # M5
    W2[0, 6] = 1 # M6
    
    # Now expand to 2x4x4.
    # The tensor is sum_{j,k} A_{i,j} B_{j,k} C_{k,i}
    # We can split j into j' (0,1) and j'' (0,1) such that j = 2*j'' + j'
    # And k into k' (0,1) and k'' (0,1) such that k = 2*k'' + k'
    # This is effectively 4 independent 2x2x2 multiplications.
    # The Kronecker structure suggests that for each (j'', k'') pair, we have a 2x2x2 Strassen block.
    # This is identical to _strassen_28.
    # To get a different structure, we might try mixing the j and k indices differently.
    # However, the task asks to combine strategies. The main improvement should come from better reduction.
    # Let's just return the same for now, or a slightly permuted version.
    
    # Permute the columns to potentially expose different dependencies during reduction.
    perm = list(range(rank))
    np.random.shuffle(perm)
    U = U2[:, perm].reshape(4, 7, 4).transpose(0, 2, 1).reshape(4, 28)
    # Wait, the reshape logic is tricky. Let's just stick to the standard construction
    # and rely on the improved reduction algorithm.
    
    return _strassen_28()

def _reduce_rank_advanced(U, V, W, n, m, p, max_time=100.0):
    """
    Advanced rank reduction using:
    1. Greedy column removal.
    2. Linear dependency check: if column c is a linear combination of others,
       we can potentially remove it if the W factor allows.
       Specifically, if U_c = sum_a alpha_a U_a, V_c = sum_a beta_a V_a,
       then W_c must be zero for the term to be removable? No.
       The term is sum_r U_i,r V_j,r W_k,r.
       If we have a set of columns R, and column c in R can be written as
       U_c = sum_{a in R\{c}} x_a U_a
       V_c = sum_{a in R\{c}} y_a V_a
       Then the contribution of c is U_c^T V_c W_c = (sum x_a U_a)^T (sum y_a V_a) W_c
       This doesn't simplify to zero unless W_c is orthogonal to something.
       
       Actually, the condition for a column c to be removable is that the tensor
       T_c = U_c (x) V_c (x) W_c is in the span of {T_a | a != c}.
       This is hard to check directly.
       
       A simpler heuristic:
       Check if U_c, V_c, W_c are all zero.
       Check if U_c is zero, or V_c is zero, or W_c is zero.
       Check if U_c is a multiple of another column U_a, and V_c is a multiple of V_a,
       and W_c is a multiple of W_a? No, that would mean they are the same term.
       
       Let's stick to greedy removal first, then try a "merge" strategy:
       If we have two columns c1, c2 such that U_c1 = U_c2, V_c1 = V_c2,
       then W_c1 and W_c2 can be merged into W_c1 + W_c2, and one column removed.
    """
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    U, V, W = best_U, best_V, best_W
    rank = U.shape[1]
    
    start_time = time.time()
    
    while time.time() - start_time < max_time and rank > 14:
        if _remaining() < 15.0:
            break
            
        progress = False
        
        # 1. Greedy removal
        for c in range(rank):
            if _remaining() < 15.0:
                break
            mask = [i for i in range(rank) if i != c]
            U_test = U[:, mask]
            V_test = V[:, mask]
            W_test = W[:, mask]
            if _verify(U_test, V_test, W_test, n, m, p):
                U, V, W = U_test, V_test, W_test
                rank = len(mask)
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = rank
                progress = True
                break
        
        if progress:
            continue
            
        # 2. Merge identical U,V columns
        merged = False
        for c1 in range(rank):
            if _remaining() < 15.0:
                break
            for c2 in range(c1 + 1, rank):
                if np.allclose(U[:, c1], U[:, c2]) and np.allclose(V[:, c1], V[:, c2]):
                    # Merge W
                    W[:, c1] += W[:, c2]
                    mask = [i for i in range(rank) if i != c2]
                    U, V, W = U[:, mask], V[:, mask], W[:, mask]
                    rank -= 1
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = rank
                    merged = True
                    break
            if merged:
                break
        
        if merged:
            continue
            
        # 3. Try scaling/combining columns to create zeros?
        # This is harder. Let's stop if no progress.
        if not progress and not merged:
            break
            
    return best_U, best_V, best_W, best_rank

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 
    matrix by a 4 by 4 matrix.
    """
    n, m, p = DIMENSIONS
    
    # Start with Strassen-based decomposition (rank 28)
    U, V, W = _strassen_28()
    
    # Try to reduce rank via advanced reduction
    U, V, W, rank = _reduce_rank_advanced(U, V, W, n, m, p)
    
    # Convert to lists of integers/half-integers
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                # Round to nearest half-integer
                val_rounded = round(val * 2) / 2
                if abs(val_rounded) < 1e-9:
                    row_list.append(0)
                elif abs(val_rounded - round(val_rounded)) < 1e-9:
                    row_list.append(int(round(val_rounded)))
                else:
                    row_list.append(val_rounded)
            result.append(row_list)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
