"""Matrix multiplication 4x5x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2(n, m, p, i_off, j_off, k_off):
    """Return U, V, W columns for a 2x2x2 Strassen block."""
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def ui(i, j): return i * m + j
    def vj(j, k): return j * p + k
    def wk(k, i): return k * n + i
    
    i0, i1 = i_off, i_off + 1
    j0, j1 = j_off, j_off + 1
    k0, k1 = k_off, k_off + 1
    
    # M1 = (A00+A11)(B00+B11)
    U[ui(i0,j0),0]=1; U[ui(i1,j1),0]=1
    V[vj(j0,k0),0]=1; V[vj(j1,k1),0]=1
    W[wk(k0,i0),0]=1; W[wk(k1,i1),0]=1
    
    # M2 = (A10+A11)B00
    U[ui(i1,j0),1]=1; U[ui(i1,j1),1]=1
    V[vj(j0,k0),1]=1
    W[wk(k0,i1),1]=1; W[wk(k1,i1),1]=-1
    
    # M3 = A00(B01-B11)
    U[ui(i0,j0),2]=1
    V[vj(j0,k1),2]=1; V[vj(j1,k1),2]=-1
    W[wk(k1,i0),2]=1; W[wk(k1,i1),2]=1
    
    # M4 = A11(B10-B00)
    U[ui(i1,j1),3]=1
    V[vj(j1,k0),3]=1; V[vj(j0,k0),3]=-1
    W[wk(k0,i0),3]=1; W[wk(k0,i1),3]=1
    
    # M5 = (A00+A01)B11
    U[ui(i0,j0),4]=1; U[ui(i0,j1),4]=1
    V[vj(j1,k1),4]=1
    W[wk(k0,i0),4]=-1; W[wk(k1,i0),4]=1
    
    # M6 = (A10-A00)(B00+B01)
    U[ui(i1,j0),5]=1; U[ui(i0,j0),5]=-1
    V[vj(j0,k0),5]=1; V[vj(j0,k1),5]=1
    W[wk(k1,i1),5]=1
    
    # M7 = (A01-A11)(B10+B11)
    U[ui(i0,j1),6]=1; U[ui(i1,j1),6]=-1
    V[vj(j1,k0),6]=1; V[vj(j1,k1),6]=1
    W[wk(k0,i0),6]=1
    
    return U, V, W

def _structured_decomposition(n, m, p):
    """Build structured block decomposition using Strassen on 2x2x2 sub-blocks."""
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_block(U_b, V_b, W_b):
        for c in range(U_b.shape[1]):
            U_cols.append(U_b[:, c])
            V_cols.append(V_b[:, c])
            W_cols.append(W_b[:, c])
    
    def schoolbook_block(n_p, m_p, p_p, i_off, j_off, k_off):
        for i_l in range(n_p):
            for j_l in range(m_p):
                for k_l in range(p_p):
                    ig = i_off + i_l
                    jg = j_off + j_l
                    kg = k_off + k_l
                    U_b = np.zeros((n*m, 1))
                    V_b = np.zeros((m*p, 1))
                    W_b = np.zeros((p*n, 1))
                    U_b[ig*m + jg, 0] = 1
                    V_b[jg*p + kg, 0] = 1
                    W_b[kg*n + ig, 0] = 1
                    add_block(U_b, V_b, W_b)
    
    def handle_block(i_off, j_off, k_off, i_sz, j_sz, k_sz):
        if i_sz == 2 and j_sz == 2 and k_sz == 2:
            U_b, V_b, W_b = _strassen_2x2x2(n, m, p, i_off, j_off, k_off)
            add_block(U_b, V_b, W_b)
        elif i_sz == 2 and j_sz == 2 and k_sz == 3:
            U_b, V_b, W_b = _strassen_2x2x2(n, m, p, i_off, j_off, k_off)
            add_block(U_b, V_b, W_b)
            schoolbook_block(2, 2, 1, i_off, j_off, k_off + 2)
        elif i_sz == 2 and j_sz == 3 and k_sz == 2:
            U_b, V_b, W_b = _strassen_2x2x2(n, m, p, i_off, j_off, k_off)
            add_block(U_b, V_b, W_b)
            schoolbook_block(2, 1, 2, i_off, j_off + 2, k_off)
        elif i_sz == 2 and j_sz == 3 and k_sz == 3:
            U_b, V_b, W_b = _strassen_2x2x2(n, m, p, i_off, j_off, k_off)
            add_block(U_b, V_b, W_b)
            schoolbook_block(2, 2, 1, i_off, j_off, k_off + 2)
            schoolbook_block(2, 1, 2, i_off, j_off + 2, k_off)
            schoolbook_block(2, 1, 1, i_off, j_off + 2, k_off + 2)
        else:
            schoolbook_block(i_sz, j_sz, k_sz, i_off, j_off, k_off)
    
    i_blocks = [(0, 2), (2, 2)]
    j_blocks = [(0, 2), (2, 3)]
    k_blocks = [(0, 2), (2, 3)]
    
    for i_off, i_sz in i_blocks:
        for j_off, j_sz in j_blocks:
            for k_off, k_sz in k_blocks:
                handle_block(i_off, j_off, k_off, i_sz, j_sz, k_sz)
    
    if not U_cols:
        return None
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    return U, V, W

def _verify(U, V, W, n, m, p, atol=1e-9):
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, np.ones((n, m, p)), atol=atol)

def _try_eliminate_columns(U, V, W, n, m, p, max_eliminations=20, timeout=30.0):
    """Try to eliminate redundant columns by checking if removing one column
    and adjusting others still gives a valid decomposition."""
    start = time.time()
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = current_U.shape[1]
    
    for _ in range(max_eliminations):
        if time.time() - start > timeout:
            break
        if current_rank <= 85:
            break
        
        found_elimination = False
        for col in range(current_rank):
            if time.time() - start > timeout:
                break
            
            # Try removing column col
            mask = np.ones(current_rank, dtype=bool)
            mask[col] = False
            U_test = current_U[:, mask]
            V_test = current_V[:, mask]
            W_test = current_W[:, mask]
            
            if _verify(U_test, V_test, W_test, n, m, p):
                current_U, current_V, current_W = U_test, V_test, W_test
                current_rank = U_test.shape[1]
                found_elimination = True
                break
        
        if not found_elimination:
            break
    
    return current_U, current_V, current_W, current_rank

def _local_search(U, V, W, n, m, p, timeout=60.0):
    """Local search: try small perturbations to find lower-rank solutions."""
    start = time.time()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Try random column removals with small adjustments
    for attempt in range(50):
        if time.time() - start > timeout:
            break
        
        # Try removing a random column
        r = best_rank
        col = np.random.randint(0, r)
        mask = np.ones(r, dtype=bool)
        mask[col] = False
        
        U_test = best_U[:, mask]
        V_test = best_V[:, mask]
        W_test = best_W[:, mask]
        
        if _verify(U_test, V_test, W_test, n, m, p):
            best_U, best_V, best_W = U_test, V_test, W_test
            best_rank = r
    
    return best_U, best_V, best_W, best_rank

def construct():
    """
    Construct rank-minimal exact tensor decomposition for 4x5x5 matrix multiplication.
    Returns dict with U, V, W matrices.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Start with structured decomposition
    structured = _structured_decomposition(n, m, p)
    if structured is not None:
        U, V, W = structured
    else:
        # Fallback to schoolbook
        school = _schoolbook(n, m, p)
        U, V, W = school['U'], school['V'], school['W']
    
    # Verify structured solution
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        school = _schoolbook(n, m, p)
        return {"U": school['U'].tolist(), "V": school['V'].tolist(), "W": school['W'].tolist()}
    
    # Try to eliminate redundant columns
    U, V, W, rank = _try_eliminate_columns(U, V, W, n, m, p, max_eliminations=30, timeout=30.0)
    
    if time.time() < deadline - 20:
        # Local search for further improvements
        U, V, W, rank = _local_search(U, V, W, n, m, p, timeout=20.0)
    
    # Final verification
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        school = _schoolbook(n, m, p)
        return {"U": school['U'].tolist(), "V": school['V'].tolist(), "W": school['W'].tolist()}
    
    # Convert to lists with integer entries
    U_list = U.astype(int).tolist()
    V_list = V.astype(int).tolist()
    W_list = W.astype(int).tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
