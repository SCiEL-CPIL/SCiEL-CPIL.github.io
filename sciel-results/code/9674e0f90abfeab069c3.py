import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]
_START = time.time()
_DEADLINE = _START + 145.0  # safety margin before 150s kill

def _remaining():
    return _DEADLINE - time.time()

def _validate(U, V, W, n, m, p):
    """Validate exact decomposition with half-integer check."""
    if not U or not V or not W:
        return False
    r = len(U[0])
    if r == 0:
        return False
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    if not all(len(row) == r for row in U):
        return False
    if not all(len(row) == r for row in V):
        return False
    if not all(len(row) == r for row in W):
        return False
    # Check half-integer entries
    for arr in (U, V, W):
        for row in arr:
            for x in row:
                if abs(2*x - round(2*x)) > 1e-9:
                    return False
    # Check tensor identity
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    U3 = Ua.reshape(n, m, r)
    V3 = Va.reshape(m, p, r)
    W3 = Wa.reshape(p, n, r)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        return False
    return True

def _snap(arr):
    """Snap to nearest half-integer."""
    return np.round(arr * 2.0) / 2.0

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return U, V, W

def _strassen_2x2x2():
    """Strassen decomposition for 2x2x2, rank 7.
    Returns U (4,7), V (4,7), W (4,7)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1: (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1
    # M2: (A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1
    # M3: A00(B01-B11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1
    # M4: A11(B10-B00)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1
    # M5: (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1
    # M6: (A10-A00)(B00+B01)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1
    # M7: (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1
    # W: C00=M1+M4-M5-M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3-M6
    W[0,0]=1; W[0,3]=1; W[0,4]=-1; W[0,6]=-1  # C00
    W[1,2]=1; W[1,4]=1  # C01
    W[2,1]=1; W[2,3]=1  # C10
    W[3,0]=1; W[3,1]=-1; W[3,2]=1; W[3,5]=-1  # C11
    return U, V, W

def _block_strassen(n, m, p):
    """
    Construct a Block Strassen decomposition.
    We partition the dimensions into blocks of size 2 where possible,
    and use schoolbook for the remaining 1x1 blocks.
    
    For 3x5x6:
    n=3: blocks [2, 1]
    m=5: blocks [2, 2, 1]
    p=6: blocks [2, 2, 2]
    
    We iterate over all block combinations.
    For each (n_block, m_block, p_block) size:
    - If 2x2x2: use Strassen (rank 7)
    - If 1x1x1: use Schoolbook (rank 1)
    - If 1x2x2 or 2x1x2 etc: use Schoolbook (rank 4) -- actually Strassen can be adapted but schoolbook is simpler and valid.
      Wait, for 1x2x2, rank is 4. Strassen is for 2x2x2.
      For any block that is not 2x2x2, we just use schoolbook for that block's sub-tensor.
    """
    # Define block sizes
    def get_blocks(dim):
        blocks = []
        while dim > 0:
            if dim >= 2:
                blocks.append(2)
                dim -= 2
            else:
                blocks.append(1)
                dim -= 1
        return blocks

    n_blocks = get_blocks(n)
    m_blocks = get_blocks(m)
    p_blocks = get_blocks(p)
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # We need to accumulate into large matrices
    # First, determine total rank
    total_rank = 0
    block_ranks = []
    for bn in n_blocks:
        for bm in m_blocks:
            for bp in p_blocks:
                if bn == 2 and bm == 2 and bp == 2:
                    total_rank += 7
                    block_ranks.append(7)
                else:
                    total_rank += bn * bm * bp
                    block_ranks.append(bn * bm * bp)
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    rank_offset = 0
    i_offset = 0
    for bn in n_blocks:
        j_offset = 0
        for bm in m_blocks:
            k_offset = 0
            for bp in p_blocks:
                # Extract sub-tensor indices
                i_idx = range(i_offset, i_offset + bn)
                j_idx = range(j_offset, j_offset + bm)
                k_idx = range(k_offset, k_offset + bp)
                
                if bn == 2 and bm == 2 and bp == 2:
                    S_U, S_V, S_W = _strassen_2x2x2()
                    # Map S_U (2x2) to U rows
                    for di in range(2):
                        for dj in range(2):
                            U[i_idx[di]*m + j_idx[dj], rank_offset:rank_offset+7] = S_U[di*2+dj, :]
                    for dj in range(2):
                        for dk in range(2):
                            V[j_idx[dj]*p + k_idx[dk], rank_offset:rank_offset+7] = S_V[dj*2+dk, :]
                    for dk in range(2):
                        for di in range(2):
                            W[k_idx[dk]*n + i_idx[di], rank_offset:rank_offset+7] = S_W[dk*2+di, :]
                    rank_offset += 7
                else:
                    # Schoolbook for this block
                    for di in range(bn):
                        for dj in range(bm):
                            for dk in range(bp):
                                U[i_idx[di]*m + j_idx[dj], rank_offset] = 1
                                V[j_idx[dj]*p + k_idx[dk], rank_offset] = 1
                                W[k_idx[dk]*n + i_idx[di], rank_offset] = 1
                                rank_offset += 1
                
                k_offset += bp
            j_offset += bm
        i_offset += bn
        
    return U.tolist(), V.tolist(), W.tolist(), total_rank

def _alternating_minimize(U, V, W, n, m, p, max_iter=100):
    """Fast alternating minimization."""
    Ua = np.array(U, dtype=np.float64).copy()
    Va = np.array(V, dtype=np.float64).copy()
    Wa = np.array(W, dtype=np.float64).copy()
    r = Ua.shape[1]
    
    best_err = np.inf
    best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
    
    for it in range(max_iter):
        if _remaining() < 2.0:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                # Target: 1 for all k
                # Equation: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
                # For fixed i,j: sum_r U[i,j,r] * (V[j,:,r] * W[:,i,r]) = 1
                # Let A[r, k] = V[j,k,r] * W[k,i,r]
                # Then A @ U[i,j,:] = 1
                A = np.zeros((r, p))
                for k in range(p):
                    A[:, k] = Va[j*p+k] * Wa[k*n+i]
                try:
                    # Solve A.T @ x = ones(p) ? No, A is (r, p). We want x (r) such that sum_r x_r A_{r,k} = 1.
                    # This is A.T @ x = ones(p).
                    # But A.T is (p, r).
                    # If r > p, we have underdetermined system. Use lstsq.
                    Ua[i*m+j] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = np.zeros((r, n))
                for i in range(n):
                    A[:, i] = Ua[i*m+j] * Wa[k*n+i]
                try:
                    Va[j*p+k] = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = np.zeros((r, m))
                for j in range(m):
                    A[:, j] = Ua[i*m+j] * Va[j*p+k]
                try:
                    Wa[k*n+i] = np.linalg.lstsq(A.T, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Compute error
        U3 = Ua.reshape(n, m, r)
        V3 = Va.reshape(m, p, r)
        W3 = Wa.reshape(p, n, r)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err = np.sum((T - 1.0)**2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
        
        if best_err < 1e-10:
            break
    
    return best_U, best_V, best_W, best_err

def _try_rank_search(n, m, p, target_rank, seed=0):
    """
    Attempt to find a decomposition of target_rank.
    Uses random initialization and alternating minimization.
    """
    rng = random.Random(seed)
    
    # Initialize with random values
    U = np.array([[rng.gauss(0, 1) for _ in range(target_rank)] for _ in range(n*m)])
    V = np.array([[rng.gauss(0, 1) for _ in range(target_rank)] for _ in range(m*p)])
    W = np.array([[rng.gauss(0, 1) for _ in range(target_rank)] for _ in range(p*n)])
    
    # Run optimization
    U_opt, V_opt, W_opt, err = _alternating_minimize(U, V, W, n, m, p, max_iter=200)
    
    if err < 1e-6:
        # Snap to half-integers
        U_snap = _snap(U_opt)
        V_snap = _snap(V_opt)
        W_snap = _snap(W_opt)
        
        U_list = U_snap.tolist()
        V_list = V_snap.tolist()
        W_list = W_snap.tolist()
        
        if _validate(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # 1. Get Block Strassen solution (Rank 84)
    U_bs, V_bs, W_bs, rank_bs = _block_strassen(n, m, p)
    best_U, best_V, best_W = U_bs, V_bs, W_bs
    best_rank = rank_bs
    
    # Verify Block Strassen
    if not _validate(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook if Block Strassen failed (should not happen)
        U_sb, V_sb, W_sb = _schoolbook(n, m, p)
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = n * m * p

    # 2. Try to find lower rank using search
    # Lower bound is max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # We know 84 is possible. Let's try 83, 82, ...
    # Time budget: ~140s
    # Each attempt takes ~1-2s
    # We can try ~50-100 attempts
    
    # Prioritize ranks close to best_rank
    ranks_to_try = list(range(best_rank - 1, 30, -1))
    
    attempts_per_rank = 5
    
    for rank in ranks_to_try:
        if time.time() - start_time > 130:
            break
            
        found = False
        for attempt in range(attempts_per_rank):
            if time.time() - start_time > 135:
                break
                
            seed = rank * 100 + attempt
            result = _try_rank_search(n, m, p, rank, seed=seed)
            
            if result is not None:
                U_list, V_list, W_list = result
                best_U, best_V, best_W = U_list, V_list, W_list
                best_rank = rank
                found = True
                break
        
        if found:
            # If we found a lower rank, we could stop or continue.
            # Continuing might find even lower, but it's rare.
            # Let's continue for a few more ranks just in case, but stop if we hit the lower bound or time.
            # Actually, finding 83 is a huge jump. Finding 82 is unlikely.
            # Let's break if we find anything below 84 to save time for safety, 
            # unless we have lots of time left.
            if best_rank < 84:
                 # We improved!
                 # Let's try to go a bit lower if we have time
                 if time.time() - start_time > 120:
                     break
            else:
                 # Didn't find improvement at this rank, move to next lower rank
                 pass
                 
    return {"U": best_U, "V": best_V, "W": best_W}
