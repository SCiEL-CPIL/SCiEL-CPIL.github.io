import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 4, 6]

def _build_strassen_2x2_2x2():
    """
    Build the rank-7 Strassen decomposition for 2x2 @ 2x2.
    Returns U, V, W each of shape (4, 7) with the trace convention:
    U indexed by (i*2+j), V by (j*2+k), W by (k*2+i).
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C[i,k] -> W[k*2+i]
    # C00: i=0,k=0 -> W[0]
    # C01: i=0,k=1 -> W[2]
    # C10: i=1,k=0 -> W[1]
    # C11: i=1,k=1 -> W[3]
    
    # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    # M2 = (A10+A11)B00 -> C10+=1, C11-=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1
    W[1,1]=1; W[3,1]=-1
    # M3 = A00(B10-B11) -> C01+=1, C11+=1
    U[0,2]=1; V[2,2]=1; V[3,2]=-1
    W[2,2]=1; W[3,2]=1
    # M4 = A11(B01-B10) -> C00+=1, C10+=1
    U[3,3]=1; V[1,3]=1; V[2,3]=-1
    W[0,3]=1; W[1,3]=1
    # M5 = (A00+A01)B11 -> C00-=1, C01+=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1
    W[0,4]=-1; W[2,4]=1
    # M6 = (A10-A00)(B00+B01) -> C11+=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1
    W[3,5]=1
    # M7 = (A01-A11)(B10+B11) -> C00+=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1
    W[0,6]=1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify the decomposition."""
    Ur = U.reshape(n, m, -1)
    Vr = V.reshape(m, p, -1)
    Wr = W.reshape(p, n, -1)
    rank = Ur.shape[2]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ur[i,j,:], Vr[j,k,:]*Wr[k,i,:])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_block_decomposition(n, m, p, split_n, split_p, split_m):
    """
    Build a block decomposition using Strassen for 2x2x2 blocks.
    n is split into blocks of size in split_n
    p is split into blocks of size in split_p
    m is split into blocks of size in split_m
    """
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    U = np.zeros((size_u, 1))
    V = np.zeros((size_v, 1))
    W = np.zeros((size_w, 1))
    
    # We'll build a list of terms, each term is (U_block, V_block, W_block)
    # where U_block is n x m, V_block is m x p, W_block is p x n
    terms = []
    
    # Iterate over blocks
    n_blocks = list(zip(*[iter(range(n))] * split_n)) if n % split_n == 0 else None
    p_blocks = list(zip(*[iter(range(p))] * split_p)) if p % split_p == 0 else None
    m_blocks = list(zip(*[iter(range(m))] * split_m)) if m % split_m == 0 else None
    
    # Handle cases where dimensions don't divide evenly
    if n_blocks is None:
        n_blocks = [tuple(range(i, min(i+split_n, n))) for i in range(0, n, split_n)]
    if p_blocks is None:
        p_blocks = [tuple(range(i, min(i+split_p, p))) for i in range(0, p, split_p)]
    if m_blocks is None:
        m_blocks = [tuple(range(i, min(i+split_m, m))) for i in range(0, m, split_m)]
    
    strassen_U, strassen_V, strassen_W = _build_strassen_2x2_2x2()
    
    for n_block in n_blocks:
        for p_block in p_blocks:
            for m_block in m_blocks:
                n_size = len(n_block)
                p_size = len(p_block)
                m_size = len(m_block)
                
                if n_size == 2 and p_size == 2 and m_size == 2:
                    # Use Strassen
                    for r in range(7):
                        U_term = np.zeros((n, m))
                        V_term = np.zeros((m, p))
                        W_term = np.zeros((p, n))
                        
                        # Map Strassen indices to global indices
                        # Strassen A indices: 0=A00, 1=A01, 2=A10, 3=A11
                        # Global: A[i,j] where i in n_block, j in m_block
                        for a_idx in range(4):
                            val = strassen_U[a_idx, r]
                            if val != 0:
                                i_local = a_idx // 2
                                j_local = a_idx % 2
                                i_global = n_block[i_local]
                                j_global = m_block[j_local]
                                U_term[i_global, j_global] = val
                        
                        for b_idx in range(4):
                            val = strassen_V[b_idx, r]
                            if val != 0:
                                j_local = b_idx // 2
                                k_local = b_idx % 2
                                j_global = m_block[j_local]
                                k_global = p_block[k_local]
                                V_term[j_global, k_global] = val
                        
                        for c_idx in range(4):
                            val = strassen_W[c_idx, r]
                            if val != 0:
                                k_local = c_idx // 2
                                i_local = c_idx % 2
                                k_global = p_block[k_local]
                                i_global = n_block[i_local]
                                W_term[k_global, i_global] = val
                        
                        terms.append((U_term, V_term, W_term))
                else:
                    # Fallback: schoolbook for this block
                    for i in n_block:
                        for j in m_block:
                            for k in p_block:
                                U_term = np.zeros((n, m))
                                V_term = np.zeros((m, p))
                                W_term = np.zeros((p, n))
                                U_term[i, j] = 1.0
                                V_term[j, k] = 1.0
                                W_term[k, i] = 1.0
                                terms.append((U_term, V_term, W_term))
    
    total_rank = len(terms)
    U = np.zeros((size_u, total_rank))
    V = np.zeros((size_v, total_rank))
    W = np.zeros((size_w, total_rank))
    
    for col, (U_term, V_term, W_term) in enumerate(terms):
        U[:, col] = U_term.flatten()
        V[:, col] = V_term.flatten()
        W[:, col] = W_term.flatten()
    
    return U, V, W, total_rank

def _try_rank_optimization(U_init, V_init, W_init, n, m, p, r, max_nfev=5000, tol=1e-10):
    """Try to optimize the decomposition at a given rank."""
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    def residuals(params, r=r):
        U = params[0:size_u*r].reshape(size_u, r)
        V = params[size_u*r:size_u*r + size_v*r].reshape(size_v, r)
        W = params[size_u*r + size_v*r:].reshape(size_w, r)
        
        D = np.zeros((n, m, p))
        for c in range(r):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            D += Uc[:, :, None] * Vc[None, :, :] * Wc.T[:, None, :]
        
        return (D - 1.0).flatten()
    
    try:
        result = least_squares(residuals, params0, method='lm', max_nfev=max_nfev, 
                               ftol=1e-12, xtol=1e-12, gtol=1e-12)
        
        if result.cost < tol:
            U_sol = result.x[0:size_u*r].reshape(size_u, r)
            V_sol = result.x[size_u*r:size_u*r + size_v*r].reshape(size_v, r)
            W_sol = result.x[size_u*r + size_v*r:].reshape(size_w, r)
            
            # Snap to nearest half-integer
            U_snap = np.round(U_sol * 2) / 2
            V_snap = np.round(V_sol * 2) / 2
            W_snap = np.round(W_sol * 2) / 2
            
            if _verify(U_snap, V_snap, W_snap, n, m, p):
                return U_snap, V_snap, W_snap
    except Exception:
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    # Build schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Strategy: Use block decomposition with Strassen
    # Split n=3 into 1+2, p=6 into 2+2+2, m=4 into 2+2
    # This gives:
    # - (1,2) x (2,2) blocks: 1*2*3*2 = 12 blocks of 1x2x2 (each rank 1, total 12)
    # - (2,2) x (2,2) blocks: 2*2*3*2 = 24 blocks of 2x2x2 (each rank 7, total 168)
    # Wait, that's worse. Let me reconsider.
    
    # Better: n=3 -> 2+1, p=6 -> 2+2+2, m=4 -> 2+2
    # (2,2) x (2,2) blocks: 2*2*3*2 = 24 blocks of 2x2x2 (rank 7 each) = 168
    # (1,2) x (2,2) blocks: 1*2*3*2 = 12 blocks of 1x2x2 (rank 1 each) = 12
    # Total: 180, much worse than schoolbook.
    
    # The issue is that block decomposition doesn't share across blocks.
    # We need a different approach.
    
    # Let's try: use Strassen on the full problem by padding.
    # Pad 3x4x6 to 4x4x8, use Strassen-like algorithm, then extract.
    # 4x4x8 can be seen as 2x2 blocks of 2x2x4, or 2x2x2 blocks.
    
    # Alternative: directly optimize with a good initial guess.
    # Start from the block decomposition (even if rank is high) and try to reduce.
    
    # For now, let's try to find a rank-64 or lower solution by optimization.
    # Use the block decomposition as initial guess, then optimize.
    
    # First, build a reasonable initial guess
    # Use the structure: C[i,k] = sum_j A[i,j] B[j,k]
    # Initialize with random values scaled appropriately
    
    size_u = n * m  # 12
    size_v = m * p  # 24
    size_w = p * n  # 18
    
    # Try several ranks
    ranks_to_try = [56, 52, 48, 44, 40, 36, 32, 28]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        for trial in range(3):
            if time.time() > deadline:
                break
                
            np.random.seed(12345 + r * 100 + trial)
            
            # Initialize with structure-aware random values
            U_init = np.zeros((size_u, r))
            V_init = np.zeros((size_v, r))
            W_init = np.zeros((size_w, r))
            
            # Assign each rank component to a (i,j,k) pattern
            for c in range(r):
                # Random but structured initialization
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                
                U_init[i*m+j, c] = np.random.uniform(-2, 2)
                V_init[j*p+k, c] = np.random.uniform(-2, 2)
                W_init[k*n+i, c] = np.random.uniform(-2, 2)
                
                # Add some cross terms
                if c < r//2:
                    i2 = (i+1) % n
                    j2 = (j+1) % m
                    k2 = (k+1) % p
                    U_init[i2*m+j2, c] = np.random.uniform(-1, 1)
                    V_init[j2*p+k2, c] = np.random.uniform(-1, 1)
                    W_init[k2*n+i2, c] = np.random.uniform(-1, 1)
            
            result = _try_rank_optimization(U_init, V_init, W_init, n, m, p, r, max_nfev=3000)
            
            if result is not None:
                U_snap, V_snap, W_snap = result
                best_result = (U_snap, V_snap, W_snap)
                best_rank = r
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        U_out, V_out, W_out = best_result
        return {
            "U": U_out.tolist(),
            "V": V_out.tolist(),
            "W": W_out.tolist()
        }
    
    # Fallback: return schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
