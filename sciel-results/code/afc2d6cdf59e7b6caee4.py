import numpy as np
import time

DIMENSIONS = [2, 2, 11]

def _build_strassen_2x2x2():
    """Return the standard Strassen rank-7 decomposition for 2x2*2x2."""
    terms = []
    terms.append(([1,0,0,1], [1,0,0,1], [1,1,0,0]))
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,1]))
    terms.append(([0,0,0,1], [0,0,1,-1], [1,0,1,0]))
    terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    terms.append([[-1,0,1,0], [1,1,0,0], [0,0,0,1]])
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    return terms

def _build_rank39():
    """Verified rank-39 decomposition: 5 Strassen blocks (rank 7 each) + 1 schoolbook column (rank 4)."""
    n, m, p = 2, 2, 11
    rank = 5 * 7 + 4
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    r = 0
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = 11 + k0  # B[1, k0]
        v_h = 11 + k1  # B[1, k1]
        
        w_c00 = 2 * k0     # C[0, k0]
        w_c10 = 2 * k0 + 1 # C[1, k0]
        w_c01 = 2 * k1     # C[0, k1]
        w_c11 = 2 * k1 + 1 # C[1, k1]
        
        # m1 = (a+d)(e+h)
        U[0, r] += 1; U[3, r] += 1
        V[v_e, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1; W[w_c11, r] += 1
        r += 1
        
        # m2 = (c+d)e
        U[2, r] += 1; U[3, r] += 1
        V[v_e, r] += 1
        W[w_c10, r] += 1; W[w_c11, r] += -1
        r += 1
        
        # m3 = a(f-h)
        U[0, r] += 1
        V[v_f, r] += 1; V[v_h, r] += -1
        W[w_c01, r] += 1; W[w_c11, r] += 1
        r += 1
        
        # m4 = d(g-e)
        U[3, r] += 1
        V[v_g, r] += 1; V[v_e, r] += -1
        W[w_c00, r] += 1; W[w_c10, r] += 1
        r += 1
        
        # m5 = (a+b)h
        U[0, r] += 1; U[1, r] += 1
        V[v_h, r] += 1
        W[w_c00, r] += -1; W[w_c01, r] += 1
        r += 1
        
        # m6 = (c-a)(e+f)
        U[2, r] += 1; U[0, r] += -1
        V[v_e, r] += 1; V[v_f, r] += 1
        W[w_c11, r] += 1
        r += 1
        
        # m7 = (b-d)(g+h)
        U[1, r] += 1; U[3, r] += -1
        V[v_g, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1
        r += 1
    
    # Single column k=10: schoolbook (4 terms)
    k2 = 10
    v_e2 = k2       # B[0,10]
    v_g2 = 11 + k2  # B[1,10]
    w_c02 = 2 * k2     # C[0,10]
    w_c12 = 2 * k2 + 1 # C[1,10]
    
    # ae -> C[0,10]
    U[0, r] += 1; V[v_e2, r] += 1; W[w_c02, r] += 1; r += 1
    # bg -> C[0,10]
    U[1, r] += 1; V[v_g2, r] += 1; W[w_c02, r] += 1; r += 1
    # ce -> C[1,10]
    U[2, r] += 1; V[v_e2, r] += 1; W[w_c12, r] += 1; r += 1
    # dg -> C[1,10]
    U[3, r] += 1; V[v_g2, r] += 1; W[w_c12, r] += 1; r += 1
    
    return U, V, W

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=1e-10)

def _try_lower_rank_search(U, V, W, deadline):
    """
    Try to reduce rank by detecting linearly dependent columns.
    Uses a smarter approach: check if any column can be expressed as a linear
    combination of others in the tensor sense, then verify removal.
    """
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W
    
    n, m, p = DIMENSIONS
    current_U = U.copy().astype(float)
    current_V = V.copy().astype(float)
    current_W = W.copy().astype(float)
    current_rank = rank
    
    # Precompute target tensor for verification
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Greedy removal with more iterations and better ordering
    removed = set()
    for _ in range(20):  # Allow up to 20 removals
        if time.time() > deadline - 5:
            break
        
        best_remove = -1
        best_score = float('inf')
        
        # Score each column by its "redundancy" - how much it can be
        # approximated by the others
        for r in range(current_rank):
            if r in removed:
                continue
            
            # Try removing column r
            mask = [i for i in range(current_rank) if i != r]
            U_test = current_U[:, mask]
            V_test = current_V[:, mask]
            W_test = current_W[:, mask]
            
            # Quick verify
            T_recon = np.einsum('ar,br,cr->abc', U_test, V_test, W_test)
            error = np.max(np.abs(T_recon - T_target))
            
            if error < 1e-10:
                best_remove = r
                break
            
            # Track best candidate for potential later use
            if error < best_score:
                best_score = error
                best_remove = r
        
        if best_remove == -1 or best_score >= 1e-10:
            break
        
        mask = [i for i in range(current_rank) if i != best_remove]
        current_U = current_U[:, mask]
        current_V = current_V[:, mask]
        current_W = current_W[:, mask]
        current_rank -= 1
        removed.add(best_remove)
    
    # Convert back to integer if all values are close to integers
    U_out = np.round(current_U).astype(np.int64)
    V_out = np.round(current_V).astype(np.int64)
    W_out = np.round(current_W).astype(np.int64)
    
    # Final verification
    if _verify(U_out, V_out, W_out):
        return U_out, V_out, W_out
    
    return U, V, W

def _try_structural_optimization(U, V, W, deadline):
    """
    Try to find a lower-rank decomposition by exploiting structural properties.
    
    Key insight: The 11 columns of the B matrix can be grouped. If we can find
    shared structure between column pairs, we might reduce the total rank.
    
    Strategy: 
    1. Start with rank-39 solution
    2. Try to merge terms from different Strassen blocks
    3. Use linear algebra to find if any term is redundant
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # For now, this is a placeholder for more sophisticated optimization
    # In a full implementation, we would:
    # 1. Set up the system of equations for rank r
    # 2. Use iterative refinement or gradient-based methods
    # 3. Check for solutions with r < 39
    
    # Since this is complex and time-consuming, we'll just return the input
    # for now and rely on the greedy removal in _try_lower_rank_search
    return U, V, W

def _build_schoolbook():
    """Fallback: rank 44 schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    # Build rank-39 construction
    U, V, W = _build_rank39()
    
    # Verify the base construction
    if _verify(U, V, W):
        # Try to reduce rank via greedy column removal
        if time.time() < deadline - 10:
            U_red, V_red, W_red = _try_lower_rank_search(U, V, W, deadline)
            if U_red.shape[1] < U.shape[1]:
                return {
                    "U": U_red.tolist(),
                    "V": V_red.tolist(),
                    "W": W_red.tolist()
                }
        
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    # Fallback: Schoolbook (Rank 44)
    U_sb, V_sb, W_sb = _build_schoolbook()
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
