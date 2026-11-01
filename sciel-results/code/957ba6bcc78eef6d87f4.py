"""Find rank-minimal exact tensor decomposition for 2x2 * 2x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 4]

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # U row: i*m+j, V row: j*p+k, W row: k*n+i
    
    # Build the multiplication tensor T as a 3D array
    # T[a, b, c] where a = i*m+j, b = j'*p+k', c = k'*n+i'
    # T[a,b,c] = 1 if i==i' and j==j' and k==k' else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try rank-14 Strassen-based solution first
    rank14 = _build_rank14_solution()
    if rank14 is not None:
        U14, V14, W14 = rank14
        if _verify_solution(U14, V14, W14, n, m, p):
            return {"U": U14, "V": V14, "W": W14}
    
    # Try to search for lower rank solutions
    # For rank r, we need to find U (4xr), V (8xr), W (8xr) such that
    # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b,c
    
    # Strategy: For each rank r from 13 down to 12, try to find a solution
    # Using the fact that for fixed U and W, V can be solved linearly
    # Or use algebraic approaches
    
    # Try structured search first (preserves Strassen structure)
    for target_rank in [13, 12]:
        if time.time() > deadline - 10:
            break
        
        success = _search_rank_structured(target_rank, n, m, p, T, deadline)
        if success is not None:
            U, V, W = success
            if _verify_solution(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Try random search as backup
    for target_rank in [13, 12]:
        if time.time() > deadline - 5:
            break
        
        success = _search_rank(target_rank, n, m, p, T, deadline)
        if success is not None:
            U, V, W = success
            if _verify_solution(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Try structured search for rank 13 and 12
    for target_rank in [13, 12]:
        if time.time() > deadline - 15:
            break
        
        success = _search_rank_structured(target_rank, n, m, p, T, deadline)
        if success is not None:
            U, V, W = success
            if _verify_solution(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook rank 16
    rank_fb = n * m * p
    U_fb = np.zeros((n*m, rank_fb))
    V_fb = np.zeros((m*p, rank_fb))
    W_fb = np.zeros((p*n, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1.0
                V_fb[j*p+k, col] = 1.0
                W_fb[k*n+i, col] = 1.0
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}


def _search_rank_structured(target_rank, n, m, p, T, deadline):
    """Structured search preserving Strassen-like patterns."""
    import numpy as np
    
    start_time = time.time()
    
    # Start from the rank-14 solution
    U14, V14, W14 = _build_rank14_solution()
    U14 = np.array(U14)
    V14 = np.array(V14)
    W14 = np.array(W14)
    
    # Try to reduce rank by finding linear dependencies
    # For rank 13: try dropping one column and adjusting others
    # For rank 12: try dropping two columns
    
    if target_rank == 13:
        # Try dropping each column and see if we can compensate
        for drop_idx in range(14):
            if time.time() > deadline - 5:
                return None
            
            # Keep all columns except drop_idx
            keep = [i for i in range(14) if i != drop_idx]
            U_sub = U14[:, keep]
            W_sub = W14[:, keep]
            
            # Build M from remaining columns
            M = np.zeros((n*m * p*n, 13))
            for r in range(13):
                M[:, r] = np.outer(U_sub[:, r], W_sub[:, r]).flatten()
            
            # Check if we can solve for V
            V_sub = np.zeros((m*p, 13))
            valid = True
            for b in range(m*p):
                rhs = T[:, b, :].flatten()
                try:
                    v, residuals, rank_m, sv = np.linalg.lstsq(M, rhs, rcond=None)
                    V_sub[b, :] = v
                except:
                    valid = False
                    break
            
            if valid and _verify_solution(U_sub.tolist(), V_sub.tolist(), W_sub.tolist(), n, m, p):
                return U_sub.tolist(), V_sub.tolist(), W_sub.tolist()
    
    elif target_rank == 12:
        # Try dropping pairs of columns
        for i in range(14):
            for j in range(i+1, 14):
                if time.time() > deadline - 5:
                    return None
                
                keep = [k for k in range(14) if k != i and k != j]
                U_sub = U14[:, keep]
                W_sub = W14[:, keep]
                
                M = np.zeros((n*m * p*n, 12))
                for r in range(12):
                    M[:, r] = np.outer(U_sub[:, r], W_sub[:, r]).flatten()
                
                V_sub = np.zeros((m*p, 12))
                valid = True
                for b in range(m*p):
                    rhs = T[:, b, :].flatten()
                    try:
                        v, residuals, rank_m, sv = np.linalg.lstsq(M, rhs, rcond=None)
                        V_sub[b, :] = v
                    except:
                        valid = False
                        break
                
                if valid and _verify_solution(U_sub.tolist(), V_sub.tolist(), W_sub.tolist(), n, m, p):
                    return U_sub.tolist(), V_sub.tolist(), W_sub.tolist()
    
    return None


def _build_rank14_solution():
    """Build the rank-14 Strassen-based solution."""
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(0,2), 3=(0,3), 4=(1,0), 5=(1,1), 6=(1,2), 7=(1,3)
    # W indices (k*n+i): C[0,0]=0, C[0,1]=2, C[0,2]=4, C[0,3]=6, C[1,0]=1, C[1,1]=3, C[1,2]=5, C[1,3]=7
    
    # Left Strassen block (B columns 0,1)
    # Col 0: L1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0], C[1,1]
    
    # Col 1: L2 = (A10+A11)(B00)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1  # C[1,0], C[1,1]
    
    # Col 2: L3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[5, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1  # C[0,1], C[1,1]
    
    # Col 3: L4 = A11(B10-B00)
    U[3, 3] = 1
    V[4, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1  # C[0,0], C[1,0]
    
    # Col 4: L5 = (A00+A01)(B11)
    U[0, 4] = 1; U[1, 4] = 1
    V[5, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1  # C[0,0], C[0,1]
    
    # Col 5: L6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C[1,1]
    
    # Col 6: L7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[4, 6] = 1; V[5, 6] = 1
    W[0, 6] = 1  # C[0,0]
    
    # Right Strassen block (B columns 2,3)
    # B indices: B[0,2]=2, B[0,3]=3, B[1,2]=6, B[1,3]=7
    
    # Col 7: R1 = (A00+A11)(B02+B13)
    U[0, 7] = 1; U[3, 7] = 1
    V[2, 7] = 1; V[7, 7] = 1
    W[4, 7] = 1; W[7, 7] = 1  # C[0,2], C[1,3]
    
    # Col 8: R2 = (A10+A11)(B02)
    U[2, 8] = 1; U[3, 8] = 1
    V[2, 8] = 1
    W[5, 8] = 1; W[7, 8] = -1  # C[1,2], C[1,3]
    
    # Col 9: R3 = A00(B03-B13)
    U[0, 9] = 1
    V[3, 9] = 1; V[7, 9] = -1
    W[6, 9] = 1; W[7, 9] = 1  # C[0,3], C[1,3]
    
    # Col 10: R4 = A11(B12-B02)
    U[3, 10] = 1
    V[6, 10] = 1; V[2, 10] = -1
    W[4, 10] = 1; W[5, 10] = 1  # C[0,2], C[1,2]
    
    # Col 11: R5 = (A00+A01)(B13)
    U[0, 11] = 1; U[1, 11] = 1
    V[7, 11] = 1
    W[4, 11] = -1; W[6, 11] = 1  # C[0,2], C[0,3]
    
    # Col 12: R6 = (A10-A00)(B02+B03)
    U[2, 12] = 1; U[0, 12] = -1
    V[2, 12] = 1; V[3, 12] = 1
    W[7, 12] = 1  # C[1,3]
    
    # Col 13: R7 = (A01-A11)(B12+B13)
    U[1, 13] = 1; U[3, 13] = -1
    V[6, 13] = 1; V[7, 13] = 1
    W[4, 13] = 1  # C[0,2]
    
    return U.tolist(), V.tolist(), W.tolist()


def _verify_solution(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            a = i * m + j
                            b = jp * p + k
                            c = kp * n + ip
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            actual = np.sum(U[a] * V[b] * W[c])
                            if abs(actual - expected) > 1e-8:
                                return False
    return True


def _search_rank(target_rank, n, m, p, T, deadline):
    """Search for a rank-r decomposition using numerical methods."""
    import numpy as np
    
    start_time = time.time()
    
    # For rank r, we need U (4xr), V (8xr), W (8xr)
    # Strategy: Randomly initialize U and W, solve for V linearly, verify
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # For fixed U,W: T[:,b,:] = sum_r V[b,r] * outer(U[:,r], W[:,r])
    # Linear system: M @ V[:,b] = T[:,b,:].flatten()
    # M shape: (4*8, rank) = (32, rank), M[(a*8+c), r] = U[a,r]*W[c,r]
    
    max_attempts = 500 if target_rank == 13 else 300
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break
        
        np.random.seed(attempt * 7919 + 12345)
        
        # Initialize U and W
        U = np.random.randn(n*m, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5
        
        # Build M: M[(a*8+c), r] = U[a,r]*W[c,r]
        M = np.zeros((n*m * p*n, target_rank))
        for r in range(target_rank):
            M[:, r] = np.outer(U[:, r], W[:, r]).flatten()
        
        # Check if M has full column rank
        if np.linalg.matrix_rank(M) < target_rank:
            continue
        
        # For each b, solve M @ V[b,:] = T[:,b,:].flatten()
        V = np.zeros((m*p, target_rank))
        valid = True
        for b in range(m*p):
            rhs = T[:, b, :].flatten()
            try:
                v, residuals, rank_m, sv = np.linalg.lstsq(M, rhs, rcond=None)
                V[b, :] = v
            except:
                valid = False
                break
        
        if not valid:
            continue
        
        # Verify the solution with tight tolerance
        if _verify_solution(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None
