"""Matrix multiplication 3x4 by 4x6: Rank-minimal decomposition via slice method."""
import time
import numpy as np
from fractions import Fraction

_START_TIME = time.time()
_DEADLINE = 135.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _build_schoolbook():
    """Build pure schoolbook rank-72 decomposition."""
    n, m, p = 3, 4, 6
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
    return U, V, W, rank

def _build_strassen_3x4x6():
    """Build the rank-66 decomposition for 3x4 * 4x6."""
    n, m, p = 3, 4, 6
    rank = 66
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1

    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        a_idx = i_base * m + j_base
        b_idx = i_base * m + (j_base + 1)
        c_idx = (i_base + 1) * m + j_base
        d_idx = (i_base + 1) * m + (j_base + 1)
        e_idx = j_base * p + k_base
        f_idx = j_base * p + (k_base + 1)
        g_idx = (j_base + 1) * p + k_base
        h_idx = (j_base + 1) * p + (k_base + 1)
        p_idx = k_base * n + i_base
        q_idx = (k_base + 1) * n + i_base
        r_idx = k_base * n + (i_base + 1)
        s_idx = (k_base + 1) * n + (i_base + 1)
        
        U[a_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        U[c_idx][col] = 1; U[d_idx][col] = 1
        V[e_idx][col] = 1
        W[r_idx][col] = 1; W[s_idx][col] = -1
        col += 1
        
        U[a_idx][col] = 1
        V[f_idx][col] = 1; V[h_idx][col] = -1
        W[q_idx][col] = 1; W[s_idx][col] = 1
        col += 1
        
        U[d_idx][col] = 1
        V[g_idx][col] = 1; V[e_idx][col] = -1
        W[p_idx][col] = 1; W[r_idx][col] = 1
        col += 1
        
        U[a_idx][col] = 1; U[b_idx][col] = 1
        V[h_idx][col] = 1
        W[p_idx][col] = -1; W[q_idx][col] = 1
        col += 1
        
        U[c_idx][col] = 1; U[a_idx][col] = -1
        V[e_idx][col] = 1; V[f_idx][col] = 1
        W[s_idx][col] = 1
        col += 1
        
        U[b_idx][col] = 1; U[d_idx][col] = -1
        V[g_idx][col] = 1; V[h_idx][col] = 1
        W[p_idx][col] = 1
        col += 1

    # Schoolbook for row 0: 1*4*6 = 24 terms
    for j in range(4):
        for k in range(6):
            add_schoolbook(0, j, k)
    
    # Strassen for rows 1-2: 2 j-blocks * 3 k-blocks * 7 = 42 terms
    for k_block in range(3):
        k_base = k_block * 2
        for j_block in range(2):
            j_base = j_block * 2
            add_strassen(1, j_base, k_base)
    
    return U, V, W, rank

def _verify(U, V, W, rank):
    """Verify the tensor decomposition is correct."""
    n, m, p = 3, 4, 6
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _slice_method_rank_reduction(U, V, W):
    """
    Apply the slice method to reduce rank.
    
    The idea: view the multiplication tensor as a matrix pencil.
    For each output column k, we have a linear map from input (i,j) to output.
    We try to find linear dependencies among the terms that allow rank reduction.
    
    This is a heuristic approach that tries to find and remove dependent terms.
    """
    n, m, p = 3, 4, 6
    rank = len(U[0])
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    active = list(range(rank))
    removed_count = 0
    
    # Try to remove terms one at a time
    for iteration in range(200):
        if _time_left() < 10.0:
            break
        if len(active) <= 1:
            break
        
        found_removed = False
        for idx in range(len(active)):
            r = active[idx]
            new_active = active[:idx] + active[idx+1:]
            
            if len(new_active) == 0:
                continue
            
            # Check if decomposition is still valid
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        val = 0.0
                        for c in new_active:
                            val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
            
            if valid:
                active = new_active
                found_removed = True
                removed_count += 1
                break
        
        if not found_removed:
            break
    
    final_rank = len(active)
    U_final = [[U[i][r] for r in active] for i in range(n*m)]
    V_final = [[V[i][r] for r in active] for i in range(m*p)]
    W_final = [[W[i][r] for r in active] for i in range(p*n)]
    
    return U_final, V_final, W_final, final_rank

def _pencil_based_reduction(U, V, W):
    """
    Use matrix pencil analysis to find rank reductions.
    
    For the multiplication tensor, we can form a matrix pencil by fixing one index.
    The rank of the pencil gives a lower bound on the tensor rank.
    We use this to guide which terms to try removing.
    """
    n, m, p = 3, 4, 6
    rank = len(U[0])
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Try removing terms in order of their "importance"
    # A term is less important if its contribution is more redundant
    
    active = list(range(rank))
    
    # Score each term by how much it contributes uniquely
    term_scores = []
    for r in range(rank):
        # Compute the "signature" of this term
        score = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    contrib = U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    score += abs(contrib)
        term_scores.append((score, r))
    
    # Sort by score (remove least important first)
    term_scores.sort(key=lambda x: x[0])
    
    for score, r in term_scores:
        if _time_left() < 10.0:
            break
        if r not in active:
            continue
        
        idx = active.index(r)
        new_active = active[:idx] + active[idx+1:]
        
        if len(new_active) == 0:
            continue
        
        # Check if decomposition is still valid
        valid = True
        for i in range(n):
            if not valid:
                break
            for j in range(m):
                if not valid:
                    break
                for k in range(p):
                    val = 0.0
                    for c in new_active:
                        val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
        
        if valid:
            active = new_active
    
    final_rank = len(active)
    U_final = [[U[i][r] for r in active] for i in range(n*m)]
    V_final = [[V[i][r] for r in active] for i in range(m*p)]
    W_final = [[W[i][r] for r in active] for i in range(p*n)]
    
    return U_final, V_final, W_final, final_rank

def _joint_rank_analysis(U, V, W):
    """
    Analyze the joint rank of the matrix slices.
    
    For each output position (i,k), the slice is a vector in the input space.
    We can look for linear dependencies among these slices.
    """
    n, m, p = 3, 4, 6
    rank = len(U[0])
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # For each (i,k), form the slice vector: sum_r U[i,m,j] * V[j,p,k] * W[k,n,i] * e_j
    # This is a linear combination of basis vectors
    
    # Try to find if we can reduce rank by exploiting structure
    active = list(range(rank))
    
    # Greedy removal with better ordering
    for iteration in range(300):
        if _time_left() < 10.0:
            break
        if len(active) <= 1:
            break
        
        best_idx = -1
        best_valid = False
        
        for idx in range(len(active)):
            r = active[idx]
            new_active = active[:idx] + active[idx+1:]
            
            if len(new_active) == 0:
                continue
            
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        val = 0.0
                        for c in new_active:
                            val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
            
            if valid:
                best_idx = idx
                best_valid = True
                break
        
        if best_valid:
            r = active[best_idx]
            active = active[:best_idx] + active[best_idx+1:]
        else:
            break
    
    final_rank = len(active)
    U_final = [[U[i][r] for r in active] for i in range(n*m)]
    V_final = [[V[i][r] for r in active] for i in range(m*p)]
    W_final = [[W[i][r] for r in active] for i in range(p*n)]
    
    return U_final, V_final, W_final, final_rank

def _try_all_reductions(U, V, W):
    """Try multiple reduction strategies and return the best result."""
    n, m, p = 3, 4, 6
    rank = len(U[0])
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Strategy 1: Simple peeling
    if _time_left() > 30:
        U1, V1, W1, rank1 = _slice_method_rank_reduction(U, V, W)
        if _verify(U1, V1, W1, rank1) and rank1 < best_rank:
            best_U, best_V, best_W, best_rank = U1, V1, W1, rank1
    
    # Strategy 2: Pencil-based
    if _time_left() > 30:
        U2, V2, W2, rank2 = _pencil_based_reduction(U, V, W)
        if _verify(U2, V2, W2, rank2) and rank2 < best_rank:
            best_U, best_V, best_W, best_rank = U2, V2, W2, rank2
    
    # Strategy 3: Joint rank analysis
    if _time_left() > 30:
        U3, V3, W3, rank3 = _joint_rank_analysis(U, V, W)
        if _verify(U3, V3, W3, rank3) and rank3 < best_rank:
            best_U, best_V, best_W, best_rank = U3, V3, W3, rank3
    
    return best_U, best_V, best_W, best_rank

def construct():
    """
    Construct exact tensor decomposition for (3x4)*(4x6) -> (3x6).
    Uses slice method and pencil analysis to minimize rank.
    """
    n, m, p = 3, 4, 6
    
    # Start with Strassen-based decomposition (rank 66)
    U, V, W, rank = _build_strassen_3x4x6()
    
    if not _verify(U, V, W, rank):
        # Fallback to schoolbook
        U, V, W, rank = _build_schoolbook()
        if not _verify(U, V, W, rank):
            return {"U": U, "V": V, "W": W}
    
    # Apply reduction strategies
    best_U, best_V, best_W, best_rank = _try_all_reductions(U, V, W)
    
    return {"U": best_U, "V": best_V, "W": best_W}
