import random
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def validate_exact(U, V, W, n, m, p):
    """Validate with exact fraction arithmetic."""
    r = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = Fraction(0, 1)
                for c in range(r):
                    val += Fraction(U[idx_u][c]) * Fraction(V[idx_v][c]) * Fraction(W[idx_w][c])
                if val != Fraction(1, 1):
                    return False
    return True

def validate_float(U, V, W, n, m, p, tol=1e-9):
    """Fast float validation."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = W.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def schoolbook_decomposition(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U, V, W

def build_strassen_blocks(n, m, p):
    """
    Build a decomposition using Strassen's algorithm for 2x2x2 blocks
    where possible, and schoolbook for the rest.
    This gives a rank lower than n*m*p.
    """
    terms = []
    
    def add_strassen(i_start, j_start, k_start):
        # Adds 7 terms for 2x2x2 block at (i_start, j_start, k_start)
        # i in {i_start, i_start+1}
        # j in {j_start, j_start+1}
        # k in {k_start, k_start+1}
        
        # M1
        u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        # M2
        u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1}
        w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
        terms.append((u, v, w))
        
        # M3
        u = {(i_start, j_start): 1}
        v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
        w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        # M4
        u = {(i_start+1, j_start+1): 1}
        v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
        w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
        terms.append((u, v, w))
        
        # M5
        u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
        v = {(j_start+1, k_start+1): 1}
        w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
        terms.append((u, v, w))
        
        # M6
        u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
        v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
        w = {(k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        # M7
        u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
        v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1}
        terms.append((u, v, w))

    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))

    # Decomposition plan for 2x5x5:
    # We have i in {0,1}, j in {0,1,2,3,4}, k in {0,1,2,3,4}
    
    # j=0,1:
    #   k=0,1: Strassen(0,0,0)
    add_strassen(0, 0, 0)
    #   k=2,3: Strassen(0,0,2)
    add_strassen(0, 0, 2)
    #   k=4: Schoolbook for i in {0,1}, j in {0,1}, k={4}
    add_schoolbook(0, 2, 0, 2, 4, 5)
    
    # j=2,3:
    #   k=0,1: Strassen(0,2,0)
    add_strassen(0, 2, 0)
    #   k=2,3: Strassen(0,2,2)
    add_strassen(0, 2, 2)
    #   k=4: Schoolbook for i in {0,1}, j in {2,3}, k={4}
    add_schoolbook(0, 2, 2, 4, 4, 5)
    
    # j=4:
    #   k=0,1: Schoolbook for i in {0,1}, j={4}, k in {0,1}
    add_schoolbook(0, 2, 4, 5, 0, 2)
    #   k=2,3: Schoolbook for i in {0,1}, j={4}, k in {2,3}
    add_schoolbook(0, 2, 4, 5, 2, 4)
    #   k=4: Schoolbook for i in {0,1}, j={4}, k={4}
    add_schoolbook(0, 2, 4, 5, 4, 5)

    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        u_dict, v_dict, w_dict = terms[c]
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][c] = val
            
    return U, V, W

def tensor_error(U, V, W, n, m, p):
    """Calculate L2 error of the tensor decomposition."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = W.shape[1]
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                error += (val - 1.0) ** 2
    return error

def try_reduce_rank(U, V, W, n, m, p, deadline):
    """
    Try to reduce rank by finding linear dependencies among terms.
    For each pair of terms, check if one is a scalar multiple of the other
    (in the sense that removing one and adjusting the other preserves the tensor).
    Also try: for each term c, check if it can be expressed as a linear combination
    of other terms by solving a linear system.
    """
    rank = len(W[0])
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Build the tensor constraint matrix
    # For each (i,j,k), we need sum_c U[i*m+j][c]*V[j*p+k][c]*W[k*n+i][c] = 1
    # If we remove term c0, we need to check if the remaining terms still satisfy
    # this, possibly with adjusted coefficients.
    
    # Strategy: for each term c0, try to see if it's redundant
    # i.e., if the tensor contribution of term c0 can be replicated by others
    # This is equivalent to checking if the "residual" tensor (total - term c0)
    # has the same rank-1 decomposition as the others.
    
    # Simpler: check if any two terms are proportional
    for c1 in range(rank):
        if time.time() > deadline - 10:
            break
        for c2 in range(c1 + 1, rank):
            if time.time() > deadline - 10:
                break
            # Check if term c2 is a scalar multiple of term c1
            # i.e., U[:,c2] = alpha * U[:,c1], V[:,c2] = beta * V[:,c1], W[:,c2] = gamma * W[:,c1]
            # with alpha*beta*gamma = 1
            # This is a strong condition, let's check it
            u1 = np.array(U[:, c1], dtype=np.float64)
            u2 = np.array(U[:, c2], dtype=np.float64)
            v1 = np.array(V[:, c1], dtype=np.float64)
            v2 = np.array(V[:, c2], dtype=np.float64)
            w1 = np.array(W[:, c1], dtype=np.float64)
            w2 = np.array(W[:, c2], dtype=np.float64)
            
            # Check proportionality
            if np.allclose(u1, 0) or np.allclose(u2, 0):
                continue
            if np.allclose(v1, 0) or np.allclose(v2, 0):
                continue
            if np.allclose(w1, 0) or np.allclose(w2, 0):
                continue
            
            # Find alpha from U
            mask_u = np.abs(u1) > 1e-12
            if not np.any(mask_u):
                continue
            alpha_vals = u2[mask_u] / u1[mask_u]
            if not np.allclose(alpha_vals, alpha_vals[0], atol=1e-10):
                continue
            alpha = alpha_vals[0]
            
            mask_v = np.abs(v1) > 1e-12
            if not np.any(mask_v):
                continue
            beta_vals = v2[mask_v] / v1[mask_v]
            if not np.allclose(beta_vals, beta_vals[0], atol=1e-10):
                continue
            beta = beta_vals[0]
            
            mask_w = np.abs(w1) > 1e-12
            if not np.any(mask_w):
                continue
            gamma_vals = w2[mask_w] / w1[mask_w]
            if not np.allclose(gamma_vals, gamma_vals[0], atol=1e-10):
                continue
            gamma = gamma_vals[0]
            
            if abs(alpha * beta * gamma - 1.0) < 1e-10:
                # Terms are redundant! Remove c2
                U_new = [row[:] for row in best_U]
                V_new = [row[:] for row in best_V]
                W_new = [row[:] for row in best_W]
                for i in range(n * m):
                    U_new[i].pop(c2)
                for j in range(m * p):
                    V_new[j].pop(c2)
                for k in range(p * n):
                    W_new[k].pop(c2)
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank -= 1
                break
        if best_rank < rank:
            rank = best_rank
            continue
    
    return best_U, best_V, best_W

def try_optimize_rank(n, m, p, rank, deadline, max_iter=500, lr=0.05):
    """
    Try to find a decomposition of given rank using structured random search
    with exact verification. Uses sparse integer/half-integer candidates.
    """
    # Use a structured approach: random sparse matrices with entries from {-2,-1,-0.5,0,0.5,1,2}
    candidate_vals = [-2, -1, -0.5, 0, 0.5, 1, 2]
    
    best_error = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Generate sparse random factors
        density = 0.25
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for c in range(rank):
            # Each column has a few non-zero entries
            u_nz = max(1, int(np.random.exponential(2)))
            v_nz = max(1, int(np.random.exponential(2)))
            w_nz = max(1, int(np.random.exponential(2)))
            
            u_idx = random.sample(range(n * m), min(u_nz, n * m))
            v_idx = random.sample(range(m * p), min(v_nz, m * p))
            w_idx = random.sample(range(p * n), min(w_nz, p * n))
            
            for idx in u_idx:
                U[idx, c] = random.choice(candidate_vals)
            for idx in v_idx:
                V[idx, c] = random.choice(candidate_vals)
            for idx in w_idx:
                W[idx, c] = random.choice(candidate_vals)
        
        err = tensor_error(U, V, W, n, m, p)
        if err < best_error:
            best_error = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-6:
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if validate_exact(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
    
    if best_error < 1e-4 and best_U is not None:
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        if validate_exact(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer
    
    # Start with Strassen-based decomposition
    U, V, W = build_strassen_blocks(n, m, p)
    best_result = (U, V, W)
    best_rank = len(W[0])
    
    # Verify the Strassen decomposition is valid
    if not validate_float(U, V, W, n, m, p):
        # Fall back to schoolbook
        U, V, W = schoolbook_decomposition(n, m, p)
        best_result = (U, V, W)
        best_rank = len(W[0])
    
    # Try to find lower ranks
    # For 2x5x5, Strassen gives rank 34 (4*7 + 6*1 = 34)
    # Try to go lower
    candidate_ranks = [30, 28, 25, 20, 15]
    
    for rank in candidate_ranks:
        if rank >= best_rank:
            continue
        if time.time() > deadline - 20:
            break
            
        # Try multiple random starts
        found = False
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
                
            result = try_optimize_rank(n, m, p, rank, deadline, max_iter=100)
            if result:
                U, V, W = result
                best_result = (U, V, W)
                best_rank = rank
                found = True
                break
        
        if found:
            pass

    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
