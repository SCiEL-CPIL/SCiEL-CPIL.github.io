import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def _als_solve_232(r, deadline, seed=42):
    """
    Find a rank-r decomposition for 2x3x2 matrix multiplication.
    Returns (U, V, W) as numpy arrays or None if not found.
    """
    n, m, p = 2, 3, 2
    np.random.seed(seed)
    
    # Shapes: U (6, r), V (6, r), W (4, r)
    for attempt in range(20):
        if time.time() > deadline:
            return None
            
        U = np.random.uniform(-1, 1, (n * m, r))
        V = np.random.uniform(-1, 1, (m * p, r))
        W = np.random.uniform(-1, 1, (p * n, r))
        
        # ALS iterations
        for it in range(100):
            if time.time() > deadline:
                return None
                
            # Update U
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    # Construct M (p x r) and b (p)
                    M = np.zeros((p, r))
                    b = np.ones(p)
                    for k in range(p):
                        v_idx = j * p + k
                        w_idx = k * n + i
                        M[k, :] = V[v_idx, :] * W[w_idx, :]
                    # Solve M @ U[u_idx, :] = b
                    # U[u_idx, :] is (r, 1)
                    sol, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
                    U[u_idx, :] = sol
                    
            # Update V
            for j in range(m):
                for k in range(p):
                    v_idx = j * p + k
                    M = np.zeros((n, r))
                    b = np.ones(n)
                    for i in range(n):
                        u_idx = i * m + j
                        w_idx = k * n + i
                        M[i, :] = U[u_idx, :] * W[w_idx, :]
                    sol, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
                    V[v_idx, :] = sol
                    
            # Update W
            for k in range(p):
                for i in range(n):
                    w_idx = k * n + i
                    M = np.zeros((m, r))
                    b = np.ones(m)
                    for j in range(m):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        M[j, :] = U[u_idx, :] * V[v_idx, :]
                    sol, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
                    W[w_idx, :] = sol
            
            # Project to half-integers
            U_half = np.round(U * 2) / 2
            V_half = np.round(V * 2) / 2
            W_half = np.round(W * 2) / 2
            
            # Check if exact
            if _check_232(U_half, V_half, W_half):
                return U_half, V_half, W_half
                
    return None

def _check_232(U, V, W):
    n, m, p = 2, 3, 2
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    best_rank = n * m * p
    best_result = _schoolbook(n, m, p)
    
    # Try to find better rank using 2x3x2 blocks
    # 2x3x7 can be split into 3x (2x3x2) + 1x (2x3x1)
    # Rank of 2x3x1 is 6.
    # We search for rank of 2x3x2 from 11 down to 6.
    
    found_232 = None
    for r_232 in range(11, 5, -1):
        if time.time() > deadline - 20:
            break
        sol = _als_solve_232(r_232, deadline - 10)
        if sol is not None:
            found_232 = sol
            break
            
    if found_232 is not None:
        U_232, V_232, W_232 = found_232
        r_232 = U_232.shape[1]
        total_rank = 3 * r_232 + 6
        
        # Construct full U, V, W
        # U shape (6, total_rank)
        # V shape (21, total_rank)
        # W shape (14, total_rank)
        
        U_full = np.zeros((6, total_rank))
        V_full = np.zeros((21, total_rank))
        W_full = np.zeros((14, total_rank))
        
        # Fill 3 blocks of 2x3x2
        # Block 1: columns 0,1 of B (k=0,1)
        # Block 2: columns 2,3 of B (k=2,3)
        # Block 3: columns 4,5 of B (k=4,5)
        # Block 4: column 6 of B (k=6) - schoolbook
        
        for b_idx in range(3):
            offset = b_idx * r_232
            # U is same for all blocks?
            # In 2x3x2, U depends on i, j.
            # In 2x3x7, U depends on i, j.
            # The U matrix is the same for all 3 blocks because the A matrix is the same.
            # So we just copy U_232 into the columns.
            U_full[:, offset:offset+r_232] = U_232
            
            # V depends on j, k.
            # For block b_idx, k ranges from 2*b_idx to 2*b_idx+1.
            # V_232 has rows indexed by j*p+k where p=2.
            # In full V, rows are indexed by j*7+k.
            # So for j in 0..2, k in 2*b_idx .. 2*b_idx+1:
            for j in range(3):
                for local_k in range(2):
                    full_k = 2 * b_idx + local_k
                    v_full_idx = j * 7 + full_k
                    v_232_idx = j * 2 + local_k
                    V_full[v_full_idx, offset:offset+r_232] = V_232[v_232_idx, :]
                    
            # W depends on k, i.
            # For block b_idx, k ranges from 2*b_idx to 2*b_idx+1.
            # W_232 has rows indexed by k*n+i where n=2.
            # In full W, rows are indexed by k*2+i.
            # So for local_k in 0..1:
            for local_k in range(2):
                full_k = 2 * b_idx + local_k
                for i in range(2):
                    w_full_idx = full_k * 2 + i
                    w_232_idx = local_k * 2 + i
                    W_full[w_full_idx, offset:offset+r_232] = W_232[w_232_idx, :]
                    
        # Fill schoolbook part for k=6
        offset = 3 * r_232
        # Schoolbook for 2x3x1
        # U_sb (6, 6), V_sb (3, 6), W_sb (2, 6)
        # U_sb[i*m+j, col] = 1 if col corresponds to (i,j)
        # V_sb[j*1+0, col] = 1 if col corresponds to (i,j)
        # W_sb[0*2+i, col] = 1 if col corresponds to (i,j)
        
        col = 0
        for i in range(2):
            for j in range(3):
                u_idx = i * 3 + j
                v_idx = j # j*1 + 0
                w_idx = 0 * 2 + i # k=0 for the 1x1 block? No, k=6 in full.
                # Wait, the W index in the full matrix is k*2+i.
                # Here k=6. So w_idx = 6*2 + i = 12 + i.
                # But in the sub-problem, W is size (2, 6).
                # Let's just construct the schoolbook terms directly into the full arrays.
                
                U_full[u_idx, offset + col] = 1
                V_full[j * 7 + 6, offset + col] = 1
                W_full[6 * 2 + i, offset + col] = 1
                col += 1
                
        # Convert to list of half-integers
        best_result = _to_list(U_full, V_full, W_full)
        best_rank = total_rank
        
    return best_result

def _schoolbook(n, m, p):
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
    return _to_list(U, V, W)

def _to_list(U, V, W):
    def to_half_int(arr):
        return [[float(int(x * 2)) / 2 for x in row] for row in arr]
    return {"U": to_half_int(U), "V": to_half_int(V), "W": to_half_int(W)}
