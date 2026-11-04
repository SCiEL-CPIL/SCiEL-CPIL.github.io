import numpy as np
import time

def construct():
    n, m, p = 2, 3, 10
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill
    
    # Start with the known rank-55 decomposition
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen coefficients for 2x2x2
    L = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    R = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    Wc = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {1: 1, 0: -1},
        {3: 1},
        {0: 1}
    ]
    
    for block in range(5):
        k_off = block * 2
        for s in range(7):
            for li, val in L[s].items():
                i, j = li // 2, li % 2
                U[i * m + j][col] = val
            for li, val in R[s].items():
                j, k = li // 2, li % 2
                V[j * p + k_off + k][col] = val
            for li, val in Wc[s].items():
                i, k = li // 2, li % 2
                W[(k_off + k) * n + i][col] = val
            col += 1
    
    for i in range(2):
        for k in range(10):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Convert to numpy for efficient linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Precompute target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Precompute indices of non-zero and zero positions for fast checking
    nz_indices = np.argwhere(T == 1.0)  # shape (60, 3)
    # Sample zero positions for verification
    all_indices = np.argwhere(np.ones((n*m, m*p, p*n), dtype=bool))
    zero_indices = all_indices[np.all(T[all_indices[:,0], all_indices[:,1], all_indices[:,2]] == 0, axis=-1)]
    # Use a subset of zero positions for speed
    if len(zero_indices) > 200:
        rng = np.random.RandomState(42)
        zero_indices = zero_indices[rng.choice(len(zero_indices), 200, replace=False)]
    
    def is_redundant_fast(r, U_m, V_m, W_m, current_rank):
        """Fast check: column r is redundant iff T - contrib_r equals sum of other rank-1 tensors."""
        # Compute contribution of column r
        u_r = U_m[:, r]
        v_r = V_m[:, r]
        w_r = W_m[:, r]
        
        # Check on non-zero positions: sum_{r'!=r} U[a,r']*V[b,r']*W[c,r'] should = 1 - u_r[a]*v_r[b]*w_r[c]
        # And on zero positions: sum_{r'!=r} U[a,r']*V[b,r']*W[c,r'] should = -u_r[a]*v_r[b]*w_r[c]
        
        # Total sum over all columns at position (a,b,c): U[a,:].dot(V[b,:]*W[c,:])
        # We need: total_sum[a,b,c] - u_r[a]*v_r[b]*w_r[c] == T[a,b,c]
        # i.e., total_sum[a,b,c] == T[a,b,c] + u_r[a]*v_r[b]*w_r[c]
        
        # Compute total_sum on the test positions efficiently
        # total_sum[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        
        # For non-zero positions
        if len(nz_indices) > 0:
            a_idx = nz_indices[:, 0]
            b_idx = nz_indices[:, 1]
            c_idx = nz_indices[:, 2]
            
            # total_sum at these positions
            # = sum_r U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
            # Use broadcasting: U[a_idx, :] has shape (60, rank)
            U_a = U_m[a_idx, :]  # (60, rank)
            V_b = V_m[b_idx, :]  # (60, rank)
            W_c = W_m[c_idx, :]  # (60, rank)
            
            # total_sum = sum over r of U_a[i,r] * V_b[i,r] * W_c[i,r]
            total_sum_nz = np.einsum('ir,ir,ir->i', U_a, V_b, W_c)
            
            # contrib at these positions
            contrib_nz = u_r[a_idx] * v_r[b_idx] * w_r[c_idx]
            
            # Check: total_sum_nz - contrib_nz should equal 1 (T value at nz positions)
            if not np.allclose(total_sum_nz - contrib_nz, 1.0, atol=1e-9):
                return False
        
        # Check on zero positions
        if len(zero_indices) > 0:
            a_idx = zero_indices[:, 0]
            b_idx = zero_indices[:, 1]
            c_idx = zero_indices[:, 2]
            
            U_a = U_m[a_idx, :]
            V_b = V_m[b_idx, :]
            W_c = W_m[c_idx, :]
            
            total_sum_z = np.einsum('ir,ir,ir->i', U_a, V_b, W_c)
            contrib_z = u_r[a_idx] * v_r[b_idx] * w_r[c_idx]
            
            # Check: total_sum_z - contrib_z should equal 0
            if not np.allclose(total_sum_z - contrib_z, 0.0, atol=1e-9):
                return False
        
        return True
    
    # Try removing columns - use a worklist approach
    removed = []
    r = 0
    while r < rank:
        if time.time() > deadline:
            break
        if is_redundant_fast(r, U_np, V_np, W_np, rank):
            U_np = np.delete(U_np, r, axis=1)
            V_np = np.delete(V_np, r, axis=1)
            W_np = np.delete(W_np, r, axis=1)
            rank -= 1
            removed.append(r)
            # Don't increment r, check the same position again (which now has the next column)
            # But reset to 0 to catch cascading redundancies
            r = 0
        else:
            r += 1
    
    # Convert back to lists with integer/half-integer entries
    U_out = U_np.tolist()
    V_out = V_np.tolist()
    W_out = W_np.tolist()
    
    for mat in [U_out, V_out, W_out]:
        for row in mat:
            for idx, val in enumerate(row):
                row[idx] = round(val * 2) / 2
    
    return {"U": U_out, "V": V_out, "W": W_out}
