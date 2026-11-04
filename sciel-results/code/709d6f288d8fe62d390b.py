import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor entries for 4x5x6
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    target_entries = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries[(a, b, c)] = 1.0

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return U, V, W, rank

    def verify_fast(U, V, W, tol=1e-6):
        """Fast verification using einsum."""
        R = np.einsum('ad,bd,cd->abc', U, V, W)
        # Check target entries
        for (a, b, c), val in target_entries.items():
            if abs(R[a, b, c] - val) > tol:
                return False
        # Check non-target: create a mask
        T = np.zeros((nm, mp, pn))
        for (a, b, c), val in target_entries.items():
            T[a, b, c] = val
        diff = R - T
        if np.max(np.abs(diff)) > tol:
            return False
        return True

    # Fallback to schoolbook
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb

    # Strategy: Decompose into 4 blocks of 2x5x3
    # A is 4x5, split into A1 (2x5) and A2 (2x5)
    # B is 5x6, split into B1 (5x3) and B2 (5x3)
    # Sub-problems: A1*B1, A1*B2, A2*B1, A2*B2
    # Each is 2x5x3.
    
    n_sub, m_sub, p_sub = 2, 5, 3
    nm_sub = n_sub * m_sub # 10
    mp_sub = m_sub * p_sub # 15
    pn_sub = p_sub * n_sub # 6
    
    # Build target for sub-problem
    sub_target = {}
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                a = i * m_sub + j
                b = j * p_sub + k
                c = k * n_sub + i
                sub_target[(a, b, c)] = 1.0

    def solve_sub_problem(max_time=10.0):
        """Solve for 2x5x3 multiplication rank."""
        start = time.time()
        best_sub_rank = n_sub * m_sub * p_sub
        best_sub_U, best_sub_V, best_sub_W = None, None, None
        
        # Try ranks from lower bound upwards
        lower_bound = max(nm_sub, mp_sub, pn_sub)
        
        for r in range(lower_bound, best_sub_rank):
            if time.time() - start > max_time:
                break
            
            # Try a few random initializations
            for seed in range(10):
                if time.time() - start > max_time:
                    break
                
                np.random.seed(seed)
                U = np.random.randn(nm_sub, r)
                V = np.random.randn(mp_sub, r)
                W = np.random.randn(pn_sub, r)
                
                # ALS iterations
                for it in range(50):
                    if time.time() - start > max_time:
                        break
                    
                    # Update U
                    for a in range(nm_sub):
                        # Build M: (mp_sub * pn_sub) x r
                        # M[b*pn_sub+c, d] = V[b,d] * W[c,d]
                        M = np.zeros((mp_sub * pn_sub, r))
                        t = np.zeros(mp_sub * pn_sub)
                        for b in range(mp_sub):
                            for c in range(pn_sub):
                                idx = b * pn_sub + c
                                M[idx, :] = V[b, :] * W[c, :]
                                if (a, b, c) in sub_target:
                                    t[idx] = 1.0
                        # Solve M @ U[a,:] = t
                        try:
                            U[a, :] = np.linalg.lstsq(M, t, rcond=None)[0]
                        except:
                            pass
                    
                    # Update V
                    for b in range(mp_sub):
                        M = np.zeros((nm_sub * pn_sub, r))
                        t = np.zeros(nm_sub * pn_sub)
                        for a in range(nm_sub):
                            for c in range(pn_sub):
                                idx = a * pn_sub + c
                                M[idx, :] = U[a, :] * W[c, :]
                                if (a, b, c) in sub_target:
                                    t[idx] = 1.0
                        try:
                            V[b, :] = np.linalg.lstsq(M, t, rcond=None)[0]
                        except:
                            pass

                    # Update W
                    for c in range(pn_sub):
                        M = np.zeros((nm_sub * mp_sub, r))
                        t = np.zeros(nm_sub * mp_sub)
                        for a in range(nm_sub):
                            for b in range(mp_sub):
                                idx = a * mp_sub + b
                                M[idx, :] = U[a, :] * V[b, :]
                                if (a, b, c) in sub_target:
                                    t[idx] = 1.0
                        try:
                            W[c, :] = np.linalg.lstsq(M, t, rcond=None)[0]
                        except:
                            pass
                
                # Verify sub-solution
                R_sub = np.einsum('ad,bd,cd->abc', U, V, W)
                valid = True
                T_sub = np.zeros((nm_sub, mp_sub, pn_sub))
                for (a, b, c), val in sub_target.items():
                    T_sub[a, b, c] = val
                    if abs(R_sub[a, b, c] - val) > 1e-6:
                        valid = False
                        break
                if valid:
                    if np.max(np.abs(R_sub - T_sub)) > 1e-6:
                        valid = False
                
                if valid:
                    best_sub_rank = r
                    best_sub_U, best_sub_V, best_sub_W = U, V, W
                    break # Found lowest r for this seed, try lower r? No, r is increasing.
                    # Actually, we break the seed loop, but continue to next r? No, r is increasing.
                    # We want the lowest r. So if we find one at r, we can stop.
                    return best_sub_U, best_sub_V, best_sub_W, r
        
        return best_sub_U, best_sub_V, best_sub_W, best_sub_rank

    # Attempt to solve sub-problem
    if time.time() < deadline - 10:
        sub_U, sub_V, sub_W, sub_rank = solve_sub_problem(max_time=30.0)
        
        if sub_rank < n_sub * m_sub * p_sub:
            total_rank = 4 * sub_rank
            if total_rank < best_rank:
                # Construct full U, V, W
                U_full = np.zeros((nm, total_rank))
                V_full = np.zeros((mp, total_rank))
                W_full = np.zeros((pn, total_rank))
                
                # Map sub-indices to full indices
                # Sub i (0,1) -> Full i (2*i, 2*i+1)
                # Sub k (0,1,2) -> Full k (3*k, 3*k+1, 3*k+2)
                # Sub j (0..4) -> Full j (0..4)
                
                for block_i in range(2): # A1, A2
                    for block_k in range(2): # B1, B2
                        offset = (block_i * 2 + block_k) * sub_rank
                        # A block rows: 2*block_i, 2*block_i+1
                        # B block cols: 3*block_k, 3*block_k+1, 3*block_k+2
                        # C block rows: 3*block_k, 3*block_k+1, 3*block_k+2
                        # C block cols: 2*block_i, 2*block_i+1
                        
                        for s_i in range(n_sub): # 0,1
                            full_i = 2 * block_i + s_i
                            for j in range(m_sub): # 0..4
                                full_j = j
                                for s_k in range(p_sub): # 0..2
                                    full_k = 3 * block_k + s_k
                                    
                                    # U index
                                    a_sub = s_i * m_sub + j
                                    a_full = full_i * m + full_j
                                    U_full[a_full, offset:offset+sub_rank] += sub_U[a_sub, :]
                                    
                                    # V index
                                    b_sub = j * p_sub + s_k
                                    b_full = full_j * p + full_k
                                    V_full[b_full, offset:offset+sub_rank] += sub_V[b_sub, :]
                                    
                                    # W index
                                    c_sub = s_k * n_sub + s_i
                                    c_full = full_k * n + full_i
                                    W_full[c_full, offset:offset+sub_rank] += sub_W[c_sub, :]
                
                # Verify full solution
                if verify_fast(U_full, V_full, W_full):
                    best_U, best_V, best_W = U_full, V_full, W_full
                    best_rank = total_rank

    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
