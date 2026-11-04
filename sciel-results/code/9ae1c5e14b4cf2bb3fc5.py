import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    nm, mp, pn = n * m, m * p, p * n
    
    deadline = time.time() + 115  # Safer margin (evaluator kills at 150s)
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    def verify_solution(U, V, W, T, tol=1e-9):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T, atol=tol)
    
    def is_valid_entries(mat, tol=1e-6):
        """Check if all entries are integer or half-integer."""
        scaled = 2.0 * mat
        rounded = np.round(scaled)
        return np.all(np.abs(scaled - rounded) < tol)
    
    def snap_to_half_integer(mat):
        return np.round(2.0 * mat) / 2.0
    
    def find_2x3x3_rank5():
        """Find a rank-5 decomposition for 2x3x3 matrix multiplication."""
        n2, m2, p2 = 2, 3, 3
        nm2, mp2, pn2 = n2 * m2, m2 * p2, p2 * n2  # 6, 9, 6
        
        # Build 2x3x3 tensor
        T2 = np.zeros((nm2, mp2, pn2))
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    T2[i*m2+j, j*p2+k, k*n2+i] = 1.0
        
        r = 5
        # Try various random V, W with integer entries to solve for U
        for seed in range(2000):
            if time.time() > deadline - 10:
                return None
            np.random.seed(seed)
            # Use sparse random matrices with entries in {-1, 0, 1}
            V = np.random.choice([-1, 0, 1], size=(mp2, r))
            W = np.random.choice([-1, 0, 1], size=(pn2, r))
            
            # Solve for U
            M = (V[:, None, :] * W[None, :, :]).reshape(mp2 * pn2, r)
            rank_m = np.linalg.matrix_rank(M)
            if rank_m < r:
                continue
            
            U = np.zeros((nm2, r))
            valid = True
            for a in range(nm2):
                target = T2[a, :, :].flatten()
                sol, residuals, rank_s, sv = np.linalg.lstsq(M, target, rcond=None)
                if len(residuals) > 0 and residuals[0] > 1e-8:
                    valid = False
                    break
                U[a, :] = sol
            
            if not valid:
                continue
            
            # Check if U is (close to) half-integer
            U_snapped = snap_to_half_integer(U)
            if not is_valid_entries(U_snapped):
                continue
            
            # Verify
            T_approx = np.einsum('ar,br,cr->abc', U_snapped, V, W)
            if np.allclose(T_approx, T2, atol=1e-9):
                return U_snapped, V, W
        
        return None
    
    # Try to find the 2x3x3 rank-5 decomposition and assemble full rank-15
    result_233 = find_2x3x3_rank5()
    
    if result_233 is not None:
        U_233, V_233, W_233 = result_233
        r_block = 5
        
        # Assemble the full 2x3x9 decomposition
        # The 9 columns of B are partitioned into 3 blocks of 3
        # Block b (b=0,1,2) corresponds to columns 3b, 3b+1, 3b+2 of B
        U_full = np.zeros((nm, 3 * r_block))
        V_full = np.zeros((mp, 3 * r_block))
        W_full = np.zeros((pn, 3 * r_block))
        
        for b in range(3):
            # U: same for all blocks (A indices don't change)
            U_full[:, b*r_block:(b+1)*r_block] = U_233
            
            # V: for block b, B index j*9 + (3b + k_local)
            # V_233 has rows indexed by j*3 + k_local
            for j in range(3):
                for k_local in range(3):
                    v_local_idx = j * 3 + k_local
                    v_global_idx = j * 9 + (3 * b + k_local)
                    V_full[v_global_idx, b*r_block:(b+1)*r_block] = V_233[v_local_idx]
            
            # W: for block b, C index (3b + k_local)*2 + i
            # W_233 has rows indexed by k_local*2 + i
            for k_local in range(3):
                for i in range(2):
                    w_local_idx = k_local * 2 + i
                    w_global_idx = (3 * b + k_local) * 2 + i
                    W_full[w_global_idx, b*r_block:(b+1)*r_block] = W_233[w_local_idx]
        
        # Verify the full solution
        if verify_solution(U_full, V_full, W_full, T):
            U_list = to_list(U_full)
            V_list = to_list(V_full)
            W_list = to_list(W_full)
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    U_list = to_list(U_sb)
    V_list = to_list(V_sb)
    W_list = to_list(W_sb)
    
    return {"U": U_list, "V": V_list, "W": W_list}
