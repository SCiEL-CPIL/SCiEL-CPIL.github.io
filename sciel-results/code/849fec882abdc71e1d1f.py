import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    deadline = time.time() + 145
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build tensor entries
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                entries.append((u_idx, v_idx, w_idx))
    
    num_entries = len(entries)  # 24
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(nm)]
    V_sb = [[0] * rank_sb for _ in range(mp)]
    W_sb = [[0] * rank_sb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    np.random.seed(42)
    
    def verify_solution(U, V, W, target_rank):
        for (u_idx, v_idx, w_idx) in entries:
            s = 0.0
            for r in range(target_rank):
                s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
            if abs(s - 1.0) > 1e-9:
                return False
        return True
    
    def snap_to_half_int(x):
        return np.round(2.0 * x) / 2.0
    
    def solve_for_W(U, V, target_rank):
        """Given U and V, solve linear system for W"""
        M = np.zeros((num_entries, pn * target_rank))
        for e_idx, (u_idx, v_idx, w_idx) in enumerate(entries):
            for r in range(target_rank):
                val = U[u_idx, r] * V[v_idx, r]
                if val != 0:
                    M[e_idx, w_idx * target_rank + r] = val
        try:
            W_flat, _, _, _ = np.linalg.lstsq(M, np.ones(num_entries), rcond=None)
            return W_flat.reshape(pn, target_rank)
        except np.linalg.LinAlgError:
            return None
    
    def solve_for_U(V, W, target_rank):
        """Given V and W, solve linear system for U"""
        M = np.zeros((num_entries, nm * target_rank))
        for e_idx, (u_idx, v_idx, w_idx) in enumerate(entries):
            for r in range(target_rank):
                val = V[v_idx, r] * W[w_idx, r]
                if val != 0:
                    M[e_idx, u_idx * target_rank + r] = val
        try:
            U_flat, _, _, _ = np.linalg.lstsq(M, np.ones(num_entries), rcond=None)
            return U_flat.reshape(nm, target_rank)
        except np.linalg.LinAlgError:
            return None
    
    def solve_for_V(U, W, target_rank):
        """Given U and W, solve linear system for V"""
        M = np.zeros((num_entries, mp * target_rank))
        for e_idx, (u_idx, v_idx, w_idx) in enumerate(entries):
            for r in range(target_rank):
                val = U[u_idx, r] * W[w_idx, r]
                if val != 0:
                    M[e_idx, v_idx * target_rank + r] = val
        try:
            V_flat, _, _, _ = np.linalg.lstsq(M, np.ones(num_entries), rcond=None)
            return V_flat.reshape(mp, target_rank)
        except np.linalg.LinAlgError:
            return None
    
    def refine(U, V, W, target_rank, max_iters=30):
        """Iteratively refine by alternating solves"""
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            W_new = solve_for_W(U, V, target_rank)
            if W_new is not None:
                W = W_new
            U_new = solve_for_V(V, W, target_rank)
            if U_new is not None:
                U = U_new
            V_new = solve_for_U(U, W, target_rank)
            if V_new is not None:
                V = V_new
        return U, V, W
    
    # Candidate values for sparse initialization
    sparse_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    # Try ranks from 16 upward (known lower bound for 2x3x4 is around 16-19)
    for target_rank in range(16, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        found = False
        num_restarts = 3000
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Sparse random initialization
            U_init = np.zeros((nm, target_rank))
            V_init = np.zeros((mp, target_rank))
            W_init = np.zeros((pn, target_rank))
            
            # Each column has 1-3 nonzeros
            for r in range(target_rank):
                nnz_u = np.random.randint(1, 4)
                nnz_v = np.random.randint(1, 4)
                nnz_w = np.random.randint(1, 4)
                
                u_rows = np.random.choice(nm, size=nnz_u, replace=False)
                v_rows = np.random.choice(mp, size=nnz_v, replace=False)
                w_rows = np.random.choice(pn, size=nnz_w, replace=False)
                
                for idx in u_rows:
                    U_init[idx, r] = np.random.choice(sparse_vals)
                for idx in v_rows:
                    V_init[idx, r] = np.random.choice(sparse_vals)
                for idx in w_rows:
                    W_init[idx, r] = np.random.choice(sparse_vals)
            
            # Refine
            U_ref, V_ref, W_ref = refine(U_init, V_init, W_init, target_rank)
            
            # Snap to half-integers
            U_snapped = snap_to_half_int(U_ref)
            V_snapped = snap_to_half_int(V_ref)
            W_snapped = snap_to_half_int(W_ref)
            
            # Verify
            if verify_solution(U_snapped, V_snapped, W_snapped, target_rank):
                U_list = [[float(x) for x in row] for row in U_snapped]
                V_list = [[float(x) for x in row] for row in V_snapped]
                W_list = [[float(x) for x in row] for row in W_snapped]
                
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    return best_result
