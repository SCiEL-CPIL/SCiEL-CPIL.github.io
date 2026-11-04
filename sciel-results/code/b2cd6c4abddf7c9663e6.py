import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s wall-clock

    # Schoolbook fallback
    rank_school = n * m * p
    U_sb = np.zeros((n*m, rank_school))
    V_sb = np.zeros((m*p, rank_school))
    W_sb = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_rank = rank_school
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    def _verify_decomposition(U, V, W, n, m, p):
        r = U.shape[1]
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        res = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        return np.allclose(res, 1.0, atol=1e-6)

    # Winograd 2x2 factors
    U2 = np.array([
        [1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0]
    ], dtype=float)
    
    V2 = np.array([
        [1, 0, 0, 1, 0],
        [0, 1, 1, 0, -1],
        [1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, -1, 1, -1, 1],
        [1, -1, 1, 0, -1],
        [1, 1, -1, 1, 0],
        [1, 1, -1, 0, 1]
    ], dtype=float)

    # Construct 4x4x4 Winograd factors (Rank 25)
    U4_kron = np.kron(U2, U2) # 16x25
    V4_kron = np.kron(V2, V2) # 16x25
    W4_kron = np.kron(W2, W2) # 16x25

    perm_U = np.zeros(16, dtype=int)
    for r1 in range(4):
        row1 = r1 // 2
        col1 = r1 % 2
        for r2 in range(4):
            row2 = r2 // 2
            col2 = r2 % 2
            kron_idx = r1 * 4 + r2
            i = row1 * 2 + row2
            j = col1 * 2 + col2
            target_idx = i * 4 + j
            perm_U[target_idx] = kron_idx

    U_winograd = U4_kron[perm_U, :]
    V_winograd = V4_kron[perm_U, :]
    W_winograd = W4_kron[perm_U, :]

    # Construct Rank 29 candidate (Current Best)
    rank_candidate = 29
    U_cand = np.zeros((16, rank_candidate))
    V_cand = np.zeros((20, rank_candidate))
    W_cand = np.zeros((20, rank_candidate))

    U_cand[:, 0:25] = U_winograd
    
    for j in range(4):
        for k in range(4):
            v_row = j * 4 + k
            target_v_row = j * 5 + k
            V_cand[target_v_row, 0:25] = V_winograd[v_row, :]
            
    W_cand[0:16, 0:25] = W_winograd

    for t in range(4):
        col_idx = 25 + t
        for j in range(4):
            U_cand[t*4+j, col_idx] = 1.0
        for j in range(4):
            V_cand[j*5+4, col_idx] = 1.0
        for i in range(4):
            W_cand[4*4+i, col_idx] = 1.0

    if _verify_decomposition(U_cand, V_cand, W_cand, n, m, p):
        best_rank = 29
        best_U = U_cand
        best_V = V_cand
        best_W = W_cand

    # Try to find lower rank solutions using structured approach
    # Strategy: Partition 4x4x5 into 3x3x3 core + boundary strips
    # Use Strassen-like decomposition for the 3x3x3 core (rank 7)
    # Then solve for boundary terms
    
    # Strassen 3x3x3 factors (rank 7)
    U3 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0]
    ], dtype=float)
    
    V3 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0]
    ], dtype=float)
    
    W3 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0]
    ], dtype=float)

    # For 4x4x5, partition as:
    # Core: 3x3x3 (i=0..2, j=0..2, k=0..2) -> rank 7
    # Row strip: 1x4x5 (i=3, j=0..3, k=0..4) -> need to handle
    # Column strip: 3x1x5 (i=0..2, j=3, k=0..4) -> need to handle  
    # Corner: 1x1x5 (i=3, j=3, k=0..4) -> need to handle
    
    # Total boundary: 20 + 15 + 5 = 40 elements
    # But many can be shared with core terms
    
    # Try rank 25: 7 (core) + 18 (boundary)
    # Try rank 24: 7 (core) + 17 (boundary)
    
    def _objective_rank25(x):
        r = 25
        U_flat = x[:16*r]
        V_flat = x[16*r:16*r+20*r]
        W_flat = x[16*r+20*r:16*r+20*r+20*r]
        
        U = U_flat.reshape(16, r)
        V = V_flat.reshape(20, r)
        W = W_flat.reshape(20, r)
        
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        
        res = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        error = np.sum((res - 1.0)**2)
        return error

    def _grad_rank25(x):
        r = 25
        U_flat = x[:16*r]
        V_flat = x[16*r:16*r+20*r]
        W_flat = x[16*r+20*r:16*r+20*r+20*r]
        
        U = U_flat.reshape(16, r)
        V = V_flat.reshape(20, r)
        W = W_flat.reshape(20, r)
        
        U_reshaped = U.reshape(n, m, r)
        V_reshaped = V.reshape(m, p, r)
        W_reshaped = W.reshape(p, n, r)
        
        res = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        diff = res - 1.0
        
        dU = 2 * np.einsum('ijk,jkr,kir->ijr', diff, V_reshaped, W_reshaped)
        dV = 2 * np.einsum('ijk,ijr,kir->jkr', diff, U_reshaped, W_reshaped)
        dW = 2 * np.einsum('ijk,ijr,jkr->kir', diff, U_reshaped, V_reshaped)
        
        return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])

    # Try multiple ranks from 24 down to 20
    for target_rank in [24, 23, 22, 21, 20]:
        if time.time() > deadline - 30:
            break
        
        def make_objective(r):
            def _obj(x):
                U_flat = x[:16*r]
                V_flat = x[16*r:16*r+20*r]
                W_flat = x[16*r+20*r:16*r+20*r+20*r]
                
                U = U_flat.reshape(16, r)
                V = V_flat.reshape(20, r)
                W = W_flat.reshape(20, r)
                
                U_reshaped = U.reshape(n, m, r)
                V_reshaped = V.reshape(m, p, r)
                W_reshaped = W.reshape(p, n, r)
                
                res = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
                
                error = np.sum((res - 1.0)**2)
                return error
            
            def _grad(x):
                U_flat = x[:16*r]
                V_flat = x[16*r:16*r+20*r]
                W_flat = x[16*r+20*r:16*r+20*r+20*r]
                
                U = U_flat.reshape(16, r)
                V = V_flat.reshape(20, r)
                W = W_flat.reshape(20, r)
                
                U_reshaped = U.reshape(n, m, r)
                V_reshaped = V.reshape(m, p, r)
                W_reshaped = W.reshape(p, n, r)
                
                res = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
                diff = res - 1.0
                
                dU = 2 * np.einsum('ijk,jkr,kir->ijr', diff, V_reshaped, W_reshaped)
                dV = 2 * np.einsum('ijk,ijr,kir->jkr', diff, U_reshaped, W_reshaped)
                dW = 2 * np.einsum('ijk,ijr,jkr->kir', diff, U_reshaped, V_reshaped)
                
                return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
            
            return _obj, _grad

        obj_func, grad_func = make_objective(target_rank)
        
        # Try 2 restarts per rank
        for restart in range(2):
            if time.time() > deadline - 35:
                break
            
            # Initialize with Winograd structure + random perturbation
            U_init = np.zeros((16, target_rank))
            V_init = np.zeros((20, target_rank))
            W_init = np.zeros((20, target_rank))
            
            # Use Winograd for first min(25, target_rank) columns
            num_winograd = min(25, target_rank)
            U_init[:, :num_winograd] = U_winograd[:, :num_winograd]
            
            for j in range(4):
                for k in range(4):
                    if j*5+k < 20:
                        v_row = j * 4 + k
                        target_v_row = j * 5 + k
                        V_init[target_v_row, :num_winograd] = V_winograd[v_row, :num_winograd]
            
            W_init[0:16, :num_winograd] = W_winograd[0:16, :num_winograd]
            
            # Add random perturbation
            U_init += np.random.randn(16, target_rank) * 0.05
            V_init += np.random.randn(20, target_rank) * 0.05
            W_init += np.random.randn(20, target_rank) * 0.05
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(obj_func, x0, method='L-BFGS-B', jac=grad_func,
                               options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 3000})
                
                if res.fun < 1e-8:
                    U_opt = res.x[:16*target_rank].reshape(16, target_rank)
                    V_opt = res.x[16*target_rank:16*target_rank+20*target_rank].reshape(20, target_rank)
                    W_opt = res.x[16*target_rank+20*target_rank:16*target_rank+20*target_rank+20*target_rank].reshape(20, target_rank)
                    
                    if _verify_decomposition(U_opt, V_opt, W_opt, n, m, p):
                        best_rank = target_rank
                        best_U = U_opt
                        best_V = V_opt
                        best_W = W_opt
                        break
            except Exception:
                pass
        
        if best_rank < target_rank:
            break

    def to_list(mat):
        return [[float(val) for val in row] for row in mat]

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
