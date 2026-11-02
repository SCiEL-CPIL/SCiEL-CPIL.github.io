import numpy as np
import time
import random

DIMENSIONS = [2, 2, 3]

def get_target_tensor():
    n, m, p = DIMENSIONS
    # U dimension: n*m = 4
    # V dimension: m*p = 6
    # W dimension: p*n = 6
    
    # Tensor T of shape (4, 6, 6)
    # T[u, v, w] is 1 if A_u * B_v contributes to C_w
    # A index u: i * m + k (0-based) -> i in 0..1, k in 0..1
    # B index v: k * p + j (0-based) -> k in 0..1, j in 0..2
    # C index w: i * p + j ? No, the prompt W convention was column major for C?
    # Let's re-read the prompt W convention: "W[k*n+i,r]".
    # In the code: W indices 0,1,2,3,4,5 correspond to C:
    # 0: c11, 1: c21, 2: c12, 3: c22, 4: c13, 5: c23.
    # This is index = i + j*n.
    # i is row of C (0,1), j is col of C (0,1,2).
    # So W index w = i + j*n.
    
    T = np.zeros((n*m, m*p, p*n))
    
    # Loop over all combinations of A index, B index, C index
    for i in range(n):
        for j in range(p):
            # C index
            w = i + j * n
            
            # Sum over k
            for k in range(m):
                # A index: row i, col k
                u = i * m + k
                # B index: row k, col j
                v = k * p + j
                
                T[u, v, w] += 1
                
    return T

def compute_error(U, V, W, T):
    # U: (nm, R), V: (mp, R), W: (pn, R)
    # Reconstruction: sum_r U[:,r] V[:,r]^T W[:,r]^T
    # This is a sum of outer products.
    # We can compute the difference T - Rec.
    # T is (nm, mp, pn).
    # Rec[u, v, w] = sum_r U[u,r] V[v,r] W[w,r]
    
    # Use einsum for efficiency
    # rec = np.einsum('ur, vr, wr -> uvw', U, V, W)
    # error = np.linalg.norm(T - rec)
    
    # For speed in optimization, we can return the squared error
    rec = np.einsum('ur, vr, wr -> uvw', U, V, W)
    diff = T - rec
    return np.sum(diff**2)

def construct():
    n, m, p = DIMENSIONS
    T = get_target_tensor()
    
    # Default Rank 11 Solution
    def get_rank_11():
        rank = 11
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Indices mapping from prompt code:
        # U: 0:a11, 1:a12, 2:a21, 3:a22
        # V: 0:b11, 1:b12, 2:b13, 3:b21, 4:b22, 5:b23
        # W: 0:c11, 1:c21, 2:c12, 3:c22, 4:c13, 5:c23
        
        # Col 0: m1
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[4,0] = 1
        W[0,0] = 1; W[3,0] = 1
        
        # Col 1: m2
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[1,1] = 1; W[3,1] = -1
        
        # Col 2: m3
        U[0,2] = 1
        V[1,2] = 1; V[4,2] = -1
        W[2,2] = 1; W[3,2] = 1
        
        # Col 3: m4
        U[3,3] = 1
        V[3,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[1,3] = 1
        
        # Col 4: m5
        U[0,4] = 1; U[1,4] = 1
        V[4,4] = 1
        W[0,4] = -1; W[2,4] = 1
        
        # Col 5: m6
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1
        
        # Col 6: m7
        U[1,6] = 1; U[3,6] = -1
        V[3,6] = 1; V[4,6] = 1
        W[0,6] = 1
        
        # Col 7: a11*b13
        U[0,7] = 1
        V[2,7] = 1
        W[4,7] = 1
        
        # Col 8: a12*b23
        U[1,8] = 1
        V[5,8] = 1
        W[4,8] = 1
        
        # Col 9: a21*b13
        U[2,9] = 1
        V[2,9] = 1
        W[5,9] = 1
        
        # Col 10: a22*b23
        U[3,10] = 1
        V[5,10] = 1
        W[5,10] = 1
        
        return U, V, W

    U_def, V_def, W_def = get_rank_11()
    
    # Check if Rank 10 is possible within time budget
    # We will try to optimize for Rank 10
    best_rank = 11
    best_U, best_V, best_W = U_def, V_def, W_def
    
    # Time limit 150s, keep margin for overhead
    start_time = time.time()
    time_limit = 145.0
    
    try:
        import scipy.optimize as opt
        
        target_rank = 10
        nm, mp, pn = n*m, m*p, p*n
        num_vars = nm*target_rank + mp*target_rank + pn*target_rank
        
        # We will try a few random restarts
        max_restarts = 20
        
        for r in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)
            
            # Flatten
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Objective function
            def objective(x):
                U_flat, V_flat, W_flat = x[:nm*target_rank], x[nm*target_rank:nm*target_rank+mp*target_rank], x[nm*target_rank+mp*target_rank:]
                U_mat = U_flat.reshape(nm, target_rank)
                V_mat = V_flat.reshape(mp, target_rank)
                W_mat = W_flat.reshape(pn, target_rank)
                return compute_error(U_mat, V_mat, W_mat, T)
            
            # Use Powell method (derivative-free)
            res = opt.minimize(objective, x0, method='Powell', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < 1e-6:
                # Found a valid rank 10 solution
                U_sol = res.x[:nm*target_rank].reshape(nm, target_rank)
                V_sol = res.x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                W_sol = res.x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                
                # Round to nearest integer or half-integer?
                # The problem asks for integer/half-integer entries.
                # If the solution is close to integers, we can snap them.
                # Let's check if snapping maintains validity.
                U_snapped = np.round(U_sol * 2) / 2.0
                V_snapped = np.round(V_sol * 2) / 2.0
                W_snapped = np.round(W_sol * 2) / 2.0
                
                if compute_error(U_snapped, V_snapped, W_snapped, T) < 1e-9:
                    return {
                        "U": U_snapped.tolist(),
                        "V": V_snapped.tolist(),
                        "W": W_snapped.tolist()
                    }
                else:
                    # If snapping fails, maybe the solution is not exactly integer/half-integer?
                    # But the problem statement implies such a solution exists.
                    # If not, we cannot return non-integer.
                    # We'll stick to the rank 11 for now if snapping fails.
                    pass

    except Exception as e:
        pass
        
    return {
        "U": U_def.tolist(),
        "V": V_def.tolist(),
        "W": W_def.tolist()
    }
