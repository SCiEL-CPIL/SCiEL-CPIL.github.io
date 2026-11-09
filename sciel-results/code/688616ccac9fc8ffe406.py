import time
import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Known rank 17 solution
    rank_17 = 17
    U17 = np.zeros((n * m, rank_17))
    V17 = np.zeros((m * p, rank_17))
    W17 = np.zeros((p * n, rank_17))
    
    # Term 1
    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1
    
    # Term 2
    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[1, 1] = 1; W17[3, 1] = -1
    
    # Term 3
    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1
    
    # Term 4
    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1
    
    # Term 5
    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[0, 4] = -1; W17[2, 4] = 1
    
    # Term 6
    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1
    
    # Term 7
    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1
    
    # Term 8
    U17[0, 7] = 1
    V17[2, 7] = 1
    W17[4, 7] = 1
    
    # Term 9
    U17[1, 8] = 1
    V17[5, 8] = 1
    W17[4, 8] = 1
    
    # Term 10
    U17[3, 9] = 1
    V17[2, 9] = 1
    W17[5, 9] = 1
    
    # Term 11
    U17[4, 10] = 1
    V17[5, 10] = 1
    W17[5, 10] = 1
    
    # Term 12
    U17[2, 11] = 1
    V17[6, 11] = 1
    W17[0, 11] = 1
    
    # Term 13
    U17[2, 12] = 1
    V17[7, 12] = 1
    W17[2, 12] = 1
    
    # Term 14
    U17[2, 13] = 1
    V17[8, 13] = 1
    W17[4, 13] = 1
    
    # Term 15
    U17[5, 14] = 1
    V17[6, 14] = 1
    W17[1, 14] = 1
    
    # Term 16
    U17[5, 15] = 1
    V17[7, 15] = 1
    W17[3, 15] = 1
    
    # Term 17
    U17[5, 16] = 1
    V17[8, 16] = 1
    W17[5, 16] = 1

    # Convert to lists as required
    def to_list(mat):
        return mat.tolist()
    
    # Attempt to find rank 16 solution
    # This is a heuristic search. If it fails, we return rank 17.
    start_time = time.time()
    deadline = start_time + 10.0 # 10 seconds for search
    
    for _ in range(1000):
        if time.time() > deadline:
            break
            
        R = 16
        # Random sparse initialization
        U = np.random.choice([-1, 0, 1], size=(n*m, R))
        V = np.random.choice([-1, 0, 1], size=(m*p, R))
        
        # Build linear system for W
        # Equation: sum_r U[u,r] * V[v,r] * W[w,r] = T[u,v,w]
        # Unknowns: W[w,r] for w in 0..5, r in 0..15. Total 96 unknowns.
        # We can group by w. For each w, the equation is sum_r (sum_u,v U[u,r]V[v,r] delta_{w, target}) ...
        # Actually, for each (u,v), there is at most one w such that T[u,v,w]=1.
        # Let's collect equations.
        
        A_mat = np.zeros((18, 96)) # 18 non-zero entries in tensor
        b_vec = np.zeros(18)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    
                    # Check if this is a valid tensor entry (it always is for this structure)
                    # The coefficient is 1.
                    b_vec[eq_idx] = 1.0
                    for r in range(R):
                        coeff = U[u, r] * V[v, r]
                        # Unknown is W[w, r]. Index in A_mat is w*R + r
                        A_mat[eq_idx, w * R + r] = coeff
                    eq_idx += 1
        
        # Solve least squares
        try:
            W_sol, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            W_sol = W_sol.reshape((p*n, R))
            
            # Check if W_sol is close to integer
            W_int = np.round(W_sol)
            if np.allclose(W_sol, W_int, atol=1e-6):
                # Verify the solution exactly
                # Reconstruct tensor and check
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u = i * m + j
                            v = j * p + k
                            w = k * n + i
                            val = np.sum(U[u, :] * V[v, :] * W_int[w, :])
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if valid:
                    # Also check zeros
                    # We only checked the 1s. We should check that other entries are 0.
                    # But if the linear system was satisfied for the 1s, and the structure is sparse,
                    # it's likely. Let's do a full check.
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            u = i * m + j
                            for k in range(p):
                                v = j * p + k
                                # Calculate contribution to all w
                                contrib = np.sum(U[u, :] * V[v, :] * W_int, axis=1)
                                # The only non-zero should be at w = k*n+i
                                w_target = k * n + i
                                if abs(contrib[w_target] - 1.0) > 1e-9:
                                    valid = False
                                    break
                                # Check others
                                for ww in range(p*n):
                                    if ww != w_target and abs(contrib[ww]) > 1e-9:
                                        valid = False
                                        break
                                if not valid: break
                            if not valid: break
                        if not valid: break
                    
                    if valid:
                        return {"U": to_list(U), "V": to_list(V), "W": to_list(W_int)}
        except:
            pass

    return {"U": to_list(U17), "V": to_list(V17), "W": to_list(W17)}
