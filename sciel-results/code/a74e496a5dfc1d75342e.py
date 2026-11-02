import numpy as np
import time
import scipy.optimize as opt

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def try_rank(R, seed, maxiter=200, popsize=15):
        if time.time() > deadline - 20:
            return None
        
        def objective(params):
            U = params[:nm*R].reshape(nm, R)
            V = params[nm*R:nm*R+mp*R].reshape(mp, R)
            W = params[nm*R+mp*R:].reshape(pn, R)
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            res = T_calc - T
            return np.sum(res**2)
        
        bounds = [(-3, 3)] * (nm*R + mp*R + pn*R)
        
        try:
            sol = opt.differential_evolution(
                objective, 
                bounds, 
                seed=seed, 
                maxiter=maxiter, 
                tol=1e-12, 
                polish=True,
                update_interval=10,
                popsize=popsize,
                mutation=(0.5, 1.0),
                recombination=0.8
            )
            
            if sol.fun < 1e-10:
                U = sol.x[:nm*R].reshape(nm, R)
                V = sol.x[nm*R:nm*R+mp*R].reshape(mp, R)
                W = sol.x[nm*R+mp*R:].reshape(pn, R)
                
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                if np.allclose(T_check, T, atol=1e-8):
                    return (U_r, V_r, W_r)
                
                # Try rounding to integers
                U_r2 = np.round(U)
                V_r2 = np.round(V)
                W_r2 = np.round(W)
                
                T_check2 = np.einsum('ar,br,cr->abc', U_r2, V_r2, W_r2)
                if np.allclose(T_check2, T, atol=1e-8):
                    return (U_r2, V_r2, W_r2)
        except Exception:
            pass
        return None
    
    # Try rank 20
    for seed in range(10):
        if time.time() > deadline - 40:
            break
        result = try_rank(20, seed, maxiter=150, popsize=12)
        if result:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 21 using Strassen
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]
    
    U = np.zeros((nm, 21))
    V = np.zeros((mp, 21))
    W = np.zeros((pn, 21))
    
    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            for l in range(4):
                i = l // 2
                j = l % 2
                a = i * m + j
                U[a, r] = step['u'][l]
                
                # For each k in this group
                for ki in range(2):
                    k = 2 * g + ki
                    b = j * p + k
                    c = k * n + i
                    # The Strassen step's v and w are for the 2x2 case
                    # We need to map the 2x2 Strassen to this sub-problem
                    # In the 2x2 sub-problem, the k-index is 0 or 1
                    # v[l] corresponds to (j, k_sub) where k_sub is 0 or 1
                    # But Strassen's v is indexed by (j, k_sub) for the 2x2 case
                    # Actually, the Strassen steps are for the full 2x2x2 tensor
                    # So v[l] is the coefficient for B[j, k_sub] where k_sub depends on l
                    # This is getting complicated. Let me use a different approach.
                    pass
    
    # Simpler: just use the known rank-21 construction
    # For each pair of k-values, use Strassen
    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            k0 = 2 * g
            k1 = 2 * g + 1
            for l in range(4):
                i = l // 2
                j = l % 2
                a = i * m + j
                U[a, r] = step['u'][l]
                
                # v: B[j, k] for k in {k0, k1}
                # In the 2x2 sub-problem, k_sub=0 maps to k0, k_sub=1 maps to k1
                # Strassen's v[l] is for the 2x2 B matrix
                # The 2x2 B matrix has entries B[j, k_sub] for j,k_sub in {0,1}
                # So v[l] corresponds to B[j, k_sub] where k_sub is determined by l
                # Actually, in the standard 2x2x2 tensor, the b-index is j*2+k_sub
                # So for l, the b-index in the sub-problem is l (since a=i*2+j=l)
                # Wait, that's not right either.
                
                # Let me just directly construct it
                # For the sub-problem with k in {k0, k1}:
                # The tensor is T_sub[i, j, k_sub] = 1 for i,j,k_sub in {0,1}
                # Strassen gives 7 terms for this
                # Term s has: u_s[i,j], v_s[j,k_sub], w_s[k_sub,i]
                # In the full tensor, this becomes:
                # U[a, r] = u_s[i,j] where a = i*2+j
                # V[b, r] = v_s[j,k_sub] where b = j*6+k, k = k0+k_sub
                # W[c, r] = w_s[k_sub,i] where c = k*2+i, k = k0+k_sub
                
                # But Strassen's steps are given as 4-element vectors for u, v, w
                # u[l] is for the l-th (i,j) pair
                # v[l] is for the l-th (j,k_sub) pair? No, that doesn't make sense.
                
                # Let me look at the Strassen steps more carefully.
                # The 2x2x2 multiplication tensor has 8 non-zero entries.
                # Strassen's 7 terms each have support on a subset of these 8 entries.
                # The u, v, w vectors are 4-element each, corresponding to the 4 entries
                # of the 2x2 matrices A, B, C.
                
                # A has entries A[i,j] for i,j in {0,1}, indexed as i*2+j
                # B has entries B[j,k] for j,k in {0,1}, indexed as j*2+k
                # C has entries C[i,k] for i,k in {0,1}, indexed as i*2+k (or k*2+i?)
                
                # The problem uses C[k,i] = k*2+i, so C is indexed as k*2+i
                # But standard matrix multiplication gives C[i,k] = sum_j A[i,j]B[j,k]
                # So C[i,k] corresponds to index i*2+k in row-major, or k*2+i if we think of it as C[k,i]
                
                # I think the Strassen steps are defined for the standard 2x2x2 tensor
                # where T[i,j,k] = 1, and the matrices are:
                # A[i,j] (2x2), B[j,k] (2x2), C[i,k] = sum_j A[i,j]B[j,k]
                # With C indexed as i*2+k (row-major)
                
                # But our problem has C indexed as k*2+i, which is C[k,i]
                # So C[k,i] = C[i,k] in the standard sense? No, C[k,i] is the (k,i) entry
                # which is not the same as C[i,k] unless the matrix is symmetric.
                
                # I think there's a mismatch. Let me just use the known rank-21
                # construction that's already working in the current program.
    
    # Use the working rank-21 construction from the current program
    U = np.zeros((nm, 21))
    V = np.zeros((mp, 21))
    W = np.zeros((pn, 21))
    
    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            for l in range(4):
                U[l, r] = step['u'][l]
                k0 = 2 * g
                k1 = 2 * g + 1
                b_map = [6*0+k0, 6*0+k1, 6*1+k0, 6*1+k1]
                V[b_map[l], r] = step['v'][l]
                c_map = [2*k0+0, 2*k1+0, 2*k0+1, 2*k1+1]
                W[c_map[l], r] = step['w'][l]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
