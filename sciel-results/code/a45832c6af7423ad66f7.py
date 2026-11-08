"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]
import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    nm = n * m  # 6
    mp = m * p  # 15
    pn = p * n  # 10
    
    # Build target tensor T[a,b,c]
    # a = i*m + j (i in [0,n), j in [0,m))
    # b = j*p + k (j in [0,m), k in [0,p))
    # c = k*n + i (k in [0,p), i in [0,n))
    # T[a,b,c] = 1 if i_a == i_c and j_a == j_b and k_b == k_c, else 0
    
    def tensor_entry(a, b, c):
        i_a = a // m
        j_a = a % m
        j_b = b // p
        k_b = b % p
        k_c = c // n
        i_c = c % n
        if i_a == i_c and j_a == j_b and k_b == k_c:
            return 1
        return 0
    
    # Try to find low-rank decomposition using iterative refinement
    # Start with rank 18 and try to converge
    
    def try_rank(r, max_iter=2000, seed=42):
        """Try to find rank-r decomposition via ALS-style iteration."""
        if time.time() > deadline - 10.0:
            return None
        
        np.random.seed(seed)
        
        # Initialize
        U = np.random.randn(nm, r) * 0.5
        V = np.random.randn(mp, r) * 0.5
        W = np.random.randn(pn, r) * 0.5
        
        # Precompute target tensor as 3D array
        T = np.zeros((nm, mp, pn))
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    T[a, b, c] = tensor_entry(a, b, c)
        
        for it in range(max_iter):
            if time.time() > deadline - 10.0:
                return None
            
            # Update U: for each a, U[a,:] should satisfy
            # sum_c' U[a,c'] * V[b,c'] * W[c,c'] = T[a,b,c] for all b,c
            # This is: U[a,:] @ (V[b,:] * W[c,:]) = T[a,b,c]
            # For fixed a, this is a linear system in U[a,:]
            # Let M_a[b,c] = sum_c' V[b,c'] * W[c,c']  (this is V @ W^T but element-wise product... no)
            # Actually: sum_c' V[b,c'] * W[c,c'] is not a standard matrix product.
            # Let me reshape: for fixed a, we need U[a,:] such that
            # (V * W) contracted with U[a,:] gives T[a,:,:]
            # where (V*W)[b,c,c'] = V[b,c']*W[c,c']
            # So the "matrix" for U[a,:] is: M_a[b,c] = sum over... no.
            # Let me think again.
            # T[a,b,c] = sum_{c'} U[a,c'] * V[b,c'] * W[c,c']
            # = sum_{c'} U[a,c'] * (V[b,c'] * W[c,c'])
            # So if I define for each c': a matrix M_{c'}[b,c] = V[b,c']*W[c,c']
            # Then T[a,:,:] = sum_{c'} U[a,c'] * M_{c'}
            # This is a linear system: [M_0, M_1, ..., M_{r-1}] * U[a,:]^T = T[a,:,:]
            # where [M_0,...,M_{r-1}] is a (mp*pn) x r matrix
            
            # Build the design matrix for U
            # D_U[b*pn+c, c'] = V[b,c'] * W[c,c']
            D_U = np.zeros((mp * pn, r))
            for b in range(mp):
                for c in range(pn):
                    for c_prime in range(r):
                        D_U[b * pn + c, c_prime] = V[b, c_prime] * W[c, c_prime]
            
            # Solve for U[a,:] for each a
            target_flat = T.reshape(nm, mp * pn)
            # U_new[a,:] = solve(D_U, target_flat[a,:])
            try:
                U_new, residuals, rank_d, sv = np.linalg.lstsq(D_U, target_flat.T, rcond=None)
                U = U_new.T  # Now U is (nm, r)
            except:
                pass
            
            # Update V: for each b, V[b,:] should satisfy
            # sum_c' U[a,c'] * V[b,c'] * W[c,c'] = T[a,b,c] for all a,c
            # D_V[a*pn+c, c'] = U[a,c'] * W[c,c']
            D_V = np.zeros((nm * pn, r))
            for a in range(nm):
                for c in range(pn):
                    for c_prime in range(r):
                        D_V[a * pn + c, c_prime] = U[a, c_prime] * W[c, c_prime]
            
            target_flat_v = T.reshape(nm, mp, pn).transpose(1, 0, 2).reshape(mp, nm * pn)
            try:
                V_new, residuals, rank_d, sv = np.linalg.lstsq(D_V, target_flat_v.T, rcond=None)
                V = V_new.T
            except:
                pass
            
            # Update W: for each c, W[c,:] should satisfy
            # sum_c' U[a,c'] * V[b,c'] * W[c,c'] = T[a,b,c] for all a,b
            # D_W[a*mp+b, c'] = U[a,c'] * V[b,c']
            D_W = np.zeros((nm * mp, r))
            for a in range(nm):
                for b in range(mp):
                    for c_prime in range(r):
                        D_W[a * mp + b, c_prime] = U[a, c_prime] * V[b, c_prime]
            
            target_flat_w = T.reshape(nm, mp, pn).transpose(2, 0, 1).reshape(pn, nm * mp)
            try:
                W_new, residuals, rank_d, sv = np.linalg.lstsq(D_W, target_flat_w.T, rcond=None)
                W = W_new.T
            except:
                pass
            
            # Check convergence
            if it % 50 == 0:
                # Compute residual
                res = 0.0
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            val = 0.0
                            for c_prime in range(r):
                                val += U[a, c_prime] * V[b, c_prime] * W[c, c_prime]
                            res += (val - T[a, b, c]) ** 2
                if res < 1e-12:
                    break
        
        # Verify
        res = 0.0
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    val = 0.0
                    for c_prime in range(r):
                        val += U[a, c_prime] * V[b, c_prime] * W[c, c_prime]
                    res += (val - T[a, b, c]) ** 2
        
        if res < 1e-10:
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 15 upward
    for r in range(15, 31):
        if time.time() > deadline - 15.0:
            break
        
        result = try_rank(r)
        if result is not None:
            return result
    
    # Fallback: schoolbook
    rank = nm * mp // m  # = n*m*p = 30
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
