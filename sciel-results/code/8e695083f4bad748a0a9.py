"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T[a, b, c]
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find a low-rank decomposition
    # Start with schoolbook rank and try to reduce
    best_result = None
    best_rank = nm * mp * pn // p  # schoolbook
    deadline = time.time() + 140  # leave 10s margin
    
    # Try various ranks from low to high
    # Lower bound is max(nm, mp, pn) = max(6, 21, 14) = 21
    lower_bound = max(nm, mp, pn)
    
    for rank in range(lower_bound, 43):
        if time.time() > deadline - 30:
            break
        if best_result is not None and rank >= best_rank:
            break
            
        # Try to find a decomposition of this rank
        # Use iterative method: initialize with random, then optimize
        found = False
        for trial in range(50):
            if time.time() > deadline - 10:
                break
                
            np.random.seed(rank * 1000 + trial)
            
            # Initialize
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Iteratively optimize
            for it in range(2000):
                if time.time() > deadline - 5:
                    break
                    
                # Update U: for each a, minimize sum_b,c T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r]
                # This is linear in U[a,:] given V and W
                # For each a: we want sum_r U[a,r] * (sum_b,c T[a,b,c]*V[b,r]*W[c,r]...) 
                # Actually, the objective is sum_{a,b,c} (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
                # Gradient wrt U[a,r] = -2 * sum_{b,c} T[a,b,c] * V[b,r] * W[c,r] + 2 * sum_{b,c} (sum_{r'} U[a,r']*V[b,r']*W[c,r']) * V[b,r] * W[c,r]
                
                # Let's use a simpler approach: fix two matrices, solve for the third
                # Fix V and W, solve for U:
                # For each a, we want: sum_r U[a,r] * M[a,r] = T_row[a] where M depends on V,W
                # Actually this is not linear in a simple way.
                
                # Let me use a different approach: alternating least squares style
                # Fix V and W, for each a, we need:
                # sum_r U[a,r] * (sum over b,c of V[b,r]*W[c,r]*delta(T[a,b,c])) ... 
                
                # This is getting complicated. Let me use a different strategy.
                # 
                # Alternative: use the fact that we can express the problem as:
                # For each (a,b): sum_r U[a,r]*V[b,r] should relate to T[a,b,:]
                # Specifically, for fixed a,b: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all c
                # This means the vector (sum_r U[a,r]*V[b,r]*W[:,r]) = T[a,b,:]
                
                # Let's define for each pair (a,b): w_ab = sum_r U[a,r]*V[b,r] * W[:,r]
                # We need w_ab = T[a,b,:]
                
                # This is still non-linear. Let me just try a gradient descent approach.
                
                # Compute current reconstruction
                # R[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # Use einsum for efficiency
                R = np.einsum('ar,br,cr->abc', U, V, W)
                
                err = np.sum((R - T)**2)
                if err < 1e-10:
                    found = True
                    break
                
                # Gradient descent
                # dL/dU[a,r] = -2 * sum_{b,c} (T[a,b,c] - R[a,b,c]) * V[b,r] * W[c,r]
                # dL/dV[b,r] = -2 * sum_{a,c} (T[a,b,c] - R[a,b,c]) * U[a,r] * W[c,r]
                # dL/dW[c,r] = -2 * sum_{a,b} (T[a,b,c] - R[a,b,c]) * U[a,r] * V[b,r]
                
                diff = T - R  # (nm, mp, pn)
                
                # dU[a,r] = sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
                dU = np.einsum('abc,br,cr->ar', diff, V, W)
                dV = np.einsum('abc,ar,cr->br', diff, U, W)
                dW = np.einsum('abc,ar,br->cr', diff, U, V)
                
                # Adaptive learning rate
                lr = 0.01 / (1 + it * 0.001)
                U += lr * dU
                V += lr * dV
                W += lr * dW
                
                if it % 200 == 0 and err < 1e-6:
                    # Try to verify
                    R_check = np.einsum('ar,br,cr->abc', U, V, W)
                    if np.max(np.abs(R_check - T)) < 1e-8:
                        found = True
                        break
            
            if found:
                best_rank = rank
                best_result = (U.copy(), V.copy(), W.copy())
                break
        
        if found:
            break
    
    if best_result is None:
        # Fall back to schoolbook
        rank = nm * mp // p  # = n*m*p / p... no, schoolbook is n*m*p
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W = best_result
    # Convert to list of lists with rounded values
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
