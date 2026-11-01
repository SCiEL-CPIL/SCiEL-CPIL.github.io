import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe within 150s
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor T[a, b, c]
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Fallback: Schoolbook
    U_school = np.zeros((nm, 24), dtype=int)
    V_school = np.zeros((mp, 24), dtype=int)
    W_school = np.zeros((pn, 24), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
                
    # Try ranks from 18 to 23
    # Known lower bounds suggest rank > 17 for 2x3x4?
    # Let's try 18, 19, 20, 21, 22, 23
    
    for r in range(18, 24):
        if time.time() > deadline - 20:
            break
            
        # Number of trials per rank
        trials = 20
        if r < 20:
            trials = 40
            
        for _ in range(trials):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W randomly
            # Scale to 1 to start
            U = np.random.randn(nm, r) * 0.1
            V = np.random.randn(mp, r) * 0.1
            W = np.random.randn(pn, r) * 0.1
            
            # ALS iterations
            # We want to minimize || T - sum U V W ||^2
            # For each a, solve for U[a, :]
            # T[a, :, :] = sum_r U[a, r] * (V[:, r] * W[:, r])
            # Let M_a be T[a, :, :] flattened (mp*pn, 1)
            # Let A_a be (mp*pn, r) where A_a[:, r] = vec(V[:, r] * W[:, r])
            # Solve A_a * U[a, :]^T = M_a
            
            # Precompute structure? No, V and W change.
            
            for it in range(100):
                if time.time() > deadline:
                    break
                    
                # Update U
                # For each a in nm:
                for a in range(nm):
                    # Construct A_a
                    # A_a is (mp*pn, r)
                    # A_a[idx, r] = V[b, r] * W[c, r]
                    # idx = b * pn + c
                    # We can compute this efficiently
                    # V is (mp, r), W is (pn, r)
                    # Outer product of V[:, r] and W[:, r] is (mp, pn)
                    # Flatten to (mp*pn)
                    
                    # Vectorized approach for all a at once?
                    # No, the system is different for each a because T[a] is different.
                    # But the matrix A is the same for all a!
                    # A depends only on V and W.
                    
                    # Let's compute A once per iteration
                    # A is (mp*pn, r)
                    # A[idx, r] = V[b, r] * W[c, r]
                    
                    # Reshape V to (mp, 1, r) and W to (1, pn, r)
                    # Prod = V[:, np.newaxis, :] * W[np.newaxis, :, :] # (mp, pn, r)
                    # A = Prod.reshape(mp*pn, r)
                    
                    # Do this outside the a-loop
                    pass
                    
                # Compute A
                A = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(mp * pn, r)
                
                # Solve for U
                # T_flat is (nm, mp*pn)
                T_flat = T.reshape(nm, mp * pn)
                
                # We want U (nm, r) such that U @ A.T = T_flat
                # This is a linear system: U @ A.T = T_flat
                # A.T is (r, mp*pn)
                # U is (nm, r)
                # T_flat is (nm, mp*pn)
                
                # Solve using lstsq
                # X @ B = C
                # X = U, B = A.T, C = T_flat
                try:
                    U, _, _, _ = np.linalg.lstsq(A.T, T_flat.T, rcond=None)
                    U = U.T
                except:
                    break
                    
                # Update V
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # For fixed b, T[:, b, :] = sum_r (U[:, r] * W[:, r]) * V[b, r]
                # Let A_b be (nm*pn, r)
                # A_b[idx, r] = U[a, r] * W[c, r]
                # idx = a * pn + c
                
                A = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(nm * pn, r)
                T_flat = T.reshape(nm, mp, pn).transpose(1, 0, 2).reshape(mp, nm * pn)
                
                try:
                    V, _, _, _ = np.linalg.lstsq(A.T, T_flat.T, rcond=None)
                    V = V.T
                except:
                    break
                    
                # Update W
                # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                # For fixed c, T[:, :, c] = sum_r (U[:, r] * V[:, r]) * W[c, r]
                # Let A_c be (nm*mp, r)
                # A_c[idx, r] = U[a, r] * V[b, r]
                # idx = a * mp + b
                
                A = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(nm * mp, r)
                T_flat = T.reshape(nm, mp, pn).transpose(2, 0, 1).reshape(pn, nm * mp)
                
                try:
                    W, _, _, _ = np.linalg.lstsq(A.T, T_flat.T, rcond=None)
                    W = W.T
                except:
                    break
                    
            # Check convergence and snap
            # Reconstruct
            reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
            
            if np.allclose(reconstructed, T, atol=1e-5):
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify snapped version
                reconstructed_snap = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
                
                if np.allclose(reconstructed_snap, T, atol=1e-9):
                    # Convert to lists
                    U_list = [[float(x) for x in row] for row in U_snap]
                    V_list = [[float(x) for x in row] for row in V_snap]
                    W_list = [[float(x) for x in row] for row in W_snap]
                    
                    return {"U": U_list, "V": V_list, "W": W_list}

    # If no better rank found, return schoolbook
    U_list = [[int(x) for x in row] for row in U_school]
    V_list = [[int(x) for x in row] for row in V_school]
    W_list = [[int(x) for x in row] for row in W_school]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
