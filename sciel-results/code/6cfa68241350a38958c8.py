import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Schoolbook fallback
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1

    # Precompute non-zero indices of the tensor T
    # T[i, j, jp, k, ip, kp] = 1 if i==ip, j==jp, k==kp
    # u = i*m + j
    # v = jp*p + k
    # w = kp*n + ip
    # Non-zeros:
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    
    nz_u = []
    nz_v = []
    nz_w = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz_u.append(i*m + j)
                nz_v.append(j*p + k)
                nz_w.append(k*n + i)
                
    nz_u = np.array(nz_u)
    nz_v = np.array(nz_v)
    nz_w = np.array(nz_w)
    
    # Build the sparse tensor representation for ALS
    # We need to solve U, V, W such that sum_c U[:,c] V[:,c] W[:,c] = T
    # This is a CP decomposition.
    
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Try ranks 16, 17, 18
    # 18 is guaranteed, so we try 16, 17 first.
    for rank in [16, 17]:
        if time.time() > deadline - 5:
            break
            
        # Multiple restarts
        for attempt in range(20):
            if time.time() > deadline - 2:
                break
                
            np.random.seed(attempt + rank * 100)
            
            # Initialize
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # ALS iterations
            converged = False
            for it in range(50):
                if time.time() > deadline - 2:
                    break
                
                # Update U
                # T^{(1)} is nm x (mp*pn)
                # We only need the non-zero rows? No, T is zero elsewhere.
                # But the residual is sum over all.
                # However, since T is sparse, we can compute the gradient efficiently?
                # Or just use the dense form for small sizes?
                # nm=6, mp=9, pn=6. Total size 324. Very small.
                
                # Let's use the dense form for simplicity and speed in this small case.
                # T is 6x9x6.
                
                # Reshape U, V, W to 3D for einsum?
                # Or compute the "contraction" matrices.
                
                # To update U:
                # Minimize || T - U (V \odot W)^T ||_F^2
                # Let Z = V \odot W (element-wise product of columns? No.)
                # Z has shape (r, mp*pn).
                # Z[c, v*pn + w] = V[v, c] * W[w, c]
                
                Z = np.zeros((rank, m*p * p*n))
                for c in range(rank):
                    Z[c, :] = (V[:, c] * W[:, c]).flatten() # Wait, V is mp x rank, W is pn x rank.
                    # V[:, c] is mp. W[:, c] is pn.
                    # Product is mp*pn.
                    
                # T_flat is nm x (mp*pn)
                T_flat = np.zeros((n*m, m*p * p*n))
                for idx in range(len(nz_u)):
                    T_flat[nz_u[idx], nz_v[idx]*p*n + nz_w[idx]] = 1.0
                
                # Solve U = T_flat @ Z.T @ inv(Z @ Z.T)
                # Z is rank x (mp*pn).
                # Z.T is (mp*pn) x rank.
                # T_flat is nm x (mp*pn).
                # T_flat @ Z.T is nm x rank.
                # Z @ Z.T is rank x rank.
                
                try:
                    ZZT = Z @ Z.T
                    # Add small regularization for stability
                    ZZT += 1e-10 * np.eye(rank)
                    U_new = T_flat @ Z.T @ np.linalg.inv(ZZT)
                except np.linalg.LinAlgError:
                    U_new = U
                    
                # Update V
                Z2 = np.zeros((rank, n*m * p*n))
                for c in range(rank):
                    Z2[c, :] = (U[:, c] * W[:, c]).flatten()
                    
                T_flat_v = np.zeros((m*p, n*m * p*n))
                for idx in range(len(nz_u)):
                    T_flat_v[nz_v[idx], nz_u[idx]*p*n + nz_w[idx]] = 1.0
                
                try:
                    Z2Z2T = Z2 @ Z2.T
                    Z2Z2T += 1e-10 * np.eye(rank)
                    V_new = T_flat_v @ Z2.T @ np.linalg.inv(Z2Z2T)
                except np.linalg.LinAlgError:
                    V_new = V
                    
                # Update W
                Z3 = np.zeros((rank, n*m * m*p))
                for c in range(rank):
                    Z3[c, :] = (U[:, c] * V[:, c]).flatten()
                    
                T_flat_w = np.zeros((p*n, n*m * m*p))
                for idx in range(len(nz_u)):
                    T_flat_w[nz_w[idx], nz_u[idx]*m*p + nz_v[idx]] = 1.0
                
                try:
                    Z3Z3T = Z3 @ Z3.T
                    Z3Z3T += 1e-10 * np.eye(rank)
                    W_new = T_flat_w @ Z3.T @ np.linalg.inv(Z3Z3T)
                except np.linalg.LinAlgError:
                    W_new = W
                    
                # Check convergence
                diff_U = np.linalg.norm(U_new - U)
                diff_V = np.linalg.norm(V_new - V)
                diff_W = np.linalg.norm(W_new - W)
                
                U, V, W = U_new, V_new, W_new
                
                if diff_U + diff_V + diff_W < 1e-8:
                    converged = True
                    break
                    
            if not converged:
                continue
                
            # Verify and round
            # Compute residual
            # T_pred = sum_c U[:,c] V[:,c] W[:,c]
            # We can check the non-zero entries.
            
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Verify
            if verify(U_r, V_r, W_r, n, m, p):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                
            # Try integer rounding
            U_i = np.round(U)
            V_i = np.round(V)
            W_i = np.round(W)
            if verify(U_i, V_i, W_i, n, m, p):
                return {"U": U_i.tolist(), "V": V_i.tolist(), "W": W_i.tolist()}
                
    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

def verify(U, V, W, n, m, p):
    # U: (nm, r), V: (mp, r), W: (pn, r)
    r = U.shape[1]
    # Check half-integers
    if not np.allclose(U * 2, np.round(U * 2)): return False
    if not np.allclose(V * 2, np.round(V * 2)): return False
    if not np.allclose(W * 2, np.round(W * 2)): return False
    
    # Check tensor identity
    # T[i, j, jp, k, ip, kp] = sum_c U[i*m+j, c] V[jp*p+k, c] W[kp*n+ip, c]
    # We only need to check the 18 non-zero entries and a sample of zeros?
    # Or check all 324 entries (small).
    
    U_reshaped = np.array(U).reshape(n, m, r)
    V_reshaped = np.array(V).reshape(m, p, r)
    W_reshaped = np.array(W).reshape(p, n, r)
    
    T_pred = np.einsum('ijc,jkc,kpc->ijkkpc', U_reshaped, V_reshaped, W_reshaped, optimize=True)
    
    # Check non-zeros
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not np.isclose(T_pred[i, j, j, k, i, k], 1.0, atol=1e-9):
                    return False
                
    # Check zeros (sample or all)
    # Since size is small, check all
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            if (i, j, k) != (ip, jp, kp):
                                if not np.isclose(T_pred[i, j, jp, k, ip, kp], 0.0, atol=1e-9):
                                    return False
    return True
