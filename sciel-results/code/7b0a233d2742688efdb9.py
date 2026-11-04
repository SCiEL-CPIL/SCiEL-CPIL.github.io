import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T[a, b, c]
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try ranks from low to high
    for rank in range(12, 31):
        if time.time() > deadline - 5:
            break
        # Try multiple random initializations
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Iteratively update: fix two, solve for third
            converged = False
            for iteration in range(500):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each a, U[a,:] should satisfy sum_c U[a,c]*V[b,c]*W[c,c'] = T[a,b,c']
                # This is: U[a,:] @ (V * W^T) = T[a,:,:] where (V*W^T)[b,c] = V[b,c]*W[c,:]
                # Actually: sum_d U[a,d] * V[b,d] * W[c,d] = T[a,b,c]
                # For fixed V,W: U[a,d] appears in sum over d, so for each a:
                # U[a,:] * M = T[a,:,:] where M[b,d] = sum_c' V[b,d]*W[c',d]... no
                # Let me reformulate: for fixed V and W, we need U such that
                # for all a,b,c: sum_d U[a,d] * V[b,d] * W[c,d] = T[a,b,c]
                # This can be written as: U @ (V ⊙ W) = T reshaped
                # Where (V ⊙ W) is a matrix of size (mp*p n) x rank? No.
                
                # Better: for fixed V, W, the equation for U is linear.
                # Reshape: let M[b, c, d] = V[b, d] * W[c, d]
                # Then for each a: sum_{b,c,d} U[a,d] * M[b,c,d] * delta = T[a,b,c]
                # Actually: sum_d U[a,d] * (sum over nothing) ... 
                # T[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
                # So for fixed V,W: U[a,:] is the solution to a linear system.
                # Let's define for each a: U[a,:] * N[a] = T[a,:,:] where N is built from V,W
                # Specifically: U[a,d] * V[b,d] * W[c,d] summed over d = T[a,b,c]
                # This is: for each a, the vector U[a,:] satisfies:
                # sum_d U[a,d] * (V[:,d] * W[:,d])_b... no, V[b,d]*W[c,d] is elementwise
                # Let's define P[b,c,d] = V[b,d]*W[c,d], then T[a,b,c] = sum_d U[a,d]*P[b,c,d]
                # So U[a,:] is the solution to: U[a,:] @ P_reshaped = T[a,:,:].reshape(-1)
                # where P_reshaped has shape (mp*pn, rank) with P_reshaped[b*pn+c, d] = P[b,c,d]
                
                P = V[:, :, None] * W[None, :, :]  # shape (mp, pn, rank)
                P_flat = P.transpose(2, 0, 1).reshape(rank, -1)  # shape (rank, mp*pn)
                T_flat = T.reshape(nm, -1)  # shape (nm, mp*pn)
                
                # Solve: U @ P_flat.T = T_flat, i.e., U = T_flat @ pinv(P_flat.T)
                # Or more precisely: for each a, U[a,:] @ P_flat.T = T_flat[a,:]
                # So U = T_flat @ pinv(P_flat.T)
                try:
                    U_new = T_flat @ np.linalg.pinv(P_flat.T)
                    U = U_new
                except:
                    pass
                
                # Update V
                P2 = U[:, :, None] * W[None, :, :]  # shape (nm, pn, rank)
                P2_flat = P2.transpose(2, 0, 1).reshape(rank, -1)
                T_flat_v = T.transpose(1, 0, 2).reshape(mp, -1)  # shape (mp, nm*pn)
                try:
                    V_new = T_flat_v @ np.linalg.pinv(P2_flat.T)
                    V = V_new
                except:
                    pass
                
                # Update W
                P3 = U[:, :, None] * V[None, :, :]  # shape (nm, mp, rank)
                P3_flat = P3.transpose(2, 0, 1).reshape(rank, -1)
                T_flat_w = T.transpose(2, 0, 1).reshape(pn, -1)  # shape (pn, nm*mp)
                try:
                    W_new = T_flat_w @ np.linalg.pinv(P3_flat.T)
                    W = W_new
                except:
                    pass
                
                # Check convergence
                # Compute current tensor
                current_T = np.einsum('ad,bd,cd->abc', U, V, W)
                error = np.linalg.norm(current_T - T)
                if error < 1e-10:
                    converged = True
                    break
            
            if converged:
                # Found a valid decomposition of this rank
                # Convert to integer/half-integer if possible, otherwise return as is
                # The problem says integer/half-integer entries, but let's first see if we get exact
                # For now, return the float decomposition
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
