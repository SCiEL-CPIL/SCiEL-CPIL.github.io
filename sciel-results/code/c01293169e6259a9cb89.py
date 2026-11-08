import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target multiplication tensor T[u, v, w]
    # u = i*m + j, v = j*p + k, w = k*n + i
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    def check_error(U, V, W):
        """Compute ||sum_r U[u,r]V[v,r]W[w,r] - T||"""
        # Rec[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r]
        Rec = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.linalg.norm(Rec - T)
    
    def solve_one_factor(fixed1, fixed2, target_slices, dim_size):
        """
        Solve for one factor given the other two.
        target_slices: array of shape (dim_size, dim2, dim3) or similar, flattened appropriately.
        Actually, let's pass the full T and the index to slice.
        """
        R = fixed1.shape[1]
        # Precompute the product matrix P
        # If solving for U: fixed1=V, fixed2=W.
        # Equation: sum_r U[u,r] * (V[v,r]*W[w,r]) = T[u,v,w]
        # For fixed u: A @ U[u,:] = b
        # A[i, r] = V[v_i, r] * W[w_i, r] where i indexes (v,w)
        
        # P shape: (dim2, dim3, R)
        P = fixed1[:, np.newaxis, :] * fixed2[np.newaxis, :, :] 
        # A shape: (dim2*dim3, R)
        A = P.reshape(-1, R)
        
        result = np.zeros((dim_size, R), dtype=np.float64)
        for idx in range(dim_size):
            # Get the target slice for this idx
            # This function needs to know how to slice T.
            # Let's refactor to pass the slice directly or a function.
            pass # This logic is embedded in the main loop below for clarity and speed

    def optimize(U, V, W, max_iter=100, tol=1e-8, start_time=0, deadline=0):
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        R = U.shape[1]
        
        for it in range(max_iter):
            if time.time() > deadline - 5: # Safety margin
                break
                
            # Update W: for each w, solve A_w @ W[w,:] = T[:,:,w].reshape(-1)
            # A_w[u*mp+v, r] = U[u,r] * V[v,r]
            P_UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]  # (nm, mp, R)
            A_W = P_UV.reshape(nm * mp, R)
            
            W_new = np.zeros((pn, R), dtype=np.float64)
            for w in range(pn):
                b = T[:, :, w].reshape(-1)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_W, b, rcond=None)
                    W_new[w] = sol
                except np.linalg.LinAlgError:
                    W_new[w] = 0.0
            W = W_new
            
            # Update V: for each v, solve A_v @ V[v,:] = T[:,v,:].reshape(-1)
            # A_v[u*pn+w, r] = U[u,r] * W[w,r]
            P_UW = U[:, np.newaxis, :] * W[np.newaxis, :, :]  # (nm, pn, R)
            A_V = P_UW.reshape(nm * pn, R)
            
            V_new = np.zeros((mp, R), dtype=np.float64)
            for v in range(mp):
                b = T[:, v, :].reshape(-1)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
                    V_new[v] = sol
                except np.linalg.LinAlgError:
                    V_new[v] = 0.0
            V = V_new
            
            # Update U: for each u, solve A_u @ U[u,:] = T[u,:,:].reshape(-1)
            # A_u[v*pn+w, r] = V[v,r] * W[w,r]
            P_VW = V[:, np.newaxis, :] * W[np.newaxis, :, :]  # (mp, pn, R)
            A_U = P_VW.reshape(mp * pn, R)
            
            U_new = np.zeros((nm, R), dtype=np.float64)
            for u in range(nm):
                b = T[u, :, :].reshape(-1)
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_U, b, rcond=None)
                    U_new[u] = sol
                except np.linalg.LinAlgError:
                    U_new[u] = 0.0
            U = U_new
            
            # Check error
            err = check_error(U, V, W)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < tol:
                break
                
        return best_U, best_V, best_W, best_err

    deadline = time.time() + 110 # 10s safety margin
    
    best_solution = None
    best_rank = n * m * p # Fallback to schoolbook rank 48
    
    # Try rank 44
    R_target = 44
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Scaling can matter. 1.0 is usually fine.
        U = np.random.randn(nm, R_target)
        V = np.random.randn(mp, R_target)
        W = np.random.randn(pn, R_target)
        
        # Run optimization
        U, V, W, err = optimize(U, V, W, max_iter=50, tol=1e-8, start_time=time.time(), deadline=deadline)
        
        if err < 1e-5:
            best_solution = (U, V, W)
            best_rank = R_target
            break
            
    if best_solution is None:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1.0
                    V[j * p + k, column] = 1.0
                    W[k * n + i, column] = 1.0
                    column += 1
        best_solution = (U, V, W)
        best_rank = rank

    U, V, W = best_solution
    
    # Convert to list of lists
    # The prompt asks for integer/half-integer entries if possible, but floats are accepted 
    # as long as they exactly decompose. Since we use lstsq, we get floats.
    # We should check if they are close to integers/half-integers and snap them.
    
    def to_list(matrix):
        L = []
        for row in matrix:
            r = []
            for val in row:
                # Check if close to half-integer
                nearest = round(val * 2) / 2.0
                if abs(val - nearest) < 1e-6:
                    r.append(nearest)
                else:
                    r.append(float(val))
            L.append(r)
        return L

    return {
        "U": to_list(U),
        "V": to_list(V),
        "W": to_list(W)
    }
