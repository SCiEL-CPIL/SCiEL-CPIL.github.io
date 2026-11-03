import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute indices for the 90 terms
    # i in [0, n), j in [0, m), k in [0, p)
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i, j, k))
    
    N = len(indices) # 90
    
    # We want to find U, V, W such that sum_r U[i,j,r] V[j,k,r] W[i,k,r] = 1
    # U shape: (n*m, R)
    # V shape: (m*p, R)
    # W shape: (p*n, R) -> Note: prompt says W[k*n+i, r], so index is k*n+i.
    
    # Helper to verify
    def verify(U, V, W):
        # Check if entries are integer or half-integer
        # And if the sum is exactly 1 (within float precision, then check exact rational if possible)
        # For simplicity, check if close to 1 and values are close to half-integers.
        if not np.allclose(U * 2, np.round(U * 2)): return False
        if not np.allclose(V * 2, np.round(V * 2)): return False
        if not np.allclose(W * 2, np.round(W * 2)): return False
        
        # Snap to half-integers
        U_snap = np.round(U * 2) / 2.0
        V_snap = np.round(V * 2) / 2.0
        W_snap = np.round(W * 2) / 2.0
        
        # Check tensor equation
        for i, j, k in indices:
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            val = np.dot(U_snap[u_idx, :], V_snap[v_idx, :]) * W_snap[w_idx, :]
            # Wait, the sum is over r.
            # val = sum_r U[u,r] V[v,r] W[w,r]
            # This is not a simple dot product of U and V multiplied by W.
            # It is element-wise product of U_row, V_row, W_row, then sum.
            prod = U_snap[u_idx, :] * V_snap[v_idx, :] * W_snap[w_idx, :]
            s = np.sum(prod)
            if abs(s - 1.0) > 1e-9:
                return False
        return True

    # Helper to run ALS for a given rank
    def try_rank(R, max_iters=500, tol=1e-8):
        # Initialize
        # Random init
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Scale to avoid overflow/underflow initially
        U /= np.sqrt(R)
        V /= np.sqrt(R)
        W /= np.sqrt(R)
        
        for it in range(max_iters):
            # Update U
            # For each (i,j), we want to minimize sum_k (1 - sum_r U[i,j,r] V[j,k,r] W[i,k,r])^2
            # This is a linear least squares problem for U[i,j,:].
            # Let A_u be a matrix where row t corresponds to a term (i,j,k).
            # Actually, for fixed i,j, the variable is U[i,j,:].
            # The equation is: sum_r U[i,j,r] * (V[j,k,r] * W[i,k,r]) = 1
            # Let's construct the system for U.
            # There are N equations.
            # For equation t (i,j,k):
            #   U[i*m+j, :] dot (V[j*p+k, :] * W[k*n+i, :]) = 1
            
            # We can solve for U row by row? Or all at once?
            # All at once:
            # Construct matrix M_u of size N x (n*m*R)? No.
            # Each row of U is independent in the linear system?
            # Yes, the equations are linear in each row of U.
            # So we can solve for each row of U independently.
            
            # For row u_idx = i*m+j:
            # We need to solve: A_u[u_idx] * U[u_idx, :] = b_u
            # Where A_u is N x R.
            # A_u[t, r] = V[v_idx, r] * W[w_idx, r]
            # b_u[t] = 1
            
            # We can build A_u for all u_idx at once?
            # A_u shape: (n*m) x N x R?
            # Let's just loop over u_idx. There are only 15.
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    A_row = np.zeros((N, R))
                    for t, (ii, jj, kk) in enumerate(indices):
                        if ii == i and jj == j:
                            v_idx = jj * p + kk
                            w_idx = kk * n + ii
                            A_row[t, :] = V[v_idx, :] * W[w_idx, :]
                        # else 0
                    b = np.ones(N)
                    # Solve A_row U[u_idx] = b
                    # Use lstsq
                    try:
                        U[u_idx, :] = np.linalg.lstsq(A_row, b, rcond=None)[0]
                    except:
                        pass
                        
            # Update V
            for j in range(m):
                for k in range(p):
                    v_idx = j * p + k
                    A_row = np.zeros((N, R))
                    for t, (ii, jj, kk) in enumerate(indices):
                        if jj == j and kk == k:
                            u_idx = ii * m + jj
                            w_idx = kk * n + ii
                            A_row[t, :] = U[u_idx, :] * W[w_idx, :]
                    b = np.ones(N)
                    try:
                        V[v_idx, :] = np.linalg.lstsq(A_row, b, rcond=None)[0]
                    except:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    w_idx = k * n + i
                    A_row = np.zeros((N, R))
                    for t, (ii, jj, kk) in enumerate(indices):
                        if kk == k and ii == i:
                            u_idx = ii * m + jj
                            v_idx = jj * p + kk
                            A_row[t, :] = U[u_idx, :] * V[v_idx, :]
                    b = np.ones(N)
                    try:
                        W[w_idx, :] = np.linalg.lstsq(A_row, b, rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            # Calculate error
            err = 0
            for i, j, k in indices:
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                err += (val - 1.0)**2
            if err < tol:
                # Snap and verify
                if verify(U, V, W):
                    return U, V, W
            if time.time() > deadline - 5:
                break
                
        return None

    # Try ranks from 90 down to 1
    # We know 90 works.
    # Try 89, 88...
    # To save time, we can jump.
    # But let's try sequentially for a bit.
    
    best_solution = None
    
    # First, ensure we have the schoolbook solution as fallback
    # But we want to beat it.
    
    for R in range(89, 0, -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few restarts for this rank
        found = False
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            res = try_rank(R)
            if res is not None:
                U, V, W = res
                # Convert to lists of lists for output
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
        
    # If nothing found, return schoolbook
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
