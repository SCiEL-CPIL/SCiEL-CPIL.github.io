import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 6]

def _verify_half_integer(arr, tol=1e-9):
    doubled = arr * 2
    rounded = np.round(doubled)
    if np.max(np.abs(doubled - rounded)) > tol:
        return False
    return True

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_half(arr):
    return np.round(arr * 2) / 2.0

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    T[i,j,k] = sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c] = 1 for all i,j,k in {0,1}
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen's 7 products for 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For tensor decomposition, we need:
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # In our convention: T[i,j,k] where i=row of A, j=col of B, k=... 
    # Actually our convention is: U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # For 2x2x2: U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c]
    # This represents: (A*B)[i,j] = sum_k A[i,k]*B[k,j]
    # So U[i*2+k, c] corresponds to A[i,k], V[k*2+j, c] to B[k,j], W[j*2+i, c] to C[i,j]
    # Wait, let me re-read: U[i*m+j,c], V[j*p+k,c], W[k*n+i,c]
    # For n=m=p=2: U[i*2+j,c], V[j*2+k,c], W[k*2+i,c]
    # This means: sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c] = 1
    # This is NOT the standard matrix multiplication tensor in the way I described.
    # Let me re-examine.
    
    # The tensor T[i,j,k] = 1 for all (i,j,k) in {0,1}^3
    # And the decomposition is: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For n=m=p=2: T[i,j,k] = sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c]
    
    # This is the tensor for "multiply a 2x2 matrix by a 2x2 matrix" where:
    # - i indexes rows of the first matrix (0,1)
    # - j indexes columns of first / rows of second (0,1) 
    # - k indexes columns of the second matrix (0,1)
    # So T[i,j,k] = 1 means (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    
    # Standard Strassen for 2x2*2x2:
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = A*B = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    
    # In our tensor: T[i,j,k] where i=row of A, j=col of A / row of B, k=col of B
    # T[0,0,0]=1: A[0,0]*B[0,0] contributes to C[0,0]
    # T[0,0,1]=1: A[0,0]*B[0,1] contributes to C[0,1]
    # T[0,1,0]=1: A[0,1]*B[1,0] contributes to C[0,0]
    # T[0,1,1]=1: A[0,1]*B[1,1] contributes to C[0,1]
    # T[1,0,0]=1: A[1,0]*B[0,0] contributes to C[1,0]
    # T[1,0,1]=1: A[1,0]*B[0,1] contributes to C[1,1]
    # T[1,1,0]=1: A[1,1]*B[1,0] contributes to C[1,0]
    # T[1,1,1]=1: A[1,1]*B[1,1] contributes to C[1,1]
    
    # Strassen's 7 multiplications:
    # M1 = (A00+A11)*(B00+B11) -> contributes to C00, C01, C10, C11
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    
    # For the tensor decomposition, each "multiplication" M_c corresponds to a rank-1 term.
    # M1 = (A00+A11)*(B00+B11): This is a product of a linear form in A and a linear form in B.
    # In our tensor: sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c]
    # For M1 to contribute, we need:
    # U[0*2+0, c] * V[0*2+0, c] * W[0*2+0, c] contributes to T[0,0,0]
    # U[0*2+1, c] * V[1*2+0, c] * W[0*2+0, c] contributes to T[0,1,0]
    # etc.
    
    # This is getting complex. Let me just use the known rank-7 values.
    # The standard Strassen decomposition in tensor form:
    
    # For each of the 7 terms, we have coefficients in U, V, W.
    # Term c: U[:,c], V[:,c], W[:,c]
    
    # M1 = (A00+A11)(B00+B11):
    # A00 = A[0,0], A11 = A[1,1] -> U indices: 0*2+0=0, 1*2+1=3
    # B00 = B[0,0], B11 = B[1,1] -> V indices: 0*2+0=0, 1*2+1=3
    # C00 gets M1, C01 gets -M1, C10 gets -M1, C11 gets M1 (from the combination)
    # Actually, let me just look up the standard form.
    
    # I'll use a simpler approach: just verify that a known rank-7 decomposition works.
    # The key values for Strassen 2x2x2:
    
    U[0, 0] = 1; U[3, 0] = 1  # A00 + A11
    V[0, 0] = 1; V[3, 0] = 1  # B00 + B11
    W[0, 0] = 1; W[3, 0] = 1  # C00 + C11 (but we need C00, C01, C10, C11 separately)
    
    # This isn't working cleanly. Let me just use the numerical approach with good initializations.
    # The Strassen structure gives us a good starting point for the optimizer.
    
    # For now, return a placeholder - we'll use numerical optimization
    return U, V, W

def _optimize_rank(r, n, m, p, time_limit=10):
    start_time = time.time()
    total_vars = n*m*r + m*p*r + p*n*r
    
    def objective(x):
        U = x[:n*m*r].reshape(n*m, r)
        V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = x[n*m*r + m*p*r :].reshape(p*n, r)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0)**2
        return err

    def gradient(x):
        U = x[:n*m*r].reshape(n*m, r)
        V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = x[n*m*r + m*p*r :].reshape(p*n, r)
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    diff = 2 * (val - 1.0)
                    
                    grad_U[u_idx, :] += diff * (V[v_idx, :] * W[w_idx, :])
                    grad_V[v_idx, :] += diff * (U[u_idx, :] * W[w_idx, :])
                    grad_W[w_idx, :] += diff * (U[u_idx, :] * V[v_idx, :])
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    def _try_polish(x, max_iter=2000):
        """Final high-precision polish pass with tighter tolerances."""
        try:
            res = minimize(objective, x, jac=gradient, method='L-BFGS-B', 
                           options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-12})
            return res.x, res.fun
        except Exception:
            return x, objective(x)

    def _generate_structured_init(seed_val):
        """Generate a structured initialization based on partial tensor structure."""
        np.random.seed(seed_val)
        x0 = np.zeros(total_vars)
        
        # Initialize with small random perturbations
        x0 += np.random.randn(total_vars) * 0.3
        
        # Add some structure: ensure each row has at least one nonzero
        for row_start in [0, n*m*r, n*m*r + m*p*r]:
            row_count = (total_vars - row_start) // r
            for row in range(row_count):
                if np.all(np.abs(x0[row_start + row*r:(row_start + row+1)*r]) < 0.1):
                    idx = np.random.randint(0, r)
                    x0[row_start + row*r + idx] = np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.5)
        
        return x0

    def _generate_sparse_init(seed_val, density=0.2):
        """Generate a sparse initialization."""
        np.random.seed(seed_val)
        x0 = np.zeros(total_vars)
        num_nonzero = int(total_vars * density)
        indices = np.random.choice(total_vars, num_nonzero, replace=False)
        x0[indices] = np.random.choice([-1, 1], num_nonzero) * np.random.uniform(0.5, 1.5, num_nonzero)
        return x0

    def _generate_block_init(seed_val):
        """Generate a block-structured initialization."""
        np.random.seed(seed_val)
        x0 = np.zeros(total_vars)
        
        # Create block structure
        block_size = max(1, r // 4)
        for start in range(0, total_vars, block_size):
            end = min(start + block_size, total_vars)
            if np.random.random() < 0.7:
                x0[start:end] = np.random.choice([-1, 1], end - start) * np.random.uniform(0.3, 1.0)
        
        return x0

    best_loss = float('inf')
    best_x = None
    
    # Generate diverse initializations
    inits = []
    
    # Random initializations with different scales
    for scale in [0.05, 0.1, 0.2, 0.3, 0.5]:
        np.random.seed(int(scale * 10000) + 42)
        inits.append(np.random.randn(total_vars) * scale)
    
    # Structured initializations
    for seed in [100, 200, 300]:
        inits.append(_generate_structured_init(seed))
    
    # Sparse initializations
    for density in [0.1, 0.2, 0.3]:
        inits.append(_generate_sparse_init(500 + int(density * 100), density))
    
    # Block initializations
    for seed in [600, 700]:
        inits.append(_generate_block_init(seed))
    
    # Optimize from each initialization
    for idx, x0 in enumerate(inits):
        if time.time() - start_time > time_limit - 2:
            break
        
        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_x = res.x.copy()
                
                # Quick check if we're close enough to snap
                U = best_x[:n*m*r].reshape(n*m, r)
                V = best_x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W = best_x[n*m*r + m*p*r :].reshape(p*n, r)
                
                U_snap = _snap_to_half(U)
                V_snap = _snap_to_half(V)
                W_snap = _snap_to_half(W)
                
                if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-5):
                    # Polish for higher precision
                    if time.time() - start_time < time_limit - 1:
                        polished_x, polished_loss = _try_polish(best_x, max_iter=500)
                        if polished_loss < best_loss:
                            best_x = polished_x
                            best_loss = polished_loss
                        U = best_x[:n*m*r].reshape(n*m, r)
                        V = best_x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                        W = best_x[n*m*r + m*p*r :].reshape(p*n, r)
                        U_snap = _snap_to_half(U)
                        V_snap = _snap_to_half(V)
                        W_snap = _snap_to_half(W)
                    
                    if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-6):
                        return U_snap, V_snap, W_snap, True
                        
        except Exception:
            pass
    
    # Final attempt with best solution found
    if best_x is not None and best_loss < 1e-4:
        if time.time() - start_time < time_limit - 1:
            polished_x, polished_loss = _try_polish(best_x, max_iter=1000)
            if polished_loss < best_loss:
                best_x = polished_x
                best_loss = polished_loss
        
        U = best_x[:n*m*r].reshape(n*m, r)
        V = best_x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = best_x[n*m*r + m*p*r :].reshape(p*n, r)
        
        U_snap = _snap_to_half(U)
        V_snap = _snap_to_half(V)
        W_snap = _snap_to_half(W)
        
        if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-6):
            return U_snap, V_snap, W_snap, True
            
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    U_school, V_school, W_school = _make_schoolbook(n, m, p)
    
    # Try ranks from a reasonable lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # Known: R(3,4,6) is likely around 28-36
    lower_bound = 24
    
    # Try a few promising ranks first with more time, then sweep
    promising_ranks = [28, 30, 32, 26, 24, 25, 27, 29, 31, 33, 34, 35, 36]
    tried_ranks = set()
    
    for r in promising_ranks:
        if r < lower_bound or r > n*m*p:
            continue
        if r in tried_ranks:
            continue
        if time.time() > deadline - 15:
            break
        
        tried_ranks.add(r)
        
        U, V, W, success = _optimize_rank(r, n, m, p, time_limit=min(18, deadline - time.time() - 10))
        
        if success:
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Then sweep remaining ranks
    for r in range(lower_bound, n*m*p + 1):
        if r in tried_ranks:
            continue
        if time.time() > deadline - 8:
            break
            
        U, V, W, success = _optimize_rank(r, n, m, p, time_limit=min(10, deadline - time.time() - 5))
        
        if success:
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
            
    U_list = [[float(x) for x in row] for row in U_school]
    V_list = [[float(x) for x in row] for row in V_school]
    W_list = [[float(x) for x in row] for row in W_school]
    return {"U": U_list, "V": V_list, "W": W_list}
