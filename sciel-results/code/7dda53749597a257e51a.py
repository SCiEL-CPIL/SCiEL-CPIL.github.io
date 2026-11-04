import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit

    n, m, p = 2, 4, 7

    # 1. Construct the known rank 50 solution as a baseline
    def construct_rank_50():
        rank = 50
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        col = 0
        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1
            U[i0*m + j0][col] += 1
            U[(i0+1)*m + (j0+1)][col] += 1
            V[j0*p + k0][col] += 1
            V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] += 1
            W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M2
            U[(i0+1)*m + j0][col] += 1
            U[(i0+1)*m + (j0+1)][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + (i0+1)][col] += 1
            W[(k0+1)*n + (i0+1)][col] -= 1
            col += 1
            # M3
            U[i0*m + j0][col] += 1
            V[j0*p + (k0+1)][col] += 1
            V[(j0+1)*p + (k0+1)][col] -= 1
            W[(k0+1)*n + i0][col] += 1
            W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M4
            U[(i0+1)*m + (j0+1)][col] += 1
            V[(j0+1)*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + i0][col] += 1
            W[k0*n + (i0+1)][col] += 1
            col += 1
            # M5
            U[i0*m + j0][col] += 1
            U[i0*m + (j0+1)][col] += 1
            V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] -= 1
            W[(k0+1)*n + i0][col] += 1
            col += 1
            # M6
            U[(i0+1)*m + j0][col] += 1
            U[i0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + (k0+1)][col] += 1
            W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M7
            U[i0*m + (j0+1)][col] += 1
            U[(i0+1)*m + (j0+1)][col] -= 1
            V[(j0+1)*p + k0][col] += 1
            V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] += 1
            col += 1

        # Part 1: 2x4x4 (k=0..3) - 4 blocks of 2x2x2
        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_strassen(0, 2, 0)
        add_strassen(0, 2, 2)

        # Part 2: 2x4x2 (k=4..5) - 2 blocks of 2x2x2
        add_strassen(0, 0, 4)
        add_strassen(0, 2, 4)

        # Part 3: 2x4x1 (k=6) - 8 rank-1 terms
        for i in range(n):
            for j in range(m):
                U[i*m + j][col] = 1
                V[j*p + 6][col] = 1
                W[6*n + i][col] = 1
                col += 1
                
        if col != rank:
            raise ValueError(f"Rank mismatch: {col} != {rank}")
            
        return np.array(U), np.array(V), np.array(W)

    U50, V50, W50 = construct_rank_50()
    
    # 2. Define the tensor T and residual function
    # T[i, j, k] = 1 for all i, j, k
    T_true = np.ones((n, m, p))
    
    def compute_tensor(U, V, W):
        """Compute the decomposed tensor from factor matrices."""
        r = U.shape[1]
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        # T[i, j, k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Note: W index is k*n+i, so W_3d[k,i,r]
        return np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)

    def residual(params, r):
        """Compute residual for optimization. params is a flat array."""
        u_size = n * m * r
        v_size = m * p * r
        w_size = p * n * r
        
        U = params[:u_size].reshape(n * m, r)
        V = params[u_size:u_size + v_size].reshape(m * p, r)
        W = params[u_size + v_size:].reshape(p * n, r)
        
        T_decomp = compute_tensor(U, V, W)
        return (T_decomp - T_true).flatten()

    def check_solution(U, V, W):
        """Verify if the solution is exact (within tight tolerance)."""
        T_decomp = compute_tensor(U, V, W)
        diff = np.abs(T_decomp - T_true).max()
        return diff < 1e-9, diff

    # 3. Try to find lower rank solutions
    best_solution = None
    best_rank = 50
    
    # Try ranks from 49 down to 35 (arbitrary cutoff, Strassen suggests 36 is possible but hard)
    # We prioritize lower ranks first.
    for target_rank in range(49, 34, -1):
        if time.time() > deadline - 5.0:
            break
            
        r = target_rank
        total_params = (n*m + m*p + p*n) * r
        
        # Try a few random initializations
        max_attempts = 10
        for attempt in range(max_attempts):
            if time.time() > deadline - 5.0:
                break
                
            # Initialize with random values
            # Scaling is important. Strassen uses 1s.
            # Let's try small random values first.
            params0 = np.random.randn(total_params) * 0.5
            
            try:
                # Use least_squares to minimize the residual
                # Method 'lm' is Levenberg-Marquardt, good for least squares
                # But it doesn't support bounds. We don't strictly need bounds.
                res = least_squares(
                    residual, 
                    params0, 
                    args=(r,), 
                    method='lm', 
                    max_nfev=2000,
                    xtol=1e-12,
                    ftol=1e-12,
                    gtol=1e-12
                )
                
                if res.success:
                    # Unpack
                    u_size = n * m * r
                    v_size = m * p * r
                    U = res.x[:u_size].reshape(n * m, r)
                    V = res.x[u_size:u_size + v_size].reshape(m * p, r)
                    W = res.x[u_size + v_size:].reshape(p * n, r)
                    
                    is_valid, diff = check_solution(U, V, W)
                    if is_valid:
                        best_solution = (U, V, W)
                        best_rank = r
                        break # Found a solution for this rank, try lower rank
            except Exception as e:
                # Catch any errors and continue
                pass
        
        # If we found a solution for this rank, we can stop trying this rank
        # and move to the next lower rank if we haven't hit the time limit
        if best_solution is not None and best_rank == target_rank:
            continue # Loop continues to next lower rank

    # 4. Return the best solution found
    if best_solution is not None:
        U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to rank 50
        return {"U": U50.tolist(), "V": V50.tolist(), "W": W50.tolist()}
