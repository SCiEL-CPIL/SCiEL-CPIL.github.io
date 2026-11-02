import numpy as np
import time
import scipy.linalg as la

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Build the target tensor T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition (Rank 48)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    size_U = n * m
    size_V = m * p
    size_W = p * n

    def try_solve_rank(R, num_trials=50):
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                return None
            
            np.random.seed(42 + R * 1000 + trial)
            
            # Randomly initialize U and V
            U = np.zeros((size_U, R))
            V = np.zeros((size_V, R))
            
            # Populate with random values
            for r in range(R):
                # Ensure each rank has some activity
                nz_u = np.random.choice(size_U, size=np.random.randint(1, 4), replace=False)
                nz_v = np.random.choice(size_V, size=np.random.randint(1, 4), replace=False)
                
                U[nz_u, r] = np.random.randn(len(nz_u))
                V[nz_v, r] = np.random.randn(len(nz_v))
                
            W = np.zeros((size_W, R))
            valid = True
            
            # Solve for W block by block (for each k)
            for k in range(p):
                # Build A_k of shape (n*m, n*R)
                # Equation: sum_{i', r} A_k[i*m+j, i'*R+r] W[k*n+i', r] = T[i*m+j, j*p+k, k*n+i]
                # A_k[i*m+j, i'*R+r] = U[i, r] * V[j, r] if i == i' else 0
                
                A_k = np.zeros((n * m, n * R))
                for i in range(n):
                    for j in range(m):
                        # Only i' == i contributes
                        A_k[i * m + j, i * R : (i + 1) * R] = U[i, :] * V[j, :]
                
                # Build b_k
                b_k = np.zeros(n * m)
                for i in range(n):
                    for j in range(m):
                        b_k[i * m + j] = T[i * m + j, j * p + k, k * n + i]
                
                # Solve A_k w_k = b_k
                try:
                    w_k, residuals, rank, sv = la.lstsq(A_k, b_k)
                except:
                    valid = False
                    break
                    
                # Check residual
                if np.linalg.norm(A_k @ w_k - b_k) > 1e-9:
                    valid = False
                    break
                    
                W[k * n : (k + 1) * n, :] = w_k.reshape(n, R)
                
            if valid:
                # Verify full tensor
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(recon, T, atol=1e-9):
                    # Try to round to integers/half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    recon_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                    if np.allclose(recon_r, T, atol=1e-9):
                        return U_r, V_r, W_r
                        
                    # If rounding fails, try scaling?
                    # For now, if float solution is exact, we can try to find a rational representation
                    # But the task requires integer/half-integer entries in the output.
                    # If the linear system has a unique solution, it is rational.
                    # We can try to find a common denominator for the entries of U, V, W.
                    
                    # Simple heuristic: scale by LCM of denominators?
                    # Let's just return the float solution if it's very close to half-integers?
                    # The check above does that.
                    # If it fails, we might have a valid rank R solution with non-integer coefficients.
                    # The task says "integer/half-integer entries".
                    # So we must ensure this.
                    
                    # If the solution is rational, we can scale U, V, W by a constant?
                    # No, scaling U by c and V by 1/c keeps product same.
                    # We can try to find c such that c*U, V/c, W are half-integers?
                    # This is hard.
                    
                    # Let's assume the random initialization leads to a solution that is "nice"
                    # or we can try to adjust the random initialization to be integers.
                    # Let's retry with integer initialization for U and V.
                    
                    for trial2 in range(10):
                        np.random.seed(42 + R * 2000 + trial2)
                        U_int = np.zeros((size_U, R))
                        V_int = np.zeros((size_V, R))
                        for r in range(R):
                            nz_u = np.random.choice(size_U, size=np.random.randint(1, 4), replace=False)
                            nz_v = np.random.choice(size_V, size=np.random.randint(1, 4), replace=False)
                            U_int[nz_u, r] = np.random.randint(1, 3, size=len(nz_u))
                            V_int[nz_v, r] = np.random.randint(1, 3, size=len(nz_v))
                            
                        W_int = np.zeros((size_W, R))
                        valid2 = True
                        for k in range(p):
                            A_k = np.zeros((n * m, n * R))
                            for i in range(n):
                                for j in range(m):
                                    A_k[i * m + j, i * R : (i + 1) * R] = U_int[i, :] * V_int[j, :]
                            b_k = np.zeros(n * m)
                            for i in range(n):
                                for j in range(m):
                                    b_k[i * m + j] = T[i * m + j, j * p + k, k * n + i]
                            try:
                                w_k, residuals, rank, sv = la.lstsq(A_k, b_k)
                            except:
                                valid2 = False
                                break
                            if np.linalg.norm(A_k @ w_k - b_k) > 1e-9:
                                valid2 = False
                                break
                            W_int[k * n : (k + 1) * n, :] = w_k.reshape(n, R)
                            
                        if valid2:
                            recon2 = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
                            if np.allclose(recon2, T, atol=1e-9):
                                # Check if W_int is close to integers/half-integers
                                W_r = np.round(W_int * 2) / 2
                                recon_r = np.einsum('ar,br,cr->abc', U_int, V_int, W_r)
                                if np.allclose(recon_r, T, atol=1e-9):
                                    return U_int, V_int, W_r
                                    # Note: U_int and V_int are integers, W_r is half-integer.
                                    # This satisfies the "integer/half-integer" requirement.
                        
                    # If no integer solution found, return None
                    return None

        return None

    # Try to find lower rank
    # Start from 47 down to 30
    for R in range(47, 29, -1):
        if time.time() > deadline - 5:
            break
        
        res = try_solve_rank(R, num_trials=100)
        if res is not None:
            U_sol, V_sol, W_sol = res
            best_U, best_V, best_W = U_sol, V_sol, W_sol
            best_rank = R
            # If we found a significantly lower rank, we can stop or continue?
            # Let's stop to save time, as lower ranks are harder.
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
