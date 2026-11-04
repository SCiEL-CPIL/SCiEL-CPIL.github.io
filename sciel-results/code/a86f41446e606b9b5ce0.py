import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # The tensor T has entries T[i,j,k] = 1 for all i,j,k
    # We want to find U, V, W of rank R such that
    # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # Start with a lower bound. Rank >= max(n*m, m*p, p*n) is not true.
    # Rank >= max(n, m, p) * something?
    # For 2x2x2, rank is 7.
    # For 3x5x5x7, let's try R from 20 up to 105.
    
    best_result = None
    best_rank = 106
    
    # Schoolbook is always a valid fallback
    U_school = [[0.0] * (n*m*p) for _ in range(n*m)]
    V_school = [[0.0] * (n*m*p) for _ in range(m*p)]
    W_school = [[0.0] * (n*m*p) for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1.0
                V_school[j*p+k][col] = 1.0
                W_school[k*n+i][col] = 1.0
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n*m*p
    
    # Try to find lower rank
    # Use a simple gradient descent or random search with local refinement
    # Given the time limit, we can try a few ranks.
    
    for R in range(20, best_rank + 1):
        if time.time() > deadline - 10:
            break
            
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Normalize to prevent overflow
            U = U / np.linalg.norm(U)
            V = V / np.linalg.norm(V)
            W = W / np.linalg.norm(W)
            
            # Simple iterative refinement
            # We want to minimize || sum_r U_r V_r W_r - 1 ||^2
            # This is hard. Let's just check if a random guess works (unlikely).
            # Instead, we can use the fact that if R is large enough, 
            # we can solve for one factor given the others.
            
            # Fix U and V, solve for W.
            # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
            # For fixed i,k, this is a linear system in W[:, i, k]?
            # No, W is indexed by k,i.
            # Let's reshape U, V, W to 3D tensors.
            U3 = U.reshape(n, m, R)
            V3 = V.reshape(m, p, R)
            W3 = W.reshape(p, n, R)
            
            # The equation is sum_r U3[i,j,r] V3[j,k,r] W3[k,i,r] = 1
            # This is a system of linear equations in W3 if U, V are fixed.
            # Variables: W3[k,i,r] for all k,i,r. Total p*n*R variables.
            # Equations: n*m*p equations.
            # We can set up the linear system A x = b.
            
            # x is a vector of length p*n*R.
            # b is a vector of length n*m*p, all 1s.
            
            # Construct A.
            # For each (i,j,k), the equation is:
            # sum_r U3[i,j,r] V3[j,k,r] W3[k,i,r] = 1
            # This couples W3[k,i,r] for all r.
            # It's not a simple linear system in W because U and V are fixed but multiplied.
            # Actually, for fixed i,j,k, the term is sum_r (U3[i,j,r] V3[j,k,r]) W3[k,i,r].
            # Let c_r = U3[i,j,r] V3[j,k,r].
            # Then sum_r c_r W3[k,i,r] = 1.
            # This is a linear equation in W3[k,i, :].
            # So for each (i,k), we have m equations (one for each j) in R variables W3[k,i, :].
            # Wait, for fixed i,k, j varies from 1 to m.
            # So we have m equations:
            # sum_r U3[i,j,r] V3[j,k,r] W3[k,i,r] = 1 for j=1..m.
            # This is a system A_{i,k} W3[k,i, :] = 1.
            # A_{i,k} is an m x R matrix.
            # (A_{i,k})_{j, r} = U3[i,j,r] V3[j,k,r].
            
            # We can solve this for W3[k,i, :] for each (i,k).
            # There are n*p such systems.
            # Each system is m x R.
            # If R > m, the system is underdetermined. We can find a least squares solution.
            # If the solution is exact, then the decomposition is valid.
            
            W_new = np.zeros((p, n, R))
            valid = True
            for i in range(n):
                for k in range(p):
                    # Construct A_ik (m x R)
                    A_ik = np.zeros((m, R))
                    for j in range(m):
                        for r in range(R):
                            A_ik[j, r] = U3[i, j, r] * V3[j, k, r]
                    
                    # Solve A_ik @ W_ik = 1
                    # W_ik is a vector of length R.
                    try:
                        # Use least squares
                        W_ik, residuals, rank, sv = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                        # Check if the solution is exact
                        if np.allclose(A_ik @ W_ik, np.ones(m)):
                            W_new[k, i, :] = W_ik
                        else:
                            valid = False
                            break
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # We found a solution!
                # Convert back to matrices
                U_mat = U.tolist()
                V_mat = V.tolist()
                W_mat = W_new.reshape(p*n, R).tolist()
                
                # Verify one more time
                # Check sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for r in range(R):
                                s += U_mat[i*m+j][r] * V_mat[j*p+k][r] * W_mat[k*n+i][r]
                            error += abs(s - 1.0)
                
                if error < 1e-9:
                    return {"U": U_mat, "V": V_mat, "W": W_mat}
    
    return best_result
