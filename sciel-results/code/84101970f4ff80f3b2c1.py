import numpy as np
import time
import sys
from fractions import Fraction

def construct():
    n, m, p = 3, 5, 6
    rank_sb = n * m * p
    
    # Schoolbook decomposition
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Verify schoolbook is valid (it should be)
    # For exactness, we can try to reduce rank using linear algebra.
    # The multiplication tensor T has entries T[i,j,k] = 1.
    # A rank-R decomposition is a set of R rank-1 tensors u^r \otimes v^r \otimes w^r
    # such that sum_r u^r_i v^r_j w^r_k = delta_{i,j,k} (effectively).
    # Wait, the tensor is defined by C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T has components T_{i,j,k} = 1.
    # We want sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} = 1 for all i,j,k.
    # And for any (i', j', k') such that the indices don't match?
    # No, the indices are coupled.
    # The condition is:
    # For all i in [0, n), j in [0, m), k in [0, p):
    #   sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p equations.
    # We have R * (n*m + m*p + p*n) variables.
    
    # Let's try to find a lower rank R.
    # We can set up a linear system if we fix two of the matrices? No, it's bilinear/trilinear.
    # However, for small R, we can try random integer solutions.
    # Or we can use the fact that the tensor is "simple" in some sense?
    
    # Let's check if R < n*m*p is possible.
    # Lower bounds:
    # R >= max(n*m, m*p, p*n) is not a lower bound.
    # R >= n*m*p / (something)?
    
    # Let's try a simple optimization:
    # Start with R = rank_sb - 1.
    # Can we satisfy the equations?
    
    # A more robust approach for small sizes:
    # Use numpy to solve for W given U and V?
    # The equation is:
    # sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1
    # This is linear in W if U and V are fixed.
    # Let's fix U and V to random integer values and solve for W.
    # If a solution W exists (and is integer/half-integer), we have a valid decomposition.
    
    # But U and V are also unknowns.
    
    # Let's try a heuristic search for R < rank_sb.
    # We will try R from rank_sb - 1 down to some limit.
    # For each R, we try a few random initializations of U and V (small integers).
    # Then we solve the linear system for W.
    # If the solution is valid (integer/half-integer and satisfies the equations), we return it.
    
    start_time = time.time()
    timeout = 140.0  # 10s buffer
    
    # We need to be careful with the linear system.
    # The equations are:
    # For each (i, j, k):
    #   sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
    # This is NOT a linear system in W if we view W as a vector of length R for each (k,i).
    # Actually, for a fixed (k, i), W[k*n+i, :] is a row of W.
    # The term involving W[k*n+i, r] is:
    #   sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r])
    # This looks like a dot product between the row W[k*n+i, :] and the vector (U[i*m+j, :] * V[j*p+k, :]).
    # So for each (i, j, k), we have one equation.
    # The unknowns are the entries of W.
    # But the coefficients depend on U and V.
    
    # Let's reorganize.
    # Let X_r = U[:, r] * V[:, r] ? No, U and V have different dimensions.
    # U is (n*m) x R. V is (m*p) x R.
    # Let u_r be the r-th column of U.
    # Let v_r be the r-th column of V.
    # Let w_r be the r-th column of W.
    # The equation for (i, j, k) is:
    # sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1
    
    # This is a system of n*m*p equations in R * (n*m + m*p + p*n) variables.
    
    # If we fix U and V, the system is linear in W.
    # Let's define a matrix M of size (n*m*p) x (p*n*R).
    # The unknowns are flattened W.
    # M[row, col] = ...
    # Row corresponds to (i, j, k).
    # Col corresponds to (k', i', r).
    # M[(i,j,k), (k',i',r)] = u_r[i*m+j] * v_r[j*p+k] if (k', i') == (k, i) and r==r?
    # No, the sum is over r.
    # The term for a specific r and specific (k, i) is:
    # u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    # So the coefficient of w_r[k*n+i] in the equation for (i, j, k) is u_r[i*m+j] * v_r[j*p+k].
    # Note that w_r[k*n+i] appears only in the equation for (i, j, k) if the indices match?
    # No, w_r[k*n+i] is a variable. It appears in ALL equations for all j?
    # Let's check.
    # The equation is for a specific (i, j, k).
    # The variable w_r[k*n+i] is multiplied by u_r[i*m+j] * v_r[j*p+k].
    # So for a fixed r and fixed (k, i), the variable w_r[k*n+i] appears in the equations for all j.
    # Specifically, for each j, the coefficient is u_r[i*m+j] * v_r[j*p+k].
    
    # So the linear system for W (given U, V) is:
    # For each (i, j, k):
    #   sum_{r} (u_r[i*m+j] * v_r[j*p+k]) * w_r[k*n+i] = 1
    
    # Let's construct the matrix A and vector b for W.
    # Number of equations: N_eq = n * m * p
    # Number of unknowns: N_unk = p * n * R
    
    # We can try to solve this system.
    # If U and V are random, the system might be overdetermined (N_eq > N_unk) or underdetermined.
    # For R = n*m*p, N_unk = p*n*n*m*p = n^2 * m * p^2.
    # N_eq = n * m * p.
    # For n=3, m=5, p=6:
    # N_eq = 90.
    # N_unk (R=90) = 3*5*6 * 3*6 * 90 = 90 * 18 * 90 = 145800.
    # Highly underdetermined.
    
    # If we reduce R, N_unk decreases.
    # We want the smallest R such that a solution exists.
    
    # Let's try to find a solution for R < 90.
    # We will iterate R from 90 down to 1.
    # For each R, we try a few random integer U and V.
    # We solve the linear system for W.
    # We check if W is integer/half-integer.
    
    # To speed up, we can use numpy's least squares solver.
    # If the residual is close to 0, we have a solution.
    # Then we snap to nearest integer/half-integer and verify.
    
    best_rank = rank_sb
    best_U = U
    best_V = V
    best_W = W
    
    # We'll try R from rank_sb - 1 down to 1.
    # But 90 is a lot of iterations if we do many random tries.
    # Let's limit the tries per R.
    
    # Actually, for this specific tensor, it is likely that the rank is 90.
    # But let's give it a shot.
    
    # We can also try to use the structure of the tensor.
    # T[i,j,k] = 1.
    
    # Let's try R = 90, 89, ... down to 50.
    # If we find a solution, we return it.
    
    # To make the search more effective, we can use the fact that the schoolbook solution is a valid one.
    # We can perturb it?
    
    # Let's just do a random search.
    
    max_tries_per_rank = 5
    min_rank_to_try = 50  # Don't go too low, it's unlikely
    
    for R in range(rank_sb - 1, min_rank_to_try - 1, -1):
        if time.time() - start_time > timeout:
            break
            
        for attempt in range(max_tries_per_rank):
            if time.time() - start_time > timeout:
                break
                
            # Random U and V with small integer entries
            # Using -1, 0, 1 might be too sparse.
            # Let's use -2, -1, 0, 1, 2.
            U_rand = np.random.randint(-2, 3, size=(n * m, R))
            V_rand = np.random.randint(-2, 3, size=(m * p, R))
            
            # Construct the linear system for W
            # A * W_flat = b
            # A has shape (n*m*p, p*n*R)
            # b has shape (n*m*p,)
            
            # We can build A efficiently.
            # For each (i, j, k), the equation is:
            # sum_r U_rand[i*m+j, r] * V_rand[j*p+k, r] * W_flat[(k*n+i)*R + r] = 1
            
            # Let's create A.
            A = np.zeros((n * m * p, p * n * R))
            
            # This loop might be slow if done in Python.
            # Let's vectorize.
            
            # Indices:
            # i: 0..n-1
            # j: 0..m-1
            # k: 0..p-1
            
            # We can precompute the coefficients for each r.
            # Let's iterate over r.
            for r in range(R):
                # u_r = U_rand[:, r]
                # v_r = V_rand[:, r]
                # w_r is part of W_flat at indices (k*n+i)*R + r
                
                # For each (i, j, k):
                # coeff = u_r[i*m+j] * v_r[j*p+k]
                # This coeff multiplies W_flat[(k*n+i)*R + r]
                
                # We can create a matrix C_r of shape (n*m*p, p*n)
                # C_r[i*m*p + j*p + k, k*n + i] = u_r[i*m+j] * v_r[j*p+k]
                
                # Then A[:, r] (which is a slice of A) is formed by scattering C_r.
                # Actually, A is a block matrix? No.
                # The column index in A is (k*n+i)*R + r.
                # So for a fixed r, the columns are (k*n+i)*R + r for all (k, i).
                # There are p*n such columns.
                
                # Let's create C_r.
                C_r = np.zeros((n * m * p, p * n))
                
                # Vectorized construction of C_r
                # i, j, k ranges
                # We can use broadcasting.
                
                # Let's use loops for clarity and speed (small n, m, p)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_eq = i * m * p + j * p + k
                            idx_unk = k * n + i
                            val = U_rand[i * m + j, r] * V_rand[j * p + k, r]
                            C_r[idx_eq, idx_unk] = val
                
                # Now scatter C_r into A
                # The columns of A corresponding to r are at indices:
                # col_idx = [ (k*n+i)*R + r for k in range(p) for i in range(n) ]
                # We can create this index array.
                col_indices = []
                for k in range(p):
                    for i in range(n):
                        col_indices.append((k * n + i) * R + r)
                col_indices = np.array(col_indices)
                
                A[:, col_indices] = C_r
            
            # b is all 1s
            b = np.ones(n * m * p)
            
            # Solve A * W_flat = b
            # Using least squares
            try:
                W_flat, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
                
            # Check residual
            if residuals.size > 0 and residuals[0] < 1e-6:
                # Potential solution
                # Reshape W_flat to (p*n, R)
                W_sol = W_flat.reshape((p * n, R))
                
                # Snap to nearest integer or half-integer
                # Multiply by 2, round, divide by 2
                W_snap = np.round(W_sol * 2) / 2.0
                
                # Verify exactness with snapped W
                # We need to check if U_rand, V_rand, W_snap satisfy the equations exactly.
                # Since U and V are integers, and W is half-integer, the products are half-integers.
                # The sum should be 1.
                
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(R):
                                s += U_rand[i * m + j, r] * V_rand[j * p + k, r] * W_snap[k * n + i, r]
                            if abs(s - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Found a valid decomposition!
                    # Convert to lists
                    best_rank = R
                    best_U = U_rand.tolist()
                    best_V = V_rand.tolist()
                    best_W = W_snap.tolist()
                    # Return immediately?
                    # We want the minimum rank, so we could continue to lower R.
                    # But for now, let's return this one.
                    return {"U": best_U, "V": best_V, "W": best_W}
    
    # If no better rank found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
