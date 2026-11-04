import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find rank 19
    target_rank = 19
    deadline = time.time() + 100  # 100s limit
    
    # Precompute indices for equations
    # Equation for (i, j, j_prime, k, k_prime, i_prime)
    # Sum_r U[i*m+j, r] * V[j_prime*p+k, r] * W[k_prime*n+i_prime, r] = delta
    # We solve for W.
    # Variables are W[k_prime*n+i_prime, r] for all k_prime, i_prime, r.
    # Total variables = p*n*target_rank
    
    # Let's build the linear system A * W_flat = b
    # A has shape (num_eqs, num_vars)
    # num_eqs = n*m * m*p * p*n = 2*3 * 3*4 * 4*2 = 6 * 12 * 8 = 576
    # num_vars = p*n*target_rank = 8 * 19 = 152
    
    num_eqs = (n * m) * (m * p) * (p * n)
    num_vars = p * n * target_rank
    
    # We iterate over random U, V
    # To speed up, we can use a fixed random seed or structured random
    
    best_result = None
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Random U, V with entries in {-1, 0, 1} or {-0.5, 0, 0.5, 1}
        # Scale by 2 to use integers
        U_rand = np.random.randint(-1, 2, size=(n * m, target_rank))
        V_rand = np.random.randint(-1, 2, size=(m * p, target_rank))
        
        # Build A matrix
        A = np.zeros((num_eqs, num_vars))
        b = np.zeros(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                # LHS: sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r]
                                # This is linear in W.
                                # Coefficient for W[kp*n+ip, r] is U[i*m+j, r] * V[jp*p+k, r]
                                u_val = U_rand[i * m + j]
                                v_val = V_rand[jp * p + k]
                                # dot product with W part?
                                # No, for each r, the coefficient is u_val[r] * v_val[r]
                                # So for each r, A[eq_idx, (kp*n+ip)*target_rank + r] = u_val[r] * v_val[r]
                                
                                # RHS
                                if i == ip and j == jp and k == kp:
                                    b[eq_idx] = 1
                                else:
                                    b[eq_idx] = 0
                                    
                                # Fill A
                                # We can do this more efficiently
                                # A[eq_idx, :] = ...
                                # But we need to map (kp, ip, r) to index
                                # Index = (kp*n + ip) * target_rank + r
                                
                                # Vectorized fill
                                # coeff_r = u_val * v_val
                                # for r in range(target_rank):
                                #    A[eq_idx, (kp*n+ip)*target_rank + r] = coeff_r[r]
                                
                                # We can precompute the coefficient vector for this equation
                                # coeff = U_rand[i*m+j] * V_rand[jp*p+k]
                                coeff = u_val * v_val
                                # This coeff is added to the row for each (kp, ip) block?
                                # No, the sum is over r.
                                # The variable is W[kp*n+ip, r].
                                # So for a fixed (kp, ip), the variables are W[kp*n+ip, 0..R-1].
                                # The coefficient for W[kp*n+ip, r] is coeff[r].
                                # So the block of A corresponding to (kp, ip) is coeff.
                                
                                # Wait, the equation is:
                                # sum_r coeff[r] * W[kp*n+ip, r] = b[eq_idx]
                                # This is one equation.
                                # But W has entries for ALL kp, ip.
                                # The equation involves ONLY W[kp*n+ip, *]?
                                # No!
                                # The sum is over r.
                                # The term is U[i,j,r] V[jp,k,r] W[kp,ip,r].
                                # For a fixed (i,j,jp,k,kp,ip), the equation is:
                                # sum_r (U[i,j,r] V[jp,k,r]) W[kp,ip,r] = b
                                # This equation ONLY involves W[kp,ip,r] for all r.
                                # It does NOT involve W[kp',ip',r] for other kp',ip'.
                                # So the system is decoupled!
                                # For each (kp, ip), we have a separate system?
                                # No, the equation index depends on (i,j,jp,k,kp,ip).
                                # For a fixed (kp, ip), we have equations for all (i,j,jp,k).
                                # The number of such equations is n*m*m*p = 2*3*3*4 = 72.
                                # The number of variables for this (kp, ip) is target_rank = 19.
                                # So we have 72 equations and 19 variables for each (kp, ip).
                                # This is overdetermined.
                                # We can solve each block independently!
                                
                                # Let's restructure.
                                # For each kp in range(p), ip in range(n):
                                #   Build A_block (72 x 19) and b_block (72)
                                #   Solve A_block * W_block = b_block
                                #   Check if W_block is close to half-integers
                                
                                eq_idx += 1
                                
        # The loop above was just counting. Now let's actually solve.
        
        valid = True
        W_sol = np.zeros((p * n, target_rank))
        
        for kp in range(p):
            for ip in range(n):
                # Build A_block
                A_block = np.zeros((n * m * m * p, target_rank))
                b_block = np.zeros(n * m * m * p)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for k in range(p):
                                u_val = U_rand[i * m + j]
                                v_val = V_rand[jp * p + k]
                                coeff = u_val * v_val
                                A_block[eq_idx, :] = coeff
                                
                                if i == ip and j == jp and k == kp:
                                    b_block[eq_idx] = 1
                                else:
                                    b_block[eq_idx] = 0
                                eq_idx += 1
                                
                # Solve A_block * x = b_block
                # Use least squares
                try:
                    x, residuals, rank, sv = np.linalg.lstsq(A_block, b_block, rcond=None)
                except:
                    valid = False
                    break
                    
                # Check if x is close to half-integers
                # x should be close to 0.5 * integer
                x_scaled = x * 2
                x_rounded = np.round(x_scaled)
                if np.allclose(x_scaled, x_rounded, atol=1e-5):
                    W_sol[kp * n + ip, :] = x_rounded / 2.0
                else:
                    valid = False
                    break
                    
        if valid:
            # Verify exact solution
            # Convert to integers for exact check
            U_int = U_rand
            V_int = V_rand
            W_int = np.round(W_sol * 2)
            
            # Check all equations
            exact = True
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            for kp in range(p):
                                for ip in range(n):
                                    lhs = np.dot(U_int[i * m + j] * V_int[jp * p + k], W_int[kp * n + ip])
                                    rhs = 1 if (i == ip and j == jp and k == kp) else 0
                                    if lhs != rhs:
                                        exact = False
                                        break
                                if not exact: break
                            if not exact: break
                        if not exact: break
                    if not exact: break
                if not exact: break
            if exact:
                # Found a solution!
                # Convert to list of lists
                U_list = U_int.tolist()
                V_list = V_int.tolist()
                W_list = (W_int / 2.0).tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
                
    # Fallback
    return {"U": U_school, "V": V_school, "W": W_school}
