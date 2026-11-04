import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school

    # Precompute Target Tensor T as a list of slices T_c
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k'*n+i' with j=j', i=i', k=k'
    # Wait, the standard definition is T_{(i,j), (j',k), (k',i')} = delta_{j,j'} delta_{i,i'} delta_{k,k'}
    # So T[a, b, c] is non-zero only if:
    # a = i*m + j
    # b = j*p + k  (Note: j in b must match j in a)
    # c = k'*n + i' (Note: i' in c must match i in a, and k' in c must match k in b)
    
    # Let's build T_slices[c] which is a vector of size nm*mp
    T_slices = []
    for c in range(pn):
        T_c = np.zeros(nm * mp)
        # c = k*n + i
        k = c // n
        i = c % n
        # We need to find a, b such that T[a,b,c] = 1
        # a = i*m + j
        # b = j*p + k
        # So for a fixed i, k, we iterate over j
        for j in range(m):
            a = i * m + j
            b = j * p + k
            # Check if b is valid index for V (0..mp-1)
            if 0 <= b < mp:
                T_c[a * mp + b] = 1.0
        T_slices.append(T_c)
        
    T_slices = np.array(T_slices) # Shape (pn, nm*mp)

    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin

    # Try to find lower ranks
    # Strategy: Random U, V, solve for W
    # M_UV is (nm*mp, R). M_UV[:, r] = U[:, r] (nm) kron V[:, r] (mp)?
    # No. M_UV[(a, b), r] = U[a, r] * V[b, r].
    # Vector index for (a, b) is a * mp + b.
    
    def check_and_solve(U, V, R):
        # Construct M_UV
        # U: (nm, R), V: (mp, R)
        # M_UV: (nm*mp, R)
        # M_UV[idx, r] = U[a, r] * V[b, r] where idx = a*mp + b
        
        # Efficient construction:
        # U_expanded: (nm, 1, R)
        # V_expanded: (1, mp, R)
        # Product: (nm, mp, R)
        # Reshape to (nm*mp, R)
        M_UV = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(nm * mp, R)
        
        # For each c in 0..pn-1, we need M_UV @ W_c = T_slices[c]
        # This is a system of linear equations.
        # We can solve for all W_c at once?
        # M_UV @ W = T_slices.T ?
        # W is (pn, R). T_slices.T is (nm*mp, pn).
        # M_UV (nm*mp, R) @ W (R, pn) = T_slices.T (nm*mp, pn)
        
        # Solve M_UV @ W = T_slices.T
        # Using least squares or direct solve if square/invertible
        try:
            # Check rank consistency
            # If M_UV is full column rank (R), and T_slices.T is in col space, solution exists.
            # We can use np.linalg.lstsq
            W_sol, residuals, rank, s = np.linalg.lstsq(M_UV, T_slices.T, rcond=None)
            
            # Check if the solution is exact (residuals should be 0)
            # residuals is a list of sums of squares for each column
            if np.allclose(residuals, 0, atol=1e-9):
                # Verify exactness with higher precision or integer check?
                # The prompt allows half-integers.
                # Let's check if W_sol is close to half-integers.
                # Or just return it. The evaluator checks exactness.
                # If it's a floating point error, it might fail.
                # But if U, V are integers, and T is integer, the solution W is rational.
                # np.linalg.lstsq might introduce float errors.
                
                # Let's verify by reconstruction
                W_sol = W_sol.T # Back to (pn, R)
                # Reconstruct T
                # S = sum_r U[:, r] V[:, r]^T W[:, r]^T ? No.
                # S[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
                S = np.einsum('ar,br,cr->abc', U, V, W_sol)
                
                # Construct full T for comparison
                T_full = np.zeros((nm, mp, pn))
                for c in range(pn):
                    T_full[:, :, c] = T_slices[c].reshape(nm, mp)
                
                if np.allclose(S, T_full, atol=1e-8):
                    return W_sol
        except np.linalg.LinAlgError:
            pass
        except Exception:
            pass
        return None

    # Search loop
    # Start from best_rank - 1
    # We use random integer/half-integer initializations for U, V to increase chance of rational solutions
    # Or just random floats.
    
    for R in range(best_rank - 1, max(best_rank - 20, 0), -1):
        if time.time() > deadline:
            break
            
        # Number of restarts depends on time
        # Try 100 restarts per rank?
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Use small integers or half-integers to help with exactness
            # Range [-1, 1] or [-2, 2]
            U_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(nm, R))
            V_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(mp, R))
            
            W_sol = check_and_solve(U_init, V_init, R)
            
            if W_sol is not None:
                # Found a solution!
                # Check if W_sol entries are close to half-integers
                # If not, it might be a valid real solution, but prompt asks for integer/half-integer.
                # "integer/half-integer entries"
                # Let's snap to nearest half-integer and re-verify
                W_snapped = np.round(W_sol * 2) / 2.0
                
                # Verify snapped version
                S_snapped = np.einsum('ar,br,cr->abc', U_init, V_init, W_snapped)
                T_full = np.zeros((nm, mp, pn))
                for c in range(pn):
                    T_full[:, :, c] = T_slices[c].reshape(nm, mp)
                
                if np.allclose(S_snapped, T_full, atol=1e-9):
                    best_result = {
                        "U": U_init.tolist(), 
                        "V": V_init.tolist(), 
                        "W": W_snapped.tolist()
                    }
                    best_rank = R
                    break # Break restarts, go to lower R? Or keep searching for even lower?
                    # The loop continues to lower R automatically.
                else:
                    # If snapping fails, maybe the raw solution is valid but not half-integer?
                    # The prompt requires integer/half-integer.
                    # If the raw solution is not close to half-integers, it's invalid for the specific constraint.
                    # However, if the problem allows any real numbers, it would be valid.
                    # "integer/half-integer entries" is a hard constraint in the interface description.
                    pass
                    
    return best_result
