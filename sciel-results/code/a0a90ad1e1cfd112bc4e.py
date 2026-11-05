import numpy as np
import time
import random
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Target tensor T[i, k, i', j', k']
    # T is 0 unless i'=i and k'=k and j'=j' (inner index matches)
    # Actually, for fixed i, k, the non-zero terms are A[i, j] B[j, k]
    # So T[i, k, i, j, k] = 1 for all j.
    # All other entries are 0.
    
    # We want to find U, V, W of rank r
    # U: (n*m, r)
    # V: (m*p, r)
    # W: (p*n, r)  -> W[k*n+i, c]
    
    # Flatten indices
    # u_idx = i*m + j
    # v_idx = j*p + k
    # w_idx = k*n + i
    
    best_rank = n * m * p
    best_solution = None
    
    # Generate target tensor as a function
    # Residual for a given set of variables X
    # X is a vector of length r * (nm + mp + pn)
    
    def get_residuals(X, r):
        U = X[:r*n*m].reshape(r, n*m)
        V = X[r*n*m : r*(n*m + m*p)].reshape(r, m*p)
        W = X[r*(n*m + m*p) : ].reshape(r, p*n)
        
        residuals = []
        # For each i, k
        for i in range(n):
            for k in range(p):
                w_idx = k * n + i
                # Target: sum_j A[i, j] B[j, k]
                # This corresponds to monomials A[i, j] B[j, k]
                # Coefficient of A[i', j'] B[j', k'] in C[i, k]
                # is 1 if i'=i, k'=k, and j'=j' (any j')? 
                # No, the term is A[i, j] B[j, k].
                # So for monomial A[i', j'] B[j', k']:
                # It contributes to C[i, k] only if i'=i and k'=k.
                # And then it is the term for j=j'.
                # So coeff is 1 if i'=i, k'=k.
                # Wait, if i'=i, k'=k, the term is A[i, j'] B[j', k].
                # This is exactly the term in the sum for j=j'.
                # So for all j', the coefficient is 1?
                # No, the sum is over j.
                # C[i, k] = sum_j A[i, j] B[j, k].
                # So the coefficient of A[i, j'] B[j', k] is 1.
                # The coefficient of A[i, j1] B[j2, k] with j1 != j2 is 0.
                # But A[i, j1] B[j2, k] is a specific monomial.
                # In the expansion of the product, we only have terms where the col of A matches row of B.
                # So A[i, j1] B[j2, k] with j1 != j2 does NOT appear in the product.
                # So its coefficient is 0.
                
                # So for fixed i, k:
                # Target coeff for A[i, j] B[j, k] is 1.
                # Target coeff for A[i', j'] B[j', k'] is 0 if (i', k') != (i, k) OR (j' != j')?
                # Actually, if (i', k') == (i, k), the monomial is A[i, j'] B[j', k].
                # This is the term for j=j'.
                # So it should be 1.
                # What about A[i, j1] B[j2, k] with j1 != j2?
                # This monomial has i'=i, k'=k.
                # But j' (col of A) is j1, j' (row of B) is j2.
                # In our notation, the monomial is identified by (i', j', j'', k')?
                # No, A has indices (i, j), B has (j, k).
                # A monomial is A[i', j'] B[j'', k'].
                # It appears in C[i, k] if i'=i, k'=k, and j'=j''.
                # So if j' != j'', it does not appear.
                
                # So we need to check all 144 monomials for each (i, k).
                # This is slow in Python loop.
                # We can vectorize.
                
                # Let's construct the full residual vector for all (i, k, i', j', j'', k')
                # But we only need to check consistency.
                # Actually, we can just compute the product C = A B for random A, B and compare with sum_c ...
                # This is a better way to check the decomposition!
                # Instead of equating coefficients, we can check the identity on random inputs.
                # But we need to find the parameters.
                # Gradient descent on random inputs is also valid.
                
                pass
        return residuals

    # Better approach:
    # Use random A, B to compute residuals.
    # Residual = || C_target - C_decomp ||_F
    
    def compute_residuals(X, r, A, B):
        U = X[:r*n*m].reshape(r, n*m)
        V = X[r*n*m : r*(n*m + m*p)].reshape(r, m*p)
        W = X[r*(n*m + m*p) : ].reshape(r, p*n)
        
        # C_target = A @ B
        C_target = A @ B
        
        # C_decomp = sum_c (U_c . A) (V_c . B) W_c
        # U_c is row c of U. A is flattened?
        # U is (r, nm). A is (n, m). Flatten A to (nm,).
        # <U_c, A> = U[c] @ A.flatten()
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        u_vals = U @ A_flat  # (r,)
        v_vals = V @ B_flat  # (r,)
        
        # C_decomp[i, k] = sum_c u_vals[c] * v_vals[c] * W[c, k*n+i]
        # W is (r, pn). W[c, idx]
        # Reshape W to (r, p, n) -> W[c, k, i]
        W_reshaped = W.reshape(r, p, n)
        
        # C_decomp = sum_c u_vals[c] v_vals[c] W_reshaped[c, :, :]
        # This is (p, n)
        C_decomp = np.tensordot(u_vals * v_vals, W_reshaped, axes=([0], [0]))
        C_decomp = C_decomp.T  # (n, p)
        
        return (C_target - C_decomp).flatten()

    # We will use least_squares with random restarts.
    # To make it faster, we can use a fixed set of random A, B for the residuals.
    # This approximates the coefficient matching.
    
    num_samples = 10
    random.seed(42)
    A_samples = [np.random.randn(n, m) for _ in range(num_samples)]
    B_samples = [np.random.randn(m, p) for _ in range(num_samples)]
    
    def objective(X, r):
        res = []
        for A, B in zip(A_samples, B_samples):
            res.append(compute_residuals(X, r, A, B))
        return np.concatenate(res)
    
    found = False
    # Try ranks from lower bound up
    # Lower bound 7, but likely higher.
    # Let's try 20 to 48.
    for r in range(20, 49):
        if time.time() > deadline:
            break
            
        # Try a few restarts
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            # Initialize
            X0 = np.random.randn(r * (n*m + m*p + p*n))
            # Scale down
            X0 *= 0.1
            
            try:
                result = least_squares(objective, X0, args=(r,), method='lm', max_nfev=1000)
            except Exception:
                continue
                
            if result.cost < 1e-6:
                # Check if it's actually correct with more samples
                # And check if it can be cleaned to half-integers
                X_sol = result.x
                U = X_sol[:r*n*m].reshape(r, n*m)
                V = X_sol[r*n*m : r*(n*m + m*p)].reshape(r, m*p)
                W = X_sol[r*(n*m + m*p) : ].reshape(r, p*n)
                
                # Verify with 100 random samples
                valid = True
                for _ in range(100):
                    A = np.random.randn(n, m)
                    B = np.random.randn(m, p)
                    C_target = A @ B
                    A_flat = A.flatten()
                    B_flat = B.flatten()
                    u_vals = U @ A_flat
                    v_vals = V @ B_flat
                    W_reshaped = W.reshape(r, p, n)
                    C_decomp = np.tensordot(u_vals * v_vals, W_reshaped, axes=([0], [0])).T
                    if np.linalg.norm(C_target - C_decomp) > 1e-5:
                        valid = False
                        break
                
                if valid:
                    # Try to clean to half-integers
                    # Round to nearest 0.5
                    U_clean = np.round(U * 2) / 2
                    V_clean = np.round(V * 2) / 2
                    W_clean = np.round(W * 2) / 2
                    
                    # Verify cleaned
                    valid_clean = True
                    for _ in range(100):
                        A = np.random.randn(n, m)
                        B = np.random.randn(m, p)
                        C_target = A @ B
                        A_flat = A.flatten()
                        B_flat = B.flatten()
                        u_vals = U_clean @ A_flat
                        v_vals = V_clean @ B_flat
                        W_reshaped = W_clean.reshape(r, p, n)
                        C_decomp = np.tensordot(u_vals * v_vals, W_reshaped, axes=([0], [0])).T
                        if np.linalg.norm(C_target - C_decomp) > 1e-5:
                            valid_clean = False
                            break
                    
                    if valid_clean:
                        # Found a valid half-integer solution
                        # Convert to lists
                        U_list = U_clean.tolist()
                        V_list = V_clean.tolist()
                        W_list = W_clean.tolist()
                        
                        # Check if rank is truly r (some columns might be zero?)
                        # The problem asks for rank-minimal.
                        # If some columns are zero, we can remove them.
                        # But least_squares might find a solution with rank < r.
                        # We should check the effective rank.
                        # However, the interface expects matrices of shape (..., r).
                        # If we have a solution of rank r, we return it.
                        # If we find a solution for r, it's valid.
                        
                        return {"U": U_list, "V": V_list, "W": W_list}
                    
                    # If cleaning failed, try to find a better real solution or accept real?
                    # Task says "integer/half-integer entries".
                    # So we must return half-integers.
                    # If we can't clean, we continue searching.
        
    # If no better found, return schoolbook
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
