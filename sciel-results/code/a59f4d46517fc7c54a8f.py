"""Improved tensor decomposition search for 3x5 x 5x7 matrix multiplication."""
import time
import random
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def _is_valid(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W form an exact decomposition."""
    rank = len(U[0])
    A = np.array(U).reshape(n, m, rank)
    B = np.array(V).reshape(m, p, rank)
    C = np.array(W).reshape(p, n, rank)
    
    # For each i,k, check sum_r A[i,:,r]*B[:,k,r] == C[k,i,r] structure
    # Actually the tensor equation is: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # The standard matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
    # No wait - matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # The multiplication tensor is a 3-tensor T[i,j,k] where the bilinear form 
    # sum_{i,j,k} T[i,j,k] a_i b_j = c_k... 
    
    # Actually for matrix multiplication (n x m) * (m x p) -> (n x p):
    # Output C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor decomposition: C[i,k] = sum_r sum_j (sum_j' U[i,j,j',r]) ... 
    # No. Standard form: The multiplication tensor T of format (n*p) x (n*m) x (m*p)
    # T[c, (i,j), (j,k)] = 1 if and only if c = (i,k) where C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Let me re-read the convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # The decomposition means: for the multiplication tensor, 
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # where a = i*m+j (input A element), b = j'*p+k (input B element), c = k*n+i (output element)
    # T[a,b,c] = 1 iff the output c corresponds to output position (i,k) and a corresponds to A[i,j] and b corresponds to B[j,k] for the same j
    
    # Actually, the bilinear multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # As a trilinear form: sum_{i,k,j} C[i,k] * f(A[i,j], B[j,k])
    # The tensor is: T[(i,k), (i,j), (j,k)] = 1 for all valid indices
    # In our indexing: W[k*n+i] corresponds to output (i,k), U[i*m+j] to input A[i,j], V[j*p+k] to input B[j,k]
    # T is nonzero (=1) exactly when the three indices (i,k), (i,j), (j,k) are consistent, i.e., same i, j, k.
    
    # So the decomposition requires: for all (i,k), (i',j), (j',k'):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k'*n+i', r] = 1 if i=i', j=j', k=k' else 0
    
    # Let's verify numerically
    # Build the target tensor (flattened)
    # For each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # Sum over r should give the tensor
    
    # More efficient check: for each (i,k) output position
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # = sum_r sum_j U[i*m+j,r]*V[j*p+k,r] * W[k*n+i,r]
    # We need: sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] * V[j*p+k,r]) = C[i,k] for all i,k
    # And for the cross terms to vanish.
    
    # Simplest numerical check: pick random matrices and verify
    rng = np.random.RandomState(42)
    for trial in range(3):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        # Compute via tensor decomposition
        C_from_decomp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    s = 0.0
                    for j in range(m):
                        s += U[i*m+j][r] * V[j*p+k][r]
                    val += W[k*n+i][r] * s
                C_from_decomp[i][k] = val
        
        if np.max(np.abs(C_from_decomp - C_expected)) > tol:
            return False
    return True

def _is_valid_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster numerical validation using matrix operations."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, rank)
    V_arr = np.array(V, dtype=np.float64).reshape(m, p, rank)
    W_arr = np.array(W, dtype=np.float64).reshape(p, n, rank)
    
    rng = np.random.RandomState(42)
    for trial in range(5):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        # C[i,k] = sum_r W_arr[k,i,r] * sum_j U_arr[i,j,r] * V_arr[j,k,r]
        C_from_decomp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                # sum_j U[i,j,r] * V[j,k,r] for each r, then sum_r with W[k,i,r]
                # = sum_r W[k,i,r] * (U[i,:,r] dot V[:,k,r])
                # U[i,:,:] is (m, rank), V[:,k,:] is (m, rank)
                # (U[i,:,r] * V[j,k,r]) summed over j = element-wise product then sum
                # Actually: sum_j U[i,j,r]*V[j,k,r] = row i of U col r dotted with col k of V col r... 
                # U_arr[i] shape (m, rank), V_arr[:,k] shape (m, rank)
                # For each r: dot(U_arr[i,:,r], V_arr[:,k,r]) = np.sum(U_arr[i,:,r] * V_arr[:,k,r])
                # = np.sum(np.einsum('jr,jr->r', U_arr[i], V_arr[:,k]))
                # Then dot with W_arr[k,i,:]
                s_r = np.sum(U_arr[i] * V_arr[:, k], axis=0)  # shape (rank,)
                C_from_decomp[i, k] = np.dot(W_arr[k, i], s_r)
        
        if np.max(np.abs(C_from_decomp - C_expected)) > tol:
            return False
    return True

def _try_round_and_verify(U_float, V_float, W_float, n, m, p, max_denom=8):
    """Try to round entries to rationals with small denominator and verify."""
    rank = U_float.shape[1]
    U_frac = []
    V_frac = []
    W_frac = []
    
    for i in range(U_float.shape[0]):
        row = []
        for j in range(rank):
            val = U_float[i, j]
            # Try to find a = val
            # Check if it's close to an integer
            closest_int = round(val)
            if abs(val - closest_int) < 1e-6:
                row.append(int(closest_int))
            else:
                found = False
                for d in range(2, max_denom + 1):
                    num = round(val * d)
                    if abs(val - num / d) < 1e-6:
                        # Store as (num, d) but for simplicity use float
                        row.append(num / d)
                        found = True
                        break
                if not found:
                    row.append(round(val, 4))
        U_frac.append(row)
    
    V_frac = []
    for i in range(V_float.shape[0]):
        row = []
        for j in range(rank):
            val = V_float[i, j]
            closest_int = round(val)
            if abs(val - closest_int) < 1e-6:
                row.append(int(closest_int))
            else:
                found = False
                for d in range(2, max_denom + 1):
                    num = round(val * d)
                    if abs(val - num / d) < 1e-6:
                        row.append(num / d)
                        found = True
                        break
                if not found:
                    row.append(round(val, 4))
        V_frac.append(row)
    
    W_frac = []
    for i in range(W_float.shape[0]):
        row = []
        for j in range(rank):
            val = W_float[i, j]
            closest_int = round(val)
            if abs(val - closest_int) < 1e-6:
                row.append(int(closest_int))
            else:
                found = False
                for d in range(2, max_denom + 1):
                    num = round(val * d)
                    if abs(val - num / d) < 1e-6:
                        row.append(num / d)
                        found = True
                        break
                if not found:
                    row.append(round(val, 4))
        W_frac.append(row)
    
    if _is_valid_fast(U_frac, V_frac, W_frac, n, m, p, tol=1e-6):
        return U_frac, V_frac, W_frac
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Start with schoolbook
    U_sb = [[0] * (best_rank) for _ in range(n * m)]
    V_sb = [[0] * (best_rank) for _ in range(m * p)]
    W_sb = [[0] * (best_rank) for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    
    # Try to find lower rank decompositions
    # For 3x5*5x7, known lower bounds and results:
    # Lower bound: n*p = 21 (trivial), also max(n*m, m*p, n*p) = 35
    # Strassen-like methods might give lower ranks
    # Let's try random search for rank from a reasonable minimum up
    
    # The multiplication tensor rank for C^{n,m,p} - we search
    # Use randomized tensor decomposition approach
    
    # Strategy: random initialization + gradient-free optimization
    # The condition: for all i,k: sum_r W[k*n+i,r] * sum_j U[i*m+j,r] * V[j*p+k,r] = 1
    # And cross-conditions: for i != i' or k != k': should be 0
    
    # Actually the full condition is the 3-tensor equation.
    # Let's use the fact that this is equivalent to:
    # For the matrix multiplication, the tensor is:
    # T[a, b, c] = 1 where a=(i,j), b=(j,k), c=(i,k) - all consistent
    # This is a "permutation" type tensor.
    
    # Randomized search: try various ranks, initialize randomly, try to fit
    # We'll use a local search / simulated annealing approach on the rank
    
    for trial_rank in range(max(n*p, 21), min(n*m*p, 60)):  # try ranks 21 to ~59
        if time.time() > deadline:
            break
        
        # Multiple random inits per rank
        attempts = 0
        max_attempts = 5 if trial_rank <= 30 else 2
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            if trial_rank <= best_rank:
                break  # no point trying a rank we already have or worse
                
            attempts += 1
            rng = np.random.RandomState(hash((trial_rank, attempt)) % 10000)
            
            # Random initialization with small values
            U_rand = (rng.randn(n*m, trial_rank) * 0.5).tolist()
            V_rand = (rng.randn(m*p, trial_rank) * 0.5).tolist()
            W_rand = (rng.randn(p*n, trial_rank) * 0.5).tolist()
            
            # Try a few rounds of local perturbation
            best_err = float('inf')
            current_U = [row[:] for row in U_rand]
            current_V = [row[:] for row in V_rand]
            current_W = [row[:] for row in W_rand]
            
            def compute_error(U, V, W):
                """Compute the tensor decomposition error."""
                U_a = np.array(U).reshape(n, m, trial_rank)
                V_a = np.array(V).reshape(m, p, trial_rank)
                W_a = np.array(W).reshape(p, n, trial_rank)
                
                err = 0.0
                # Target: for each (i,j,k): sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] should equal 1
                # For "wrong" combinations (i,j) with (j',k) where j != j', or (i,k) inconsistent:
                # Actually the tensor T[(i,j),(j',k),(k',i')] = delta(i,i')*delta(j,j')*delta(k,k')
                # Wait no. Let me re-think.
                
                # The matrix multiplication bilinear form:
                # (A,B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]
                # As a trilinear map: C = T(A,B) where
                # C[i,k] = sum_j A[i,j] B[j,k]
                # This can be written as: sum_{i,k,j} C[i,k] * A[i,j] * B[j,k] = sum_{i,k} (sum_j A[i,j]B[j,k]) * C[i,k]... 
                
                # For a rank-r decomposition:
                # C[i,k] = sum_r alpha_r[i,k] * sum_j (beta_r[i,j] * gamma_r[j,k])
                # where alpha_r[i,k] = W[k*n+i, r], beta_r[i,j] = U[i*m+j, r], gamma_r[j,k] = V[j*p+k, r]
                
                # So the condition is:
                # For all i,k: sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] * V[j*p+k,r]) = 1
                # (since C[i,k] = sum_j A[i,j]*B[j,k] and we're looking at coefficient of A[i,j]*B[j,k])
                
                # But we also need the "wrong" coefficients to be zero:
                # The full trilinear expansion:
                # sum_{i,k,j,j'} W[k*n+i,r] * U[i*m+j,r] * V[j'*p+k,r] 
                # should equal delta(j,j') for the decomposition to work.
                
                # Hmm, actually no. Let me think again.
                # C[i,k] = sum_r [sum_j U[i*m+j,r] * V[j*p+k,r]] * W[k*n+i,r]
                # For this to equal sum_j A[i,j]*B[j,k] for all A,B, we need:
                # sum_r W[k*n+i,r] * U[i*m+j,r] * V[j'*p+k,r] = delta(j,j') * 1
                # i.e., it equals 1 when j=j' and the (i,k) pair, and 0 otherwise.
                
                # Wait that's not right either. Let me be more careful.
                # We want: for all A (n×m) and B (m×p),
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # to equal
                # C_decomp[i,k] = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] * V[j*p+k,r])... 
                
                # No wait. The decomposition gives a bilinear form:
                # C_decomp = sum_r W_r * (U_r^T * A) * (V_r^T * B)... this isn't right.
                
                # Let me use the standard tensor decomposition formulation.
                # The multiplication tensor M ∈ R^{(n*p) × (n*m) × (m*p)} is defined by:
                # M[(i,k), (i,j), (j,k)] = 1 (for all valid i, j, k)
                # and 0 otherwise.
                
                # A rank-r decomposition is:
                # M = sum_{r=1}^{rank} u_r ⊗ v_r ⊗ w_r
                # where u_r ∈ R^{n*p}, v_r ∈ R^{n*m}, w_r ∈ R^{m*p}
                
                # In our notation:
                # u_r[(k,i)] = W[k*n+i, r] (output vector, indexed by (i,k) but stored as k*n+i)
                # v_r[(i,j)] = U[i*m+j, r] (first input, indexed by (i,j))  
                # w_r[(j,k)] = V[j*p+k, r] (second input, indexed by (j,k))
                
                # So: M[(i,k),(i',j),(j',k')] = sum_r W[k'*n+i,r] * U[i'*m+j,r] * V[j*p+k',r]
                # This should equal 1 iff i=i', j=j', k=k' (and the indices are consistent), else 0.
                
                # Wait, M[(i,k),(i',j),(j',k')] = 1 iff (i,k)=(i',k') AND (i,j) and (j',k) are "compatible"
                # Actually M[(i,k),(a,b),(c,d)] = 1 iff a=i, b=j where j is from A[i,j], c=j where B[j,k], d=k
                # So: M[(i,k),(i',j),(j',k')] = 1 iff i'=i, j'=j, and k'=k... 
                # More precisely: = 1 iff i'=i AND j'=j AND k=k' (i.e., the three index tuples are consistent)
                
                # Hmm, I think the condition is:
                # sum_r W[k*n+i, r] * U[i'*m+j, r] * V[j'*p+k', r] = 1 if i=i', j=j', k=k' else 0
                
                # Let me just check all (i,k) pairs for the main condition and a subset of cross terms
                err = 0.0
                # Main condition: sum_r W[k*n+i,r] * sum_j U[i*m+j,r] * V[j*p+k,r] = 1
                for i in range(n):
                    for k in range(p):
                        s = 0.0
                        for r in range(trial_rank):
                            inner = 0.0
                            for j in range(m):
                                inner += U_a[i, j, r] * V_a[j, k, r]
                            s += W_a[k, i, r] * inner
                        err += (s - 1.0) ** 2
                
                # Cross conditions (subset): for i != i', should be 0
                for i in range(n):
                    for i2 in range(n):
                        if i == i2:
                            continue
                        for k in range(p):
                            s = 0.0
                            for r in range(trial_rank):
                                inner = 0.0
                                for j in range(m):
                                    inner += U_a[i2, j, r] * V_a[j, k, r]
                                s += W_a[k, i, r] * inner
                            err += s ** 2
                
                # Cross conditions on j: for j != j'
                for j in range(m):
                    for j2 in range(m):
                        if j == j2:
                            continue
                        for i in range(n):
                            for k in range(p):
                                s = 0.0
                                for r in range(trial_rank):
                                    s += W_a[k, i, r] * U_a[i, j2, r] * V_a[j, k, r]
                                err += s ** 2
                
                return err
            
            # Local search with perturbations
            err = compute_error(current_U, current_V, current_W)
            best_err = err
            
            for step in range(200):
                if time.time() > deadline:
                    break
                
                # Perturb a random element
                mat_choice = rng.randint(0, 3)
                if mat_choice == 1:
                    row_idx = rng.randint(0, n*m-1)
                    col_idx = rng.randint(0, trial_rank-1)
                    current_U[row_idx][col_idx] += rng.randn() * 0.3
                elif mat_choice == 2:
                    row_idx = rng.randint(0, m*p-1)
                    col_idx = rng.randint(0, trial_rank-1)
                    current_V[row_idx][col_idx] += rng.randn() * 0.3
                else:
                    row_idx = rng.randint(0, p*n-1)
                    col_idx = rng.randint(0, trial_rank-1)
                    current_W[row_idx][col_idx] += rng.randn() * 0.3
                
                new_err = compute_error(current_U, current_V, current_W)
                if new_err < best_err:
                    best_err = new_err
                    if best_err < 1e-10:
                        break
                else:
                    # Revert
                    if mat_choice == 1:
                        current_U[row_idx][col_idx] -= rng.randn() * 0.3  # can't easily revert, just keep going
                    # Actually for simplicity, let's not revert, just accept improvements
                
                # Early exit if we found a good solution
                if best_err < 1e-8:
                    break
            
            if best_err < 1e-8:
                # Try to round to exact values
                rounded = _try_round_and_verify(
                    np.array(current_U), np.array(current_V), np.array(current_W),
                    n, m, p, max_denom=4)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    if trial_rank < best_rank:
                        best_rank = trial_rank
                        best_result = (U_r, V_r, W_r)
                    break
        
        # Early exit if we can't improve further
        if time.time() > deadline:
            break
    
    # Final check: verify best result
    if best_result is not None:
        U_best, V_best, W_best = best_result
        # Convert to lists of lists (ensure proper format)
        U_out = [[int(x) if float(x) == int(x) else (x * 2 if float(x*2) == int(x*2) else x) for x in row] for row in U_best]
        V_out = [[int(x) if float(x) == int(x) else (x * 2 if float(x*2) == int(x*2) else x) for x in row] for row in V_best]
        W_out = [[int(x) if float(x) == int(x) else (x * 2 if float(x*2) == int(x*2) else x) for x in row] for row in W_best]
        
        # Verify once more
        if _is_valid_fast(U_out, V_out, W_out, n, m, p, tol=1e-6):
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
