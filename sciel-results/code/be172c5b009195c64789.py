import numpy as np
import time
from fractions import Fraction
from itertools import product
from scipy.optimize import minimize

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin

    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor indices
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    nz_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nz_indices.append((a, b, c))
    
    # Precompute the linear system structure for ALS
    # For a given rank r, we have U (nm x r), V (mp x r), W (pn x r)
    # Constraint: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
    
    # We will search for integer/half-integer solutions.
    # This is equivalent to finding a low-rank tensor with entries in {0, 0.5, 1, ...}
    
    def verify_exact(U, V, W):
        """
        Verify if U, V, W exactly decompose the tensor.
        U, V, W are lists of lists of Fractions or integers.
        """
        U = np.array(U, dtype=object) # Object array to hold Fractions
        V = np.array(V, dtype=object)
        W = np.array(W, dtype=object)
        
        # Check all nonzero entries
        for a, b, c in nz_indices:
            val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(len(U[0])))
            if val != 1:
                return False
        
        # Check a sample of zero entries to save time, or all if small
        # For 4x5x6, total entries 14400, nonzero 120.
        # Checking all zero entries is 14280 checks. Each check is a sum of r multiplications.
        # If r=20, that's ~285k operations. This is fast in Python if optimized, but Fractions are slow.
        # We can use a hash or a faster check.
        
        # Alternative: Check all entries using numpy with floats first, then exact if close.
        # But we are in the "exact" verification step, so inputs are already rounded.
        
        # Let's check all zero entries to be safe, but break early.
        # To speed up, we can just check if the sum is 0.
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) not in set(nz_indices):
                         val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(len(U[0])))
                         if val != 0:
                             return False
        return True

    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(nm)]
        V = [[0]*rank for _ in range(mp)]
        W = [[0]*rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    fallback = schoolbook()

    # Search for better rank
    best_result = None
    
    # Try ranks from 20 to 40
    for r in range(20, 41):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random sparse integers
            # Use a sparsity pattern that might align with the tensor structure
            U = np.zeros((nm, r), dtype=int)
            V = np.zeros((mp, r), dtype=int)
            W = np.zeros((pn, r), dtype=int)
            
            # Random sparse initialization
            # Each row gets 2-4 non-zeros
            np.random.seed(42 + r * 100 + attempt)
            for a in range(nm):
                nnz = np.random.randint(2, 5)
                cols = np.random.choice(r, nnz, replace=False)
                for c in cols:
                    U[a, c] = np.random.choice([-1, 1])
            
            for b in range(mp):
                nnz = np.random.randint(2, 5)
                cols = np.random.choice(r, nnz, replace=False)
                for c in cols:
                    V[b, c] = np.random.choice([-1, 1])
            
            # Solve for W using Least Squares
            # We want sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
            # This is linear in W[c, :] for each c.
            # For each c, we have nm*mp equations (mostly 0, some 1).
            # Let's build the system for W.
            
            W_sol = np.zeros((pn, r))
            valid = True
            
            for c in range(pn):
                # Build matrix A_c such that A_c * W[c,:] = t_c
                # A_c has nm*mp rows and r columns
                # Row (a,b) is U[a,:] * V[b,:]
                # t_c(a,b) = 1 if (a,b,c) in nz, else 0
                
                # To save memory and time, we can use the fact that most are 0.
                # But r is small (20-40), nm*mp is 600.
                # 600x20 matrix is small.
                
                A_c = np.zeros((nm * mp, r))
                t_c = np.zeros(nm * mp)
                
                idx = 0
                for a in range(nm):
                    for b in range(mp):
                        A_c[idx, :] = U[a, :] * V[b, :]
                        if (a, b, c) in set(nz_indices):
                            t_c[idx] = 1.0
                        idx += 1
                
                # Solve A_c x = t_c
                try:
                    W_c, residuals, rank_c, sv = np.linalg.lstsq(A_c, t_c, rcond=None)
                except:
                    valid = False
                    break
                
                W_sol[c, :] = W_c
                
                # Check if the solution is consistent (residual is small)
                if len(residuals) > 0 and residuals[0] > 1e-6:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Round W to nearest integer or half-integer
            # The problem allows integer/half-integer entries.
            # Let's try rounding to nearest 0.5
            W_rounded = np.round(W_sol * 2) / 2
            
            # Now we have U, V (integers), W_rounded (half-integers).
            # We should also try to round U and V?
            # The ALS step only solved for W. U and V were fixed.
            # This is a "Fix U, V, Solve W" approach.
            # If W_rounded is a good fit, we can try to refine U and V.
            
            # Let's verify if (U, V, W_rounded) is an exact solution.
            # Convert to Fractions for exact check
            U_frac = [[Fraction(U[a, r]) for r in range(r)] for a in range(nm)]
            V_frac = [[Fraction(V[b, r]) for r in range(r)] for b in range(mp)]
            W_frac = [[Fraction(W_rounded[c, r]) for r in range(r)] for c in range(pn)]
            
            if verify_exact(U_frac, V_frac, W_frac):
                best_result = {"U": U_frac, "V": V_frac, "W": W_frac}
                return best_result # Found a solution, return immediately
            
            # If not exact, we can try to adjust U and V?
            # For now, we just rely on the random search hitting a valid configuration.
            
    # If no solution found in the specific loop, try a more general ALS with rounding
    # Or just return fallback
    
    return fallback
