"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _build_target_tensor(n, m, p):
    """Build the target multiplication tensor as a 3D array."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A = i * m + j
                B = j * p + k
                C = k * n + i
                T[A, B, C] = 1.0
    return T

def _verify_factors(U, V, W, n, m, p, tol=1e-6):
    """Verify decomposition using einsum."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    T_target = _build_target_tensor(n, m, p)
    return np.allclose(T_recon, T_target, atol=tol), T_recon - T_target

def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _try_constructive_rank(n, m, p, R, time_budget=30):
    """
    Try to find a rank-R decomposition using iterative optimization.
    Uses a smooth objective: minimize ||T_recon - T_target||_F^2
    with entries constrained to be near half-integers via soft projection.
    """
    start_time = time.time()
    T_target = _build_target_tensor(n, m, p)
    nm, mp, pn = n * m, m * p, p * n
    
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    # Multiple random restarts
    n_restarts = max(3, int(time_budget / 5))
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget:
            break
        
        rng = np.random.RandomState(1000 + restart)
        
        # Initialize with small random values
        U = rng.uniform(-1, 1, (nm, R))
        V = rng.uniform(-1, 1, (mp, R))
        W = rng.uniform(-1, 1, (pn, R))
        
        # Gradient descent with soft half-integer projection
        lr = 0.01
        for step in range(20000):
            if time.time() - start_time > time_budget:
                break
            
            # Compute reconstruction
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = T_recon - T_target
            err_f = np.sum(err * err)
            
            if err_f < best_err:
                best_err = err_f
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err_f < 1e-12:
                break
            
            # Compute gradients
            # dT/dU[A,r] = sum_{B,C} err[A,B,C] * V[B,r] * W[C,r]
            # Efficiently: for each r, grad_U[:,r] = (err * outer(V[r], W[r])) summed over B,C
            # Actually: grad_U[A,r] = sum_{B,C} err[A,B,C] * V[B,r] * W[C,r]
            # This is: for each r, grad_U[:,r] = np.einsum('abc,br,cr->ac', err, V[:,r], W[:,r])
            # But that's expensive. Let's use:
            # grad_U = np.einsum('abc,br,cr->ar', err, V, W)
            # grad_V = np.einsum('abc,ar,cr->br', err, U, W)
            # grad_W = np.einsum('abc,ar,br->cr', err, U, V)
            
            grad_U = np.einsum('abc,br,cr->ar', err, V, W)
            grad_V = np.einsum('abc,ar,cr->br', err, U, W)
            grad_W = np.einsum('abc,ar,br->cr', err, U, V)
            
            # Soft projection toward half-integers:
            # Add a penalty that pushes values toward nearest 0.5*integer
            # penalty = sum (x - round(2x)/2)^2, gradient = 2*(x - round(2x)/2)
            # Simplified: just add a small pull toward nearest half-integer
            half_int_U = np.round(U * 2) / 2
            half_int_V = np.round(V * 2) / 2
            half_int_W = np.round(W * 2) / 2
            
            pull_strength = 0.001
            grad_U += pull_strength * (U - half_int_U)
            grad_V += pull_strength * (V - half_int_V)
            grad_W += pull_strength * (W - half_int_W)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Decay learning rate
            lr *= 0.999
    
    # Try to round best solution to half-integers and verify
    if best_U is not None and best_err < 1e-6:
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        ok, _ = _verify_factors(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p)
        if ok:
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    return None

def _try_fixed_pattern(n, m, p, R):
    """
    Try a structured pattern for the decomposition.
    For M_{2,3,3}, try to exploit the n=2 structure.
    
    Idea: Split into two 1x3x3 problems (for i=0 and i=1) and combine.
    Each 1x3x3 has rank 9 (schoolbook). Total would be 18.
    But we can share some structure.
    
    Alternative: Use a block structure where we group by j (middle index).
    For each j, we have a 2x1x3 sub-problem (i=0,1; k=0,1,2).
    The sub-tensor for fixed j: U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r]
    This is like a 2x3 matrix multiplication with rank R_j per j.
    
    For a 2x3 multiplication (output 2x3 from input 2x3... wait, let me think again.
    
    Actually for fixed j: we need sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r] = 1 for all i,k.
    This is equivalent to: the matrix (U[:,j_col] * V[:,j_row]) outer product structure...
    
    Let me try a different approach: assume a specific algebraic form.
    """
    # Try a known construction pattern
    # For M_{2,3,3}, one approach:
    # Use rank 11 with specific patterns
    
    # Let's try to solve a smaller system
    # Assume W has a specific structure: W[k*2+i, r] depends on (k,i)
    
    # Pattern 1: Try to make V sparse
    # For each r, V[j*3+k, r] is nonzero for at most one (j,k) pair
    # Then U[i*3+j, r] and W[k*2+i, r] must be consistent
    
    # This is essentially the schoolbook approach with possible sharing.
    # Let's try to find sharing opportunities.
    
    # For rank 11, we need to save 7 columns from schoolbook's 18.
    # This means 7 "savings" where multiple (i,j,k) triples share a column.
    
    # A column r contributes to T[A,B,C] = U[A,r]*V[B,r]*W[C,r]
    # For it to be valid, for each (A,B,C) in support, the sum over r must be 1,
    # and for (A,B,C) not in support, the sum must be 0.
    
    # Let's try a constructive approach with specific patterns.
    
    nm, mp, pn = n*m, m*p, p*n
    
    # Try: for each r, define support sets S_U(r), S_V(r), S_W(r)
    # where U[A,r] != 0 only for A in S_U(r), etc.
    # The product U[A,r]*V[B,r]*W[C,r] is nonzero only when A in S_U, B in S_V, C in S_W.
    # For the decomposition to work, for each (A,B,C) in the tensor support,
    # exactly the right combination must sum to 1.
    
    # This is a constraint satisfaction problem. Let's try small patterns.
    
    # Pattern: each column r has U nonzero on 1-2 rows, V on 1-2 rows, W on 1-2 rows.
    # Try all combinations of small support patterns.
    
    # For efficiency, let's try a specific known construction.
    # Based on literature, M_{2,3,3} can be done in rank 11.
    
    # Let me try to hard-code a known solution if I can derive one.
    # 
    # Alternative: use the fact that M_{2,3,3} embeds M_{2,2,2} (Strassen, rank 7)
    # plus additional terms.
    
    # Let's try: decompose into M_{2,2,2} (using columns 0-6, rank 7)
    # plus M_{2,1,1} for the remaining j=2 column (rank 2)
    # plus cross terms...
    
    # This is getting complex. Let's just return None and rely on the optimizer.
    return None

def _try_integer_lp_approach(n, m, p, R, time_budget=20):
    """
    Try to find a decomposition with entries in {-1, 0, 1} or {-1, -0.5, 0, 0.5, 1}.
    Use a randomized local search.
    """
    start_time = time.time()
    T_target = _build_target_tensor(n, m, p)
    nm, mp, pn = n*m, m*p, p*n
    
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    values = [-1, -0.5, 0, 0.5, 1]
    n_values = len(values)
    
    for restart in range(10):
        if time.time() - start_time > time_budget:
            break
        
        rng = random.Random(5000 + restart)
        
        U = np.array([[rng.choice(values) for _ in range(R)] for _ in range(nm)], dtype=np.float64)
        V = np.array([[rng.choice(values) for _ in range(R)] for _ in range(mp)], dtype=np.float64)
        W = np.array([[rng.choice(values) for _ in range(R)] for _ in range(pn)], dtype=np.float64)
        
        for step in range(50000):
            if time.time() - start_time > time_budget:
                break
            
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = T_recon - T_target
            err_f = np.sum(err * err)
            
            if err_f < best_err:
                best_err = err_f
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err_f < 1e-12:
                break
            
            # Try flipping a random entry
            mat = rng.choice(['U', 'V', 'W'])
            if mat == 'U':
                A, r = rng.randint(0, nm-1), rng.randint(0, R-1)
                old = U[A, r]
                new = rng.choice(values)
                U[A, r] = new
            elif mat == 'V':
                B, r = rng.randint(0, mp-1), rng.randint(0, R-1)
                old = V[B, r]
                new = rng.choice(values)
                V[B, r] = new
            else:
                C, r = rng.randint(0, pn-1), rng.randint(0, R-1)
                old = W[C, r]
                new = rng.choice(values)
                W[C, r] = new
            
            # Recompute error (incremental would be better but this is simpler)
            T_recon_new = np.einsum('ar,br,cr->abc', U, V, W)
            err_new = T_recon_new - T_target
            err_f_new = np.sum(err_new * err_new)
            
            # Accept if better or with small probability (simulated annealing)
            if err_f_new < err_f or (err_f < 1e-6 and rng.random() < 0.1):
                pass  # keep the change
            else:
                # Revert
                if mat == 'U':
                    U[A, r] = old
                elif mat == 'V':
                    V[B, r] = old
                else:
                    W[C, r] = old
    
    if best_U is not None and best_err < 1e-9:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145  # 150s limit, 5s safety margin
    
    # First, try known lower-rank constructions
    # Try ranks from 11 upward
    for R in [11, 12, 13, 14, 15, 16, 17]:
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        remaining = deadline - elapsed - 5
        if remaining < 3:
            break
        
        # Try constructive pattern first (fast)
        result = _try_fixed_pattern(n, m, p, R)
        if result is not None:
            U, V, W = result
            ok, _ = _verify_factors(U, V, W, n, m, p)
            if ok:
                return {"U": U, "V": V, "W": W}
        
        # Try integer local search
        budget = min(remaining * 0.4, 15)
        result = _try_integer_lp_approach(n, m, p, R, time_budget=budget)
        if result is not None:
            U, V, W = result
            ok, _ = _verify_factors(U, V, W, n, m, p)
            if ok:
                return {"U": U, "V": V, "W": W}
        
        # Try continuous optimization
        budget = min(remaining * 0.6, 20)
        result = _try_constructive_rank(n, m, p, R, time_budget=budget)
        if result is not None:
            U, V, W = result
            ok, _ = _verify_factors(U, V, W, n, m, p)
            if ok:
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook (rank 18)
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
