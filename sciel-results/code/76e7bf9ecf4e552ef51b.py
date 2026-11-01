import numpy as np
import time
from fractions import Fraction
import itertools

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Schoolbook decomposition (rank 12) as fallback
    rank_school = n * m * p
    U_sb = np.zeros((n * m, rank_school))
    V_sb = np.zeros((m * p, rank_school))
    W_sb = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb.tolist(), V_sb.tolist(), W_sb.tolist()
    best_rank = rank_school

    def get_target_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T
    
    T_target = get_target_tensor()
    
    def compute_tensor(U, V, W):
        return np.einsum('ir,jr,kr->ijk', U, V, W)
    
    def verify(U, V, W):
        T = compute_tensor(U, V, W)
        return np.allclose(T, T_target, atol=1e-8)

    def try_round_and_verify(U_num, V_num, W_num, denom=2):
        U_rnd = np.round(U_num * denom) / denom
        V_rnd = np.round(V_num * denom) / denom
        W_rnd = np.round(W_num * denom) / denom
        if verify(U_rnd, V_rnd, W_rnd):
            return U_rnd, V_rnd, W_rnd, True
        return None, None, None, False

    # Strategy: Structured Search for Rank 10
    # We assume a structure based on the known lower bounds and typical decompositions for 2x2x3.
    # The tensor is T_{ijk} = 1 if indices align.
    # A common structure for rank r decompositions of matrix multiplication involves
    # "bit" or "digit" like terms.
    
    # Instead of guessing, we will use a "Symbolic Linear Solve" approach on a reduced parameter space.
    # We parameterize U, V, W with a smaller set of variables and solve the linear constraints.
    
    # For 2x2x3, the tensor has 12 non-zero entries.
    # Rank 10 is the target.
    
    # Let's try to construct a rank-10 decomposition using a specific ansatz.
    # One known approach is to use the "strassen-like" structure but adapted.
    # However, since we don't have a precomputed rank-10 solution, we will use a 
    # "Sparse Linear Programming" style approach but with a smarter initialization.
    
    # Actually, a more robust "different" strategy is **Algebraic Geometry / Groebner Basis** 
    # is too heavy. 
    # Let's use **Randomized Linear Algebra with Rounding**.
    # But the previous program did that.
    
    # The directive asks for a STRUCTURAL change. 
    # Let's try **Explicit Construction via Known Identities**.
    # For 2x2x3, is there a known rank 10 or 11 solution?
    # Lower bound is 10 (from asymptotic or specific bounds? Actually LB for 2x2x3 is often cited around 10-11).
    # Let's try to find a rank 11 solution first as it might be easier, then rank 10.
    
    # New Strategy: **Perturbation of Schoolbook with Linear Dependencies**
    # The schoolbook has rank 12. 
    # If we can find 2 linear dependencies among the 12 rank-1 tensors that allow us to 
    # replace 2 terms with 1 term (reducing rank by 1) or similar, we can reduce rank.
    # Specifically, if sum_{r=1}^12 c_r T_r = 0, we might be able to eliminate terms.
    
    # Let's look for a rank 11 solution by solving a linear system.
    # Assume we keep 11 of the schoolbook terms and add 1 new term to fix the error? No, that increases rank.
    # We need to replace 2 schoolbook terms with 1 new term? That would be rank 11.
    # Or replace 3 with 2? Rank 10.
    
    # Let's try to find a rank 10 solution by parameterizing 10 rank-1 tensors 
    # and solving for the coefficients using a subset of the tensor equations.
    
    # We have 12 non-zero target entries. 
    # We have 10 * (4 + 6 + 6) = 160 variables. 
    # This is under-determined. 
    # We can fix some variables to integers/half-integers to reduce the search space.
    
    # Let's try a **Genetic Algorithm on Integer/Half-Integer Coefficients**.
    # This is a discrete search, structurally different from continuous DE.
    
    import random
    np.random.seed(42)
    random.seed(42)
    
    def generate_random_half_integers(size, max_val=3):
        # Generate array of size 'size' with values in {-max_val/2, ..., max_val/2}
        # Multiples of 0.5
        vals = np.random.randint(-max_val, max_val + 1, size=size) * 0.5
        return vals

    def evaluate_rank_10_candidate(U, V, W):
        T = compute_tensor(U, V, W)
        return np.sum((T - T_target)**2)

    # We will search for rank 10 solutions using a discrete search space.
    # To make it efficient, we will fix the structure of U, V, W partially.
    # For example, we can assume U, V, W have entries in {-1, -0.5, 0, 0.5, 1}.
    
    # This is a huge space. 11^160 is impossible.
    # We need to exploit sparsity or structure.
    
    # Alternative Structural Strategy: **Use the known rank-10 decomposition if it exists in literature?**
    # I don't have it memorized. 
    
    # Let's try **Simulated Annealing on the Continuous Parameters** but with a 
    # **Hard Constraint Projection** to the nearest half-integers at each step?
    # No, that's not standard.
    
    # Let's try **Local Search on a Discrete Lattice**.
    # Start from a good continuous approximation (e.g., from a quick DE run or random) 
    # and then perform local moves on the lattice of half-integers.
    
    # Step 1: Get a rough continuous solution using a very fast, low-budget DE or random restarts.
    # Step 2: Round to nearest half-integers.
    # Step 3: Perform local discrete optimization (swap values in U, V, W) to minimize error.
    
    target_ranks = [10, 11]
    
    found = False
    
    for target_rank in target_ranks:
        if found or time.time() > deadline - 20.0:
            break
        
        nm, mp, pn = n*m, m*p, p*n
        total_params = (nm + mp + pn) * target_rank
        
        # 1. Find a good continuous starting point
        # Use a few random restarts with L-BFGS (fast) to find a low-error continuous solution
        best_continuous = None
        best_err = np.inf
        
        for i in range(10):
            if time.time() > deadline - 30.0:
                break
            params0 = np.random.randn(total_params) * 1.0
            # Ensure some structure? No, just random.
            
            def objective(params):
                U_flat = params[:nm*target_rank]
                V_flat = params[nm*target_rank:nm*target_rank + mp*target_rank]
                W_flat = params[nm*target_rank + mp*target_rank:]
                U = U_flat.reshape(nm, target_rank)
                V = V_flat.reshape(mp, target_rank)
                W = W_flat.reshape(pn, target_rank)
                T_prime = compute_tensor(U, V, W)
                diff = T_prime - T_target
                return np.sum(diff**2)

            try:
                from scipy.optimize import minimize
                res = minimize(objective, params0, method='L-BFGS-B',
                             options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.fun < best_err:
                    best_err = res.fun
                    best_continuous = res.x
            except Exception:
                pass
        
        if best_continuous is None:
            continue
            
        # 2. Round to half-integers
        U_flat = best_continuous[:nm*target_rank]
        V_flat = best_continuous[nm*target_rank:nm*target_rank + mp*target_rank]
        W_flat = best_continuous[nm*target_rank + mp*target_rank:]
        
        U = np.round(U_flat * 2) / 2.0
        V = np.round(V_flat * 2) / 2.0
        W = np.round(W_flat * 2) / 2.0
        
        U = U.reshape(nm, target_rank)
        V = V.reshape(mp, target_rank)
        W = W.reshape(pn, target_rank)
        
        current_err = evaluate_rank_10_candidate(U, V, W)
        
        if current_err < 1e-8:
            best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
            best_rank = target_rank
            found = True
            break
            
        # 3. Local Discrete Search (Hill Climbing)
        # We will try to improve the solution by changing single entries to nearby half-integers.
        # This is a structural change: we are navigating the discrete lattice directly.
        
        max_iters = 5000
        step_size = 0.5
        allowed_values = np.arange(-5, 5.1, step_size) # -5 to 5 in steps of 0.5
        
        for iter in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            improved = False
            # Randomly select a matrix and an entry to change
            mat_choice = np.random.randint(0, 3)
            if mat_choice == 0:
                mat_name = 'U'
                mat = U
                shape = U.shape
            elif mat_choice == 1:
                mat_name = 'V'
                mat = V
                shape = V.shape
            else:
                mat_name = 'W'
                mat = W
                shape = W.shape
                
            r, c = np.random.randint(0, shape[0]), np.random.randint(0, shape[1])
            old_val = mat[r, c]
            
            # Choose a new value from allowed_values, different from old_val
            candidates = [v for v in allowed_values if abs(v - old_val) > 1e-9]
            if not candidates:
                continue
            new_val = np.random.choice(candidates)
            
            # Apply change
            if mat_name == 'U':
                U[r, c] = new_val
            elif mat_name == 'V':
                V[r, c] = new_val
            else:
                W[r, c] = new_val
                
            new_err = evaluate_rank_10_candidate(U, V, W)
            
            if new_err < current_err:
                current_err = new_err
                improved = True
                if current_err < 1e-8:
                    best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
                    best_rank = target_rank
                    found = True
                    break
            else:
                # Revert
                if mat_name == 'U':
                    U[r, c] = old_val
                elif mat_name == 'V':
                    V[r, c] = old_val
                else:
                    W[r, c] = old_val
            
            # If stuck for a long time, break? 
            # For simplicity, just continue.
            
        if found:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
