import numpy as np
import time
import sys

def coefficients(stages):
    """
    Compute 12-stage product formula coefficients for Heisenberg chain time evolution.
    Returns (a, b) where each is a list of 12 floats summing to 1.
    """
    start_time = time.time()
    # Use 140s as a safety margin for the 150s limit
    deadline = start_time + 140.0 

    # --- 1. Build Hamiltonians ---
    N = 6
    dim = 2 ** N
    lambda_aniso = 0.8
    h_field = 0.3

    # Single-site spin operators
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)

    def op_on_site(op, site, N):
        ops = [np.eye(2, dtype=complex)] * N
        ops[site] = op
        result = ops[0]
        for i in range(1, N):
            result = np.kron(result, ops[i])
        return result

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    # Field term: h * sum Sz_i / 2
    # Problem states "each including half of the uniform field"
    field_term = 0.5 * h_field
    for i in range(N):
        sz_i = op_on_site(sz, i, N)
        A += field_term * sz_i
        B += field_term * sz_i

    # Bond terms: Sx_i Sx_j + Sy_i Sy_j + lambda Sz_i Sz_j
    def add_bond(H, i, j):
        sx_i = op_on_site(sx, i, N)
        sx_j = op_on_site(sx, j, N)
        sy_i = op_on_site(sy, i, N)
        sy_j = op_on_site(sy, j, N)
        sz_i = op_on_site(sz, i, N)
        sz_j = op_on_site(sz, j, N)
        
        H += sx_i @ sx_j
        H += sy_i @ sy_j
        H += lambda_aniso * (sz_i @ sz_j)

    # Odd bonds for A: (0,1), (2,3), (4,5)
    add_bond(A, 0, 1)
    add_bond(A, 2, 3)
    add_bond(A, 4, 5)
    
    # Even bonds for B: (1,2), (3,4)
    add_bond(B, 1, 2)
    add_bond(B, 3, 4)

    # --- 2. Precompute Eigendecompositions ---
    # A and B are Hermitian, so eigh is stable and fast
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    
    # Precompute conjugate transposes for efficiency
    eigvecs_A_dag = eigvecs_A.conj().T
    eigvecs_B_dag = eigvecs_B.conj().T

    # Exact propagator
    H_total = A + B
    eigvals_H, eigvecs_H = np.linalg.eigh(H_total)
    eigvecs_H_dag = eigvecs_H.conj().T

    t_values = [0.25, 0.5, 1.0]
    U_exacts = []
    for t in t_values:
        exp_H = np.exp(-1j * t * eigvals_H)
        U_exact = (eigvecs_H * exp_H) @ eigvecs_H_dag
        U_exacts.append(U_exact)

    # --- 3. Objective Function ---
    def compute_error(a, b):
        """
        Compute mean spectral norm error for the given coefficients.
        a, b: lists of 12 floats.
        """
        if time.time() > deadline:
            return float('inf')
        
        total_err = 0.0
        for i, t in enumerate(t_values):
            U_approx = np.eye(dim, dtype=complex)
            for k in range(12):
                # Step B: exp(-i b_k t B)
                exp_B = np.exp(-1j * b[k] * t * eigvals_B)
                U_B = (eigvecs_B * exp_B) @ eigvecs_B_dag
                U_approx = U_B @ U_approx
                
                # Step A: exp(-i a_k t A)
                exp_A = np.exp(-1j * a[k] * t * eigvals_A)
                U_A = (eigvecs_A * exp_A) @ eigvecs_A_dag
                U_approx = U_A @ U_approx
            
            diff = U_approx - U_exacts[i]
            # Spectral norm is max singular value
            # For small errors, this is the L2 norm of the matrix in a sense, 
            # but strictly it's the largest singular value.
            # np.linalg.svd is expensive. 
            # However, for 64x64, it's fast enough if we limit iterations.
            # Alternatively, we can use np.linalg.norm(diff, 2) which computes SVD.
            err = np.linalg.norm(diff, 2)
            total_err += err
            
        return total_err / len(t_values)

    def objective(params):
        """
        params: 24-element array. First 12 are a, last 12 are b.
        We normalize them to sum to 1 inside the objective to handle constraints softly,
        or we can enforce them in the optimizer. 
        Given Nelder-Mead doesn't handle constraints well, we use a penalty or normalization.
        Normalization is safer for the "sum=1" constraint.
        """
        if time.time() > deadline:
            return float('inf')
        
        a_raw = params[:12]
        b_raw = params[12:]
        
        # Normalize to sum to 1
        sum_a = np.sum(a_raw)
        sum_b = np.sum(b_raw)
        
        # If sum is close to 0, return large penalty
        if abs(sum_a) < 1e-10 or abs(sum_b) < 1e-10:
            return 1e10
            
        a_norm = a_raw / sum_a
        b_norm = b_raw / sum_b
        
        # Check bounds [-50, 50]
        if np.any(a_norm < -50) or np.any(a_norm > 50) or np.any(b_norm < -50) or np.any(b_norm > 50):
            return 1e10

        return compute_error(a_norm, b_norm)

    # --- 4. Nelder-Mead Optimizer ---
    def nelder_mead(func, x0, max_iter=500, tol=1e-8, verbose=False):
        n = len(x0)
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initialize simplex
        simplex = [np.array(x0, dtype=float)]
        for i in range(n):
            point = np.array(x0, dtype=float)
            if abs(point[i]) > 1e-10:
                point[i] *= 1 + 0.05
            else:
                point[i] = 0.05
            simplex.append(point)
        
        fvals = [func(p) for p in simplex]
        best_val = min(fvals)
        best_point = simplex[np.argmin(fvals)]
        
        for it in range(max_iter):
            if time.time() > deadline - 5: # Leave 5s for return
                break
            
            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Update best
            if fvals[0] < best_val:
                best_val = fvals[0]
                best_point = simplex[0].copy()
                
            # Check convergence
            if abs(fvals[0] - fvals[-1]) < tol:
                break
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            f_r = func(x_r)
            
            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                f_e = func(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < fvals[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (simplex[-1] - centroid)
                
                f_c = func(x_c)
                if f_c < fvals[-1]:
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
            
            if verbose and it % 50 == 0:
                print(f"Iter {it}, Best: {best_val}")

        return best_point, best_val

    # --- 5. Optimization Strategy ---
    # Start with uniform distribution
    x0_base = np.ones(24) / 12.0
    
    best_x = None
    best_score = float('inf')
    
    # Try multiple restarts
    num_restarts = 3
    rng = np.random.default_rng(42)
    
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        if r == 0:
            x0 = x0_base
        elif r == 1:
            # Symmetric initialization: a = b, symmetric sequence
            # x = [x1, x2, x3, x4, x5, x6, x6, x5, x4, x3, x2, x1]
            # Sum x = 1 => 2*sum(x1..x6) = 1 => sum(x1..x6) = 0.5
            sym_half = rng.random(6)
            sym_half = sym_half / np.sum(sym_half) * 0.5
            seq = np.array([sym_half[0], sym_half[1], sym_half[2], sym_half[3], sym_half[4], sym_half[5], 
                            sym_half[5], sym_half[4], sym_half[3], sym_half[2], sym_half[1], sym_half[0]])
            x0 = np.concatenate([seq, seq])
        else:
            # Random initialization with normalization
            a_rand = rng.random(12)
            b_rand = rng.random(12)
            a_rand = a_rand / np.sum(a_rand)
            b_rand = b_rand / np.sum(b_rand)
            x0 = np.concatenate([a_rand, b_rand])
            
        # Run Nelder-Mead
        # Limit iterations based on remaining time
        remaining_time = deadline - time.time()
        # Rough estimate: ~10-20ms per eval? Let's allow ~100 iterations per restart.
        max_iter = 150
        
        try:
            x_opt, score = nelder_mead(objective, x0, max_iter=max_iter)
            if score < best_score:
                best_score = score
                best_x = x_opt
        except Exception:
            pass

    if best_x is None:
        best_x = x0_base

    # --- 6. Finalize Result ---
    a_final = best_x[:12]
    b_final = best_x[12:]
    
    # Ensure sums are exactly 1
    sum_a = np.sum(a_final)
    sum_b = np.sum(b_final)
    
    if abs(sum_a) > 1e-10:
        a_final = a_final / sum_a
    else:
        a_final = np.ones(12) / 12.0
        
    if abs(sum_b) > 1e-10:
        b_final = b_final / sum_b
    else:
        b_final = np.ones(12) / 12.0
        
    # Clip to [-50, 50] just in case, though normalization usually keeps them small
    a_final = np.clip(a_final, -50, 50)
    b_final = np.clip(b_final, -50, 50)
    
    # Re-normalize after clipping to ensure sum=1 strictly
    sum_a = np.sum(a_final)
    if sum_a != 0:
        a_final = a_final / sum_a
    sum_b = np.sum(b_final)
    if sum_b != 0:
        b_final = b_final / sum_b

    return [float(v) for v in a_final], [float(v) for v in b_final]
