import numpy as np
import time

def _build_hamiltonians():
    """
    Builds A and B for the 6-site XXZ Heisenberg chain.
    Sites 0..5.
    Bonds:
    0: (0,1) -> Odd -> A
    1: (1,2) -> Even -> B
    2: (2,3) -> Odd -> A
    3: (3,4) -> Even -> B
    4: (4,5) -> Odd -> A
    """
    N = 6
    dim = 2**N
    lambda_xz = 0.8
    h_field = 0.3
    
    # Spin operators (0.5 * Pauli)
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    def op_on_site(op, site):
        ops = []
        for i in range(N):
            if i == site:
                ops.append(op)
            else:
                ops.append(np.eye(2, dtype=complex))
        res = ops[0]
        for i in range(1, N):
            res = np.kron(res, ops[i])
        return res

    def bond_term(site1, site2):
        def two_site_op(op1, s1, op2, s2):
            ops = []
            for i in range(N):
                if i == s1:
                    ops.append(op1)
                elif i == s2:
                    ops.append(op2)
                else:
                    ops.append(np.eye(2, dtype=complex))
            res = ops[0]
            for i in range(1, N):
                res = np.kron(res, ops[i])
            return res

        return (two_site_op(Sx, site1, Sx, site2) + 
                two_site_op(Sy, site1, Sy, site2) + 
                lambda_xz * two_site_op(Sz, site1, Sz, site2))

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    A += bond_term(0, 1)
    B += bond_term(1, 2)
    A += bond_term(2, 3)
    B += bond_term(3, 4)
    A += bond_term(4, 5)
    
    H_field = np.zeros((dim, dim), dtype=complex)
    for j in range(N):
        H_field += op_on_site(Sz, j)
        
    A += 0.5 * h_field * H_field
    B += 0.5 * h_field * H_field
    
    return A, B

def _nelder_mead(f, x0, max_iter=1000, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5, tol=1e-10, time_limit=None, start_time=None):
    n = len(x0)
    
    simplex = [x0.copy()]
    for i in range(n):
        x_new = x0.copy()
        # Initial step size
        step = 0.1
        x_new[i] += step
        simplex.append(x_new)
        
    values = [f(x) for x in simplex]
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
            
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        if abs(values[0] - values[-1]) < tol:
            break
            
        x_best = simplex[0]
        x_worst = simplex[-1]
        x_second_worst = simplex[-2]
        
        x_centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = x_centroid + alpha * (x_centroid - x_worst)
        f_r = f(x_r)
        
        if f_r < values[0]:
            # Expansion
            x_e = x_centroid + gamma * (x_r - x_centroid)
            f_e = f(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                values[-1] = f_e
            else:
                simplex[-1] = x_r
                values[-1] = f_r
        elif f_r < values[-2]:
            # Accept reflection
            simplex[-1] = x_r
            values[-1] = f_r
        else:
            # Contraction
            x_c = x_centroid + rho * (x_worst - x_centroid)
            f_c = f(x_c)
            if f_c < values[-1]:
                simplex[-1] = x_c
                values[-1] = f_c
            else:
                # Shrink
                for i in range(1, n+1):
                    simplex[i] = x_best + sigma * (simplex[i] - x_best)
                    values[i] = f(simplex[i])
                    
    order = np.argsort(values)
    return simplex[order[0]], values[order[0]]

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin for 150s limit
    
    A, B = _build_hamiltonians()
    H = A + B
    
    # Precompute exact propagators
    eigvals, eigvecs = np.linalg.eigh(H)
    
    times = [0.25, 0.5, 1.0]
    U_exact = []
    for t in times:
        phase = np.exp(-1j * t * eigvals)
        U_t = eigvecs @ np.diag(phase) @ eigvecs.conj().T
        U_exact.append(U_t)
        
    # Precompute eigendecomps of A and B
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    
    def get_exp_A(t_coeff):
        if abs(t_coeff) < 1e-12:
            return np.eye(64, dtype=complex)
        phase = np.exp(-1j * t_coeff * eigvals_A)
        return eigvecs_A @ np.diag(phase) @ eigvecs_A.conj().T
        
    def get_exp_B(t_coeff):
        if abs(t_coeff) < 1e-12:
            return np.eye(64, dtype=complex)
        phase = np.exp(-1j * t_coeff * eigvals_B)
        return eigvecs_B @ np.diag(phase) @ eigvecs_B.conj().T

    def objective(params):
        a_free = params[:11]
        b_free = params[11:22]
        
        a_full = np.concatenate([a_free, [1.0 - np.sum(a_free)]])
        b_full = np.concatenate([b_free, [1.0 - np.sum(b_free)]])
        
        # Bounds check
        if np.any(np.abs(a_full) > 50) or np.any(np.abs(b_full) > 50):
            return 1e10
            
        err_sum = 0.0
        for i, t in enumerate(times):
            U_approx = np.eye(64, dtype=complex)
            for k in range(12):
                E_A_k = get_exp_A(a_full[k] * t)
                E_B_k = get_exp_B(b_full[k] * t)
                # U = E_B1 E_A1 E_B2 E_A2 ...
                U_approx = E_B_k @ E_A_k @ U_approx
                
            diff = U_approx - U_exact[i]
            # Spectral norm
            err = np.linalg.norm(diff, 2)
            err_sum += err
            
        return err_sum / len(times)
        
    # Initial guess
    # Try a few different initializations if time permits
    # Standard uniform
    x0 = np.full(22, 1.0/12.0)
    
    # Run Nelder-Mead
    # Use a lot of iterations but respect time limit
    x_best, val_best = _nelder_mead(objective, x0, max_iter=3000, time_limit=135.0, start_time=start_time)
    
    # If time remains, try a second restart with a perturbed initial point
    if time.time() < deadline - 30.0:
        x0_2 = x0 + np.random.normal(0, 0.01, 22)
        x_best_2, val_best_2 = _nelder_mead(objective, x0_2, max_iter=1000, time_limit=25.0, start_time=start_time)
        if val_best_2 < val_best:
            x_best = x_best_2
            val_best = val_best_2
            
    a_free = x_best[:11]
    b_free = x_best[11:22]
    
    a = list(a_free)
    a.append(1.0 - np.sum(a_free))
    
    b = list(b_free)
    b.append(1.0 - np.sum(b_free))
    
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    
    return a, b

if __name__ == "__main__":
    a, b = coefficients(12)
    print(f"Sum a: {sum(a)}, Sum b: {sum(b)}")
