import numpy as np
import time

def coefficients(stages):
    start_time = time.time()
    time_limit = 145.0 # Safety margin for the 150s kill switch
    
    # 1. Build Hamiltonians
    sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2.0
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2.0
    sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2.0
    s0 = np.eye(2, dtype=complex)
    
    A = np.zeros((64, 64), dtype=complex)
    B = np.zeros((64, 64), dtype=complex)
    
    def kron_all(ops):
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res
    
    def add_bond(H, i, j, anisotropy):
        def make_term(op_i, op_j):
            ops = []
            for k in range(6):
                if k == i:
                    ops.append(op_i)
                elif k == j:
                    ops.append(op_j)
                else:
                    ops.append(s0)
            return kron_all(ops)
            
        H += make_term(sx, sx)
        H += make_term(sy, sy)
        H += anisotropy * make_term(sz, sz)
        
    add_bond(A, 0, 1, 0.8)
    add_bond(A, 2, 3, 0.8)
    add_bond(A, 4, 5, 0.8)
    
    add_bond(B, 1, 2, 0.8)
    add_bond(B, 3, 4, 0.8)
    
    field_strength = 0.3
    for j in range(6):
        ops = []
        for k in range(6):
            if k == j:
                ops.append(sz)
            else:
                ops.append(s0)
        H_field_j = kron_all(ops)
        A += 0.5 * field_strength * H_field_j
        B += 0.5 * field_strength * H_field_j

    # Helper: Matrix exponential for Hermitian matrices using eigen-decomposition
    # Since A and B are sums of Pauli terms (Hermitian), their eigenvalues are real.
    # exp(-i * t * H) = V @ diag(exp(-i * t * w)) @ V^-1
    # For Hermitian H, V is unitary, so V^-1 = V^H.
    
    def expm_hermitian(H, t):
        # H is Hermitian. We want exp(-1j * t * H)
        # Note: scipy.linalg.expm handles general matrices, but we can be faster for Hermitian
        # using eigh.
        w, V = np.linalg.eigh(H)
        # w is real, V is unitary
        # exp(-1j * t * w)
        phase = np.exp(-1j * t * w)
        # Reconstruct
        return (V * phase) @ V.conj().T

    # Precompute eigen-decompositions of A and B to speed up expm calls
    # A and B are constant.
    wA, VA = np.linalg.eigh(A)
    wB, VB = np.linalg.eigh(B)
    
    def expm_A(t):
        phase = np.exp(-1j * t * wA)
        return (VA * phase) @ VA.conj().T

    def expm_B(t):
        phase = np.exp(-1j * t * wB)
        return (VB * phase) @ VB.conj().T

    # Precompute exact propagators
    t_values = [0.25, 0.5, 1.0]
    H_total = A + B
    wT, VT = np.linalg.eigh(H_total)
    
    exact_props = []
    for t in t_values:
        phase = np.exp(-1j * t * wT)
        exact_props.append((VT * phase) @ VT.conj().T)

    # 3. Define cost function with symmetry
    # Variables: a1..a5, b1..b5
    # a6 = 0.5 - sum(a1..a5)
    # b6 = 0.5 - sum(b1..b5)
    # Full a: [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    
    def cost(x):
        a1, a2, a3, a4, a5 = x[0:5]
        b1, b2, b3, b4, b5 = x[5:10]
        
        a6 = 0.5 - (a1 + a2 + a3 + a4 + a5)
        b6 = 0.5 - (b1 + b2 + b3 + b4 + b5)
        
        a_full = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
        b_full = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
        
        # Check bounds
        if np.any(np.abs(a_full) > 50) or np.any(np.abs(b_full) > 50):
            return 1e10
            
        err = 0.0
        for i, t in enumerate(t_values):
            U = np.eye(64, dtype=complex)
            for k in range(12):
                # Order: exp(-i b_k t B) exp(-i a_k t A)
                # U = U @ exp(-i a_k t A) @ exp(-i b_k t B) ? 
                # The task says: U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A) in order k=1..12
                # Usually product formulas are applied right-to-left or left-to-right depending on convention.
                # "prod_k ... in order k=1..12" usually means U = F_1 F_2 ... F_12 where F_k = exp(-i b_k t B) exp(-i a_k t A).
                # So U = F_1 @ F_2 @ ... @ F_12.
                # Let's stick to the previous code's logic which was:
                # U = expm_B @ U ; U = expm_A @ U
                # This corresponds to U_new = F_k @ U_old.
                # So U_final = F_12 @ ... @ F_1 @ I = F_12 ... F_1.
                # Wait, standard matrix multiplication is associative.
                # If U starts as I.
                # k=1: U = F_1 @ I = F_1
                # k=2: U = F_2 @ F_1
                # ...
                # k=12: U = F_12 @ ... @ F_1
                # This matches "prod in order k=1..12" if we interpret the product as sequential application on the state vector |psi> -> F_12 ... F_1 |psi>.
                
                U = expm_B(b_full[k] * t) @ U
                U = expm_A(a_full[k] * t) @ U
            diff = U - exact_props[i]
            # Spectral norm is the largest singular value
            err += np.linalg.norm(diff, 2)
        return err / 3.0

    # 4. Custom Nelder-Mead Implementation
    def nelder_mead(func, x0, max_iter=1000, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5, tol=1e-6):
        n = len(x0)
        # Initial simplex
        simplex = [x0.copy()]
        for i in range(n):
            new_point = x0.copy()
            new_point[i] += 0.05 if new_point[i] == 0 else new_point[i] * 0.05
            simplex.append(new_point)
        
        fvals = [func(x) for x in simplex]
        
        for i in range(max_iter):
            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[j] for j in order]
            fvals = [fvals[j] for j in order]
            
            # Check convergence
            if abs(fvals[-1] - fvals[0]) < tol:
                break
            
            # Centroid
            centroid = np.sum(simplex[:-1], axis=0) / n
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = func(xr)
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (xr - centroid)
                fe = func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = func(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for j in range(1, n + 1):
                        simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                        fvals[j] = func(simplex[j])
            
            # Time check inside loop
            if time.time() - start_time > time_limit - 5:
                break
        
        # Return best point
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx]

    # Initial guess
    x0 = np.full(10, 1.0/12.0)
    best_x = x0
    best_cost = cost(x0)
    
    # Run optimization
    try:
        # We can try a few random restarts if time permits, but one good run is better.
        # Let's run a single robust Nelder-Mead.
        res_x, res_cost = nelder_mead(cost, x0, max_iter=200)
        
        if res_cost < best_cost:
            best_x = res_x
            best_cost = res_cost
            
        # If time remains, try another restart with perturbed start
        if time.time() - start_time < time_limit - 20:
            x0_2 = x0 + np.random.normal(0, 0.01, 10)
            res_x2, res_cost2 = nelder_mead(cost, x0_2, max_iter=200)
            if res_cost2 < best_cost:
                best_x = res_x2
                best_cost = res_cost2
                
    except Exception as e:
        pass # Keep best_x as is

    a1, a2, a3, a4, a5 = best_x[0:5]
    b1, b2, b3, b4, b5 = best_x[5:10]
    
    a6 = 0.5 - (a1 + a2 + a3 + a4 + a5)
    b6 = 0.5 - (b1 + b2 + b3 + b4 + b5)
    
    a = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    b = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
    
    # Ensure finite and within bounds
    a = [float(np.clip(v, -50, 50)) for v in a]
    b = [float(np.clip(v, -50, 50)) for v in b]
    
    # Re-normalize to ensure sum=1 exactly? 
    # The sum constraint is built into the symmetry, but floating point errors might creep in.
    # However, the evaluator checks sum(a)=1. 
    # sum(a) = 2*(a1+a2+a3+a4+a5+a6) = 2*(0.5) = 1. 
    # This is mathematically exact. Floating point might make it 0.9999999. 
    # Let's force the last element to correct any drift? No, the symmetry is strict.
    # We will just return the values.
    
    return a, b
