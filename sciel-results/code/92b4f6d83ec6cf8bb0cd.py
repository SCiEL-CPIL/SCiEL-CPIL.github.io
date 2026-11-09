"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site open XXZ chain."""
    N = 6
    lam = 0.8
    h = 0.3
    
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
    
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    field_term = 0.5 * h
    for i in range(N):
        A += field_term * op_on_site(sz, i, N)
        B += field_term * op_on_site(sz, i, N)
        
    def add_bond(H, i, j):
        sx_i = op_on_site(sx, i, N)
        sx_j = op_on_site(sx, j, N)
        sy_i = op_on_site(sy, i, N)
        sy_j = op_on_site(sy, j, N)
        sz_i = op_on_site(sz, i, N)
        sz_j = op_on_site(sz, j, N)
        H += sx_i @ sx_j + sy_i @ sy_j + lam * (sz_i @ sz_j)
        
    # Odd bonds for A: (0,1), (2,3), (4,5)
    add_bond(A, 0, 1)
    add_bond(A, 2, 3)
    add_bond(A, 4, 5)
    
    # Even bonds for B: (1,2), (3,4)
    add_bond(B, 1, 2)
    add_bond(B, 3, 4)
    
    return A, B

def coefficients(stages):
    """Optimized 12-stage product formula coefficients via 1D search over Suzuki-Trotter template."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    H_total = A + B
    
    # Pre-compute exact propagators
    t_values = [0.25, 0.5, 1.0]
    U_exacts = []
    for t in t_values:
        # Use eigen-decomposition for expm
        w, v = np.linalg.eigh(H_total)
        U_exact = (v * np.exp(-1j * t * w)) @ v.conj().T
        U_exacts.append(U_exact)
        
    # Pre-compute eigen-decompositions of A and B
    w_A, v_A = np.linalg.eigh(A)
    w_B, v_B = np.linalg.eigh(B)
    
    def compute_error(a, b):
        """Compute mean spectral norm error over t_values."""
        total_err = 0.0
        for t in t_values:
            U_approx = np.eye(64, dtype=complex)
            for k in range(12):
                # Apply exp(-i b_k t B)
                exp_B = np.exp(-1j * b[k] * t * w_B)
                U_B = (v_B * exp_B) @ v_B.conj().T
                U_approx = U_B @ U_approx
                
                # Apply exp(-i a_k t A)
                exp_A = np.exp(-1j * a[k] * t * w_A)
                U_A = (v_A * exp_A) @ v_A.conj().T
                U_approx = U_A @ U_approx
            
            diff = U_approx - U_exacts[t_values.index(t)]
            total_err += np.linalg.norm(diff, 2)
        return total_err / len(t_values)
    
    # Define the Suzuki-Trotter 6th order template.
    # We need a sequence of 12 steps. A standard 6th order symmetric scheme has a specific structure.
    # However, we can simply use a generic symmetric high-order set of coefficients.
    # Let's use the 6th order coefficients from Suzuki (1991) or similar.
    # c1 = 0.1578869450914819
    # c2 = -0.1241919579161346
    # c3 = 0.0782060549085181
    # c4 = 0.1241919579161346
    # c5 = 0.1578869450914819
    # Wait, that's 5 coefficients for a 3-step scheme? No.
    
    # Let's use a known 6th order 12-step scheme or just optimize a symmetric 6-param vector.
    # The directive suggests: "Parametrize the 24 coefficients via a single scalar variable that scales a precomputed, high-order (6th-order) Suzuki-Trotter coefficient template".
    
    # Let's define a base template for a and b.
    # Symmetric 12-step scheme.
    # Let x1..x6 be the first half.
    # a = [x1, x2, x3, x4, x5, x6, x6, x5, x4, x3, x2, x1]
    # b = [y1, y2, y3, y4, y5, y6, y6, y5, y4, y3, y2, y1]
    
    # We will define a base vector x_base and y_base and scale them by s.
    # But we need sum(a)=1 and sum(b)=1.
    # If we scale by s, we need s * sum(x_base) = 1 => s = 1/sum(x_base).
    # This fixes s! So we can't just scale.
    
    # Alternative interpretation: The "template" provides the *ratios* or the *shape*, and we optimize the magnitude?
    # No, the constraint sum=1 fixes the magnitude if the shape is fixed.
    
    # Perhaps the strategy means: Use a high-order template as the *initial guess* or *basis* for a small search?
    # Or maybe the "single scalar variable" is not a global scale, but a parameter within the template?
    
    # Let's re-read: "Parametrize the 24 coefficients via a single scalar variable that scales a precomputed... template... thereby reducing the search space from 24 dimensions to 1".
    # This implies the template itself might not satisfy sum=1, and we find the scale? But sum=1 is a hard constraint.
    # If the template is T, and we set a = s * T_a, b = s * T_b.
    # Then sum(a) = s * sum(T_a) = 1 => s = 1/sum(T_a).
    # This is a unique solution. There is no search.
    
    # Unless... the template is not fixed? Or the "scaling" refers to something else?
    
    # Maybe the template is a *family* of templates?
    # Or maybe the "single scalar variable" is a shift?
    
    # Let's look at the inspiration 1 code. It used Nelder-Mead on 6 variables.
    # The directive says "Combine... structural idea of one and refinement machinery of the other".
    # Current: Simple Strang.
    # Inspiration 1: Nelder-Mead on 6 vars.
    # Strategy to try: 1D search over a scaled template.
    
    # If I strictly follow "1D search", I need a parameter that changes the coefficients while maintaining sum=1.
    # Example: a_i = base_i + delta * v_i, where sum(v)=0.
    # This is a 1D subspace.
    
    # Let's pick a base template (e.g., 4th order or 6th order coefficients) and a variation vector v that sums to 0.
    # We search for the optimal delta.
    
    # Let's use the 4th order coefficients as the base.
    # 4th order 3-step: c1=1/(2-2^(1/3)), c2=-2^(1/3)/(2-2^(1/3)), c3=1-c1-c2.
    # Sum = 1.
    # We need 12 steps. We can repeat the 3-step pattern 4 times?
    # a = [c1, c2, c3, c1, c2, c3, c1, c2, c3, c1, c2, c3]
    # Sum = 4 * 1 = 4. So we scale by 0.25.
    # base_a = 0.25 * [c1, c2, c3, ...]
    
    # Let's try this.
    
    c1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    c2 = -2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))
    c3 = 1.0 - c1 - c2
    
    base_seq = [c1, c2, c3] * 4
    base_a = [0.25 * x for x in base_seq]
    base_b = [0.25 * x for x in base_seq]
    
    # Now we need a variation vector v such that sum(v) = 0.
    # Let's use a simple sinusoidal variation or a polynomial.
    # v_k = sin(2 * pi * k / 12) ? Sum is 0.
    # Or v_k = (k - 5.5) / 12 ? Sum is 0.
    
    # Let's use v_k = sin(2 * pi * k / 12) for k=0..11.
    v = [np.sin(2 * np.pi * k / 12.0) for k in range(12)]
    
    def get_coeffs(delta):
        a = [base_a[k] + delta * v[k] for k in range(12)]
        b = [base_b[k] + delta * v[k] for k in range(12)]
        # Ensure sums are exactly 1 (floating point cleanup)
        sum_a = sum(a)
        sum_b = sum(b)
        if abs(sum_a - 1.0) > 1e-10:
            a = [x / sum_a for x in a]
        if abs(sum_b - 1.0) > 1e-10:
            b = [x / sum_b for x in b]
        # Check bounds [-50, 50]
        for x in a:
            if x < -50 or x > 50:
                return None
        for x in b:
            if x < -50 or x > 50:
                return None
        return a, b
    
    best_err = float('inf')
    best_a, best_b = base_a, base_b
    
    # 1D Golden Section Search
    # Range for delta: let's try [-0.1, 0.1]
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    a_range = -0.1
    b_range = 0.1
    
    # Evaluate at initial points
    c = b_range - phi * (b_range - a_range)
    d = a_range + phi * (b_range - a_range)
    
    cd, dd = get_coeffs(c)
    if cd is None:
        cd, dd = base_a, base_b
        err_c = compute_error(cd, dd)
    else:
        err_c = compute_error(cd, dd)
        
    cd, dd = get_coeffs(d)
    if cd is None:
        cd, dd = base_a, base_b
        err_d = compute_error(cd, dd)
    else:
        err_d = compute_error(cd, dd)
    
    if err_c < best_err:
        best_err = err_c
        best_a, best_b = cd, dd
    if err_d < best_err:
        best_err = err_d
        best_a, best_b = cd, dd
        
    for _ in range(50):
        if time.time() > deadline:
            break
            
        if err_c < err_d:
            b_range = d
            d = c
            err_d = err_c
            c = b_range - phi * (b_range - a_range)
            cd, dd = get_coeffs(c)
            if cd is None:
                cd, dd = base_a, base_b
                err_c = compute_error(cd, dd)
            else:
                err_c = compute_error(cd, dd)
        else:
            a_range = c
            c = d
            err_c = err_d
            d = a_range + phi * (b_range - a_range)
            cd, dd = get_coeffs(d)
            if cd is None:
                cd, dd = base_a, base_b
                err_d = compute_error(cd, dd)
            else:
                err_d = compute_error(cd, dd)
                
        if err_c < best_err:
            best_err = err_c
            best_a, best_b = cd, dd
        if err_d < best_err:
            best_err = err_d
            best_a, best_b = cd, dd
            
    return best_a, best_b
# EVOLVE-BLOCK-END
