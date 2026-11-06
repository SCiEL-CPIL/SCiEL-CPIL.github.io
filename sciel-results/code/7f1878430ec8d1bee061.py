import time
import numpy as np
from scipy.linalg import expm

_START_TIME = time.time()

def _build_heisenberg_matrices():
    """Build A (odd bonds) and B (even bonds) for 6-site open XXZ chain.
    
    H = sum_{j=0}^{4} [S^z_j S^z_{j+1} + 0.5*(S^x_j S^x_{j+1} + S^y_j S^y_{j+1})] + 0.3 * sum_{j=0}^{5} S^z_j
    
    With anisotropy gamma=0.8, the transverse terms get factor gamma/2 in standard XXZ form:
    H = sum_j [S^z_j S^z_{j+1} + (gamma/2)(S^x_j S^x_{j+1} + S^y_j S^y_{j+1})] + field * sum_j S^z_j
    
    Actually, standard XXZ: H = sum_j [S^x_j S^x_{j+1} + S^y_j S^y_{j+1} + gamma * S^z_j S^z_{j+1}]
    But the task says "anisotropy 0.8 and field 0.3". Let me use the convention:
    H = sum_j [0.5*(S^x_j S^x_{j+1} + S^y_j S^y_{j+1}) * (1+gamma) + S^z_j S^z_{j+1} * (1-gamma)/1 ... ]
    
    Let me just use the most common convention for XXZ:
    H = sum_j [S^x_j S^x_{j+1} + S^y_j S^y_{j+1} + gamma * S^z_j S^z_{j+1}] + h * sum_j S^z_j
    
    But with anisotropy 0.8 < 1, it's in the XY regime. The problem says "anisotropy 0.8" which 
    typically means the z-component is scaled by 0.8 relative to x,y. So:
    H = sum_j [S^x_j S^x_{j+1} + S^y_j S^y_{j+1} + 0.8 * S^z_j S^z_{j+1}] + 0.3 * sum_j S^z_j
    
    Split into odd bonds (j even: 0-1, 2-3, 4-5) as A, even bonds (j odd: 1-2, 3-4) as B,
    plus half the field in each.
    
    For spin-1/2: S^x = 0.5*sigma_x, S^y = 0.5*sigma_y, S^z = 0.5*sigma_z
    """
    N = 6
    s = 0.5
    sx = np.array([[0, s], [s, 0]], dtype=complex)
    sy = np.array([[0, -1j*s], [1j*s, 0]], dtype=complex)
    sz = np.array([[s, 0], [0, -s]], dtype=complex)
    
    def tensor_all(ops):
        """Tensor product of all single-site operators."""
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    def make_site_op(op, site):
        ops = [np.eye(2, dtype=complex)] * N
        ops[site] = op
        return tensor_all(ops)
    
    gamma = 0.8
    field = 0.3
    
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    # Odd bonds: j=0 (sites 0,1), j=2 (sites 2,3), j=4 (sites 4,5)
    for j in [0, 2, 4]:
        # Sx_j Sx_{j+1}
        A += make_site_op(sx, j) @ make_site_op(sx, j+1)
        # Sy_j Sy_{j+1}
        A += make_site_op(sy, j) @ make_site_op(sy, j+1)
        # gamma * Sz_j Sz_{j+1}
        A += gamma * (make_site_op(sz, j) @ make_site_op(sz, j+1))
    
    # Even bonds: j=1 (sites 1,2), j=3 (sites 3,4)
    for j in [1, 3]:
        B += make_site_op(sx, j) @ make_site_op(sx, j+1)
        B += make_site_op(sy, j) @ make_site_op(sy, j+1)
        B += gamma * (make_site_op(sz, j) @ make_site_op(sz, j+1))
    
    # Half field in each
    for j in range(N):
        half_field = 0.5 * field
        A += half_field * make_site_op(sz, j)
        B += half_field * make_site_op(sz, j)
    
    return A, B


def _evaluate(a, b):
    """Compute the mean spectral norm error at t = 0.25, 0.5, 1.0."""
    A, B = _build_heisenberg_matrices()
    H = A + B
    
    errors = []
    for t in [0.25, 0.5, 1.0]:
        # Build product formula U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
        U = np.eye(2**6, dtype=complex)
        for k in range(len(a)):
            U = expm(-1j * b[k] * t * B) @ U
            U = expm(-1j * a[k] * t * A) @ U
        
        U_exact = expm(-1j * t * H)
        err = np.linalg.norm(U - U_exact, 2)
        errors.append(err)
    
    return np.mean(errors)


def _get_strang_coefficients(substeps):
    """Return a, b lists for Strang splitting with given substeps."""
    a = [0.0] * (2 * substeps)
    b = [0.0] * (2 * substeps)
    for i in range(substeps):
        a[2*i] = 0.5 / substeps
        a[2*i+1] = 0.5 / substeps
        b[2*i] = 1.0 / substeps
        b[2*i+1] = 0.0
    return a, b


def _get_suzuki4_coefficients():
    """Return the 4th-order Suzuki-Trotter coefficients for one stage.
    
    The 4th-order formula for exp(-(A+B)) uses 8 stages with coefficients:
    c1 = 1/(4 - 4^(1/3))
    c0 = -4^(1/3) / (4 - 4^(1/3))
    
    a = [c1, c0, c1, c0, c1, c0, c1, c0] / 2  (for A parts)
    b = [0, c1, c0, c1, c0, c1, c0, c1]  (for B parts, with first being 0)
    
    Actually, the standard 4th-order Suzuki splitting for exp(-t(A+B)) is:
    exp(-c0*t*A) exp(-c1*t*B) exp(-c0*t*A) exp(-c1*t*B) exp(-c0*t*A) exp(-c1*t*B) exp(-c0*t*A)
    where c0 = 1/(2 - 2^(1/3)) / 2 and c1 = 1/(2 - 2^(1/3))
    
    Let me use the well-known 4th order coefficients:
    c1 = 1/(4 - 4**(1/3))
    c0 = -4**(1/3)/(4 - 4**(1/3))
    
    The formula is:
    exp(-c0*t*A) * exp(-c1*t*B) * exp(-c0*t*A) * exp(-c1*t*B) * 
    exp(-c0*t*A) * exp(-c1*t*B) * exp(-c0*t*A) * exp(-c1*t*B)
    
    But our product formula is: prod_k exp(-i b_k t B) exp(-i a_k t A)
    
    So we need to rearrange. The 4th order formula applied to exp(-i t (A+B)) is:
    Let's denote the exponent as -i t (A+B). The 4th order Suzuki formula for exp(-i t H) where H = A+B:
    
    exp(-i c0 t A) exp(-i c1 t B) exp(-i c0 t A) exp(-i c1 t B) exp(-i c0 t A) exp(-i c1 t B) exp(-i c0 t A)
    
    This has 4 A-exponentials and 3 B-exponentials. To fit our form prod_k exp(-i b_k t B) exp(-i a_k t A),
    we need 12 stages. Let me think about how to compose.
    
    For 12 stages, I can use the 4th-order formula with 3 sub-steps, or find another composition.
    
    Actually, a simpler approach: use the known 4th-order coefficients and scale them.
    The 4th order Suzuki for one "step" uses 8 terms (4 A, 4 B in alternating pattern starting with A).
    For 12 stages, I can use a composition that sums to 1.
    
    Let me just use a numerical optimization approach with a time budget.
    """
    c1 = 1.0 / (4.0 - 4.0**(1.0/3.0))
    c0 = -4.0**(1.0/3.0) / (4.0 - 4.0**(1.0/3.0))
    
    # One 4th-order step: 4 A's and 3 B's (or 4 and 4 with one zero)
    # Pattern: A(c0) B(c1) A(c0) B(c1) A(c0) B(c1) A(c0)
    # In our form: (b=0, a=c0), (b=c1, a=c0), (b=0, a=c0), (b=c1, a=c0), (b=0, a=c0), (b=c1, a=0), (b=0, a=0)...
    
    # This is getting complicated. Let me use a different strategy.
    # For 12 stages, use a scaled version of a known high-order method.
    
    # The simplest improvement: use 6 substeps of Strang (which is 2nd order) but that's the same as the seed.
    # Better: use a 4th-order method with fewer stages and pad, or use a different known formula.
    
    # Let me try: use the 4th-order Suzuki formula scaled to fit 12 stages.
    # 4th order needs 8 exponentials (4 of each type in some arrangements).
    # For 12 stages (12 A and 12 B exponentials), I can use 1.5 repetitions of the 4th order formula.
    
    # Alternative: Just return optimized coefficients found offline.
    # Based on literature, for the 4th-order Suzuki-Trotter, the coefficients for the 
    # 8-stage formula (4 A, 4 B) are well-known. Let me construct a 12-stage formula
    # by using 3 applications of a 4-stage 2nd-order method but with 4th-order coefficients
    # embedded, or use the known 12-stage 4th-order method if one exists.
    
    # Actually, the cleanest approach for 12 stages: use the 4th-order Suzuki formula
    # which naturally has 8 terms. We have 12 stages. Let me use a composition:
    # Apply the 4th-order formula (8 stages) with weight 1, and add 4 zero stages.
    # But that wastes stages.
    
    # Better: use a 4th-order formula with 12 stages. One approach:
    # The 4th-order formula can be written with symmetric coefficients. 
    # For n stages of each type, the optimal coefficients for 4th order can be found.
    
    # Let me just use a practical approach: try several known coefficient sets
    # and pick the best one within the time budget.
    
    return None


def coefficients(stages):
    """Return 12 coefficients each for a and b lists."""
    deadline = _START_TIME + 145.0  # 5s safety margin before 150s limit
    
    if stages != 12:
        # Fallback to Strang for non-12 stage cases
        substeps = max(1, stages // 2)
        a = [0.0] * stages
        b = [0.0] * stages
        for i in range(substeps):
            a[2*i] = 0.5 / substeps
            a[2*i+1] = 0.5 / substeps
            b[2*i] = 1.0 / substeps
            b[2*i+1] = 0.0
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        a = a[:stages]
        b = b[:stages]
        return a, b
    
    # Try several candidate coefficient sets and pick the best
    candidates = []
    
    # Candidate 1: Strang splitting with 6 substeps (2nd order, baseline)
    a1, b1 = _get_strang_coefficients(6)
    candidates.append((a1, b1))
    
    # Candidate 2: Strang splitting with 3 substeps
    a2, b2 = _get_strang_coefficients(3)
    candidates.append((a2, b2))
    
    # Candidate 3: 4th-order Suzuki coefficients adapted to 12 stages
    # The 4th-order Suzuki for exp(-t(A+B)) uses:
    # c1 = 1/(4-4^(1/3)), c0 = -4^(1/3)/(4-4^(1/3))
    # Formula: exp(-c0 t A) exp(-c1 t B) exp(-c0 t A) exp(-c1 t B) exp(-c0 t A) exp(-c1 t B) exp(-c0 t A)
    # This is 4 A-terms and 3 B-terms = 7 exponentials, but in our paired form it's different.
    
    # In our form: U = prod_k exp(-i b_k t B) exp(-i a_k t A)
    # For the 4th order Suzuki, I need to map it to this form.
    
    c1 = 1.0 / (4.0 - 4.0**(1.0/3.0))
    c0 = -4.0**(1.0/3.0) / (4.0 - 4.0**(1.0/3.0))
    
    # One 4th-order step in paired form (8 stages):
    # Stage 1: b=0, a=c0
    # Stage 2: b=c1, a=c0
    # Stage 3: b=0, a=c0
    # Stage 4: b=c1, a=c0
    # Stage 5: b=0, a=c0
    # Stage 6: b=c1, a=0
    # Stage 7: b=0, a=0
    # Stage 8: b=0, a=0
    # Wait, this doesn't quite work. Let me think more carefully.
    
    # The 4th order Suzuki: S4(t) = exp(-c0 t A) exp(-c1 t B) exp(-c0 t A) exp(-c1 t B) exp(-c0 t A) exp(-c1 t B) exp(-c0 t A)
    # = 4 exp(-c0 t A) and 3 exp(-c1 t B) interleaved.
    
    # To express as prod_k exp(-i b_k t B) exp(-i a_k t A):
    # I need to group B-exponentials before A-exponentials in each stage.
    # 
    # S4(t) = [exp(-i*0*t*B) exp(-i*c0*t*A)] * [exp(-i*c1*t*B) exp(-i*c0*t*A)] * [exp(-i*0*t*B) exp(-i*c0*t*A)] * [exp(-i*c1*t*B) exp(-i*c0*t*A)]
    #
    # That's only 4 stages. The remaining terms: exp(-i*0*t*B) exp(-i*c0*t*A) * exp(-i*c1*t*B) * exp(-i*0*t*B) exp(-i*0*t*A)
    #
    # Hmm, this is messy. Let me just use a different approach.
    
    # For 12 stages, I'll try a symmetric 4th-order-like composition.
    # A well-known 4th-order 12-stage formula can be constructed.
    
    # Let me use the known 4th-order coefficients and scale to 12 stages.
    # If I use the 8-stage 4th-order formula with 4 stages of zeros, that's wasteful.
    
    # Better idea: use a 4th-order method with 12 stages by using the 
    # "balanced" 4th-order Suzuki with 6 A and 6 B terms.
    
    # For a symmetric 4th-order method with n=6 pairs:
    # The coefficients can be derived. Let me use known values.
    # 
    # For 4th order with 6 stages of each (12 total), one set of coefficients:
    # a = [alpha, beta, gamma, gamma, beta, alpha, ...] symmetric
    # 
    # Actually, let me just numerically optimize within the time budget.
    
    # Start with the 4th-order Suzuki 8-stage coefficients padded to 12
    a3 = [0.0] * 12
    b3 = [0.0] * 12
    # 4th order Suzuki: c0 for A (4 times), c1 for B (3 times)
    # Map to 12 stages: use the 8-stage formula in first 8, zeros in last 4
    a3[0] = c0
    a3[1] = c0
    a3[2] = c0
    a3[3] = c0
    b3[1] = c1
    b3[3] = c1
    b3[5] = c1
    # Normalize
    sum_a = sum(a3)
    sum_b = sum(b3)
    if abs(sum_a) > 1e-15:
        a3 = [x / sum_a for x in a3]
    if abs(sum_b) > 1e-15:
        b3 = [x / sum_b for x in b3]
    candidates.append((a3, b3))
    
    # Candidate 4: Try a symmetric 4th-order composition with 12 stages
    # Using the "optimal" 4th-order coefficients for 6 pairs
    # From literature: for 4th order with n substeps, coefficients are:
    # a_j = c1 * sin^2(pi*j/(2*n+2)) for j=1..n, with c0 for the middle
    # This is getting too complex without reference. Let me use a simpler heuristic.
    
    # Candidate 5: Use 4th-order Suzuki with 2 substeps of the 6-stage version
    # Just try a few more Strang variations
    
    # Candidate: 12-stage Strang with non-uniform weights
    # Weight more on the middle stages
    a5 = [0.0] * 12
    b5 = [0.0] * 12
    weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    total_w = sum(weights)
    for i in range(6):
        a5[2*i] = 0.5 * weights[i] / total_w
        a5[2*i+1] = 0.5 * weights[i+1] / total_w
        b5[2*i] = weights[i] / total_w
        b5[2*i+1] = 0.0
    # Normalize to ensure sums are 1
    sum_a5 = sum(a5)
    sum_b5 = sum(b5)
    if abs(sum_a5) > 1e-15:
        a5 = [x / sum_a5 for x in a5]
    if abs(sum_b5) > 1e-15:
        b5 = [x / sum_b5 for x in b5]
    candidates.append((a5, b5))
    
    # Evaluate all candidates
    best_score = float('inf')
    best_a, best_b = a1, b1
    
    for a_c, b_c in candidates:
        if time.time() > deadline:
            break
        try:
            score = _evaluate(a_c, b_c)
            if score < best_score:
                best_score = score
                best_a, best_b = a_c, b_c
        except Exception:
            pass
    
    # If we have time, try a simple local optimization
    if time.time() < deadline - 20.0:
        try:
            # Perturb the best solution slightly
            rng = np.random.RandomState(42)
            current_a = list(best_a)
            current_b = list(best_b)
            current_score = best_score
            
            for iteration in range(100):
                if time.time() > deadline:
                    break
                
                # Random perturbation
                step_size = 0.01 / (1 + iteration * 0.1)
                trial_a = [max(-50, min(50, x + rng.randn() * step_size)) for x in current_a]
                trial_b = [max(-50, min(50, x + rng.randn() * step_size)) for x in current_b]
                
                # Enforce sum constraint
                sum_a_t = sum(trial_a)
                sum_b_t = sum(trial_b)
                if abs(sum_a_t) > 1e-15:
                    trial_a = [x / sum_a_t for x in trial_a]
                if abs(sum_b_t) > 1e-15:
                    trial_b = [x / sum_b_t for x in trial_b]
                
                try:
                    trial_score = _evaluate(trial_a, trial_b)
                    if trial_score < current_score:
                        current_a = trial_a
                        current_b = trial_b
                        current_score = trial_score
                except Exception:
                    pass
            
            if current_score < best_score:
                best_score = current_score
                best_a, best_b = current_a, current_b
        except Exception:
            pass
    
    # Final check: ensure all values are in [-50, 50] and sums are approximately 1
    best_a = [max(-50.0, min(50.0, x)) for x in best_a]
    best_b = [max(-50.0, min(50.0, x)) for x in best_b]
    
    # Renormalize to ensure exact sums of 1
    sum_a_final = sum(best_a)
    sum_b_final = sum(best_b)
    if abs(sum_a_final) > 1e-15:
        best_a = [x / sum_a_final for x in best_a]
    else:
        best_a = [1.0/12.0] * 12
    if abs(sum_b_final) > 1e-15:
        best_b = [x / sum_b_final for x in best_b]
    else:
        best_b = [1.0/12.0] * 12
    
    # Ensure finite
    best_a = [float(x) if np.isfinite(x) else 0.0 for x in best_a]
    best_b = [float(x) if np.isfinite(x) else 0.0 for x in best_b]
    
    return best_a, best_b
