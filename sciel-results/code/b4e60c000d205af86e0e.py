# EVOLVE-BLOCK-START
import numpy as np
import math
import time

# Precompute Hermite polynomial coefficients in y = x^2
# H_0(y) = 1
# H_4(y) = 12 - 48y + 16y^2
# H_8(y) = 168 - 1344y + 2688y^2 - 1344y^3 + 256y^4
# H_12(y) = 18480 - 921600y + 1424640y^2 - 1003520y^3 + 332160y^4 - 46080y^5 + 4096y^6

h0 = [1.0]
h4 = [12.0, -48.0, 16.0]
h8 = [168.0, -1344.0, 2688.0, -1344.0, 256.0]
h12 = [18480.0, -921600.0, 1424640.0, -1003520.0, 332160.0, -46080.0, 4096.0]

def compute_q(c0, c1, c2):
    """Return q(y) coefficients [q0,...,q5] where q(y)=g(x)/x^2, and the leading coeff sign."""
    # c3 from g(0)=0: c0*1 + c1*12 + c2*168 + c3*18480 = 0
    c3 = -(c0 + 12.0*c1 + 168.0*c2) / 18480.0
    
    # g(y) = c0*h0 + c1*h4 + c2*h8 + c3*h12
    # g has degree 6 in y. q(y) = g(y)/y has degree 5.
    # q_k = g_{k+1} for k=0..5
    g = [0.0]*7
    for k in range(len(h0)): g[k] += c0*h0[k]
    for k in range(len(h4)): g[k] += c1*h4[k]
    for k in range(len(h8)): g[k] += c2*h8[k]
    for k in range(len(h12)): g[k] += c3*h12[k]
    
    q = g[1:7]  # q[0]..q[5]
    return q, c3

def q_eval(q, y):
    """Evaluate q(y) via Horner."""
    r = 0.0
    for c in q:
        r = r*y + c
    return r

def largest_real_root_of_q(q):
    """Find the largest real root of q(y) (degree 5 in y). Returns None if no real roots."""
    # Use numpy for initial roots, then refine
    try:
        roots = np.roots(q)
    except Exception:
        return None
    
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-9:
            real_roots.append(r.real)
    
    if not real_roots:
        return None
    
    # Refine the largest one with Newton's method
    y_max = max(real_roots)
    
    # Newton refinement
    for _ in range(50):
        f = q_eval(q, y_max)
        # derivative
        df = 0.0
        power = 1.0
        for i in range(len(q)-1, 0, -1):
            df += q[i] * power * i
            power *= y_max
        if abs(df) < 1e-30:
            break
        step = f/df
        y_max -= step
        if abs(step) < 1e-14:
            break
    
    if y_max < 0:
        # Check if there are any positive roots
        pos_roots = [r for r in real_roots if r > 0]
        if not pos_roots:
            return None
        y_max = max(pos_roots)
    
    return y_max

def evaluate_value(c0, c1, c2):
    """Compute the task value. Lower is better."""
    q, c3 = compute_q(c0, c1, c2)
    
    # Check leading coefficient sign
    # If q_5 > 0, then q -> +inf as y->inf, so the rightmost root is a sign change from + to -
    # If q_5 < 0, then q -> -inf as y->inf, so the rightmost root is a sign change from - to +
    # In both cases, the rightmost real root IS a sign-changing root (for a real polynomial,
    # the rightmost root always has odd multiplicity... actually no, it could be even multiplicity)
    # Wait: the rightmost real root of a real polynomial always has odd multiplicity? No!
    # Consider (y-1)^2 - 0.01: roots at 1±0.1, both simple. The rightmost is 1.1, sign change.
    # Actually, the rightmost real root of a real polynomial MUST have odd multiplicity because
    # to the right of it, the polynomial has a definite sign (the sign of the leading coeff),
    # and immediately to the left it must have the opposite sign (otherwise it wouldn't be a root
    # of the rightmost... hmm, actually it could be even multiplicity if the polynomial touches
    # zero and bounces back).
    
    # Let me think more carefully. If the rightmost root has even multiplicity, then q doesn't
    # change sign there. But then there must be another root to the left. The question is whether
    # the task counts only sign-changing roots.
    
    # The task says: "r be the largest real root of q at which q changes sign"
    # So we need the largest root with ODD multiplicity.
    
    # For safety, let's check sign change at the largest root.
    y_max = largest_real_root_of_q(q)
    if y_max is None or y_max <= 0:
        return float('inf')
    
    # Check sign change
    eps = max(1e-8, abs(y_max)*1e-8)
    f_minus = q_eval(q, max(0, y_max - eps))
    f_plus = q_eval(q, y_max + eps)
    
    if f_minus * f_plus > 0:
        # No sign change at the largest root; need to find the largest root with sign change
        # This is trickier. Let's find all real roots and check each.
        try:
            roots = np.roots(q)
        except Exception:
            return float('inf')
        
        real_roots = sorted([r.real for r in roots if abs(r.imag) < 1e-9], reverse=True)
        
        for yr in real_roots:
            if yr <= 0:
                continue
            eps_r = max(1e-8, abs(yr)*1e-8)
            fm = q_eval(q, max(0, yr - eps_r))
            fp = q_eval(q, yr + eps_r)
            if fm * fp < 0:
                return yr / (2.0 * math.pi)
        
        return float('inf')
    
    return y_max / (2.0 * math.pi)

def construct():
    """
    Strategy: 
    1. The value = y_max / (2*pi) where y_max is the largest sign-changing root of q(y).
    2. We want to minimize y_max.
    3. Key insight: if we can make q(y) have NO positive real roots, the value would be 0 (or inf if no roots).
       But the task requires q to divide exactly and have a sign-changing root, so we need at least one.
    4. Alternative: make the largest sign-changing root as small as possible.
    
    Approach: Direct optimization. The objective is piecewise smooth (changes when the identity of the
    "largest sign-changing root" changes). We'll use:
    - A grid search over a reasonable range to find good starting points
    - Multi-start local optimization (Nelder-Mead) from diverse points
    - Exploit the structure: the leading coefficient of q is
      q5 = c0*0 + c1*0 + c2*256 + c3*4096
         = 256*c2 + 4096*c3
         = 256*c2 + 4096*(-(c0+12*c1+168*c2)/18480)
         = 256*c2 - (4096/18480)*(c0+12*c1+168*c2)
         = 256*c2 - (4096/18480)*c0 - (4096*12/18480)*c1 - (4096*168/18480)*c2
         = 256*c2 - 0.22164*c0 - 2.6597*c1 - 37.333*c2
         = -0.22164*c0 - 2.6597*c1 + 218.667*c2
    
    Let me recompute: 4096*168/18480 = 4096*168/18480 = 688128/18480 = 37.236...
    256 - 37.236 = 218.764
    
    So q5 = -(4096/18480)*c0 - (49152/18480)*c1 + (256 - 4096*168/18480)*c2
         = -(4096/18480)*c0 - (49152/18480)*c1 + ((256*18480 - 4096*168)/18480)*c2
         = -(4096/18480)*c0 - (49152/18480)*c1 + ((4730880 - 688128)/18480)*c2
         = -(4096/18480)*c0 - (49152/18480)*c1 + (4042752/18480)*c2
    
    For q5 < 0 (so q -> -inf), we need:
    4096*c0 + 49152*c1 - 4042752*c2 > 0  (multiplying by -18480 and flipping)
    i.e., 4096*c0 + 49152*c1 > 4042752*c2
    i.e., c0 + 12*c1 > 987*c2  (dividing by 4096: 4042752/4096 = 987.0)
    
    Wait: 4042752/4096 = 987.0 exactly? 4096*987 = 4042752. Yes!
    So q5 < 0 iff c0 + 12*c1 + 168*c2 < 0 ... wait let me redo.
    
    c3 = -(c0 + 12*c1 + 168*c2)/18480
    q5 = 256*c2 + 4096*c3 = 256*c2 + 4096*(-(c0+12*c1+168*c2)/18480)
       = 256*c2 - 4096*(c0+12*c1+168*c2)/18480
       = (256*18480 - 4096*(c0+12*c1+168*c2))/18480
       = (4730880 - 4096*c0 - 49152*c1 - 688128*c2)/18480
    
    q5 < 0 iff 4730880 - 4096*c0 - 49152*c1 - 688128*c2 < 0
    iff 4096*c0 + 49152*c1 + 688128*c2 > 4730880
    iff c0 + 12*c1 + 168*c2 > 4730880/4096 = 1155.0
    
    So q5 < 0 iff c0 + 12*c1 + 168*c2 > 1155.
    
    Similarly, q5 > 0 iff c0 + 12*c1 + 168*c2 < 1155.
    
    The seed has c0=-1.13, c1=0.04, c2=1/3240≈0.0003086.
    c0+12*c1+168*c2 = -1.13 + 0.48 + 0.05185 = -0.598 < 1155, so q5 > 0 for the seed.
    """
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin
    
    # Seed
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_coeffs = seed.copy()
    best_value = evaluate_value(seed[0], seed[1], seed[2])
    
    if not math.isfinite(best_value):
        best_value = 1e10
    
    try:
        from scipy.optimize import minimize
        have_scipy = True
    except ImportError:
        have_scipy = False
    
    def objective(coeffs):
        val = evaluate_value(coeffs[0], coeffs[1], coeffs[2])
        if not math.isfinite(val):
            return 1e10
        return val
    
    if have_scipy:
        # Multi-start Nelder-Mead from diverse points
        # Generate diverse starting points
        rng = np.random.RandomState(12345)
        
        # Region 1: q5 > 0 (c0+12*c1+168*c2 < 1155) - includes seed
        # Region 2: q5 < 0 (c0+12*c1+168*c2 > 1155)
        
        start_points = [seed.copy()]
        
        # Random points in a box around the seed
        for _ in range(8):
            p = seed + rng.normal(0, 0.5, 3)
            start_points.append(p)
        
        # Points with larger c0 (to push roots around)
        for c0_val in [-2.0, -1.5, -3.0, -0.5]:
            start_points.append(np.array([c0_val, 0.04, 0.0003]))
        
        # Points with different c1
        for c1_val in [0.1, 0.2, -0.1, 0.05, 0.15]:
            start_points.append(np.array([-1.13, c1_val, 0.0003]))
        
        # Points with different c2
        for c2_val in [0.001, 0.005, -0.001, 0.0001, 0.01]:
            start_points.append(np.array([-1.13, 0.04, c2_val]))
        
        # Points in the q5 < 0 region (c0+12*c1+168*c2 > 1155)
        # Need large positive c0 or c1
        for c0_val in [1200.0, 1300.0, 1500.0]:
            start_points.append(np.array([c0_val, 0.0, 0.0]))
        
        for c1_val in [100.0, 150.0, 200.0]:
            start_points.append(np.array([0.0, c1_val, 0.0]))
        
        # Mixed
        start_points.append(np.array([500.0, 50.0, 0.0]))
        start_points.append(np.array([800.0, 30.0, 1.0]))
        
        # Try each starting point with Nelder-Mead
        for sp in start_points:
            if time.time() > deadline - 15:
                break
            
            try:
                result = minimize(
                    objective,
                    sp,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-9,
                        'fatol': 1e-9,
                        'adaptive': True
                    }
                )
                if result.fun < best_value:
                    best_value = result.fun
                    best_coeffs = result.x.copy()
            except Exception:
                pass
        
        # Second round: more focused search around the best found so far
        if time.time() < deadline - 20:
            for _ in range(5):
                perturbation = rng.normal(0, 0.01, 3)
                sp = best_coeffs + perturbation
                try:
                    result = minimize(
                        objective,
                        sp,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 300,
                            'xatol': 1e-11,
                            'fatol': 1e-11,
                            'adaptive': True
                        }
                    )
                    if result.fun < best_value:
                        best_value = result.fun
                        best_coeffs = result.x.copy()
                except Exception:
                    pass
        
        # COBYLA polish for higher precision
        if time.time() < deadline - 10:
            try:
                result = minimize(
                    objective,
                    best_coeffs,
                    method='COBYLA',
                    options={
                        'maxiter': 500,
                        'rhobeg': 1e-6,
                        'tol': 1e-12
                    }
                )
                if result.fun < best_value:
                    best_value = result.fun
                    best_coeffs = result.x.copy()
            except Exception:
                pass
    else:
        # Fallback: simple random search
        rng = np.random.RandomState(42)
        for _ in range(500):
            if time.time() > deadline - 10:
                break
            p = seed + rng.normal(0, 0.3, 3)
            val = objective(p)
            if val < best_value:
                best_value = val
                best_coeffs = p.copy()
                # Local refinement
                step = 0.1
                for _ in range(30):
                    improved = False
                    for i in range(3):
                        for sign in [-1, 1]:
                            new_p = best_coeffs.copy()
                            new_p[i] += sign * step
                            v = objective(new_p)
                            if v < best_value:
                                best_value = v
                                best_coeffs = new_p.copy()
                                improved = True
                    if not improved:
                        step *= 0.5
                    if step < 1e-6:
                        break

    # Final verification
    final_val = evaluate_value(best_coeffs[0], best_coeffs[1], best_coeffs[2])
    if math.isfinite(final_val) and final_val < best_value:
        best_value = final_val
    
    return list(best_coeffs)
# EVOLVE-BLOCK-END
