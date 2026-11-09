# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.special import lpmv

# Constants
L = 10
N_THETA = 70
N_PHI = 130

# Precompute quadrature nodes and weights
u_nodes, w_weights = np.polynomial.legendre.leggauss(N_THETA)
phi_nodes = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)

# Precompute flattened weights
w_flat = np.repeat(w_weights, N_PHI) * (2.0 * np.pi / N_PHI)

# Precompute Spherical Harmonics Y_matrix
# Real orthonormal basis:
# m=0: Y_0 = N_0 P_10^0
# m>0: Y_m = sqrt(2) N_m P_10^m cos(m phi)
# m<0: Y_m = sqrt(2) N_|m| P_10^|m| sin(|m| phi)
Y_matrix = np.zeros((N_THETA * N_PHI, 21), dtype=np.float64)

def fact(n):
    return math.factorial(n)

for m in range(-L, L + 1):
    abs_m = abs(m)
    N_m = math.sqrt(21.0 / (4.0 * np.pi) * fact(L - abs_m) / fact(L + abs_m))
    P_vals = lpmv(abs_m, L, u_nodes)
    
    if m == 0:
        Y_flat = np.tile(P_vals * N_m, (1, N_PHI)).flatten()
    else:
        if m > 0:
            cos_term = np.cos(m * phi_nodes)
            Y_2d = np.sqrt(2.0) * N_m * np.outer(P_vals, cos_term)
            Y_flat = Y_2d.flatten()
        else:
            sin_term = np.sin(abs_m * phi_nodes)
            Y_2d = np.sqrt(2.0) * N_m * np.outer(P_vals, sin_term)
            Y_flat = Y_2d.flatten()
    
    Y_matrix[:, m + 10] = Y_flat

def compute_ratio(c):
    """
    Computes R = ||f||_4 / ||f||_2
    c is a real vector of size 21.
    """
    f = Y_matrix @ c
    f2 = f * f
    f4 = f2 * f2
    
    norm2_sq = np.dot(f2, w_flat)
    norm4_sq = np.dot(f4, w_flat)
    
    if norm2_sq <= 1e-15:
        return 1e15
        
    norm2 = math.sqrt(norm2_sq)
    norm4 = math.sqrt(math.sqrt(norm4_sq))
    
    return norm4 / norm2

def map_complex_to_real(z):
    """
    Maps a complex vector z of length 11 (indices 0..10 corresponding to m=0..10)
    to a real vector c of length 21 (indices 0..20 corresponding to m=-10..10).
    
    z[0] corresponds to m=0.
    z[k] for k>0 corresponds to the complex amplitude for m=k.
    c[m] for m>0 is sqrt(2) * Re(z[m])
    c[-m] for m>0 is sqrt(2) * Im(z[m])
    c[0] is Re(z[0])
    
    Wait, the basis is:
    Y_m (m>0) = sqrt(2) N_m P^m cos(m phi)
    Y_-m (m>0) = sqrt(2) N_m P^m sin(m phi)
    
    Standard complex spherical harmonics Y_l^m usually involve e^{im phi}.
    Y_l^m ~ P_l^m(cos theta) e^{i m phi}.
    Y_l^m + Y_l^{-m} ~ 2 cos(m phi)
    Y_l^m - Y_l^{-m} ~ -2i sin(m phi)
    
    Let's define the complex vector z such that:
    c_0 = z_0 (real)
    For k=1..10:
    c_k = sqrt(2) * Re(z_k)
    c_{-k} = sqrt(2) * Im(z_k)
    
    This is a valid bijection between C^11 and R^21.
    The norm in C^11: ||z||^2 = |z_0|^2 + sum |z_k|^2
    The norm in R^21: ||c||^2 = c_0^2 + sum (c_k^2 + c_{-k}^2) = z_0^2 + sum (2 Re(z_k)^2 + 2 Im(z_k)^2) = z_0^2 + 2 sum |z_k|^2
    
    This metric mismatch is fine as long as we are consistent.
    We want to minimize R(c(z)).
    Since R is scale invariant, we can fix ||z||_C = 1 (or some constant) or just optimize on the complex sphere.
    Actually, it's easier to optimize on the real sphere S^20 but use the complex structure to guide steps?
    Or simply optimize z on the complex sphere S^20 (real dimension 21).
    
    Let's stick to optimizing z directly.
    Objective J(z) = R(c(z)).
    Gradient with respect to z:
    dJ/dz = (dJ/dc) * (dc/dz)
    
    dc_0/dz_0 = 1, dc_0/dz_0^* = 1 (if we treat z and z* as independent? No, z is complex variable).
    Let's use Wirtinger derivatives or just map to real/imag parts.
    Let z = x + i y.
    c_0 = x_0
    c_k = sqrt(2) x_k
    c_{-k} = sqrt(2) y_k
    
    This is a linear map M such that c = M z_real_imag.
    z_real_imag is a vector of size 22 (x_0..x_10, y_0..y_10).
    But z_0 should be real? The m=0 term is real. So y_0 is always 0.
    So we have 11 real variables (x_0..x_10) and 10 imaginary variables (y_1..y_10). Total 21 variables.
    
    Let v = [x_0, x_1, ..., x_10, y_1, ..., y_10] (size 21).
    c_0 = v[0]
    c_k = sqrt(2) v[k] for k=1..10
    c_{-k} = sqrt(2) v[11+k-1] = sqrt(2) v[10+k] for k=1..10
    
    So c = A @ v where A is 21x21.
    A[0, 0] = 1
    A[k, k] = sqrt(2) for k=1..10
    A[21-k, 10+k] = sqrt(2) for k=1..10
    
    We can just define this mapping and use standard real-space CG on v.
    Why is this "different"? It's just a linear coordinate transform.
    However, the "Conjugate Gradient" method itself is a different optimizer than the momentum-based GD used before.
    And the "Complex Sphere" framing suggests we should maybe enforce unit norm in the complex sense?
    Let's just use CG on the 21-dim real vector v representing the complex coefficients.
    """
    c = np.zeros(21)
    c[10] = z[0].real # m=0 index is 10
    
    for k in range(1, 11):
        c[10 + k] = math.sqrt(2.0) * z[k].real
        c[10 - k] = math.sqrt(2.0) * z[k].imag
    return c

def map_real_to_complex(c):
    z = np.zeros(11, dtype=np.complex128)
    z[0] = c[10]
    for k in range(1, 11):
        z[k] = complex(c[10 + k] / math.sqrt(2.0), c[10 - k] / math.sqrt(2.0))
    return z

def objective_complex(z):
    c = map_complex_to_real(z)
    return compute_ratio(c)

def gradient_complex(z):
    """
    Computes gradient of objective with respect to the real and imaginary parts of z.
    Returns a complex vector g such that Re(g) is grad wrt Re(z) and Im(g) is grad wrt Im(z).
    """
    c = map_complex_to_real(z)
    f = Y_matrix @ c
    f2 = f * f
    f4 = f2 * f2
    
    norm2_sq = np.dot(f2, w_flat)
    norm4_sq = np.dot(f4, w_flat)
    
    if norm2_sq <= 1e-15 or norm4_sq <= 1e-15:
        return np.zeros(11, dtype=np.complex128)
        
    norm2 = math.sqrt(norm2_sq)
    norm4 = math.sqrt(math.sqrt(norm4_sq))
    
    A = (f**3 * w_flat) @ Y_matrix
    B = (f * w_flat) @ Y_matrix
    
    coeff_A = (norm4 ** -3) / norm2
    coeff_B = norm4 / (norm2 ** 3)
    
    grad_c = coeff_A * A - coeff_B * B
    
    # Map grad_c back to grad_z
    # c_0 = z_0_real -> dc_0/dz_0_real = 1
    # c_k = sqrt(2) z_k_real -> dc_k/dz_k_real = sqrt(2)
    # c_-k = sqrt(2) z_k_imag -> dc_-k/dz_k_imag = sqrt(2)
    
    grad_z = np.zeros(11, dtype=np.complex128)
    grad_z[0] = grad_c[10] # m=0
    
    for k in range(1, 11):
        grad_z[k].real = math.sqrt(2.0) * grad_c[10 + k]
        grad_z[k].imag = math.sqrt(2.0) * grad_c[10 - k]
        
    return grad_z

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0
    
    best_c = None
    best_score = 1e15
    
    # Conjugate Gradient method on the complex sphere (represented as real 21-dim vector v)
    # v = [Re(z_0), ..., Re(z_10), Im(z_1), ..., Im(z_10)]
    # We will normalize v to unit norm in the "complex" metric?
    # Metric: ||z||^2 = |z_0|^2 + sum_{k=1}^{10} |z_k|^2
    # In terms of v: v[0]^2 + sum_{k=1}^{10} (v[k]^2 + v[10+k]^2)
    # This is just the standard Euclidean norm of v!
    # Because v contains Re(z_0) at index 0, Re(z_k) at k, Im(z_k) at 10+k.
    # So ||v||_2^2 = v[0]^2 + sum v[k]^2 + sum v[10+k]^2 = |z_0|^2 + sum |z_k|^2.
    # Perfect. The complex unit sphere corresponds to the real unit sphere S^20 in v-space.
    
    n_runs = 100
    max_iter = 200
    
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = []
    
    # 1. Single m seeds in complex space
    for k in range(11):
        z = np.zeros(11, dtype=np.complex128)
        z[k] = 1.0
        starts.append(z)
        
    # 2. Pairs
    for i in range(11):
        for j in range(i+1, 11):
            z = np.zeros(11, dtype=np.complex128)
            z[i] = 1.0 / math.sqrt(2.0)
            z[j] = 1.0 / math.sqrt(2.0)
            starts.append(z)
            
    # 3. Random
    for _ in range(50):
        z = (rng.standard_normal(11) + 1j * rng.standard_normal(11)).astype(np.complex128)
        z = z / np.linalg.norm(z)
        starts.append(z)
        
    for z0 in starts:
        if time.time() > deadline:
            break
            
        # Normalize start
        z = z0 / np.linalg.norm(z0)
        
        # Conjugate Gradient on the sphere
        # We use a projected CG method.
        # p_0 = -grad(z_0) projected onto tangent space
        # Tangent space: {v | v . z = 0} (using complex inner product? No, real inner product on v)
        # Let's work with v = [Re(z), Im(z)] explicitly for CG to avoid confusion.
        
        def z_to_v(z):
            v = np.zeros(21)
            v[0] = z[0].real
            for k in range(1, 11):
                v[k] = z[k].real
                v[10+k] = z[k].imag
            return v
            
        def v_to_z(v):
            z = np.zeros(11, dtype=np.complex128)
            z[0] = v[0]
            for k in range(1, 11):
                z[k] = complex(v[k], v[10+k])
            return z

        v = z_to_v(z)
        v = v / np.linalg.norm(v)
        
        # Initial gradient
        g = gradient_complex(z)
        g_v = z_to_v(g) # This maps complex grad to real vector grad
        # Project gradient onto tangent space: g_tan = g - (g.v) v
        g_tan = g_v - np.dot(g_v, v) * v
        
        p = -g_tan
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-10:
             # Try a random step
             p = rng.standard_normal(21)
             p = p - np.dot(p, v) * v
             p_norm = np.linalg.norm(p)
             if p_norm < 1e-10:
                 continue
             p = p / p_norm
        else:
             p = p / p_norm
             
        z_curr = v_to_z(v)
        f_curr = objective_complex(z_curr)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # Line search: Minimize f(v + alpha * p) subject to ||v + alpha p|| = 1
            # This is a quadratic constraint.
            # Let q = v + alpha p.
            # q . q = 1 => v.v + 2 alpha v.p + alpha^2 p.p = 1
            # Since v.v = 1 and p is tangent (v.p = 0), this is 1 + alpha^2 ||p||^2 = 1.
            # This implies alpha must be 0 if p is tangent?
            # No, we need to re-normalize after the step.
            # Standard spherical CG:
            # v_new = v + alpha p
            # v_new = v_new / ||v_new||
            
            # Simple backtracking line search
            alpha = 0.1
            best_alpha = 0
            best_f = f_curr
            
            for _ in range(20):
                v_trial = v + alpha * p
                norm_v = np.linalg.norm(v_trial)
                if norm_v < 1e-6:
                    alpha *= 0.5
                    continue
                v_trial = v_trial / norm_v
                z_trial = v_to_z(v_trial)
                f_trial = objective_complex(z_trial)
                
                if f_trial < best_f:
                    best_f = f_trial
                    best_alpha = alpha
                    v_best = v_trial
                    z_best = z_trial
                else:
                    alpha *= 0.5
                    if alpha < 1e-6:
                        break
                        
            if best_alpha == 0:
                break
                
            v = v_best
            z = z_best
            f_curr = best_f
            
            if f_curr < best_score:
                best_score = f_curr
                best_c = map_complex_to_real(z)
                
            # Compute new gradient
            g_new = gradient_complex(z)
            g_new_v = z_to_v(g_new)
            g_new_tan = g_new_v - np.dot(g_new_v, v) * v
            
            # Check convergence
            if np.linalg.norm(g_new_tan) < 1e-7:
                break
                
            # CG update
            r_old = -g_tan # residual is negative gradient
            # beta = (r_new . r_new) / (r_old . r_old)
            # But we need to ensure p is orthogonal to v?
            # The new direction p_new = -g_new_tan + beta p
            # We must project p_new to be tangent to the sphere at v.
            
            r_new = -g_new_tan
            beta = np.dot(r_new, r_new) / max(np.dot(r_old, r_old), 1e-15)
            
            p_new = r_new + beta * p
            # Project p_new to tangent space
            p_new = p_new - np.dot(p_new, v) * v
            p_norm_new = np.linalg.norm(p_new)
            
            if p_norm_new < 1e-10:
                # Restart
                p = r_new
                p_norm = np.linalg.norm(p)
                if p_norm < 1e-10:
                    break
                p = p / p_norm
            else:
                p = p_new / p_norm_new
                
            r_old = r_new
            g_tan = g_new_tan
            
    if best_c is None:
        best_c = np.zeros(21)
        best_c[10] = 1.0
        
    # Ensure validity
    max_c = np.max(np.abs(best_c))
    if max_c < 1e-3:
        best_c = best_c / max_c * 1.0
    elif max_c > 1e5:
        best_c = best_c / max_c * 1.0
        
    return best_c

# EVOLVE-BLOCK-END
