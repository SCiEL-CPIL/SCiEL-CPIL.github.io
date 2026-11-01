"""Uncertainty bound minimization via unit sphere reparameterization and adaptive stochastic local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 195.0

# Precompute integrals I[n] = int x^{2n} exp(-x^2) dx
# I[0] = sqrt(pi)
# I[n] = I[n-1] * (2n-1)/2
_I = np.zeros(13)
_I[0] = np.sqrt(np.pi)
for _n in range(1, 13):
    _I[_n] = _I[_n-1] * (2*_n - 1) / 2.0

def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty product for g(x) = P(x) exp(-x^2/2).
    P(x) is a Hermite combination with g(0)=0.
    a12 is derived from a0, a4, a8 to enforce P(0)=0.
    """
    # Enforce P(0) = 0
    # P(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # H0(0) = 1
    # H4(0) = 3
    # H8(0) = 105
    # H12(0) = 945
    # a0 + 3a4 + 105a8 + 945a12 = 0
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    # Polynomial P(x) = sum_{k=0}^6 p_k x^{2k}
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    
    # Coefficients p_k for x^{2k}
    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12
    
    p = np.array([p0, p1, p2, p3, p4, p5, p6])

    # m[k] = integral x^{2k} |P(x)|^2 exp(-x^2) dx
    # m[k] = sum_{i,j} p_i p_j I[k+i+j]
    m0 = 0.0
    m2 = 0.0
    # We need m[0] and m[2] (variance uses <x^2>/<1> = m[2]/m[0]? No, <x^2> = m[2]/m[0] if m[0] is <1>? 
    # Wait, m[k] defined as int x^{2k} P^2 e^{-x^2}.
    # <1> = m[0]
    # <x^2> = m[2] / m[0]
    
    # Optimization: Only compute m[0] and m[2]
    # m[0] = sum_{i,j} p_i p_j I[i+j]
    # m[2] = sum_{i,j} p_i p_j I[2+i+j]
    
    # Loop is small (7x7), so direct computation is fine.
    for i in range(7):
        pi = p[i]
        if pi == 0.0: continue
        for j in range(7):
            pp = pi * p[j]
            if pp == 0.0: continue
            m0 += pp * _I[i+j]
            m2 += pp * _I[2+i+j]

    # Fourier side
    # G(xi) = sqrt(2*pi) P(-i xi) exp(-xi^2/2)
    # P(-i xi) = sum p_k (-1)^k xi^{2k}
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    # n[k] = int xi^{2k} |G(xi)|^2 dxi = 2*pi * int xi^{2k} |Q(xi)|^2 exp(-xi^2) dxi
    # n[k] = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    
    n0 = 0.0
    n2 = 0.0
    for i in range(7):
        qi = q[i]
        if qi == 0.0: continue
        for j in range(7):
            qq = qi * q[j]
            if qq == 0.0: continue
            n0 += qq * _I[i+j]
            n2 += qq * _I[2+i+j]
            
    n0 *= 2.0 * np.pi
    n2 *= 2.0 * np.pi

    if m0 <= 1e-15 or n0 <= 1e-15:
        return 1e18
        
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18
        
    return np.sqrt(vf * vF)

def _bound_from_angles(t1, t2):
    """
    Compute bound for unit vector defined by spherical angles.
    a0 = sin(t2) cos(t1)
    a4 = sin(t2) sin(t1)
    a8 = cos(t2)
    """
    st2 = np.sin(t2)
    a0 = st2 * np.cos(t1)
    a4 = st2 * np.sin(t1)
    a8 = np.cos(t2)
    return _compute_bound(a0, a4, a8)

def construct():
    """
    Minimize uncertainty bound on the unit sphere using adaptive stochastic local search.
    """
    # Seed values
    seed = np.array([-1.13, 0.04, 0.000308642])
    r_seed = np.linalg.norm(seed)
    if r_seed < 1e-15:
        t1_init = 0.0
        t2_init = 0.0
    else:
        t1_init = np.arctan2(seed[1], seed[0])
        t2_init = np.arccos(np.clip(seed[2] / r_seed, -1.0, 1.0))
        
    seed_bound = _bound_from_angles(t1_init, t2_init)
    
    best_bound = seed_bound
    best_t1 = t1_init
    best_t2 = t2_init
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts to escape local minima
    # We will run several local searches from different starting points
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > _DEADLINE - 20:
            break
            
        if restart == 0:
            t1 = t1_init
            t2 = t2_init
        else:
            t1 = rng.uniform(-np.pi, np.pi)
            t2 = rng.uniform(0, np.pi)
            
        current_bound = _bound_from_angles(t1, t2)
        
        # Adaptive step sizes for t1, t2
        step_t1 = 0.1
        step_t2 = 0.1
        
        # Simulated Annealing / Stochastic Local Search
        temp = 1.0
        cooling = 0.9995
        
        max_iters = 5000
        
        for _ in range(max_iters):
            if time.time() > _DEADLINE - 10:
                break
                
            # Perturb
            dt1 = rng.normal(0, step_t1)
            dt2 = rng.normal(0, step_t2)
            
            new_t1 = (t1 + dt1 + np.pi) % (2 * np.pi) - np.pi
            new_t2 = np.clip(t2 + dt2, 0.01, np.pi - 0.01)
            
            new_bound = _bound_from_angles(new_t1, new_t2)
            
            if not np.isfinite(new_bound):
                new_bound = 1e18
                
            delta = new_bound - current_bound
            
            if delta < 0 or np.exp(-delta / temp) > rng.random():
                t1 = new_t1
                t2 = new_t2
                current_bound = new_bound
                
                # Update best
                if current_bound < best_bound:
                    best_bound = current_bound
                    best_t1 = t1
                    best_t2 = t2
                    
                # Shrink step on acceptance (local improvement)
                step_t1 *= 0.98
                step_t2 *= 0.98
            else:
                # Expand step on rejection to explore
                step_t1 *= 1.05
                step_t2 *= 1.05
                
            # Adaptive step size adjustment based on acceptance rate over last 100 steps
            # (Simplified: just cool down)
            temp *= cooling
            if temp < 1e-8:
                temp = 1e-8
                
            # Prevent steps from becoming too small or too large
            step_t1 = np.clip(step_t1, 1e-4, 1.0)
            step_t2 = np.clip(step_t2, 1e-4, 1.0)

    # Convert best angles back to coefficients
    st2 = np.sin(best_t2)
    a0 = st2 * np.cos(best_t1)
    a4 = st2 * np.sin(best_t1)
    a8 = np.cos(best_t2)
    
    result = np.array([a0, a4, a8])
    
    # Final verification
    final_bound = _compute_bound(*result)
    if not np.isfinite(final_bound) or final_bound > best_bound * 1.01:
        # Fallback to seed if something went wrong
        return seed.copy()
        
    return result
# EVOLVE-BLOCK-END
