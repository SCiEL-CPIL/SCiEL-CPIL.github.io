import numpy as np
from scipy import optimize
import time
import sympy
from sympy import Rational, symbols, Poly, hermite_poly, real_roots, nsolve, S, sqrt, pi, Integer, sympify
from sympy.polys.polytools import Poly as SymPyPoly
import random

# Precompute exact Hermite polynomials and their values at 0
_x = symbols('x')
_H0 = hermite_poly(0, _x)
_H4 = hermite_poly(4, _x)
_H8 = hermite_poly(8, _x)
_H12 = hermite_poly(12, _x)

_H0_0 = _H0.subs(_x, 0)  # 1
_H4_0 = _H4.subs(_x, 0)  # 12
_H8_0 = _H8.subs(_x, 0)  # 1008
_H12_0 = _H12.subs(_x, 0)  # 103950

def compute_value_exact(c0_r, c1_r, c2_r):
    """Compute value using exact rational arithmetic."""
    c0 = Rational(c0_r)
    c1 = Rational(c1_r)
    c2 = Rational(c2_r)
    
    # c3 = -(c0*H0(0) + c1*H4(0) + c2*H8(0)) / H12(0)
    c3 = -(c0*_H0_0 + c1*_H4_0 + c2*_H8_0) / _H12_0
    
    # g = c0*H0 + c1*H4 + c2*H8 + c3*H12
    g = c0*_H0 + c1*_H4 + c2*_H8 + c3*_H12
    g = sympify(g).expand()
    
    # q = g / x^2, must divide exactly
    try:
        q = Poly(g, _x).as_expr() / _x**2
        q_poly = Poly(q, _x)
    except:
        return float('inf')
    
    # Check if q is a valid polynomial
    if q_poly.degree() < 0:
        return float('inf')
    
    # Find real roots of q
    try:
        roots = real_roots(q_poly, _x)
    except:
        return float('inf')
    
    if len(roots) == 0:
        return float('inf')
    
    # Find roots where q changes sign (odd multiplicity)
    changing_roots = []
    for r in roots:
        # Check multiplicity
        mult = q_poly.degree() - Poly(q_poly.as_expr() / (_x - r), _x).degree() if r != 0 else 0
        # Better: check if multiplicity is odd by evaluating sign change
        # Use a small perturbation
        r_float = float(r)
        if abs(r_float) < 1e-15:
            continue
        eps = 1e-8 * max(1.0, abs(r_float))
        f_minus = float((q_poly.as_expr()).subs(_x, r - sympify(eps)))
        f_plus = float((q_poly.as_expr()).subs(_x, r + sympify(eps)))
        if f_minus * f_plus < 0:
            changing_roots.append(r_float)
    
    if len(changing_roots) == 0:
        return float('inf')
    
    r_max = max(changing_roots)
    if r_max <= 0:
        return float('inf')
    
    return r_max**2 / (2.0 * float(pi))

def construct():
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin before 210s kill
    
    # Seed parameters
    seed_c0 = -113.0/100.0
    seed_c1 = 1.0/25.0
    seed_c2 = 1.0/3240.0
    
    best_value = compute_value_exact(seed_c0, seed_c1, seed_c2)
    best_params = [seed_c0, seed_c1, seed_c2]
    
    rng = random.Random(42)
    
    # Phase 1: Coarse grid search around seed
    # c0 in [-2, 0], c1 in [-0.1, 0.1], c2 in [-0.001, 0.001]
    n0, n1, n2 = 20, 15, 10
    
    for i0 in range(n0):
        if time.time() > deadline - 60:
            break
        c0 = -2.0 + (0.0 - (-2.0)) * i0 / (n0 - 1)
        for i1 in range(n1):
            c1 = -0.1 + (0.1 - (-0.1)) * i1 / (n1 - 1)
            for i2 in range(n2):
                c2 = -0.001 + (0.001 - (-0.001)) * i2 / (n2 - 1)
                try:
                    val = compute_value_exact(c0, c1, c2)
                    if val < best_value:
                        best_value = val
                        best_params = [c0, c1, c2]
                except:
                    pass
    
    # Phase 2: Local refinement using Nelder-Mead with exact evaluation
    def obj(x):
        try:
            return compute_value_exact(x[0], x[1], x[2])
        except:
            return float('inf')
    
    # Multiple restarts from best and random points
    n_restarts = 30
    for i in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        if i == 0:
            x0 = best_params
        elif i < 10:
            x0 = [
                best_params[0] + rng.gauss(0, 0.1),
                best_params[1] + rng.gauss(0, 0.01),
                best_params[2] + rng.gauss(0, 0.001)
            ]
        else:
            x0 = [
                rng.uniform(-2, 0),
                rng.uniform(-0.1, 0.1),
                rng.uniform(-0.001, 0.001)
            ]
        
        try:
            result = optimize.minimize(
                obj, x0, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            val = obj(result.x)
            if val < best_value:
                best_value = val
                best_params = list(result.x)
        except:
            pass
    
    return np.array(best_params)
