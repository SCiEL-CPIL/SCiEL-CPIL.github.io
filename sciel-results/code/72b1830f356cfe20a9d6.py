import numpy as np
from scipy.optimize import minimize
import time

# Precompute grids
_L = 15.0
_N = 30000
_x = np.linspace(-_L, _L, _N)
_dx = _x[1] - _x[0]

_Lf = 15.0
_Nf = 30000
_xi = np.linspace(-_Lf, _Lf, _Nf)
_dxi = _xi[1] - _xi[0]

# Precompute powers for efficiency
_x_powers = np.zeros((13, _N))
_xi_powers = np.zeros((13, _Nf))
for k in range(13):
    _x_powers[k] = _x**k
    _xi_powers[k] = _xi**k

_w_gauss = np.exp(-_x**2)
_w_exp2 = np.exp(-2*_x**2)
_w_gauss_f = np.exp(-_xi**2 / 4.0)

def _hermite_coeffs(n):
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 2.0])
    H_nm2 = np.array([1.0])
    H_nm1 = np.array([0.0, 2.0])
    for k in range(2, n + 1):
        term1 = np.zeros(k + 1)
        term1[1:] = 2.0 * H_nm1
        term2 = np.zeros(k + 1)
        term2[:len(H_nm2)] = -2.0 * (k - 1) * H_nm2
        H_k = term1 + term2
        H_nm2 = H_nm1
        H_nm1 = H_k
    return H_nm1

_H0 = _hermite_coeffs(0)
_H4 = _hermite_coeffs(4)
_H8 = _hermite_coeffs(8)
_H12 = _hermite_coeffs(12)

def _compute_P_polys(max_degree):
    P = [np.array([1.0])]
    for k in range(max_degree):
        pk = P[-1]
        pk_prime = np.zeros(max(0, len(pk) - 1))
        for i in range(1, len(pk)):
            pk_prime[i-1] = pk[i] * i
        xi_pk = np.zeros(len(pk) + 1)
        xi_pk[1:] = 0.5 * pk
        max_len = max(len(pk_prime), len(xi_pk))
        result = np.zeros(max_len)
        result[:len(pk_prime)] += pk_prime
        result[:len(xi_pk)] -= xi_pk
        P.append(result)
    return P

_P_polys = _compute_P_polys(12)

def _compute_bound(a0, a4, a8):
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 665280.0
    
    P_c = np.zeros(13)
    P_c[:len(_H0)] += a0 * _H0
    P_c[:len(_H4)] += a4 * _H4
    P_c[:len(_H8)] += a8 * _H8
    P_c[:len(_H12)] += a12 * _H12
    
    # P(x) on grid
    P_x = np.zeros(_N)
    for k in range(13):
        if P_c[k] != 0:
            P_x += P_c[k] * _x_powers[k]
    
    norm1 = np.sum(np.abs(P_x) * _w_gauss) * _dx
    norm2_sq = np.sum(P_x**2 * _w_exp2) * _dx
    
    if norm2_sq <= 1e-30:
        return 1e10
    
    # Q(xi)
    Q_c = np.zeros(13, dtype=complex)
    for k in range(13):
        if P_c[k] != 0:
            Q_c[:len(_P_polys[k])] += P_c[k] * (1j**k) * _P_polys[k]
    
    Q_xi = np.zeros(_Nf, dtype=complex)
    for k in range(13):
        if Q_c[k] != 0:
            Q_xi += Q_c[k] * _xi_powers[k]
    
    norm1_hat = np.sum(np.abs(Q_xi) * _w_gauss_f) * _dxi * np.sqrt(np.pi)
    
    return norm1 * norm1_hat / norm2_sq

def construct():
    start_time = time.time()
    deadline = 170.0  # 170s to be safe (limit is 180s, evaluator kills at 210s)
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    best_x = seed.copy()
    best_val = _compute_bound(*seed)
    
    def obj(x):
        try:
            v = _compute_bound(x[0], x[1], x[2])
            return v
        except:
            return 1e10
    
    # Phase 1: Nelder-Mead from seed
    try:
        res = minimize(obj, seed, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x.copy()
    except:
        pass
    
    # Phase 2: Random restarts
    rng = np.random.RandomState(12345)
    n_restarts = 30
    
    for i in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20:
            break
        
        # Different perturbation scales
        if i < 10:
            scale = 0.5
        elif i < 20:
            scale = 2.0
        else:
            scale = 5.0
        
        x_start = best_x + rng.normal(0, scale, size=3)
        
        try:
            res = minimize(obj, x_start, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except:
            pass
    
    # Phase 3: Final polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        try:
            res = minimize(obj, best_x, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except:
            pass
    
    return best_x
