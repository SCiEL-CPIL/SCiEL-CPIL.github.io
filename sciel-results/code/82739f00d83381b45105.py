import numpy as np
import scipy.linalg
import scipy.optimize
import time

# Constants
ANISOTROPY = 0.8
FIELD = 0.3
TIMES = [0.25, 0.5, 1.0]
N_SITES = 6
DIM = 2 ** N_SITES
MAX_TIME = 40.0 # Safety margin

def _build_hamiltonians():
    Sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    Sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)

    def two_site_op(op1, op2, site1, site2):
        ops = [np.eye(2, dtype=complex) for _ in range(N_SITES)]
        ops[site1] = op1
        ops[site2] = op2
        res = ops[0]
        for i in range(1, N_SITES):
            res = np.kron(res, ops[i])
        return res

    A = np.zeros((DIM, DIM), dtype=complex)
    B = np.zeros((DIM, DIM), dtype=complex)

    H_field = np.zeros((DIM, DIM), dtype=complex)
    for i in range(N_SITES):
        ops = [np.eye(2, dtype=complex) for _ in range(N_SITES)]
        ops[i] = Sz
        res = ops[0]
        for j in range(1, N_SITES):
            res = np.kron(res, ops[j])
        H_field += res
    H_field *= FIELD

    A += 0.5 * H_field
    B += 0.5 * H_field

    bonds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    for idx, (s1, s2) in enumerate(bonds):
        term = two_site_op(Sx, Sx, s1, s2)
        term += two_site_op(Sy, Sy, s1, s2)
        term += ANISOTROPY * two_site_op(Sz, Sz, s1, s2)
        
        if idx % 2 == 0:
            A += term
        else:
            B += term
            
    return A, B

_A, _B = _build_hamiltonians()
_H_total = _A + _B
_U_exact = {t: scipy.linalg.expm(-1j * t * _H_total) for t in TIMES}

def _compute_error(a, b):
    err_sum = 0.0
    for t in TIMES:
        U_approx = np.eye(DIM, dtype=complex)
        for k in range(12):
            exp_B = scipy.linalg.expm(-1j * b[k] * t * _B)
            exp_A = scipy.linalg.expm(-1j * a[k] * t * _A)
            P_k = exp_B @ exp_A
            U_approx = P_k @ U_approx
        err = np.linalg.norm(U_approx - _U_exact[t], 2)
        err_sum += err
    return err_sum / len(TIMES)

def _optimize():
    start_time = time.time()
    
    def unpack(params):
        a_half = list(params[0:5])
        b_half = list(params[5:10])
        a6 = 0.5 - sum(a_half)
        b6 = 0.5 - sum(b_half)
        a_full = a_half + [a6] + a_half[::-1]
        b_full = b_half + [b6] + b_half[::-1]
        for val in a_full + b_full:
            if val < -50 or val > 50:
                return None, None
        return a_full, b_full

    def objective(params):
        if time.time() - start_time > MAX_TIME:
            return 1e6 # Stop search
        a, b = unpack(params)
        if a is None:
            return 1e6
        return _compute_error(a, b)

    bounds = [(-50, 50)] * 10
    
    try:
        res = scipy.optimize.differential_evolution(
            objective, 
            bounds=bounds, 
            maxiter=200, 
            tol=1e-5, 
            seed=42,
            polish=True,
            popsize=10
        )
        best_params = res.x
        a, b = unpack(best_params)
    except Exception:
        a, b = None, None

    if a is None:
        # Fallback to seed-like
        substeps = 6
        a = [0.5 / substeps] * 12
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
    else:
        a = [max(-50, min(50, x)) for x in a]
        b = [max(-50, min(50, x)) for x in b]

    return a, b

_OPT_CACHE = None

def coefficients(stages):
    global _OPT_CACHE
    if _OPT_CACHE is not None:
        a, b = _OPT_CACHE
        # Adjust length if necessary (though stages should be 12)
        if len(a) != stages:
             # Re-pad or truncate
             while len(a) < stages:
                 a.append(0.0)
                 b.append(0.0)
             a = a[:stages]
             b = b[:stages]
        return a, b
    
    a, b = _optimize()
    _OPT_CACHE = (a, b)
    return a, b
