import time
import numpy as np

_DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def _build_hamiltonians():
    """Build A (odd bonds 2-3,4-5 + half field) and B (even bonds 1-2,3-4,5-6 + half field)."""
    n = 6
    dim = 2 ** n
    delta = 0.8
    field = 0.3
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    def site_op(op, site):
        """op (2x2) acting on site (0-indexed)."""
        ops = []
        for s in range(n):
            ops.append(op if s == site else I)
        # build in order site 0 .. site n-1
        result = ops[0]
        for s in range(1, n):
            result = np.kron(result, ops[s])
        return result

    def bond_op(op1, op2, s1, s2):
        ops = []
        for s in range(n):
            if s == s1:
                ops.append(op1)
            elif s == s2:
                ops.append(op2)
            else:
                ops.append(I)
        result = ops[0]
        for s in range(1, n):
            result = np.kron(result, ops[s])
        return result

    # XXZ bond: (1/2)(Sx Sx + Sy Sy) + delta Sz Sz  = (1/4)(sx sx + sy sy) + delta/4 sz sz
    def xxz_bond(s1, s2):
        return (0.25 * bond_op(sx, sx, s1, s2)
                + 0.25 * bond_op(sy, sy, s1, s2)
                + (delta / 4.0) * bond_op(sz, sz, s1, s2))

    # A: odd bonds (sites 1-2, 3-4 in 0-index) + half field on all sites
    A = np.zeros((dim, dim), dtype=complex)
    A += xxz_bond(1, 2)
    A += xxz_bond(3, 4)
    for s in range(n):
        A += (field / 2.0) * site_op(sz, s)

    # B: even bonds (sites 0-1, 2-3, 4-5) + half field on all sites
    B = np.zeros((dim, dim), dtype=complex)
    B += xxz_bond(0, 1)
    B += xxz_bond(2, 3)
    B += xxz_bond(4, 5)
    for s in range(n):
        B += (field / 2.0) * site_op(sz, s)

    return A, B

_A = None
_B = None
_A_eigvals = None
_A_eigvecs = None
_B_eigvals = None
_B_eigvecs = None
_AB_eigvals = None
_AB_eigvecs = None

def _get_AB():
    global _A, _B, _A_eigvals, _A_eigvecs, _B_eigvals, _B_eigvecs, _AB_eigvals, _AB_eigvecs
    if _A is None:
        _A, _B = _build_hamiltonians()
        _A_eigvals, _A_eigvecs = np.linalg.eigh(_A)
        _B_eigvals, _B_eigvecs = np.linalg.eigh(_B)
        _AB_eigvals, _AB_eigvecs = np.linalg.eigh(_A + _B)
    return _A, _B

def _expm_hermitian(H, t, eigvals, eigvecs):
    """Compute exp(-i H t) for Hermitian H using precomputed eigendecomposition."""
    phase = np.exp(-1j * eigvals * t)
    return (eigvecs * phase) @ eigvecs.conj().T

def _clamp_to_box(x, lo=-50.0, hi=50.0):
    return np.clip(x, lo, hi)

def _nelder_mead(objective, x0, max_iter=500, tol=1e-6, deadline=None):
    """Simple Nelder-Mead implementation."""
    n = len(x0)
    # Initial simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        point = np.array(x0, dtype=float)
        if point[i] != 0:
            point[i] *= 1.05
        else:
            point[i] += 0.05
        simplex.append(point)
    
    values = [objective(p) for p in simplex]
    best_val = min(values)
    best_point = simplex[values.index(best_val)]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline - 5:
            break
            
        # Sort simplex
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        # Check convergence
        if abs(values[-1] - values[0]) < tol:
            break
            
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < values[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
        elif fr < values[-2]:
            simplex[-1] = xr
            values[-1] = fr
        else:
            # Contraction
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = objective(xc)
            if fc < values[-1]:
                simplex[-1] = xc
                values[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    values[i] = objective(simplex[i])
        
        current_best = min(values)
        if current_best < best_val:
            best_val = current_best
            best_point = simplex[values.index(current_best)]
            
    return best_point, best_val

def coefficients(stages):
    """Optimize 24 coefficients to minimize mean spectral-norm error."""
    if time.time() > _DEADLINE - 20:
        # Not enough time; return a sane Strang-like default
        sub = stages // 2
        a = [0.5 / sub] * (2 * sub)
        b = []
        for _ in range(sub):
            b += [1.0 / sub, 0.0]
        while len(a) < stages:
            a.append(0.0); b.append(0.0)
        return a[:stages], b[:stages]

    _A, _B = _get_AB()
    t_list = [0.25, 0.5, 1.0]
    
    # Precompute exact propagators
    exact = {}
    for t in t_list:
        exact[t] = _expm_hermitian(_A + _B, t, _AB_eigvals, _AB_eigvecs)

    def U_of(a, b, t):
        U = np.eye(64, dtype=complex)
        for k in range(12):
            Eb = _expm_hermitian(_B, b[k] * t, _B_eigvals, _B_eigvecs)
            Ea = _expm_hermitian(_A, a[k] * t, _A_eigvals, _A_eigvecs)
            U = Eb @ Ea @ U
        return U

    def obj(theta):
        # theta has 10 parameters: 5 for a, 5 for b
        # Symmetric a: a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1
        # Sum constraint: 2*(a1+...+a6) = 1 => a1+...+a6 = 0.5
        # Free: a1, a2, a3, a4, a5. a6 = 0.5 - (a1+...+a5)
        
        # Symmetric b: b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1
        # Sum constraint: 2*(b1+...+b6) = 1 => b1+...+b6 = 0.5
        # Free: b1, b2, b3, b4, b5. b6 = 0.5 - (b1+...+b5)
        
        a_free = theta[:5]
        b_free = theta[5:10]
        
        a6 = 0.5 - np.sum(a_free)
        b6 = 0.5 - np.sum(b_free)
        
        a = np.array([a_free[0], a_free[1], a_free[2], a_free[3], a_free[4], a6, a6, a_free[4], a_free[3], a_free[2], a_free[1], a_free[0]])
        b = np.array([b_free[0], b_free[1], b_free[2], b_free[3], b_free[4], b6, b6, b_free[4], b_free[3], b_free[2], b_free[1], b_free[0]])
        
        a = _clamp_to_box(a)
        b = _clamp_to_box(b)
        
        total = 0.0
        for t in t_list:
            U = U_of(a, b, t)
            err = U - exact[t]
            total += np.linalg.norm(err, 2)
        return total / len(t_list)

    # Initial guess based on 4th order Suzuki-Trotter
    # Coeffs for 4th order:
    # z1 = 1/(2-2^(1/3)), z2 = -2^(1/3)/(2-2^(1/3)), z3 = 2/(2-2^(1/3))
    # Structure for 12 stages (symmetric):
    # A: [z1/3, z2/3, z3/3, 0, 0, 0, 0, 0, z3/3, z2/3, z1/3, 0]? No, this is getting complicated.
    # Let's just use a uniform distribution as a safe start, but optimized.
    # Uniform: a_k = 1/12, b_k = 1/12.
    # Symmetric uniform:
    # a = [1/12]*12 -> a1..a6 = 1/12. Sum = 6/12 = 0.5. Correct.
    # b = [1/12]*12 -> b1..b6 = 1/12. Sum = 6/12 = 0.5. Correct.
    
    theta0 = np.full(10, 1.0/12.0)
    
    best_theta = theta0.copy()
    best_val = obj(theta0)
    
    # Run Nelder-Mead
    try:
        res_theta, res_val = _nelder_mead(obj, theta0, max_iter=1000, tol=1e-5, deadline=_DEADLINE)
        if res_val < best_val:
            best_val = res_val
            best_theta = res_theta
    except Exception:
        pass

    # Extract final coefficients
    a_free = best_theta[:5]
    b_free = best_theta[5:10]
    
    a6 = 0.5 - np.sum(a_free)
    b6 = 0.5 - np.sum(b_free)
    
    a = np.array([a_free[0], a_free[1], a_free[2], a_free[3], a_free[4], a6, a6, a_free[4], a_free[3], a_free[2], a_free[1], a_free[0]])
    b = np.array([b_free[0], b_free[1], b_free[2], b_free[3], b_free[4], b6, b6, b_free[4], b_free[3], b_free[2], b_free[1], b_free[0]])
    
    a = _clamp_to_box(a).tolist()
    b = _clamp_to_box(b).tolist()

    # Pad/truncate to stages
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
    return a[:stages], b[:stages]
