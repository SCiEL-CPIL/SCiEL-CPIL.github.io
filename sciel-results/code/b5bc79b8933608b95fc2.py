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

def _expm_hermitian(eigvals, eigvecs, t):
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

    A, B = _get_AB()
    t_list = [0.25, 0.5, 1.0]
    
    # Precompute exact propagators
    exact = {}
    for t in t_list:
        exact[t] = _expm_hermitian(_AB_eigvals, _AB_eigvecs, t)

    def _expm_A(t):
        return _expm_hermitian(_A_eigvals, _A_eigvecs, t)
    
    def _expm_B(t):
        return _expm_hermitian(_B_eigvals, _B_eigvecs, t)

    def U_of_sym(a_sym, b_sym, t):
        # a_sym = [a1..a6], b_sym = [b1..b6]
        # Full a = [a1..a6, a6..a1]
        # Full b = [b1..b6, b6..b1]
        # U = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)
        # Due to symmetry, we can compute the first half and mirror?
        # No, the product order matters. 
        # U = (B_12 A_12) ... (B_1 A_1)
        # With symmetry a_k = a_{13-k}, b_k = b_{13-k}.
        # We just iterate 12 times.
        U = np.eye(64, dtype=complex)
        # k=0..5 corresponds to first half (indices 0..5 in 0-based)
        # k=6..11 corresponds to second half
        # a_full = a_sym + a_sym[::-1]
        # b_full = b_sym + b_sym[::-1]
        
        # Optimization: Precompute exps for the unique values?
        # The values a_sym[0..5] are unique. a_full[6..11] are a_sym[5..0].
        # So we compute 6 unique A-exps and 6 unique B-exps.
        
        Ea_list = [_expm_A(a_sym[i] * t) for i in range(6)]
        Eb_list = [_expm_B(b_sym[i] * t) for i in range(6)]
        
        # Sequence:
        # k=0: B[0] A[0]
        # k=1: B[1] A[1]
        # k=2: B[2] A[2]
        # k=3: B[3] A[3]
        # k=4: B[4] A[4]
        # k=5: B[5] A[5]
        # k=6: B[5] A[5]  (b_6 = b_sym[5], a_6 = a_sym[5]) -- Wait.
        # Symmetry definition: a_k = a_{13-k}.
        # k=1 (idx 0) <-> k=12 (idx 11). a[0] = a[11].
        # k=2 (idx 1) <-> k=11 (idx 10). a[1] = a[10].
        # k=3 (idx 2) <-> k=10 (idx 9). a[2] = a[9].
        # k=4 (idx 3) <-> k=9 (idx 8). a[3] = a[8].
        # k=5 (idx 4) <-> k=8 (idx 7). a[4] = a[7].
        # k=6 (idx 5) <-> k=7 (idx 6). a[5] = a[6].
        
        # So a_full = [a0, a1, a2, a3, a4, a5, a5, a4, a3, a2, a1, a0]
        # b_full = [b0, b1, b2, b3, b4, b5, b5, b4, b3, b2, b1, b0]
        
        # We need to apply in order k=0 to 11.
        # U = E_b11 E_a11 ... E_b0 E_a0 U
        # Note: The formula is U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A).
        # Usually product notation $\prod_{k=1}^N X_k$ means $X_N \dots X_1$ or $X_1 \dots X_N$?
        # Standard math: $\prod_{k=1}^N X_k = X_N \dots X_1$.
        # However, in code `for k in range(12): U = Eb @ Ea @ U`, this builds $U = (B_{11} A_{11}) \dots (B_0 A_0) I$.
        # This corresponds to $X_{11} \dots X_0$.
        # So the "first" coefficient applied is $k=11$ (last in list) if we follow standard math prod?
        # Let's check the task description: "U(t) = prod_k ... in order k = 1..12".
        # This phrasing "in order k=1..12" usually implies the sequence of operations is $Op_1, Op_2, \dots, Op_{12}$.
        # In matrix multiplication acting on a state $|\psi\rangle$, $U |\psi\rangle = Op_{12} \dots Op_1 |\psi\rangle$.
        # So $U = Op_{12} \dots Op_1$.
        # The code `U = Eb @ Ea @ U` with `U` starting as `I` and loop `k=0..11`:
        # Iter 0: U = B0 A0 I
        # Iter 1: U = B1 A1 B0 A0
        # ...
        # Iter 11: U = B11 A11 ... B0 A0.
        # This matches $Op_{12} \dots Op_1$ if we map k=0 to Op_1?
        # If k=0 is Op_1, then U = Op_12 ... Op_1.
        # So the coefficient $a[0]$ is for the "first" operator applied to the state (rightmost).
        # Symmetry $a_k = a_{13-k}$ (1-based) means $a[0] = a[11]$, $a[1] = a[10]$, etc.
        # This is consistent with the time-reversal symmetry of the error.
        
        # So the list of exponents to apply in order (rightmost to leftmost in matrix product) is:
        # A: a[0], a[1], a[2], a[3], a[4], a[5], a[5], a[4], a[3], a[2], a[1], a[0]
        # B: b[0], b[1], b[2], b[3], b[4], b[5], b[5], b[4], b[3], b[2], b[1], b[0]
        
        # We already have Ea_list[0..5] and Eb_list[0..5].
        # We need to apply:
        # 1. Ea_list[0] @ Eb_list[0] ? No, order is B then A?
        # Formula: exp(-i b_k t B) exp(-i a_k t A).
        # So for a given k, the operator is $B_k A_k$.
        # In the product $U = \dots (B_k A_k) \dots$, $B_k$ is to the left of $A_k$.
        # So for k=0 (rightmost), we apply $A_0$ then $B_0$?
        # $U = B_{11} A_{11} \dots B_0 A_0$.
        # Acting on $\psi$: $A_0 \psi$, then $B_0 (A_0 \psi)$, etc.
        # So in code `U = Eb @ Ea @ U`, `Ea` is applied first to `U` (which holds the result of previous steps).
        # So `U` accumulates from right to left.
        # Iter 0: U = B0 A0. (A0 applied to I, then B0).
        # This matches the formula.
        
        # Sequence of indices for A and B:
        # k=0: A[0], B[0]
        # k=1: A[1], B[1]
        # k=2: A[2], B[2]
        # k=3: A[3], B[3]
        # k=4: A[4], B[4]
        # k=5: A[5], B[5]
        # k=6: A[5], B[5]
        # k=7: A[4], B[4]
        # k=8: A[3], B[3]
        # k=9: A[2], B[2]
        # k=10: A[1], B[1]
        # k=11: A[0], B[0]
        
        # We can just loop 12 times using the precomputed lists.
        for k in range(12):
            if k < 6:
                Ea = Ea_list[k]
                Eb = Eb_list[k]
            else:
                idx = 11 - k
                Ea = Ea_list[idx]
                Eb = Eb_list[idx]
            U = Eb @ Ea @ U
        return U

    def obj(theta):
        # theta: [a1..a5, b1..b5]
        # a6 = 0.5 - sum(a1..a5)
        # b6 = 0.5 - sum(b1..b5)
        a_free = theta[:5]
        b_free = theta[5:10]
        
        a6 = 0.5 - np.sum(a_free)
        b6 = 0.5 - np.sum(b_free)
        
        a_sym = np.concatenate([a_free, [a6]])
        b_sym = np.concatenate([b_free, [b6]])
        
        # Clamp
        a_sym = _clamp_to_box(a_sym)
        b_sym = _clamp_to_box(b_sym)
        
        total = 0.0
        for t in t_list:
            U = U_of_sym(a_sym, b_sym, t)
            err = U - exact[t]
            total += np.linalg.norm(err, 2)
        return total / len(t_list)

    # Start from symmetric Strang-like point
    # Uniform: a_i = 1/12? No, sum a_sym = 0.5. 6 * a = 0.5 => a = 1/12.
    # So a_sym = [1/12, 1/12, 1/12, 1/12, 1/12, 1/12].
    # b_sym = [1/12, 1/12, 1/12, 1/12, 1/12, 1/12].
    theta0 = np.array([1.0/12.0]*5 + [1.0/12.0]*5)
    
    best_theta = theta0.copy()
    best_val = obj(theta0)
    
    # Multi-start
    np.random.seed(42)
    starts = [theta0.copy()]
    for _ in range(10):
        # Perturb around uniform
        perturbed = theta0 + np.random.randn(10) * 0.05
        starts.append(perturbed)
        
    for start in starts:
        if time.time() > _DEADLINE - 15:
            break
        try:
            res_theta, res_val = _nelder_mead(obj, start, max_iter=300, tol=1e-8, deadline=_DEADLINE)
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
    
    a_sym = np.concatenate([a_free, [a6]])
    b_sym = np.concatenate([b_free, [b6]])
    
    a_sym = _clamp_to_box(a_sym)
    b_sym = _clamp_to_box(b_sym)
    
    # Expand to 12
    a = np.concatenate([a_sym, a_sym[::-1]])
    b = np.concatenate([b_sym, b_sym[::-1]])
    
    a = a.tolist()
    b = b.tolist()
    
    # Pad/truncate to stages
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
    return a[:stages], b[:stages]
