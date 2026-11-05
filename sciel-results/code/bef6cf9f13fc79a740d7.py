"""Product-formula coefficient search for six-site Heisenberg-chain time evolution.

Strategy:
  1. Build A (odd bonds + half field) and B (even bonds + half field) for a 6-site
     open XXZ chain with anisotropy 0.8 and field 0.3.
  2. Precompute eigendecompositions of A, B, and A+B for fast evaluation.
  3. Use Nelder-Mead (pure NumPy) to optimize 24 free coefficients (a and b, each
     length 12, with sum constraints enforced via parameterization).
  4. Track wall-clock deadline and return best result before 150 s.
"""
import time
import numpy as np
from scipy.linalg import expm  # fallback; we prefer eig-based

DEADLINE_S = 145.0  # safety margin under 150 s wall-clock
STAGES = 12
TS = [0.25, 0.5, 1.0]
ANISO = 0.8
FIELD = 0.3
N_SITES = 6


def _build_hamiltonians():
    """Return (A, B) as 2^6 x 2^6 complex matrices."""
    dim = 2 ** N_SITES
    # Basis: site i is bit i (LSB = site 0). Pauli ops on site i.
    # Build sigma_x, sigma_y, sigma_z on each site.
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    def tensor_op(op_site, op):
        """Build full Hilbert-space operator that is op on site_site and I elsewhere."""
        full = np.array([[1.0 + 0j]])
        for s in range(N_SITES - 1, -1, -1):  # MSB to LSB
            if s == op_site:
                full = np.kron(full, op)
            else:
                full = np.kron(full, np.eye(2, dtype=complex))
        return full

    # Heisenberg XXZ: H = sum_{i} [sx_i sx_{i+1} + sy_i sy_{i+1} + aniso * sz_i sz_{i+1}] + field * sum_i sz_i
    # Split into odd bonds (i=0,2,4) -> A, even bonds (i=1,3,5) -> B
    # Field split: half to each.

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    for i in range(N_SITES - 1):
        sx_i = tensor_op(i, sx)
        sx_ip1 = tensor_op(i + 1, sx)
        sy_i = tensor_op(i, sy)
        sy_ip1 = tensor_op(i + 1, sy)
        sz_i = tensor_op(i, sz)
        sz_ip1 = tensor_op(i + 1, sz)

        bond = (sx_i @ sx_ip1 + sy_i @ sy_ip1 + ANISO * sz_i @ sz_ip1)
        if i % 2 == 0:
            A += bond
        else:
            B += bond

    # Field: half to A, half to B
    field_op = np.zeros((dim, dim), dtype=complex)
    for i in range(N_SITES):
        field_op += tensor_op(i, sz)
    A += (FIELD / 2.0) * field_op
    B += (FIELD / 2.0) * field_op

    return A, B


def _eig_decomp(M):
    """Eigendecomposition: M = V diag(eig) V^{-1}. Return V, eig, Vinv."""
    eig, V = np.linalg.eig(M)
    Vinv = np.linalg.inv(V)
    return V, eig, Vinv


def _expm_eig(V, eig, Vinv, theta):
    """exp(i * theta * M) given eigendecomp of M (M = V diag(eig) V^{-1})."""
    diag = np.exp(1j * theta * eig)
    Vdiag = V * diag  # column-wise scaling: V[:, k] *= diag[k]
    return Vdiag @ Vinv


def _compute_U(a, b, V_A, eig_A, Vinv_A, V_B, eig_B, Vinv_B, t):
    """Compute product formula U(t) = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)."""
    dim = V_A.shape[1]
    U = np.eye(dim, dtype=complex)
    for k in range(STAGES):
        # exp(-i a_k t A)
        Ueig = _expm_eig(V_A, eig_A, Vinv_A, -a[k] * t)
        U = Ueig @ U
        # exp(-i b_k t B)
        Ueig = _expm_eig(V_B, eig_B, Vinv_B, -b[k] * t)
        U = Ueig @ U
    return U


def _spectral_norm_err(M):
    """Spectral norm of complex matrix M."""
    try:
        sv = np.linalg.svd(M, compute_uv=False)
        return float(sv[0])
    except Exception:
        # Fallback: frobenius norm / sqrt(dim) as rough upper bound
        return float(np.linalg.norm(M) / np.sqrt(M.shape[0]))


def _objective(x, precomp):
    """
    x: 24-vector. First 12 are a, next 12 are b.
    Constraints: sum(a)=1, sum(b)=1, each in [-50,50].
    We parameterize: a_raw = x[:12], a = a_raw - a_raw.mean() + 1/12 (so sum=1).
    Similarly for b. Then clamp to [-50, 50] and re-normalize.
    """
    a_raw = x[:12]
    b_raw = x[12:24]

    # Center to zero-mean then shift to sum=1
    a = a_raw - np.mean(a_raw) + 1.0 / 12.0
    b = b_raw - np.mean(b_raw) + 1.0 / 12.0

    # Clamp
    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)

    # Re-normalize to sum=1 (handle edge cases)
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-15:
        a = a * (1.0 / sa)
    if abs(sb) > 1e-15:
        b = b * (1.0 / sb)

    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)
    # Final safety re-normalize
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-15:
        a = a * (1.0 / sa)
    if abs(sb) > 1e-15:
        b = b * (1.0 / sb)

    V_A, eig_A, Vinv_A = precomp['VA'], precomp['eigA'], precomp['VinvA']
    V_B, eig_B, Vinv_B = precomp['VB'], precomp['eigB'], precomp['VinvB']
    U_exact = precomp['Uexact']

    err_sum = 0.0
    for idx, t in enumerate(TS):
        U = _compute_U(a, b, V_A, eig_A, Vinv_A, V_B, eig_B, Vinv_B, t)
        err = _spectral_norm_err(U - U_exact[idx])
        err_sum += err

    return err_sum / len(TS)


def _nelder_mead(f, x0, max_iter=5000, timeout=None, start_time=None):
    """Pure-NumPy Nelder-Mead. Returns (best_x, best_f)."""
    n = len(x0)
    alpha, gamma, rho, sigma = 1.0, 2.0, -0.5, 0.5
    x0 = np.array(x0, dtype=float)
    f0 = f(x0)

    # Initialize simplex
    simplex = [x0.copy()]
    fvals = [f0]
    for i in range(n):
        xi = x0.copy()
        xi[i] += 0.05 if xi[i] == 0 else 0.05 * abs(xi[i])
        simplex.append(xi)
        fvals.append(f(xi))

    best_idx = int(np.argmin(fvals))
    best_x = simplex[best_idx].copy()
    best_f = fvals[best_idx]

    for it in range(max_iter):
        if timeout is not None and time.time() > timeout:
            break

        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i].copy() for i in order]
        fvals = [fvals[i] for i in order]

        best_idx = 0
        worst_idx = n
        second_worst_idx = n - 1

        # Centroid of all but worst
        centroid = np.mean(simplex[:n], axis=0)

        # Reflection
        xr = centroid + alpha * (centroid - simplex[worst_idx])
        fr = f(xr)

        if fr < fvals[best_idx]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                simplex[worst_idx] = xe
                fvals[worst_idx] = fe
            else:
                simplex[worst_idx] = xr
                fvals[worst_idx] = fr
        elif fr < fvals[second_worst_idx]:
            simplex[worst_idx] = xr
            fvals[worst_idx] = fr
        else:
            # Contraction
            xc = centroid + rho * (simplex[worst_idx] - centroid)
            fc = f(xc)
            if fc < fvals[worst_idx]:
                simplex[worst_idx] = xc
                fvals[worst_idx] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])

        # Track best
        cur_best = int(np.argmin(fvals))
        if fvals[cur_best] < best_f:
            best_f = fvals[cur_best]
            best_x = simplex[cur_best].copy()

    return best_x, best_f


def coefficients(stages):
    """Optimize 12-stage product formula coefficients."""
    start_time = time.time()
    deadline = start_time + DEADLINE_S

    # Build Hamiltonians
    A, B = _build_hamiltonians()

    # Eigendecompositions
    VA, eigA, VinvA = _eig_decomp(A)
    VB, eigB, VinvB = _eig_decomp(B)
    VAB, eigAB, VinvAB = _eig_decomp(A + B)

    # Precompute exact propagators
    Uexact = []
    for t in TS:
        Ue = _expm_eig(VAB, eigAB, VinvAB, -t)
        Uexact.append(Ue)

    precomp = {
        'VA': VA, 'eigA': eigA, 'VinvA': VinvA,
        'VB': VB, 'eigB': eigB, 'VinvB': VinvB,
        'Uexact': Uexact,
    }

    # Initial guess: uniform coefficients
    x0 = np.ones(24)
    # Seed: a_k = 1/12, b_k = 1/12
    # Perturb slightly to break symmetry
    rng = np.random.RandomState(42)
    x0 += 0.01 * rng.randn(24)

    # Run Nelder-Mead
    timeout = min(deadline - 2.0, start_time + DEADLINE_S - 2.0)
    x_best, f_best = _nelder_mead(_objective, x0, max_iter=3000, timeout=timeout, start_time=start_time)

    # Extract a, b
    a_raw = x_best[:12]
    b_raw = x_best[12:24]
    a = a_raw - np.mean(a_raw) + 1.0 / 12.0
    b = b_raw - np.mean(b_raw) + 1.0 / 12.0
    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-15:
        a = a / sa
    if abs(sb) > 1e-15:
        b = b / sb
    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)
    # Final normalize
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-15:
        a = a / sa
    if abs(sb) > 1e-15:
        b = b / sb

    a_list = [float(v) for v in a]
    b_list = [float(v) for v in b]

    return a_list, b_list
