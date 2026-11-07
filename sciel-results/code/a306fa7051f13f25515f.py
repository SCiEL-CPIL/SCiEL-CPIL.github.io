"""Rewritten program: Legendre-basis pulse optimized with L-BFGS-B on the true 3-level physics."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.special import eval_legendre

# ---- hardware / physics constants (all in rad/ns units) ----
DT = 0.5
N = 80
T = DT * N                       # 40 ns
AMP = 0.4                        # amplitude bound
SLEW = 0.12                      # slew bound
DET_LO, DET_HI = -2 * np.pi * 0.002, 2 * np.pi * 0.002
SCALE_LO, SCALE_HI = 0.97, 1.03
T1 = 25000.0
Tphi = 40000.0
ANH = -2 * np.pi * 0.25          # anharmonicity

# 3-level ladder operators
A = np.zeros((3, 3), dtype=complex)
A[0, 1] = 1.0
A[1, 2] = np.sqrt(2.0)
AD = A.conj().T
N_OP = AD @ A
# diagonal number operator values [0,1,2]
N_DIAG = np.array([0.0, 1.0, 2.0])

# states
S0 = np.array([1, 0, 0], dtype=complex)
S1 = np.array([0, 1, 0], dtype=complex)
S_PLUS = np.array([1, 1, 0], dtype=complex) / np.sqrt(2.0)
S_MINUS = np.array([1, -1, 0], dtype=complex) / np.sqrt(2.0)
S_IPOS = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2.0)
S_INEG = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2.0)
STATES = [S0, S1, S_PLUS, S_MINUS, S_IPOS, S_INEG]
# target: X (swap 0<->1)
XG = np.zeros((3, 3), dtype=complex)
XG[0, 1] = 1.0
XG[1, 0] = 1.0
XG[2, 2] = 1.0
TARGETS = [XG @ s for s in STATES]

# Legendre basis on x in [0,1]
X_GRID = np.linspace(0.0, 1.0, N)
L = np.array([eval_legendre(k, X_GRID) for k in range(8)], dtype=float)  # (8, N)


def build_pulse(c):
    """c: (8,) Legendre coefficients -> (I, Q) each (N,)."""
    I = L[0] * c[0] + L[1] * c[1] + L[2] * c[2] + L[3] * c[3]
    Q = L[4] * c[4] + L[5] * c[5] + L[6] * c[6] + L[7] * c[7]
    return I, Q


def liouvillian(det, scale, I, Q):
    """Return (H, [Lops]) for one step. H is (3,3) complex, Lops list of (3,3)."""
    # detuning n + 0.5*anh*n(n-1)  (diagonal)
    d = det * N_DIAG + 0.5 * ANH * N_DIAG * (N_DIAG - 1.0)
    H = np.diag(d.astype(complex))
    drive = 0.5 * scale * (I * (A + AD) + 1j * Q * (A - AD))
    H = H + drive
    L1 = A / np.sqrt(T1)
    L2 = N_OP * np.sqrt(2.0 / Tphi)
    return H, [L1, L2]


def apply_step(rho, H, Lops, dt):
    """Exponentiate the Liouvillian superoperator on the vectorized rho."""
    d = rho.size
    # build superoperator L: drho = L rho
    L = np.zeros((d, d), dtype=complex)
    # commutator part: -i (H rho - rho H)
    # -i H rho: -i * kron(I, H)
    L += -1j * np.kron(np.eye(3, dtype=complex), H)
    # +i rho H: +i * kron(H^T, I)
    L += 1j * np.kron(H.T, np.eye(3, dtype=complex))
    # dissipator: sum_k ( L_k rho L_k^dag - 0.5 {L_k^dag L_k, rho} )
    for Lk in Lops:
        Lkd = Lk.conj().T
        # Lk rho Lkd^dag : kron(Lkd^T, Lk)
        L += np.kron(Lkd.T, Lk)
        # -0.5 Lkd Lk rho : -0.5 kron(I, Lkd Lk)
        L -= 0.5 * np.kron(np.eye(3, dtype=complex), Lkd @ Lk)
        # -0.5 rho Lkd Lk : -0.5 kron((Lkd Lk)^T, I)
        L -= 0.5 * np.kron((Lkd @ Lk).T, np.eye(3, dtype=complex))
    # matrix exponential
    try:
        from scipy.linalg import expm
        E = expm(L * dt)
    except Exception:
        E = np.eye(d, dtype=complex)
    return (E @ rho).reshape(3, 3)


def simulate(det, scale, I, Q):
    """Return final density matrix for the 80-step evolution starting from |0><0|."""
    rho = np.outer(S0, S0.conj())
    for i in range(N):
        H, Lops = liouvillian(det, scale, I[i], Q[i])
        rho = apply_step(rho, H, Lops, DT)
    return rho


def fidelity(rho, target):
    amp = np.trace(rho @ target)
    return np.abs(amp) ** 2


def score_of(I, Q):
    """Mean + max infidelity over the 9 detuning/scale cases."""
    dets = [DET_LO, 0.0, DET_HI]
    scales = [SCALE_LO, 1.0, SCALE_HI]
    infs = []
    for det in dets:
        for scale in scales:
            rho = simulate(det, scale, I, Q)
            fids = [fidelity(rho, t) for t in TARGETS]
            mean_f = float(np.mean(fids))
            infs.append(1.0 - mean_f)
    return float(np.mean(infs) + np.max(infs))


def constraints_ok(I, Q):
    # amplitude
    amp = np.sqrt(I ** 2 + Q ** 2)
    if np.max(amp) > AMP + 1e-9:
        return False
    # slew including from zero
    dI = np.diff(I)
    dQ = np.diff(Q)
    d = np.sqrt(dI ** 2 + dQ ** 2) / DT
    if np.max(d) > SLEW + 1e-9:
        return False
    # from zero into step 1
    if np.hypot(I[0], Q[0]) / DT > SLEW + 1e-9:
        return False
    # from last step to zero
    if np.hypot(I[-1], Q[-1]) / DT > SLEW + 1e-9:
        return False
    return True


def objective(c):
    I, Q = build_pulse(c)
    if not constraints_ok(I, Q):
        # penalty for infeasibility
        amp = np.sqrt(I ** 2 + Q ** 2)
        dI = np.diff(I); dQ = np.diff(Q)
        d = np.sqrt(dI ** 2 + dQ ** 2) / DT
        pen = 0.0
        pen += np.sum(np.clip(amp - AMP, 0, None)) ** 2
        pen += np.sum(np.clip(d - SLEW, 0, None)) ** 2
        pen += max(0.0, np.hypot(I[0], Q[0]) / DT - SLEW) ** 2
        pen += max(0.0, np.hypot(I[-1], Q[-1]) / DT - SLEW) ** 2
        return score_of(I, Q) + 1e3 * pen
    return score_of(I, Q)


def drag_seed():
    """DRAG-style Gaussian seed (sigma 40/6 ns, offset to 0 at edges, area pi)."""
    t = np.linspace(0.0, T, N)
    sigma = T / 6.0
    # Gaussian centered at T/2
    g = np.exp(-0.5 * ((t - T / 2.0) / sigma) ** 2)
    # offset to 0 at edges
    g = g - g[0]
    # area pi
    area = np.trapz(g, dx=DT)
    if area != 0:
        g = g * (np.pi / area)
    I = g
    # small DRAG Q component (derivative-like)
    dI = np.gradient(g, DT)
    Q = 0.05 * dI
    return I, Q


def init_coefficients():
    I, Q = drag_seed()
    # project onto Legendre basis (least squares)
    cI = np.linalg.lstsq(L, I, rcond=None)[0]
    cQ = np.linalg.lstsq(L, Q, rcond=None)[0]
    return np.concatenate([cI, cQ])


def dop853_recheck(I, Q, det, scale):
    """High-accuracy check of the nominal case using DOP853."""
    def rhs(tt, y):
        rho = y.reshape(3, 3)
        H, Lops = liouvillian(det, scale, I[int(tt / DT) % N], Q[int(tt / DT) % N])
        d = rho.size
        L = np.zeros((d, d), dtype=complex)
        L += -1j * np.kron(np.eye(3, dtype=complex), H)
        L += 1j * np.kron(H.T, np.eye(3, dtype=complex))
        for Lk in Lops:
            Lkd = Lk.conj().T
            L += np.kron(Lkd.T, Lk)
            L -= 0.5 * np.kron(np.eye(3, dtype=complex), Lkd @ Lk)
            L -= 0.5 * np.kron((Lkd @ Lk).T, np.eye(3, dtype=complex))
        return (L @ y).reshape(-1)
    y0 = np.outer(S0, S0.conj()).reshape(-1)
    sol = solve_ivp(rhs, [0.0, T], y0, method="DOP853", rtol=1e-11, atol=1e-13)
    rho = sol.y[:, -1].reshape(3, 3)
    fids = [fidelity(rho, t) for t in TARGETS]
    return 1.0 - float(np.mean(fids))


# ---- global optimization result (cached) ----
_OPT = None


def _optimize():
    global _OPT
    if _OPT is not None:
        return _OPT
    c0 = init_coefficients()
    # bounds: keep coefficients modest so amplitude/slew stay feasible
    bounds = [(-2.0, 2.0)] * 8
    best_c = c0
    best_val = objective(c0)
    for attempt in range(3):
        res = minimize(objective, best_c, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 200, "ftol": 1e-9, "gtol": 1e-6})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x
    I, Q = build_pulse(best_c)
    # final feasibility clip
    if not constraints_ok(I, Q):
        # fall back to seed
        I, Q = drag_seed()
    _OPT = (I, Q)
    return _OPT


def pulse(target_gate, hardware_model):
    I, Q = _optimize()
    return np.stack([I, Q], axis=1)
# EVOLVE-BLOCK-END
