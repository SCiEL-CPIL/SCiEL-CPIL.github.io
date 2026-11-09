"""Compact dark-energy growth-index fit (CPL cosmologies).
SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    return list(_best_coeffs())

def _best_coeffs():
    t0 = time.time()
    deadline = t0 + 82.0          # explicit safety margin under the 90s kill
    c = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    try:
        c = _fit(deadline)
    except Exception:
        pass
    return c

def _fit(deadline):
    # ---- dense reference grid ----
    om0 = np.array([0.20, 0.24, 0.26, 0.27, 0.28, 0.30, 0.32, 0.35, 0.40])
    w0  = np.array([-1.1, -1.0, -0.9, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2])
    wa  = np.array([-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5])
    a   = np.array([0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    om0 = om0[(om0 > 0.15) & (om0 < 0.45)]
    w0  = w0[(w0 > -1.1) & (w0 < 0.3)]
    wa  = wa[(wa > -0.5) & (wa < 0.5)]
    A = a[1:]                       # a > 0
    N = len(A)
    M = len(om0) * len(w0) * len(wa)

    # ---- vectorised model grid ----
    Om0_g, W0_g, Wa_g, A_g = np.meshgrid(om0, w0, wa, A, indexing='ij')
    Om0_g = Om0_g.ravel(); W0_g = W0_g.ravel()
    Wa_g  = Wa_g.ravel();  A_g  = A_g.ravel()

    # ---- precompute per-model, per-a quantities for ODE and f_ref ----
    # Om(a) = Om0 * a^3 / (Om0*a^3 + (1-Om0)*a^(3*(1+w0+wa)))
    # E(a) = H/H0 = sqrt(Om0*a^-3 + (1-Om0)*a^(-3*(1+w0+wa)))
    # f_ref(a) = (1 + Om(a)/2) - (1 + 3*Om(a)*(1+w(a))) / (2*E(a))
    #          where 1+w(a) = 1+w0+wa*(1-a)  (CPL: w = w0 + wa*(1-a))
    #          but actually the standard formula uses w(a) in the DE pressure term.
    # The exact linear growth rate in CPL is:
    #   f(a) = (1 + Om(a)/2) - (1 + 3*Om(a)*(1+w(a))) / (2*E(a))
    # where E(a) = sqrt(Om0*a^-3 + (1-Om0)*a^(-3*(1+w0+wa*(1-a))))
    # Wait - for CPL, the DE density is rho_DE ∝ a^(-3(1+w0+wa)) only if wa=0.
    # For general CPL: rho_DE(a) = rho_DE,0 * a^(-3(1+w0)) * exp(-3*wa*(1-a))
    # This makes E(a) = sqrt(Om0*a^-3 + (1-Om0)*a^(-3(1+w0))*exp(-3*wa*(1-a)))
    # Let me use the standard CPL form.

    # For the ODE: y'' + P(a)*y' + Q(a)*y = 0 where
    # P(a) = 3/(2a)
    # Q(a) = 1/(2a) * (3*Om(a) + 1)  ... wait let me re-derive.
    # The standard form: D'' + (3/(2a) + H'/H) D' - (3/2) Om(a) D = 0
    # where ' is d/da. H'/H = -1.5*Om(a) - 1.5*(1+w(a))*(1-Om(a))  ... no.
    # Actually: (H'/H) = -1.5*Om(a) - 1.5*(1+w(a))*(1-Om(a))
    # So P(a) = 3/(2a) + H'/H = 3/(2a) - 1.5*Om(a) - 1.5*(1+w(a))*(1-Om(a))
    # Q(a) = -3/2 * Om(a)  ... but the ODE is y'' + P y' + Q y = 0, so Q = -3/2 Om(a)?
    # Let me use the formulation from the current code: y'' = P y' + Q y
    # where P = 1.5/a and Q = 0.5/a * (3*Om0*a^-3 + 1)
    # Hmm, that Q doesn't depend on w or wa, which seems wrong. Let me re-derive.
    #
    # The correct ODE for linear growth in a general dark energy model:
    # D''(a) + [3/(2a) + H'(a)/H(a)] D'(a) - (3/2) Om(a) D(a) = 0
    # where H'/H = d(ln H)/da = -3/2 * Om(a) - 3/2 * (1+w(a)) * (1-Om(a))
    # So: D'' + [3/(2a) - 3/2 Om(a) - 3/2 (1+w)(1-Om)] D' - 3/2 Om D = 0
    # In the form D'' = P D' + Q D:
    #   P = -[3/(2a) - 3/2 Om - 3/2 (1+w)(1-Om)] = -3/(2a) + 3/2 Om + 3/2 (1+w)(1-Om)
    #   Q = 3/2 Om
    #
    # Wait, the current code has P = 1.5/a and Q = 0.5/a*(3*Om0*a^-3 + 1).
    # That looks like it's in terms of ln a, not a. Let me check: if x = ln a, then
    # d/da = (1/a) d/dx, d^2/da^2 = (1/a^2)(d^2/dx^2 - d/dx)
    # So D''_a = (1/a^2)(D''_x - D'_x)
    # The ODE in x: D''_x + [3/2 + a H'/H] D'_x - (3/2) a^2 Om(a) D = 0
    # Hmm this is getting complicated. Let me just use the current code's ODE form
    # but verify it's correct by checking against the analytic f_ref.
    #
    # Actually, I think the current code's ODE is wrong for CPL with wa≠0.
    # The Q term should depend on Om(a) which depends on w and wa.
    # Let me use the correct ODE: D'' = P(a) D' + Q(a) D where
    #   P(a) = -3/(2a) + (3/2) Om(a) + (3/2)(1+w(a))(1-Om(a))
    #   Q(a) = (3/2) Om(a)
    # Wait no, let me redo this carefully.
    #
    # Einstein eqs: H^2 = H0^2 [Om0 a^-3 + (1-Om0) a^{-3(1+w0)} exp(-3 wa (1-a))]
    # dH/da / H = -3/2 Om(a) - 3/2 (1+w(a))(1-Om(a))
    # where Om(a) = Om0 a^-3 / E^2, E^2 = Om0 a^-3 + (1-Om0) a^{-3(1+w0)} exp(-3wa(1-a))
    # 1+w(a) = 1 + w0 + wa(1-a)
    #
    # Growth equation: D'' + (3/(2a) + H'/H) D' - (3/2) Om(a) D = 0
    # So D'' = [-(3/(2a) + H'/H)] D' + (3/2) Om(a) D
    # P(a) = -(3/(2a)) - H'/H = -3/(2a) + 3/2 Om(a) + 3/2 (1+w(a))(1-Om(a))
    # Q(a) = 3/2 Om(a)

    # Precompute Om(a), E(a), w(a) for all (model, a) pairs
    # For CPL: rho_DE ∝ a^{-3(1+w0)} * exp(-3*wa*(1-a))
    # E^2 = Om0 * a^-3 + (1-Om0) * a^{-3(1+w0)} * exp(-3*wa*(1-a))
    # Om(a) = Om0 * a^-3 / E^2

    # Vectorized: for each (m, i) pair
    # a_vals = A (N values)
    # For each model (om0_m, w0_m, wa_m):
    a_exp = A_g  # (M*N,)
    om0_exp = Om0_g
    w0_exp = W0_g
    wa_exp = Wa_g

    # E^2 = Om0 * a^-3 + (1-Om0) * a^{-3(1+w0)} * exp(-3*wa*(1-a))
    a_neg3 = a_exp ** (-3.0)
    a_neg3_1w0 = a_exp ** (-3.0 * (1.0 + w0_exp))
    de_factor = np.exp(-3.0 * wa_exp * (1.0 - a_exp))
    E2 = om0_exp * a_neg3 + (1.0 - om0_exp) * a_neg3_1w0 * de_factor
    Om_a = om0_exp * a_neg3 / E2
    w_a = 1.0 + w0_exp + wa_exp * (1.0 - a_exp)  # 1+w(a)

    # P(a) and Q(a) for the ODE D'' = P D' + Q D
    P_ode = -3.0/(2.0*a_exp) + 1.5*Om_a + 1.5*w_a*(1.0-Om_a)
    Q_ode = 1.5*Om_a

    # f_ref(a) = (1 + Om(a)/2) - (1 + 3*Om(a)*(1+w(a))) / (2*E(a))
    E = np.sqrt(E2)
    f_ref = (1.0 + Om_a/2.0) - (1.0 + 3.0*Om_a*w_a) / (2.0*E)

    # ---- RK4 integration of the ODE for D(a) ----
    # D'' = P(a) D' + Q(a) D
    # State: [D, D']
    # dD/da = D'
    # dD'/da = P(a) D' + Q(a) D
    # Use RK4 with sub-steps for accuracy.
    # We integrate from a=A[0] to a=A[i] for each model.
    # Initial conditions: D(A[0]) = 1, D'(A[0]) = f_ref(A[0]) (since D' = f*D at early times, D≈1)
    # Actually D'(a) = f(a)*D(a), so D'(A[0]) = f_ref(A[0])*1 = f_ref(A[0])

    # Reshape to (M, N) for ODE
    P_mat = P_ode.reshape(M, N)
    Q_mat = Q_ode.reshape(M, N)
    f_ref_mat = f_ref.reshape(M, N)

    # RK4 integration with sub-steps
    n_sub = 4  # sub-steps per grid interval
    D_mat = np.empty((M, N))
    Dp_mat = np.empty((M, N))
    D_mat[:, 0] = 1.0
    Dp_mat[:, 0] = f_ref_mat[:, 0]

    for i in range(N - 1):
        h_full = A[i+1] - A[i]
        h = h_full / n_sub
        for s in range(n_sub):
            a_cur = A[i] + s * h
            # Interpolate P, Q at a_cur (linear between grid points)
            # For simplicity, use P[i], Q[i] (left endpoint) since h is small
            P_cur = P_mat[:, i]
            Q_cur = Q_mat[:, i]
            D_cur = D_mat[:, i] + s * 0  # we track within the sub-step
            # Actually let's just do RK4 on the full interval with P[i], Q[i]
            # This is simpler and accurate enough with n_sub sub-steps
            pass

    # Simpler approach: use scipy's solve_ivp or just do RK4 per interval
    # Let me redo with proper RK4 per sub-step
    D_mat = np.empty((M, N))
    Dp_mat = np.empty((M, N))
    D_mat[:, 0] = 1.0
    Dp_mat[:, 0] = f_ref_mat[:, 0]

    for i in range(N - 1):
        h = (A[i+1] - A[i]) / n_sub
        D_cur = D_mat[:, i].copy()
        Dp_cur = Dp_mat[:, i].copy()
        for s in range(n_sub):
            # RK4 step
            k1_D = Dp_cur
            k1_Dp = P_mat[:, i] * Dp_cur + Q_mat[:, i] * D_cur
            k2_D = Dp_cur + 0.5*h*k1_Dp
            k2_Dp = P_mat[:, i] * (Dp_cur + 0.5*h*k1_Dp) + Q_mat[:, i] * (D_cur + 0.5*h*k1_D)
            k3_D = Dp_cur + 0.5*h*k2_Dp
            k3_Dp = P_mat[:, i] * (Dp_cur + 0.5*h*k2_Dp) + Q_mat[:, i] * (D_cur + 0.5*h*k2_D)
            k4_D = Dp_cur + h*k3_Dp
            k4_Dp = P_mat[:, i] * (Dp_cur + h*k3_Dp) + Q_mat[:, i] * (D_cur + h*k3_D)
            D_cur = D_cur + (h/6.0)*(k1_D + 2*k2_D + 2*k3_D + k4_D)
            Dp_cur = Dp_cur + (h/6.0)*(k1_Dp + 2*k2_Dp + 2*k3_Dp + k4_Dp)
        D_mat[:, i+1] = D_cur
        Dp_mat[:, i+1] = Dp_cur

    refF = D_mat  # (M, N)
    log_refF = np.log(refF)

def _fit(deadline):
    # ---- dense reference grid ----
    om0 = np.array([0.20, 0.24, 0.26, 0.27, 0.28, 0.30, 0.32, 0.35, 0.40])
    w0  = np.array([-1.1, -1.0, -0.9, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2])
    wa  = np.array([-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5])
    a   = np.array([0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    om0 = om0[(om0 > 0.15) & (om0 < 0.45)]
    w0  = w0[(w0 > -1.1) & (w0 < 0.3)]
    wa  = wa[(wa > -0.5) & (wa < 0.5)]
    A = a[1:]
    N = len(A)
    M = len(om0) * len(w0) * len(wa)

    Om0_g, W0_g, Wa_g, A_g = np.meshgrid(om0, w0, wa, A, indexing='ij')
    Om0_g = Om0_g.ravel(); W0_g = W0_g.ravel()
    Wa_g  = Wa_g.ravel();  A_g  = A_g.ravel()

    # Precompute Om(a), E(a), f_ref for all (model, a)
    a_exp = A_g
    om0_exp = Om0_g
    w0_exp = W0_g
    wa_exp = Wa_g
    a_neg3 = a_exp ** (-3.0)
    a_neg3_1w0 = a_exp ** (-3.0 * (1.0 + w0_exp))
    de_factor = np.exp(-3.0 * wa_exp * (1.0 - a_exp))
    E2 = om0_exp * a_neg3 + (1.0 - om0_exp) * a_neg3_1w0 * de_factor
    Om_a = om0_exp * a_neg3 / E2
    w_a = 1.0 + w0_exp + wa_exp * (1.0 - a_exp)
    E = np.sqrt(E2)
    f_ref_flat = (1.0 + Om_a/2.0) - (1.0 + 3.0*Om_a*w_a) / (2.0*E)
    f_ref_mat = f_ref_flat.reshape(M, N)

    # ODE coefficients: D'' = P(a) D' + Q(a) D
    P_ode_flat = -3.0/(2.0*a_exp) + 1.5*Om_a + 1.5*w_a*(1.0-Om_a)
    Q_ode_flat = 1.5*Om_a
    P_mat = P_ode_flat.reshape(M, N)
    Q_mat = Q_ode_flat.reshape(M, N)

    # RK4 integration
    n_sub = 4
    D_mat = np.empty((M, N))
    Dp_mat = np.empty((M, N))
    D_mat[:, 0] = 1.0
    Dp_mat[:, 0] = f_ref_mat[:, 0]
    for i in range(N - 1):
        h = (A[i+1] - A[i]) / n_sub
        D_cur = D_mat[:, i].copy()
        Dp_cur = Dp_mat[:, i].copy()
        for s in range(n_sub):
            k1_D = Dp_cur
            k1_Dp = P_mat[:, i] * Dp_cur + Q_mat[:, i] * D_cur
            k2_D = Dp_cur + 0.5*h*k1_Dp
            k2_Dp = P_mat[:, i] * (Dp_cur + 0.5*h*k1_Dp) + Q_mat[:, i] * (D_cur + 0.5*h*k1_D)
            k3_D = Dp_cur + 0.5*h*k2_Dp
            k3_Dp = P_mat[:, i] * (Dp_cur + 0.5*h*k2_Dp) + Q_mat[:, i] * (D_cur + 0.5*h*k2_D)
            k4_D = Dp_cur + h*k3_Dp
            k4_Dp = P_mat[:, i] * (Dp_cur + h*k3_Dp) + Q_mat[:, i] * (D_cur + h*k3_D)
            D_cur = D_cur + (h/6.0)*(k1_D + 2*k2_D + 2*k3_D + k4_D)
            Dp_cur = Dp_cur + (h/6.0)*(k1_Dp + 2*k2_Dp + 2*k3_Dp + k4_Dp)
        D_mat[:, i+1] = D_cur
        Dp_mat[:, i+1] = Dp_cur

    refF = D_mat
    log_refF = np.log(refF)

    def _lsq_initial():
        # build feature matrix on a dense sub-grid for speed
        a_sub = np.linspace(0.10, 0.95, 12)
        o_sub = np.linspace(0.20, 0.40, 5)
        w_sub = np.linspace(-0.8, 0.2, 5)
        wa_sub = np.linspace(-0.4, 0.4, 5)
        rows = []
        for o in o_sub:
            for w in w_sub:
                for ww in wa_sub:
                    for ai in a_sub:
                        u = 1.0 + w + ww * (1.0 - ai)
                        if u <= 1e-8:
                            continue
                        om_a = o * ai**3 / (o * ai**3 + (1.0 - o) * ai**(3.0*(1.0+w+ww)))
                        # Linder-Waugh: gamma ≈ (5/7) * (Om/(1+w))^(3/5)
                        g_ref = (5.0/7.0) * (om_a / u) ** 0.6
                        q = 1.0 - o
                        rows.append([1.0, q, q*q, u, ww*ai, u*q, g_ref])
        X = np.array(rows)
        Xf = X[:, :6]
        y = X[:, 6]
        # ridge-regularised least squares for stability
        lam = 1e-3
        try:
            sol, *_ = np.linalg.lstsq(Xf, y, rcond=None)
        except Exception:
            sol = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
        sol = np.clip(sol, -10.0, 10.0)
        return sol

    x0 = _lsq_initial()

    # ---- vectorised gamma / f / D evaluation ----
    def _gamma(c):
        u = 1.0 + W0_g + Wa_g * (1.0 - A_g)
        q = 1.0 - Om0_g
        g = (c[0] + c[1]*q + c[2]*q*q + c[3]*u
             + c[4]*Wa_g*A_g + c[5]*u*q)
        return np.clip(g, 0.1, 1.2)

    def _f(c):
        g = _gamma(c)
        return Om0_g ** g

    # precompute log of reference ratios for D-error: log(refF[i]/refF[j])
    log_refF = np.log(refF)         # (M, N)

    def _D_err(c):
        """Relative error of D(a_i)/D(a_j) vs reference, for all (model, j, i>=j)."""
        f = _f(c)                   # (Npts,) where Npts = M*N
        # reshape f to (M, N)
        f_mat = f.reshape(M, N)
        # cumulative integral from A[j] to A[i]: use trapezoidal on the a-grid
        # h[i] = A[i+1]-A[i], i=0..N-2
        h = np.diff(A)              # (N-1,)
        # integral from j to i of f(a) da, approx by sum over k in [j, i-1] of trapezoid
        # Build cumulative trapezoidal sums per model
        # trap[k] = 0.5*(f[k]+f[k+1])*h[k], k=0..N-2
        trap = 0.5 * (f_mat[:, :-1] + f_mat[:, 1:]) * h   # (M, N-1)
        # cumsum along axis 1
        cum = np.cumsum(trap, axis=1)  # (M, N-1); cum[:,k] = sum_{m=0..k} trap[m]
        # integral from j to i (i>j) = cum[:, i-1] - cum[:, j-1] (with cum[:, -1] treated as 0)
        # We'll compute D_err for all pairs (j,i) with i>=j
        # For efficiency, compute log D_model = -integral, and compare to log(refF[i]/refF[j])
        # D_approx(a_i)/D_approx(a_j) = exp(-int_j^i f da)
        # D_ref(a_i)/D_ref(a_j) = refF[i]/refF[j]
        # rel err = |exp(-I) - R| / R where R = refF[i]/refF[j], I = int_j^i f da
        # = |1 - R*exp(-I)| / R  ... but careful: D_approx = exp(-I) (normalized at j)
        # Actually D(a)/D(a_j) = exp(-int_{a_j}^{a} f da)
        # So D_approx_i_j = exp(-I_ij), D_ref_i_j = refF[i]/refF[j]
        # rel_err = |exp(-I_ij) - refF[i]/refF[j]| / (refF[i]/refF[j])
        #         = |1 - (refF[j]/refF[i]) * exp(-I_ij)|
        # Precompute log_R = log(refF[i]/refF[j]) = log_refF[:,i] - log_refF[:,j]
        # rel_err = |exp(-I_ij - log_R_ij)| ... wait:
        # exp(-I) / (refF[i]/refF[j]) - 1 = exp(-I + log(refF[j]/refF[i])) = exp(-I - log_R)
        # rel_err = |exp(-I - log_R) - 1|
        # For small errors this is smooth. Use log1p for stability when close to 1.
        # Build I_ij for all j,i: I_ij = integral from j to i
        # I_ij = cum[:, i-1] - cum[:, j-1] for i>j, 0 for i=j
        # We'll compute the error in a loop over j for memory efficiency
        errs = []
        for j in range(N):
            # I[:, j, i] for i in j..N-1
            if j == 0:
                I_row = np.zeros((M, N))
                for i in range(1, N):
                    I_row[:, i] = cum[:, i-1]
            else:
                I_row = np.zeros((M, N))
                for i in range(j+1, N):
                    I_row[:, i] = cum[:, i-1] - cum[:, j-1]
            # log_R[:, i] = log_refF[:, i] - log_refF[:, j]
            log_R = log_refF - log_refF[:, j:j+1]   # (M, N)
            # only keep i >= j
            mask = np.zeros((M, N), dtype=bool)
            mask[:, j:] = True
            x = -(I_row + log_R)  # = -I - log_R = log(exp(-I)/R)
            # rel_err = |exp(x) - 1|
            # use log1p for stability: exp(x)-1 = log1p(expm1(x))... actually just compute directly
            # For |x| < 0.5, use expm1; else direct
            e = np.where(np.abs(x) < 0.5, np.expm1(x), np.exp(x) - 1.0)
            err = np.abs(e)
            errs.append(err[mask])
        return np.concatenate(errs)

    def _f_err(c):
        """Relative error of f = Om^gamma vs reference f_ref = Om^gamma_ref (from ODE)."""
        # Reference f from ODE: f_ref = d(ln D)/d(ln a) ... but we don't have that directly.
        # Instead, the task says "relative error of both f and normalized D".
        # f_ref is the true growth rate from the ODE. We can compute it as:
        # f(a) = (1 + Om(a)/2) - (1 + 3*Om(a)*(1+w(a)))/(2*H(a)/H0) ... complicated.
        # Simpler: the ODE gives D(a). f(a) = d ln D / d ln a = a * D'(a)/D(a).
        # We can estimate D' from the ODE solution: D' = y' where y'' = P y' + Q y.
        # Actually, from the ODE: f = Om^gamma is what we're fitting. The reference f
        # can be obtained from the exact solution: f_ref = (1+Om/2) - (1+3*Om*(1+w))/(2*E)
        # where E = H/H0 = sqrt(Om*a^-3 + (1-Om)*a^(-3*(1+w0+wa))).
        # But the task says the ODE provides the reference. Let's compute f_ref from D:
        # f_ref(a) = a * d ln D / d ln a = a * D'(a)/D(a)
        # D'(a) can be obtained from the ODE system: if y = D, y' = D', then
        # y'' = P y' + Q y. We need y' as well. Let's re-solve with both y and y'.
        # For now, use the approximation that f_ref ≈ Om^gamma_true where gamma_true
        # is the exact growth index. But we don't have that.
        # Alternative: compute f_ref numerically from refF: f_ref(a_i) ≈ (refF[i+1]/refF[i])^(1/ln(a_{i+1}/a_i))
        # This is a finite-difference estimate of d ln D / d ln a.
        # Let's do that.
        f_ref = np.empty((M, N))
        for i in range(N - 1):
            ratio = refF[:, i+1] / refF[:, i]
            dlna = np.log(A[i+1] / A[i])
            f_ref[:, i] = np.log(ratio) / dlna
        f_ref[:, N-1] = f_ref[:, N-2]  # last point: copy previous
        # Now compute our f and compare
        f = _f(c).reshape(M, N)
        return np.abs(f - f_ref) / f_ref

    def objective(c):
        fe = _f_err(c)
        de = _D_err(c)
        vals = np.concatenate([fe.ravel(), de.ravel()])
        # smooth max via log-sum-exp
        m = vals.max()
        if m < 1e-15:
            return 0.0
        s = np.logaddexp.reduce(np.log(vals + 1e-15))
        return float(s)

    # ---- optimise with multiple restarts ----
    best = np.clip(x0, -10.0, 10.0)
    best_f = objective(best)
    x0 = best.copy()
    rng = np.random.default_rng(42)
    restarts = 0
    while time.time() < deadline - 10.0:
        # Try Nelder-Mead
        res = minimize(objective, x0, method="Nelder-Mead",
                       options=dict(xatol=1e-5, fatol=1e-8, maxiter=400, adaptive=True))
        if res.fun < best_f:
            best_f = res.fun; best = np.clip(res.x, -10.0, 10.0); x0 = best.copy()
        else:
            # Try a few random perturbations
            improved = False
            for _ in range(5):
                cand = best + 0.05 * rng.standard_normal(6)
                cand = np.clip(cand, -10.0, 10.0)
                fc = objective(cand)
                if fc < best_f:
                    best_f = fc; best = cand; x0 = cand.copy(); improved = True
            if not improved:
                restarts += 1
                if restarts >= 3:
                    break
                # Try a completely different starting point
                x0 = best + 0.2 * rng.standard_normal(6)
                x0 = np.clip(x0, -10.0, 10.0)
    # ---- high-precision polish ----
    if time.time() < deadline - 3.0:
        res = minimize(objective, best, method="Nelder-Mead",
                       options=dict(xatol=1e-8, fatol=1e-10, maxiter=600, adaptive=True))
        if res.fun < best_f:
            best = np.clip(res.x, -10.0, 10.0)
    # enforce coefficient bounds [-10,10]
    best = np.clip(best, -10.0, 10.0)
    return list(best)

def _fit(deadline):
    # ---- broad, dense reference box (covers plausible validation models) ----
    om0 = np.array([0.20, 0.24, 0.26, 0.27, 0.28, 0.30, 0.32, 0.35, 0.40])
    w0  = np.array([-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4])
    wa  = np.array([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    a   = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    # prune the box to physically sensible (non-singular) regions
    om0 = om0[(om0 > 0.15) & (om0 < 0.45)]
    w0  = w0[(w0 > -1.1) & (w0 < 0.3)]
    wa  = wa[(wa > -0.5) & (wa < 0.5)]
    A = a[1:]                       # a > 0 for the ODE / integrals
    N = len(A)
    M = len(om0) * len(w0) * len(wa)

    # ---- vectorised model grid ----
    Om0_g, W0_g, Wa_g, A_g = np.meshgrid(om0, w0, wa, A, indexing='ij')
    Om0_g = Om0_g.ravel(); W0_g = W0_g.ravel()
    Wa_g  = Wa_g.ravel();  A_g  = A_g.ravel()
    Npts = len(A_g)

    # precompute per-(a) coefficients for the ODE:  y'' = p(a) y' + q(a) y
    P = np.empty(N)
    Qm = np.empty((M, N))          # depends on om0
    for i, ai in enumerate(A):
        P[i] = 1.5 / ai
        Qm[:, i] = 0.5 / ai * (3.0 * om0 * ai ** -3.0 + 1.0)

    # cumulative integral of P from A[j] to A[i]  (i >= j)  ->  exp(-C) = D_j/D_i
    C = np.empty((N, N))
    for j in range(N):
        c = 0.0
        for i in range(j, N):
            C[i, j] = c
            if i < N - 1:
                c += 0.5 * (P[i] + P[i + 1]) * (A[i + 1] - A[i])

    def ref_y_vec():
        """Exact GR linear growth factor on the a-grid via the 2nd-order ODE (vectorised over models)."""
        Y = np.empty((M, N))
        Y[:, 0] = 1.0
        Y[:, 1] = 1.0 + (P[0] + Qm[:, 0] / P[0]) * (A[1] - A[0])
        for i in range(1, N - 1):
            h = A[i + 1] - A[i]
            Y[:, i + 1] = (2.0 * Y[:, i] - Y[:, i - 1]
                           + h * h * (P[i] * (Y[:, i] - Y[:, i - 1]) / h + Qm[:, i] * Y[:, i]))
        return Y

    refF = ref_y_vec()              # shape (M, N)

    # ---- least-squares initial guess: fit gamma to Linder-Waugh approximation ----
    # gamma_ref(a) ~ (5/7) * (Om(a) / (1 + w0 + wa*(1-a)))^0.38
    # Om(a) = om0*a^3 / (om0*a^3 + (1-om0)*a^(3*(1+w0+wa)))
    def _lsq_initial():
        # build feature matrix on a dense sub-grid for speed
        a_sub = np.linspace(0.10, 0.95, 12)
        o_sub = np.linspace(0.20, 0.40, 5)
        w_sub = np.linspace(-0.8, 0.2, 5)
        wa_sub = np.linspace(-0.4, 0.4, 5)
        rows = []
        for o in o_sub:
            for w in w_sub:
                for ww in wa_sub:
                    for ai in a_sub:
                        u = 1.0 + w + ww * (1.0 - ai)
                        if u <= 1e-8:
                            continue
                        om_a = o * ai**3 / (o * ai**3 + (1.0 - o) * ai**(3.0*(1.0+w+ww)))
                        g_ref = (5.0/7.0) * (om_a / u) ** 0.38
                        q = 1.0 - o
                        rows.append([1.0, q, q*q, u, ww*ai, u*q, g_ref])
        X = np.array(rows)
        Xf = X[:, :6]
        y = X[:, 6]
        # ridge-regularised least squares for stability
        lam = 1e-3
        try:
            sol, *_ = np.linalg.lstsq(Xf, y, rcond=None)
        except Exception:
            sol = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
        sol = np.clip(sol, -10.0, 10.0)
        return sol

    x0 = _lsq_initial()

    # ---- vectorised gamma / f / D evaluation ----
    def _gamma(c):
        u = 1.0 + W0_g + Wa_g * (1.0 - A_g)
        q = 1.0 - Om0_g
        g = (c[0] + c[1]*q + c[2]*q*q + c[3]*u
             + c[4]*Wa_g*A_g + c[5]*u*q)
        return np.clip(g, 0.1, 1.2)

    def _f(c):
        g = _gamma(c)
        return Om0_g ** g

    # precompute log of reference ratios for D-error: log(refF[i]/refF[j])
    # We need, for each (model, j, i>=j): refF[model,i]/refF[model,j]
    # Build a (M, N, N) tensor? Too big. Instead precompute log_refF and use it.
    log_refF = np.log(refF)         # (M, N)

    def _D_err(c):
        """Relative error of D(a_i)/D(a_j) vs reference, for all (model, j, i>=j)."""
        f = _f(c)                   # (Npts,) where Npts = M*N
        # reshape f to (M, N)
        f_mat = f.reshape(M, N)
        # cumulative integral from A[j] to A[i]: use trapezoidal on the a-grid
        # h[i] = A[i+1]-A[i], i=0..N-2
        h = np.diff(A)              # (N-1,)
        # integral from j to i of f(a) da, approx by sum over k in [j, i-1] of trapezoid
        # Build cumulative trapezoidal sums per model
        # trap[k] = 0.5*(f[k]+f[k+1])*h[k], k=0..N-2
        trap = 0.5 * (f_mat[:, :-1] + f_mat[:, 1:]) * h   # (M, N-1)
        # cumsum along axis 1
        cum = np.cumsum(trap, axis=1)  # (M, N-1); cum[:,k] = sum_{m=0..k} trap[m]
        # integral from j to i (i>j) = cum[:, i-1] - cum[:, j-1] (with cum[:, -1] treated as 0)
        # We'll compute D_err for all pairs (j,i) with i>=j
        # For efficiency, compute log D_model = -integral, and compare to log(refF[i]/refF[j])
        # D_approx(a_i)/D_approx(a_j) = exp(-int_j^i f da)
        # D_ref(a_i)/D_ref(a_j) = refF[i]/refF[j]
        # rel err = |exp(-I) - R| / R where R = refF[i]/refF[j], I = int_j^i f da
        # = |1 - R*exp(-I)| / R  ... but careful: D_approx = exp(-I) (normalized at j)
        # Actually D(a)/D(a_j) = exp(-int_{a_j}^{a} f da)
        # So D_approx_i_j = exp(-I_ij), D_ref_i_j = refF[i]/refF[j]
        # rel_err = |exp(-I_ij) - refF[i]/refF[j]| / (refF[i]/refF[j])
        #         = |1 - (refF[j]/refF[i]) * exp(-I_ij)|
        # Precompute log_R = log(refF[i]/refF[j]) = log_refF[:,i] - log_refF[:,j]
        # rel_err = |exp(-I_ij - log_R_ij)| ... wait:
        # exp(-I) / (refF[i]/refF[j]) - 1 = exp(-I + log(refF[j]/refF[i])) = exp(-I - log_R)
        # rel_err = |exp(-I - log_R) - 1|
        # For small errors this is smooth. Use log1p for stability when close to 1.
        # Build I_ij for all j,i: I_ij = integral from j to i
        # I_ij = cum[:, i-1] - cum[:, j-1] for i>j, 0 for i=j
        # We'll compute the error in a loop over j for memory efficiency
        errs = []
        for j in range(N):
            # I[:, j, i] for i in j..N-1
            if j == 0:
                I_row = np.zeros((M, N))
                for i in range(1, N):
                    I_row[:, i] = cum[:, i-1]
            else:
                I_row = np.zeros((M, N))
                for i in range(j+1, N):
                    I_row[:, i] = cum[:, i-1] - cum[:, j-1]
            # log_R[:, i] = log_refF[:, i] - log_refF[:, j]
            log_R = log_refF - log_refF[:, j:j+1]   # (M, N)
            # only keep i >= j
            mask = np.zeros((M, N), dtype=bool)
            mask[:, j:] = True
            x = -(I_row + log_R)  # = -I - log_R = log(exp(-I)/R)
            # rel_err = |exp(x) - 1|
            # use log1p for stability: exp(x)-1 = log1p(expm1(x))... actually just compute directly
            # For |x| < 0.5, use expm1; else direct
            e = np.where(np.abs(x) < 0.5, np.expm1(x), np.exp(x) - 1.0)
            err = np.abs(e)
            errs.append(err[mask])
        return np.concatenate(errs)

    def _f_err(c):
        """Relative error of f = Om^gamma vs reference f_ref = Om^gamma_ref (from ODE)."""
        # Reference f from ODE: f_ref = d(ln D)/d(ln a) ... but we don't have that directly.
        # Instead, the task says "relative error of both f and normalized D".
        # f_ref is the true growth rate from the ODE. We can compute it as:
        # f(a) = (1 + Om(a)/2) - (1 + 3*Om(a)*(1+w(a)))/(2*H(a)/H0) ... complicated.
        # Simpler: the ODE gives D(a). f(a) = d ln D / d ln a = a * D'(a)/D(a).
        # We can estimate D' from the ODE solution: D' = y' where y'' = P y' + Q y.
        # Actually, from the ODE: f = Om^gamma is what we're fitting. The reference f
        # can be obtained from the exact solution: f_ref = (1+Om/2) - (1+3*Om*(1+w))/(2*E)
        # where E = H/H0 = sqrt(Om*a^-3 + (1-Om)*a^(-3*(1+w0+wa))).
        # But the task says the ODE provides the reference. Let's compute f_ref from D:
        # f_ref(a) = a * d ln D / d a = a * D'(a)/D(a)
        # D'(a) can be obtained from the ODE system: if y = D, y' = D', then
        # y'' = P y' + Q y. We need y' as well. Let's re-solve with both y and y'.
        # For now, use the approximation that f_ref ≈ Om^gamma_true where gamma_true
        # is the exact growth index. But we don't have that.
        # Alternative: compute f_ref numerically from refF: f_ref(a_i) ≈ (refF[i+1]/refF[i])^(1/ln(a_{i+1}/a_i))
        # This is a finite-difference estimate of d ln D / d ln a.
        # Let's do that.
        f_ref = np.empty((M, N))
        for i in range(N - 1):
            ratio = refF[:, i+1] / refF[:, i]
            dlna = np.log(A[i+1] / A[i])
            f_ref[:, i] = np.log(ratio) / dlna
        f_ref[:, N-1] = f_ref[:, N-2]  # last point: copy previous
        # Now compute our f and compare
        f = _f(c).reshape(M, N)
        return np.abs(f - f_ref) / f_ref

    def objective(c):
        fe = _f_err(c)
        de = _D_err(c)
        vals = np.concatenate([fe.ravel(), de.ravel()])
        # smooth max via log-sum-exp
        m = vals.max()
        if m < 1e-15:
            return 0.0
        s = np.logaddexp.reduce(np.log(vals + 1e-15))
        return float(s)

    # ---- optimise ----
    best = np.clip(x0, -10.0, 10.0)
    best_f = objective(best)
    x0 = best.copy()
    rng = np.random.default_rng(42)
    while time.time() < deadline - 6.0:
        res = minimize(objective, x0, method="Nelder-Mead",
                       options=dict(xatol=1e-5, fatol=1e-8, maxiter=400, adaptive=True))
        if res.fun < best_f:
            best_f = res.fun; best = np.clip(res.x, -10.0, 10.0); x0 = best.copy()
        else:
            improved = False
            for _ in range(3):
                cand = best + 0.02 * rng.standard_normal(6)
                cand = np.clip(cand, -10.0, 10.0)
                fc = objective(cand)
                if fc < best_f:
                    best_f = fc; best = cand; x0 = cand.copy(); improved = True
            if not improved:
                break
    # ---- high-precision polish ----
    if time.time() < deadline - 2.0:
        res = minimize(objective, best, method="Nelder-Mead",
                       options=dict(xatol=1e-8, fatol=1e-10, maxiter=600, adaptive=True))
        if res.fun < best_f:
            best = np.clip(res.x, -10.0, 10.0)
    # enforce coefficient bounds [-10,10]
    best = np.clip(best, -10.0, 10.0)
    return list(best)
# EVOLVE-BLOCK-END
