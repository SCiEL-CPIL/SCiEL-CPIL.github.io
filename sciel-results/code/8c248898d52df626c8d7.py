"""Broadband antireflection coating design.

EVOLVE-BLOCK: variable-length alternating H/L stack optimized with
Simulated Annealing using a fast vectorized characteristic-matrix
reflectance evaluator, with an explicit deadline guard.
"""
import time
import math
import numpy as np

# ---- constants (fixed by the task) ---------------------------------------
L_IDX = 1.38
H_IDX = 2.35
SUB_IDX = 1.52
AIR_IDX = 1.0
N_WL = 401
WL = np.linspace(400.0, 800.0, N_WL)
MAX_LAYERS = 20
MAX_TOTAL = 1000.0
MAX_LAYER = 400.0
# precompute wavelength-dependent quantities once
K = 2.0 * np.pi / WL                       # wavenumber (rad / nm)
COSL = np.sqrt(max(L_IDX * L_IDX - 1.0, 0.0))
COSH = np.sqrt(max(H_IDX * H_IDX - 1.0, 0.0))
COSN = np.sqrt(max(SUB_IDX * SUB_IDX - 1.0, 0.0))


def _reflectance_mean(theta):
    """Mean reflectance (fraction) of an alternating H/L stack.

    theta[0] = number of layers (>=1, <=MAX_LAYERS).
    theta[1:] = physical thicknesses in nm of the layers (air -> substrate).
    Layers alternate H, L, H, L, ... starting with H on the air side.
    Returns mean reflectance (0..1) over the 401 band wavelengths.
    """
    n = int(round(theta[0]))
    if n < 1:
        n = 1
    if n > MAX_LAYERS:
        n = MAX_LAYERS
    t = theta[1:1 + n].astype(np.float64)
    # keep within per-layer bounds
    t = np.clip(t, 0.0, MAX_LAYER)
    if t.sum() > MAX_TOTAL:
        # scale down to respect the total-thickness budget
        t = t * (MAX_TOTAL / t.sum())
    # alternating refractive indices: H, L, H, L, ...
    idx = np.where((np.arange(n) % 2) == 0, H_IDX, L_IDX)
    # characteristic-matrix products per wavelength (vectorised)
    c = np.empty((n, N_WL), dtype=np.float64)
    s = np.empty((n, N_WL), dtype=np.float64)
    for i in range(n):
        if i % 2 == 0:
            ph = K * t[i] * COSH
            c[i] = np.cos(ph)
            s[i] = np.sin(ph)
        else:
            ph = K * t[i] * COSL
            c[i] = np.cos(ph)
            s[i] = np.sin(ph)
    # build the total characteristic matrix M = M_1 M_2 ... M_n
    m11 = np.ones(N_WL)
    m12 = np.zeros(N_WL)
    m21 = np.zeros(N_WL)
    m22 = np.ones(N_WL)
    for i in range(n):
        ii = idx[i]
        b11 = c[i]
        b12 = (1.0j / ii) * s[i]
        b21 = ii * s[i]
        b22 = c[i]
        nm11 = m11 * b11 + m12 * b21
        nm12 = m11 * b12 + m12 * b22
        nm21 = m21 * b11 + m22 * b21
        nm22 = m21 * b12 + m22 * b22
        m11, m12, m21, m22 = nm11, nm12, nm21, nm22
    # admittance of the stack as seen from the air side
    Y = (AIR_IDX * m11 + m12) / (AIR_IDX * m21 + m22)
    R = np.abs((Y - SUB_IDX) / (Y + SUB_IDX)) ** 2
    return float(np.mean(R))


def _validity(materials, thicknesses):
    """Return True if the design satisfies all task constraints."""
    if len(materials) != len(thicknesses):
        return False
    if len(materials) == 0 or len(materials) > MAX_LAYERS:
        return False
    for m in materials:
        if m not in ("L", "H"):
            return False
    for t in thicknesses:
        if t < 0.0 or t > MAX_LAYER + 1e-9:
            return False
    if sum(thicknesses) > MAX_TOTAL + 1e-9:
        return False
    return True


def design(problem):
    """Design a broadband antireflection stack (air-side -> substrate)."""
    start = time.time()
    deadline = start + 140.0          # safety margin vs 150 s wall-clock kill
    rng = np.random.default_rng(12345)

    # --- build an initial alternating H/L stack with quarter-wave thicknesses
    def _initial_stack():
        lam = 0.5 * (400.0 + 800.0)   # centre wavelength
        # start with a modest number of layers; SA will grow/shrink it
        n = 6
        mats = []
        ths = []
        for i in range(n):
            if i % 2 == 0:
                mats.append("H")
                ths.append(lam / (4.0 * H_IDX))
            else:
                mats.append("L")
                ths.append(lam / (4.0 * L_IDX))
        return mats, ths

    mats0, ths0 = _initial_stack()
    # encode into theta: [n, t0, t1, ..., t_{n-1}]
    theta = np.array([len(ths0)] + ths0, dtype=np.float64)
    cur_cost = _reflectance_mean(theta)
    best_theta = theta.copy()
    best_cost = cur_cost

    T = 5e-4          # initial temperature (reflectance units)
    T_floor = 1e-7
    cool = 0.9990     # cooling per accepted-ish step

    while time.time() < deadline:
        # --- propose a neighbour
        n = int(round(theta[0]))
        r = rng.random()
        if r < 0.15 and n < MAX_LAYERS:
            # add a layer (keep alternating: append the next H/L)
            new_n = n + 1
            new_t = theta[1:1 + n].copy()
            # append thickness for the new (alternating) layer
            idx_new = H_IDX if (new_n - 1) % 2 == 0 else L_IDX
            new_t = np.concatenate([new_t, [rng.uniform(40.0, 200.0)]])
            theta = np.concatenate([[float(new_n)], new_t])
            n = new_n
        elif r < 0.30 and n > 1:
            # remove a layer
            new_n = n - 1
            new_t = theta[1:1 + new_n].copy()
            theta = np.concatenate([[float(new_n)], new_t])
            n = new_n
        elif r < 0.70:
            # mutate one thickness (multiplicative-ish)
            i = 1 + int(rng.integers(0, n))
            new_t = theta[i] * (1.0 + 0.3 * rng.standard_normal())
            new_t = float(np.clip(new_t, 0.0, MAX_LAYER))
            theta[i] = new_t
        else:
            # global-ish move: rescale all thicknesses
            scale = 1.0 + 0.25 * rng.standard_normal()
            theta[1:1 + n] = np.clip(theta[1:1 + n] * scale, 0.0, MAX_LAYER)

        new_cost = _reflectance_mean(theta)
        delta = new_cost - cur_cost
        if delta < 0 or rng.random() < math.exp(-delta / T):
            cur_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_theta = theta.copy()
        T *= cool
        if T < T_floor:
            T = T_floor

    # --- decode best_theta into (materials, thicknesses)
    n = int(round(best_theta[0]))
    n = int(np.clip(n, 1, MAX_LAYERS))
    t = np.clip(best_theta[1:1 + n], 0.0, MAX_LAYER)
    if t.sum() > MAX_TOTAL:
        t = t * (MAX_TOTAL / t.sum())
    materials = ["H" if (i % 2 == 0) else "L" for i in range(n)]
    thicknesses = [float(x) for x in t]

    # final safety: if anything is off, fall back to a valid quarter-wave stack
    if not _validity(materials, thicknesses):
        lam = 600.0
        materials = ["L"]
        thicknesses = [lam / (4.0 * L_IDX)]
    return materials, thicknesses
