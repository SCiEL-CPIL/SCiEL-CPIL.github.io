"""Bootstrap portfolio: model-and-optimize for stress-strain of aluminium rod."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT_LIMIT = 140.0

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.

    Design family: saturating rational (elastic-plastic knee) + tanh (hardening
    saturation) + smooth piecewise-linear zigzag basis for yield/work-hardening
    segments + temperature-dependent offsets.

    Model:
      stress = term_rational + term_tanh + term_zigzag + B(T)

    term_rational = E(T) * strain / (1 + H*strain)
      E(T) = p0 + p1*temp + p2*temp**2
      H    = p3

    term_tanh = S(T) * tanh(K(T)*strain)
      S(T) = p4 + p5*temp
      K(T) = p6 + p7*temp

    term_zigzag = A(T) * (kink(strain, c1, w) - kink(strain, c2, w))
      A(T) = p8 + p9*temp
      c1 = 0.005  (yield point, fixed near typical Al yield strain)
      c2 = 0.05   (hardening onset, fixed)
      w  = 0.002  (smoothness width, fixed small)
      kink(x, c, w) = smooth approximation of max(x - c, 0) using 0.5*((x-c)+sqrt((x-c)^2 + w^2))

    B(T) = p8*temp**2  (reuses p8 in a quadratic form for offset temperature dependence)
    Actually to keep 10 params and avoid redundancy, we use:
      B(T) = p8 + p9*temp  (linear offset)
    and the zigzag amplitude uses a separate small constant derived from params.

    Let me restructure to use all 10 params cleanly:
      p0, p1, p2 : E(T) quadratic in temp
      p3         : H (rational saturation)
      p4, p5     : S(T) linear in temp (tanh amplitude)
      p6, p7     : K(T) linear in temp (tanh rate)
      p8         : zigzag amplitude A (constant, temperature-independent for simplicity)
      p9         : offset B (constant)

    The zigzag term adds kinked structure at fixed breakpoints to help BFGS
    fit the yield point and work-hardening segments that smooth rational/tanh
    may underfit.
    """
    if time.time() - _START_TIME > _TIMEOUT_LIMIT:
        return np.zeros_like(strain, dtype=np.float64)

    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Temperature-dependent Elastic Modulus (quadratic)
    E_T = p0 + p1 * temp + p2 * temp**2

    # Rational saturation parameter (constant)
    H = p3

    # Tanh amplitude and rate (linear in temp)
    S_T = p4 + p5 * temp
    K_T = p6 + p7 * temp

    # Zigzag amplitude (constant) and offset (constant)
    A_zig = p8
    B_off = p9

    # --- Term 1: Saturating rational (elastic-plastic knee) ---
    den = 1.0 + H * strain
    eps = 1e-8
    den_safe = np.where(np.abs(den) < eps, np.sign(den + 1e-300) * eps, den)
    term_rational = (E_T * strain) / den_safe

    # --- Term 2: Tanh hardening/saturation ---
    tanh_arg = K_T * strain
    term_tanh = S_T * np.tanh(tanh_arg)

    # --- Term 3: Smooth piecewise-linear zigzag basis ---
    # kink(x, c, w) ≈ 0.5 * ((x - c) + sqrt((x - c)^2 + w^2))
    # This is a smooth approximation of max(x - c, 0), differentiable everywhere.
    # We use two kinks at fixed yield-point and hardening-onset strains.
    c1 = 0.005   # typical yield strain for aluminium
    c2 = 0.05    # hardening onset
    w  = 0.002   # smoothness width (small enough to be kink-like, large enough for BFGS)

    d1 = strain - c1
    d2 = strain - c2

    kink1 = 0.5 * (d1 + np.sqrt(d1**2 + w**2))
    kink2 = 0.5 * (d2 + np.sqrt(d2**2 + w**2))

    # Difference creates a "bump" between c1 and c2, modeling the work-hardening
    # segment that rises after yield and then flattens.
    term_zigzag = A_zig * (kink1 - kink2)

    # --- Term 4: Constant offset ---
    term_offset = B_off

    stress = term_rational + term_tanh + term_zigzag + term_offset

    # Ensure finite output
    stress = np.where(np.isfinite(stress), stress, 0.0)

    return stress
# EVOLVE-BLOCK-END
