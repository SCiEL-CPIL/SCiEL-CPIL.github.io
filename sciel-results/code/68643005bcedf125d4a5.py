"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for a nonlinear oscillator acceleration.
    
    Combines the structural physics terms from the current program with the
    learnable frequency/phase drive from the high-scoring inspirations.
    
    Terms:
    - Duffing cubic stiffness: x^3
    - Linear stiffness: x
    - Linear damping: v
    - Van der Pol nonlinear damping: x^2 * v
    - Quadratic position: x^2
    - Quadratic velocity (drag): v^2
    - Cross-coupling: x * v
    - Learnable sinusoidal drive: sin(p[7]*t + p[8]) and cos(p[7]*t + p[8])
    - Constant offset
    """
    p = params
    # Ensure we have at least 10 params (pad with zeros if needed)
    if len(p) < 10:
        p = np.concatenate([p, np.zeros(10 - len(p))])
    
    # Precompute powers and products
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    
    # Learnable drive frequency and phase
    # p[7]: frequency, p[8]: phase
    drive_freq = p[7]
    drive_phase = p[8]
    
    # Drive terms using learnable frequency and phase
    # We use two separate amplitudes for sin and cos to allow arbitrary phase shift
    # but since we only have 10 params, we combine them:
    # Actually, let's use p[7] as amplitude for sin, p[8] as amplitude for cos,
    # and keep a fixed frequency or use a param for frequency.
    # Looking at inspiration 2, it used p0*cos(p1*t + p2).
    # Let's use:
    # p[7]: amplitude for sin component
    # p[8]: amplitude for cos component
    # But we also want to learn the frequency. We don't have enough params for both
    # freq and two amplitudes if we keep all other terms.
    # Let's stick to the current program's structure but make the drive more flexible.
    # The current program uses sin(t) and cos(t) with fixed freq 1.
    # Inspiration 2 used cos(p1*t + p2) which is very flexible.
    # Let's use p[7] and p[8] for the drive amplitudes, and assume the frequency is 1
    # or maybe we can combine them into a single cosine with phase?
    # A = sin(wt + phi) = A_sin * sin(wt) + A_cos * cos(wt)
    # If w is fixed, we just need two amplitudes.
    # If w is variable, it's non-linear in params, which might be harder for BFGS.
    # Let's try using p[7] and p[8] as amplitudes for sin(t) and cos(t) respectively.
    # This is what the current program does, but the current program scored poorly.
    # The high scoring ones (18.9594) used a single cosine with learnable freq and phase.
    # Let's try that: drive = p[7] * cos(p[8] * t + p[9]) ... but we need a constant offset too.
    # We have 10 params.
    # p0: x coeff
    # p1: x^3 coeff
    # p2: v coeff
    # p3: x^2*v coeff
    # p4: x^2 coeff
    # p5: v^2 coeff
    # p6: x*v coeff
    # p7: drive amplitude
    # p8: drive frequency
    # p9: drive phase
    # This leaves no room for a constant offset.
    # The inspirations had a constant offset p9.
    # Let's see if the constant offset is crucial. The target range is [-0.7108, 0.6604], centered near 0.
    # Maybe the constant offset can be absorbed or is small.
    # Or we can drop one of the less critical physics terms.
    # The cross term x*v or the v^2 term might be less critical than the drive.
    # Let's try:
    # p0: x
    # p1: x^3
    # p2: v
    # p3: x^2*v
    # p4: x^2
    # p5: v^2
    # p6: x*v
    # p7: drive amp
    # p8: drive freq
    # p9: drive phase
    # And drop the constant offset.
    
    # Alternatively, use the structure from inspiration 2 exactly, but add the Van der Pol term x^2*v which was in the current program.
    # Inspiration 2: p0*cos(p1*t+p2) + p3*x + p4*x^3 + p5*v + p6*x^2 + p7*v^2 + p8*x*v + p9
    # This has 10 params. It does NOT have x^2*v.
    # The current program HAS x^2*v but does NOT have learnable drive freq/phase.
    # Let's combine: Use the learnable drive from Insp 2, and add x^2*v by dropping one of the weaker terms.
    # Which term to drop? x*v (cross) or v^2 (drag)?
    # Let's try dropping v^2 and adding x^2*v.
    
    # Revised mapping:
    # p0: drive amplitude
    # p1: drive frequency
    # p2: drive phase
    # p3: x coeff
    # p4: x^3 coeff
    # p5: v coeff
    # p6: x^2 coeff
    # p7: x^2*v coeff (Van der Pol)
    # p8: x*v coeff (Cross)
    # p9: constant offset
    
    # This drops v^2.
    
    drive = p[0] * np.cos(p[1] * t + p[2])
    
    result = (
        drive
        + p[3] * x
        + p[4] * x3
        + p[5] * v
        + p[6] * x2
        + p[7] * x2 * v
        + p[8] * x * v
        + p[9]
    )
    
    return result
# EVOLVE-BLOCK-END
