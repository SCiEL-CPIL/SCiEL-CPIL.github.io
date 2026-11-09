# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model combining Monod kinetics with a flexible
    Fourier (sum-of-sines) basis for temperature and pH.

    This hybrid approach uses the biologically-motivated Monod term for
    substrate saturation and a linear combination of sinusoidal features
    for temperature and pH to capture their optimal ranges and potential
    periodic constraints. The Fourier basis uses fixed frequencies to
    ensure a convex optimization landscape for the amplitudes, allowing
    BFGS to converge reliably.

    Inputs:
    b: Population density (range ~[1, 5])
    s: Substrate concentration (range ~[1, 40])
    temp: Temperature (range ~[1, 40])
    pH: pH level (range ~[1, 14])

    Parameters (10):
    p0: Half-saturation constant Ks (for Monod term, used as exp(p0) for stability)
    p1: Scaling factor for the overall growth rate
    p2: Bias term
    p3, p4: Amplitudes for temperature 1st harmonic (sin, cos)
    p5, p6: Amplitudes for temperature 2nd harmonic (sin, cos)
    p7, p8: Amplitudes for pH 1st harmonic (sin, cos)
    p9: Amplitude for pH 2nd harmonic (sin)
    """
    p = params

    # Normalize inputs to [0, 1] range for the Fourier basis
    # b: [1, 5]
    b_norm = (b - 1.0) / 4.0
    # s: [1, 40]
    s_norm = (s - 1.0) / 39.0
    # temp: [1, 40]
    temp_norm = (temp - 1.0) / 39.0
    # pH: [1, 14]
    ph_norm = (pH - 1.0) / 13.0

    # Monod term for substrate
    # Use exp(p0) to ensure Ks is positive
    Ks = np.exp(p[0])
    monod = s / (Ks + s)

    # Fourier basis for temperature
    sin_temp_1 = np.sin(2.0 * np.pi * temp_norm)
    cos_temp_1 = np.cos(2.0 * np.pi * temp_norm)
    sin_temp_2 = np.sin(4.0 * np.pi * temp_norm)
    cos_temp_2 = np.cos(4.0 * np.pi * temp_norm)

    # Fourier basis for pH
    sin_ph_1 = np.sin(2.0 * np.pi * ph_norm)
    cos_ph_1 = np.cos(2.0 * np.pi * ph_norm)
    sin_ph_2 = np.sin(4.0 * np.pi * ph_norm)

    # Linear combination of Fourier features for temp and pH
    # p3, p4 for temp 1st harmonic
    # p5, p6 for temp 2nd harmonic
    # p7, p8 for pH 1st harmonic
    # p9 for pH 2nd harmonic (sin only, to save params)
    temp_effect = p[3] * sin_temp_1 + p[4] * cos_temp_1 + p[5] * sin_temp_2 + p[6] * cos_temp_2
    ph_effect = p[7] * sin_ph_1 + p[8] * cos_ph_1 + p[9] * sin_ph_2

    # Combine: base growth rate scaled by Monod, plus Fourier corrections
    # The structure is: mu = p1 * monod * (1 + temp_effect + ph_effect) + p2
    # This allows the Fourier terms to modulate the growth rate multiplicatively
    # and additively.
    # Alternatively, a simpler additive structure might be more stable:
    # mu = p1 * monod + temp_effect + ph_effect + p2
    # Let's try the multiplicative modulation first, as it respects the
    # idea that temp/pH modulate the maximum rate set by substrate.
    # However, if temp_effect or ph_effect become large negative, we might
    # get negative growth rates, which is fine (inhibition).
    
    # To be safe and flexible, let's use an additive combination of the
    # scaled Monod term and the Fourier terms. This is a linear model in
    # the parameters (except p0 and p1 which are in the Monod term), but
    # the Fourier amplitudes are linear.
    
    # Let's stick to a structure where the Fourier terms are additive corrections
    # to the base Monod growth.
    base_growth = p[1] * monod
    correction = temp_effect + ph_effect
    result = base_growth + correction + p[2]

    # Ensure the output is finite
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
