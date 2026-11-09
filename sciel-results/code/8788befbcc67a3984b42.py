import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard: The task states 120s limit, but knowledge says 150s kill.
    # This function is called repeatedly by the optimizer. It should be very fast.
    # No complex search inside equation, just the basis evaluation.
    
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has exactly 10 elements
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Fourier Basis Strategy
    # Domain is [-2, 2]. 
    # Standard Fourier series on [-L, L] uses frequencies n*pi/L.
    # Here L=2, so fundamental frequency is pi/2.
    # We have 10 parameters. We can use:
    # p[0] * 1 (DC term)
    # p[1] * cos(pi/2 * x)
    # p[2] * sin(pi/2 * x)
    # p[3] * cos(2*pi/2 * x) = cos(pi * x)
    # p[4] * sin(2*pi/2 * x) = sin(pi * x)
    # p[5] * cos(3*pi/2 * x)
    # p[6] * sin(3*pi/2 * x)
    # p[7] * cos(4*pi/2 * x) = cos(2*pi * x)
    # p[8] * sin(4*pi/2 * x) = sin(2*pi * x)
    # p[9] * cos(5*pi/2 * x)
    
    # This covers up to the 5th harmonic (with one more cosine).
    # Orthogonality of sin/cos on symmetric intervals ensures well-conditioned
    # Gram matrix, helping BFGS from the all-ones start.
    
    # Calculate basis terms
    # To be safe and general, we compute the angles explicitly.
    # L = 2.0
    # k = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5]
    # types = [const, cos, sin, cos, sin, cos, sin, cos, sin, cos]
    
    # Vectorized calculation for speed
    # Term 0: 1
    # Term 1: cos(0.5 * pi * x)
    # Term 2: sin(0.5 * pi * x)
    # Term 3: cos(1.0 * pi * x)
    # Term 4: sin(1.0 * pi * x)
    # Term 5: cos(1.5 * pi * x)
    # Term 6: sin(1.5 * pi * x)
    # Term 7: cos(2.0 * pi * x)
    # Term 8: sin(2.0 * pi * x)
    # Term 9: cos(2.5 * pi * x)
    
    # Constructing the basis matrix B where B[i, j] is basis_j(x_i)
    # Then y_hat = B @ p
    
    # For efficiency, we can compute directly.
    # Let's create the array of angles for each harmonic.
    
    # Harmonic 1
    angle1 = 0.5 * np.pi * x
    # Harmonic 2
    angle2 = 1.0 * np.pi * x
    # Harmonic 3
    angle3 = 1.5 * np.pi * x
    # Harmonic 4
    angle4 = 2.0 * np.pi * x
    # Harmonic 5
    angle5 = 2.5 * np.pi * x
    
    y_hat = (
        p[0] * 1.0 +
        p[1] * np.cos(angle1) +
        p[2] * np.sin(angle1) +
        p[3] * np.cos(angle2) +
        p[4] * np.sin(angle2) +
        p[5] * np.cos(angle3) +
        p[6] * np.sin(angle3) +
        p[7] * np.cos(angle4) +
        p[8] * np.sin(angle4) +
        p[9] * np.cos(angle5)
    )
    
    return y_hat
