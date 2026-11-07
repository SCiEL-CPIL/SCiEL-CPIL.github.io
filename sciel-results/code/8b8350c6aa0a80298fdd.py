"""
Fourier series basis with 10 parameters: a constant term plus 9 sinusoidal
terms at harmonically related frequencies scaled to the domain [0.1, 6].
This is linear in params, so BFGS converges to the global optimum.
The Fourier basis provides global smoothness and can represent periodic
and complex smooth components that RBFs may miss, while avoiding the
numerical instability of high-degree polynomials.
"""
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series model with 10 parameters.
    params[0]: constant offset
    params[1:10]: amplitudes for 9 sinusoidal terms
    
    The frequencies are chosen as harmonics of a fundamental frequency
    that gives roughly 1-2 cycles over the domain [0.1, 6].
    We use sin and cos at different frequencies to span a rich basis.
    
    With 9 parameters for sinusoidal terms, we use:
    - sin(k * w * x) and cos(k * w * x) for k = 1, 2, 3, 4, 5
    But that's 10 terms. We need exactly 9.
    
    Let's use:
    - sin(w * x), cos(w * x)          -> 2 terms (k=1)
    - sin(2*w*x), cos(2*w*x)          -> 2 terms (k=2)
    - sin(3*w*x), cos(3*w*x)          -> 2 terms (k=3)
    - sin(4*w*x), cos(4*w*x)          -> 2 terms (k=4)
    - sin(5*w*x)                      -> 1 term  (k=5, sine only)
    Total: 9 sinusoidal terms + 1 constant = 10 params
    
    Fundamental frequency w: domain length is 5.9, so w = 2*pi/5.9 gives
    roughly 1 cycle over the domain. But let's use a slightly higher frequency
    to allow more detail. w = 2*pi/3 gives about 2 cycles over [0.1, 6].
    """
    # Fundamental angular frequency: ~2 cycles over the domain
    # Domain [0.1, 6] has length 5.9
    w = 2.0 * np.pi / 3.0
    
    # Build the Fourier basis matrix: shape (n_points, 9)
    # Term 0: sin(w*x)
    # Term 1: cos(w*x)
    # Term 2: sin(2*w*x)
    # Term 3: cos(2*w*x)
    # Term 4: sin(3*w*x)
    # Term 5: cos(3*w*x)
    # Term 6: sin(4*w*x)
    # Term 7: cos(4*w*x)
    # Term 8: sin(5*w*x)
    
    basis = np.empty((x.shape[0], 9), dtype=np.float64)
    
    for k in range(1, 6):
        arg = k * w * x
        if k <= 4:
            basis[:, 2*(k-1)] = np.sin(arg)
            basis[:, 2*(k-1)+1] = np.cos(arg)
        else:
            # k=5: only sine (to keep total at 9)
            basis[:, 8] = np.sin(arg)
    
    # prediction = constant + weighted sum of Fourier terms
    prediction = params[0] + basis @ params[1:10]
    
    return prediction
