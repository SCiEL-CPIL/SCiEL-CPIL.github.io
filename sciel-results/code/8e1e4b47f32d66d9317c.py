import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Fourier series basis tailored to the domain [0.1, 6].
    
    The domain length is L = 6 - 0.1 = 5.9.
    We define a phase variable phi such that the basis functions sin(k * phi) and cos(k * phi)
    have integer periods over the domain if we choose phi = 2 * pi * (x - x0) / L.
    However, for orthogonality over [x_min, x_max], the standard Fourier basis on [0, L] 
    uses sin(2*pi*k*x/L) and cos(2*pi*k*x/L).
    
    Here, we shift x to be relative to the start of the domain to ensure the basis is 
    well-behaved and orthogonal over the uniform sampling distribution.
    
    x_min = 0.1, x_max = 6.0, L = 5.9.
    Let t = (x - 0.1) / 5.9. t is in [0, 1].
    The orthogonal basis on [0, 1] is {1, cos(2*pi*k*t), sin(2*pi*k*t)}.
    
    We have 10 parameters.
    params[0]: Constant term (DC)
    params[1]: cos(2*pi*t)
    params[2]: sin(2*pi*t)
    params[3]: cos(4*pi*t)
    params[4]: sin(4*pi*t)
    params[5]: cos(6*pi*t)
    params[6]: sin(6*pi*t)
    params[7]: cos(8*pi*t)
    params[8]: sin(8*pi*t)
    params[9]: cos(10*pi*t)
    
    This provides 5 frequency components (plus DC), allowing representation of functions
    with up to 5 "wiggles" across the domain. The orthogonality ensures that the Hessian
    is well-conditioned (diagonal-dominant), which is ideal for BFGS.
    """
    x_min = 0.1
    L = 5.9
    
    # Normalize x to [0, 1]
    t = (x - x_min) / L
    
    # Precompute angles
    # k=1: 2*pi*t
    # k=2: 4*pi*t
    # k=3: 6*pi*t
    # k=4: 8*pi*t
    # k=5: 10*pi*t
    
    y = np.zeros_like(x, dtype=float)
    
    # DC Component
    y += params[0]
    
    # Frequency 1
    angle1 = 2.0 * np.pi * t
    y += params[1] * np.cos(angle1)
    y += params[2] * np.sin(angle1)
    
    # Frequency 2
    angle2 = 4.0 * np.pi * t
    y += params[3] * np.cos(angle2)
    y += params[4] * np.sin(angle2)
    
    # Frequency 3
    angle3 = 6.0 * np.pi * t
    y += params[5] * np.cos(angle3)
    y += params[6] * np.sin(angle3)
    
    # Frequency 4
    angle4 = 8.0 * np.pi * t
    y += params[7] * np.cos(angle4)
    y += params[8] * np.sin(angle4)
    
    # Frequency 5
    angle5 = 10.0 * np.pi * t
    y += params[9] * np.cos(angle5)
    
    return y
