import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Nonlinear oscillator model with time-dependent driving force.
        
        Model:
        a = p0*x + p1*x^2 + p2*x^3 + p3*v + p4*v^2 + p5*v^3 + 
            p6*sin(p7*t) + p8*cos(p7*t) + p9
        """
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
        
        # Nonlinear restoring forces
        restoring = p0 * x + p1 * x**2 + p2 * x**3
        
        # Nonlinear damping forces
        damping = p3 * v + p4 * v**2 + p5 * v**3
        
        # Driving force (general sinusoid with frequency p7)
        drive = p6 * np.sin(p7 * t) + p8 * np.cos(p7 * t)
        
        # Constant bias
        bias = p9
        
        return restoring + damping + drive + bias
