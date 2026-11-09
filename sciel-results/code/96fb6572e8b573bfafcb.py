"""Linear-in-parameters feature model with bounded nonlinearities for oscillator."""
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters model: a = sum_i p_i * phi_i(t, x, v)
    
    10 bounded basis functions capturing nonlinear oscillator dynamics:
      phi_0 = 1                          (offset)
      phi_1 = tanh(x)                    (linear spring, bounded)
      phi_2 = tanh(x^2)                  (quadratic spring, bounded)
      phi_3 = tanh(v)                    (linear damping, bounded)
      phi_4 = tanh(x*v)                  (cross coupling)
      phi_5 = sin(t)                     (periodic drive, sine)
      phi_6 = cos(t)                     (periodic drive, cosine)
      phi_7 = tanh(t) * tanh(x)          (time-position interaction)
      phi_8 = tanh(v) * tanh(x)          (damping-position interaction)
      phi_9 = tanh(x^2) * tanh(v)        (nonlinear spring-damping coupling)
    
    This is linear in params, so BFGS faces a convex landscape.
    """
    tanh_x = np.tanh(x)
    tanh_x2 = np.tanh(x * x)
    tanh_v = np.tanh(v)
    tanh_xv = np.tanh(x * v)
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    tanh_t = np.tanh(t)
    
    a = (params[0]
         + params[1] * tanh_x
         + params[2] * tanh_x2
         + params[3] * tanh_v
         + params[4] * tanh_xv
         + params[5] * sin_t
         + params[6] * cos_t
         + params[7] * tanh_t * tanh_x
         + params[8] * tanh_v * tanh_x
         + params[9] * tanh_x2 * tanh_v)
    
    return a
