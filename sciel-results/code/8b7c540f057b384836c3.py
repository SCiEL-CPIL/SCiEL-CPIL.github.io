import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map inputs from [0.5, 2.0] to [-1, 1]
    # t = (u - 1.25) / 0.75
    # When u=0.5, t = -0.75/0.75 = -1
    # When u=2.0, t = 0.75/0.75 = 1
    
    t_x = (x - 1.25) / 0.75
    t_y = (y - 1.25) / 0.75
    t_z = (z - 1.25) / 0.75

    # Chebyshev polynomials of the first kind T_n(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    # T_4(t) = 8t^4 - 8t^2 + 1

    # We have 10 parameters. Let's use a tensor product of Chebyshev polynomials.
    # To maximize expressiveness and conditioning, we can mix low-order terms in each variable.
    
    # Option A: Full tensor product up to degree 2 in each variable?
    # Degrees: 0, 1, 2.
    # Terms:
    # 0: T0(x)T0(y)T0(z) = 1
    # 1: T1(x)T0(y)T0(z) = tx
    # 2: T0(x)T1(y)T0(z) = ty
    # 3: T0(x)T0(y)T1(z) = tz
    # 4: T2(x)T0(y)T0(z) = 2tx^2 - 1
    # 5: T0(x)T2(y)T0(z) = 2ty^2 - 1
    # 6: T0(x)T0(y)T2(z) = 2tz^2 - 1
    # 7: T1(x)T1(y)T0(z) = tx*ty
    # 8: T1(x)T0(y)T1(z) = tx*tz
    # 9: T0(x)T1(y)T1(z) = ty*tz
    
    # This leaves out higher order interactions like T1(x)T1(y)T1(z) or T2 terms with interactions.
    # However, Chebyshev polynomials are orthogonal and well-conditioned.
    # The previous best (1.08) used: 1, tx, ty, tz, tx^3, ty^3, tz^3, tx*ty*tz, tx^2*ty^2, ty^2*tz^2
    # Note that T_3(t) = 4t^3 - 3t. So t^3 is a combination of T_3 and T_1.
    # If we use Chebyshev basis, we can represent cubic terms efficiently.
    
    # Let's try a basis that includes some higher order Chebyshev terms.
    # Maybe:
    # 0: T0(x)T0(y)T0(z)
    # 1: T1(x)T0(y)T0(z)
    # 2: T0(x)T1(y)T0(z)
    # 3: T0(x)T0(y)T1(z)
    # 4: T2(x)T0(y)T0(z)
    # 5: T0(x)T2(y)T0(z)
    # 6: T0(x)T0(y)T2(z)
    # 7: T3(x)T0(y)T0(z)  # Captures cubic behavior in x
    # 8: T0(x)T3(y)T0(z)  # Captures cubic behavior in y
    # 9: T0(x)T0(y)T3(z)  # Captures cubic behavior in z
    
    # This basis is separable (sum of functions of x, y, z individually).
    # The previous best had interaction terms (tx*ty*tz, etc.).
    # If the function is not separable, this might be worse.
    
    # Let's try a mix:
    # 0: 1
    # 1: T1(x)
    # 2: T1(y)
    # 3: T1(z)
    # 4: T2(x)
    # 5: T2(y)
    # 6: T2(z)
    # 7: T1(x)T1(y)
    # 8: T1(x)T1(z)
    # 9: T1(y)T1(z)
    
    # This is the "quadratic" Chebyshev basis.
    # It scored 7.39 as a monomial basis. Chebyshev might be better conditioned.
    
    # What if we prioritize interactions?
    # The 1.08 score had: tx*ty*tz, tx^2*ty^2, ty^2*tz^2.
    # These are high-order interactions.
    # Chebyshev polynomials can represent these.
    # T_2(x)T_2(y) = (2x^2-1)(2y^2-1) = 4x^2y^2 - 2x^2 - 2y^2 + 1.
    # So x^2y^2 is a combination of T_2(x)T_2(y), T_2(x), T_2(y), and 1.
    
    # Let's construct a basis that allows for these high-order interactions via products of Chebyshev polynomials.
    # Basis:
    # 0: T0(x)T0(y)T0(z) = 1
    # 1: T1(x)T0(y)T0(z) = tx
    # 2: T0(x)T1(y)T0(z) = ty
    # 3: T0(x)T0(y)T1(z) = tz
    # 4: T2(x)T0(y)T0(z) = 2tx^2 - 1
    # 5: T0(x)T2(y)T0(z) = 2ty^2 - 1
    # 6: T0(x)T0(y)T2(z) = 2tz^2 - 1
    # 7: T1(x)T1(y)T1(z) = tx*ty*tz
    # 8: T2(x)T2(y)T0(z) = (2tx^2-1)(2ty^2-1)
    # 9: T0(x)T2(y)T2(z) = (2ty^2-1)(2tz^2-1)
    
    # This closely mirrors the structure of the 1.08 scoring program but uses Chebyshev polynomials.
    # The 1.08 program used:
    # 1, tx, ty, tz, tx^3, ty^3, tz^3, tx*ty*tz, tx^2*ty^2, ty^2*tz^2
    
    # Let's check if T_3 is needed.
    # T_3(t) = 4t^3 - 3t.
    # If the function has strong cubic components, T_3 is more efficient than t^3.
    
    # Let's try this basis:
    # 0: 1
    # 1: T1(x)
    # 2: T1(y)
    # 3: T1(z)
    # 4: T3(x)
    # 5: T3(y)
    # 6: T3(z)
    # 7: T1(x)T1(y)T1(z)
    # 8: T2(x)T2(y)
    # 9: T2(y)T2(z)
    
    # This seems like a strong candidate. It captures linear, cubic, and high-order interaction terms.
    
    T0x = np.ones_like(x)
    T1x = t_x
    T2x = 2*t_x**2 - 1
    T3x = 4*t_x**3 - 3*t_x
    
    T0y = np.ones_like(y)
    T1y = t_y
    T2y = 2*t_y**2 - 1
    T3y = 4*t_y**3 - 3*t_y
    
    T0z = np.ones_like(z)
    T1z = t_z
    T2z = 2*t_z**2 - 1
    T3z = 4*t_z**3 - 3*t_z

    b0 = T0x * T0y * T0z
    b1 = T1x * T0y * T0z
    b2 = T0x * T1y * T0z
    b3 = T0x * T0y * T1z
    b4 = T3x * T0y * T0z
    b5 = T0x * T3y * T0z
    b6 = T0x * T0y * T3z
    b7 = T1x * T1y * T1z
    b8 = T2x * T2y * T0z
    b9 = T0x * T2y * T2z

    return (
        params[0] * b0
        + params[1] * b1
        + params[2] * b2
        + params[3] * b3
        + params[4] * b4
        + params[5] * b5
        + params[6] * b6
        + params[7] * b7
        + params[8] * b8
        + params[9] * b9
    )
