import numpy as np

def equation(x, y, params):
    # Use Chebyshev polynomial basis up to total degree 4.
    # T0 = 1
    # T1 = t
    # T2 = 2t^2 - 1
    # T3 = 4t^3 - 3t
    # T4 = 8t^4 - 8t^2 + 1
    
    # Chebyshev polynomials for x
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2 * x**2 - 1
    T3_x = 4 * x**3 - 3 * x
    T4_x = 8 * x**4 - 8 * x**2 + 1
    
    # Chebyshev polynomials for y
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2 * y**2 - 1
    T3_y = 4 * y**3 - 3 * y
    T4_y = 8 * y**4 - 8 * y**2 + 1
    
    # Basis functions: 10 selected tensor products
    # We need exactly 10 basis functions to match the 10 parameters.
    # Let's select a diverse set up to degree 4.
    
    # Degree 0:
    b0 = T0_x * T0_y  # 1
    
    # Degree 1:
    b1 = T1_x * T0_y  # x
    b2 = T0_x * T1_y  # y
    
    # Degree 2:
    b3 = T2_x * T0_y  # T2(x)
    b4 = T1_x * T1_y  # x*y
    b5 = T0_x * T2_y  # T2(y)
    
    # Degree 3:
    b6 = T3_x * T0_y  # T3(x)
    b7 = T2_x * T1_y  # T2(x)*y
    b8 = T1_x * T2_y  # x*T2(y)
    b9 = T0_x * T3_y  # T3(y)
    
    # Wait, that's already 10. If I want to include degree 4, I have to drop some.
    # The current score is 7.02. Dropping a lower degree term might hurt if the function has those components.
    # However, the directive says "KEEP this design and investigate improvements within its family".
    # Increasing the degree is a natural improvement.
    # Let's try to keep the most important low-order terms and add higher order ones.
    # Maybe replace the least "isotropic" higher order terms?
    # Or just shift the degree up.
    
    # Let's try a different selection of 10 terms that goes up to degree 4.
    # 1: 1
    # 2: x
    # 3: y
    # 4: T2(x)
    # 5: x*y
    # 6: T2(y)
    # 7: T3(x)
    # 8: T3(y)
    # 9: T4(x)
    # 10: T4(y)
    # This drops T2(x)*y and x*T2(y) (degree 3 mixed) to add T4(x) and T4(y) (degree 4 pure).
    # This might be better if the function has strong pure-x or pure-y non-linearities.
    
    b0 = T0_x * T0_y
    b1 = T1_x * T0_y
    b2 = T0_x * T1_y
    b3 = T2_x * T0_y
    b4 = T1_x * T1_y
    b5 = T0_x * T2_y
    b6 = T3_x * T0_y
    b7 = T0_x * T3_y
    b8 = T4_x * T0_y
    b9 = T0_x * T4_y
    
    return (params[0]*b0 + params[1]*b1 + params[2]*b2 + params[3]*b3 +
            params[4]*b4 + params[5]*b5 + params[6]*b6 + params[7]*b7 +
            params[8]*b8 + params[9]*b9)
