import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
        # x, y are 1D arrays of length N
        # params is 1D array of length 10
        
        # Compute Chebyshev polynomials T_0 to T_3
        # T_0(t) = 1
        # T_1(t) = t
        # T_2(t) = 2t^2 - 1
        # T_3(t) = 4t^3 - 3t
        
        T0_x = np.ones_like(x)
        T1_x = x
        T2_x = 2 * x**2 - 1
        T3_x = 4 * x**3 - 3 * x
        
        T0_y = np.ones_like(y)
        T1_y = y
        T2_y = 2 * y**2 - 1
        T3_y = 4 * y**3 - 3 * y
        
        # Construct basis columns
        # Order: (i,j) where i+j <= 3
        # 1: (0,0)
        # 2: (1,0)
        # 3: (0,1)
        # 4: (2,0)
        # 5: (1,1)
        # 6: (0,2)
        # 7: (3,0)
        # 8: (2,1)
        # 9: (1,2)
        # 10: (0,3)
        
        B = np.column_stack((
            T0_x * T0_y,
            T1_x * T0_y,
            T0_x * T1_y,
            T2_x * T0_y,
            T1_x * T1_y,
            T0_x * T2_y,
            T3_x * T0_y,
            T2_x * T1_y,
            T1_x * T2_y,
            T0_x * T3_y
        ))
        
        return B @ params
