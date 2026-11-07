import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
        # Normalize inputs to O(1) scale for better conditioning
        # Based on observed ranges: x in [-0.57, 0.53], v in [-0.18, 0.28]
        # Using slightly wider bounds to be safe
        x_scale = 0.6
        v_scale = 0.3
        
        X = x / x_scale
        V = v / v_scale
        
        # Build basis terms (Complete cubic polynomial)
        # 1: 1
        # 2: X
        # 3: V
        # 4: X^2
        # 5: X*V
        # 6: V^2
        # 7: X^3
        # 8: X^2*V
        # 9: X*V^2
        # 10: V^3
        
        basis = np.column_stack((
            np.ones_like(X),
            X,
            V,
            X * X,
            X * V,
            V * V,
            X * X * X,
            X * X * V,
            X * V * V,
            V * V * V
        ))
        
        return basis @ params
