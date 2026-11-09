import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64)
            p = np.asarray(params, dtype=np.float64)
            
            # Fixed centers in [0.1, 6]
            centers = np.linspace(0.1, 6.0, 10)
            # Fixed width
            sigma = 0.6
            
            # Compute squared distances
            # x shape: (N,), centers shape: (10,)
            # dist2 shape: (N, 10)
            dist2 = (x[:, np.newaxis] - centers[np.newaxis, :]) ** 2
            
            # Compute Gaussian basis
            # basis shape: (N, 10)
            basis = np.exp(-dist2 / (2.0 * sigma * sigma))
            
            # Linear combination
            return basis @ p
