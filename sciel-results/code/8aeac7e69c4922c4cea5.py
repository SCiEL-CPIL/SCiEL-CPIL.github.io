import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
        # Ensure x, y are 1D arrays
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        # Use a 3x3 grid of RBF centers for 9 parameters + 1 bias = 10 total
        # Centers placed at -0.7, 0, 0.7 to better cover [-1, 1]
        cx = np.array([-0.7, 0.0, 0.7])
        cy = np.array([-0.7, 0.0, 0.7])
        
        # Create meshgrid of centers
        Cx, Cy = np.meshgrid(cx, cy, indexing='ij')
        Cx = Cx.flatten()
        Cy = Cy.flatten()
        
        # params[0:9] are weights for the 9 RBFs
        # params[9] is the bias
        
        # Compute distances
        # x, y shape: (N,)
        # Cx, Cy shape: (9,)
        # We want (N, 9) distances
        dx = x[:, np.newaxis] - Cx[np.newaxis, :]
        dy = y[:, np.newaxis] - Cy[np.newaxis, :]
        
        dist_sq = dx**2 + dy**2
        
        # Use gamma = 0.8 for slightly wider basis functions
        # This allows each RBF to influence a larger region, improving generalization
        gamma = 0.8
        basis = np.exp(-gamma * dist_sq)
        
        # Prediction
        pred = basis @ params[:9] + params[9]
        
        return pred
