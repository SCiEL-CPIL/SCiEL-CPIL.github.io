import numpy as np

def construct(training_x, training_subgrid_tendency):
       x = np.asarray(training_x, dtype=np.float64)
       y = np.asarray(training_subgrid_tendency, dtype=np.float64)
       
       # Handle empty or trivial cases
       if len(x) < 6:
           return np.zeros(6)
       
       # Create Vandermonde matrix
       # V[:, j] = x ** j
       V = np.vander(x, N=6, increasing=True)
       
       # Regularization
       # Compute Gram matrix G = V.T @ V
       # Solve (G + lambda I) c = V.T @ y
       
       # Estimate lambda based on scale of G
       # A common heuristic is lambda = alpha * mean(diag(G))
       # Let's try a small lambda first, e.g., 1e-4 * mean(diag(G))
       
       G = V.T @ V
       b = V.T @ y
       
       # Add regularization
       # If the problem is ill-conditioned, this helps.
       # For degree 5, it might be okay without, but regularization prevents overfitting.
       # Let's use a small value.
       lambda_reg = 1e-6 * np.mean(np.diag(G))
       G_reg = G + lambda_reg * np.eye(6)
       
       try:
           c = np.linalg.solve(G_reg, b)
       except np.linalg.LinAlgError:
           c = np.linalg.lstsq(V, y, rcond=None)[0]
           
       # Check for finiteness
       if not np.all(np.isfinite(c)):
           # Fallback to linear fit
           c = np.zeros(6)
           c[:2] = np.polyfit(x, y, 1)
           c = c[::-1] # polyfit returns descending, we need ascending
           # Wait, polyfit returns [c_n ... c_0]. We need [c_0 ... c_n].
           # So c[:2] from polyfit is [c_1, c_0].
           # Let's just use np.polynomial.polynomial.polyfit for fallback
           c = np.polynomial.polynomial.polyfit(x, y, 1)
           
       return c
