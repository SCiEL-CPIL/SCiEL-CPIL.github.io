"""Rewritten program: uses symbolic regression with a rich basis set and model selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 285  # Safety margin for 330s limit
    
    X = np.asarray(X_train, dtype=float).reshape(-1, 1)
    y = np.asarray(y_train, dtype=float).flatten()
    n = len(y)
    
    if n < 2:
        mean_y = np.mean(y) if n > 0 else 0.0
        return lambda X: np.full(len(np.asarray(X).reshape(-1)), mean_y)
    
    x_vals = X[:, 0]
    y_vals = y
    
    # Check if y is constant
    if np.std(y_vals) < 1e-12:
        mean_y = np.mean(y_vals)
        return lambda X: np.full(len(np.asarray(X).reshape(-1)), mean_y)
    
    y_var = np.var(y_vals)
    if y_var < 1e-15:
        mean_y = np.mean(y_vals)
        return lambda X: np.full(len(np.asarray(X).reshape(-1)), mean_y)
    
    # Generate a rich set of basis functions
    basis_names = []
    basis_funcs = []
    
    # Add constant
    basis_names.append('const')
    basis_funcs.append(lambda x: np.ones_like(x))
    
    # Polynomials up to degree 8
    for d in range(1, 9):
        basis_names.append(f'x^{d}')
        basis_funcs.append(lambda x, d=d: x**d)
    
    # Trigonometric with more frequencies
    for f in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        basis_names.append(f'cos({f}*x)')
        basis_funcs.append(lambda x, f=f: np.cos(f * x))
        basis_names.append(f'sin({f}*x)')
        basis_funcs.append(lambda x, f=f: np.sin(f * x))
    
    # Exponential with more rates
    for b in [-3.0, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        basis_names.append(f'exp({b}*x)')
        basis_funcs.append(lambda x, b=b: np.exp(b * x))
    
    # Logarithmic (only if x is mostly positive)
    if np.min(x_vals) > -0.5:
        basis_names.append('log(|x|+1)')
        basis_funcs.append(lambda x: np.log(np.abs(x) + 1.0))
        basis_names.append('log(x^2+1)')
        basis_funcs.append(lambda x: np.log(x**2 + 1.0))
        basis_names.append('log(x+1)')
        basis_funcs.append(lambda x: np.log(x + 1.0))
    
    # Rational
    basis_names.append('1/(x+1)')
    basis_funcs.append(lambda x: 1.0 / (x + 1.0))
    basis_names.append('1/(x-1)')
    basis_funcs.append(lambda x: 1.0 / (x - 1.0))
    basis_names.append('1/(x^2+1)')
    basis_funcs.append(lambda x: 1.0 / (x**2 + 1.0))
    basis_names.append('1/(x+2)')
    basis_funcs.append(lambda x: 1.0 / (x + 2.0))
    basis_names.append('1/(x-2)')
    basis_funcs.append(lambda x: 1.0 / (x - 2.0))
    basis_names.append('1/(x^2+4)')
    basis_funcs.append(lambda x: 1.0 / (x**2 + 4.0))
    
    # Sqrt
    if np.min(x_vals) > -0.5:
        basis_names.append('sqrt(|x|+1)')
        basis_funcs.append(lambda x: np.sqrt(np.abs(x) + 1.0))
    
    # Tanh
    for f in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
        basis_names.append(f'tanh({f}*x)')
        basis_funcs.append(lambda x, f=f: np.tanh(f * x))
    
    # Sigmoid
    for f in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
        basis_names.append(f'sigmoid({f}*x)')
        basis_funcs.append(lambda x, f=f: 1.0 / (1.0 + np.exp(-f * x)))
    
    # Absolute value
    basis_names.append('|x|')
    basis_funcs.append(lambda x: np.abs(x))
    
    # x*|x|
    basis_names.append('x*|x|')
    basis_funcs.append(lambda x: x * np.abs(x))
    
    # Gaussian-like bumps centered at various points
    for center in [-2, -1, 0, 1, 2]:
        for width in [0.5, 1.0, 2.0]:
            basis_names.append(f'exp(-((x-{center})^2)/(2*{width}^2))')
            basis_funcs.append(lambda x, c=center, w=width: np.exp(-((x - c)**2) / (2 * w**2)))
    
    # Evaluate all basis functions on training data
    n_basis = len(basis_funcs)
    B = np.zeros((n, n_basis))
    for j in range(n_basis):
        try:
            B[:, j] = basis_funcs[j](x_vals)
            # Check for finite values
            if not np.all(np.isfinite(B[:, j])):
                B[:, j] = 0.0
        except Exception:
            B[:, j] = 0.0
    
    # Remove columns that are all zero or have very small variance
    col_var = np.var(B, axis=0)
    valid_cols = col_var > 1e-10
    B_valid = B[:, valid_cols]
    basis_names_valid = [basis_names[i] for i in range(n_basis) if valid_cols[i]]
    basis_funcs_valid = [basis_funcs[i] for i in range(n_basis) if valid_cols[i]]
    n_valid = len(basis_names_valid)
    
    if n_valid == 0:
        mean_y = np.mean(y_vals)
        return lambda X: np.full(len(np.asarray(X).reshape(-1)), mean_y)
    
    # Try different subset sizes using forward selection or random subsets
    # For efficiency, try:
    # 1. Single basis functions
    # 2. Pairs of basis functions
    # 3. Small random subsets of size 3-6
    
    candidates = []
    
    # Evaluate single basis functions
    for j in range(n_valid):
        try:
            # Fit a0 + a1 * basis_j
            X_aug = np.column_stack([np.ones(n), B_valid[:, j]])
            coeffs, _, _, _ = np.linalg.lstsq(X_aug, y_vals, rcond=None)
            y_pred = X_aug @ coeffs
            mse = np.mean((y_pred - y_vals) ** 2)
            norm_mse = mse / y_var
            if np.isfinite(norm_mse):
                candidates.append((norm_mse, [j], coeffs))
        except Exception:
            continue
    
    # Evaluate pairs of basis functions
    if n_valid >= 2:
        # Try all pairs if n_valid is small, otherwise random sample
        if n_valid <= 30:
            pairs = []
            for i in range(n_valid):
                for j in range(i + 1, n_valid):
                    pairs.append((i, j))
        else:
            np.random.seed(42)
            pairs = []
            for _ in range(500):
                i, j = np.random.choice(n_valid, size=2, replace=False)
                if i > j:
                    i, j = j, i
                pairs.append((i, j))
        
        for (i, j) in pairs:
            if time.time() > deadline - 5:
                break
            try:
                X_aug = np.column_stack([np.ones(n), B_valid[:, i], B_valid[:, j]])
                coeffs, _, _, _ = np.linalg.lstsq(X_aug, y_vals, rcond=None)
                y_pred = X_aug @ coeffs
                mse = np.mean((y_pred - y_vals) ** 2)
                norm_mse = mse / y_var
                if np.isfinite(norm_mse):
                    candidates.append((norm_mse, [i, j], coeffs))
            except Exception:
                continue
    
    # Try random subsets of size 3-6
    np.random.seed(42)
    subset_sizes = [3, 4, 5, 6]
    for size in subset_sizes:
        if n_valid < size:
            continue
        n_subsets = min(200, max(20, n_valid // 2))
        for _ in range(n_subsets):
            if time.time() > deadline - 5:
                break
            idx = np.random.choice(n_valid, size=size, replace=False)
            idx = sorted(idx)
            try:
                X_aug = np.column_stack([np.ones(n)] + [B_valid[:, i] for i in idx])
                coeffs, _, _, _ = np.linalg.lstsq(X_aug, y_vals, rcond=None)
                y_pred = X_aug @ coeffs
                mse = np.mean((y_pred - y_vals) ** 2)
                norm_mse = mse / y_var
                if np.isfinite(norm_mse):
                    candidates.append((norm_mse, idx, coeffs))
            except Exception:
                continue
    
    # Also try polynomial fits of various degrees as a fallback
    for deg in range(0, 9):
        if deg > n - 1:
            break
        try:
            coeffs = np.polyfit(x_vals, y_vals, deg)
            y_pred = np.polyval(coeffs, x_vals)
            mse = np.mean((y_pred - y_vals) ** 2)
            norm_mse = mse / y_var
            if np.isfinite(norm_mse):
                candidates.append((norm_mse, 'poly', coeffs))
        except Exception:
            continue
    
    # If no candidates, use constant
    if not candidates:
        mean_y = np.mean(y_vals)
        return lambda X: np.full(len(np.asarray(X).reshape(-1)), mean_y)
    
    # Sort by normalized MSE
    candidates.sort(key=lambda x: x[0])
    
    best_norm_mse, best_indices, best_coeffs = candidates[0]
    
    # Create prediction function
    def predict(X_pred):
        X_pred = np.asarray(X_pred, dtype=float).reshape(-1, 1)[:, 0]
        
        if isinstance(best_indices, str) and best_indices == 'poly':
            return np.polyval(best_coeffs, X_pred)
        else:
            # Reconstruct basis values for prediction
            n_pred = len(X_pred)
            X_aug = np.ones((n_pred, 1 + len(best_indices)))
            for k, idx in enumerate(best_indices):
                try:
                    X_aug[:, k + 1] = basis_funcs_valid[idx](X_pred)
                    if not np.all(np.isfinite(X_aug[:, k + 1])):
                        X_aug[:, k + 1] = 0.0
                except Exception:
                    X_aug[:, k + 1] = 0.0
            return X_aug @ best_coeffs
    
    return predict
# EVOLVE-BLOCK-END
