import numpy as np
import time
import scipy.interpolate as interp

def construct(training_x, training_subgrid_tendency):
    """
    Fits a shared degree-5 polynomial to the unresolved subgrid tendency 
    using Chebyshev-basis Ridge regression with cross-validated regularization.
    Includes a spline-based denoising step for the target variable.
    """
    start_time = time.time()
    deadline = 140.0
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    if len(x) == 0:
        return np.zeros(6)
    
    if len(x) < 6:
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 1)
            return np.pad(coeff, (0, 6 - len(coeff)))
        except Exception:
            return np.zeros(6)

    # --- Step 1: Spline Denoising ---
    # Sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Create quantile bins
    n_bins = 30
    # Ensure we have enough points for bins
    if len(x) > n_bins * 2:
        # Define bin edges using quantiles
        edges = np.quantile(x_sorted, np.linspace(0, 1, n_bins + 1))
        # Avoid duplicate edges if data is sparse
        # If duplicates, merge bins or reduce n_bins
        unique_edges = np.unique(edges)
        if len(unique_edges) < n_bins + 1:
             n_bins = len(unique_edges) - 1
             if n_bins < 2:
                 y_smoothed = y
             else:
                 edges = unique_edges
        else:
            edges = unique_edges
            
        # Assign points to bins
        # np.digitize returns indices 1..n_bins, but we want 0..n_bins-1
        # We want mean y for each bin
        bin_indices = np.digitize(x_sorted, edges[1:-1]) # 0 to n_bins-1
        
        y_smoothed = np.zeros_like(y)
        
        valid_bins = []
        bin_means = []
        bin_centers = []
        
        for b in range(n_bins):
            mask = (bin_indices == b)
            if np.sum(mask) > 0:
                mean_y = np.mean(y_sorted[mask])
                # Center of bin (mean of x in bin is better)
                center_x = np.mean(x_sorted[mask])
                bin_means.append(mean_y)
                bin_centers.append(center_x)
                valid_bins.append(b)
                
        if len(bin_centers) > 2:
            # Fit spline to (bin_centers, bin_means)
            # Use CubicSpline with natural boundary conditions
            try:
                cs = interp.CubicSpline(np.array(bin_centers), np.array(bin_means), bc_type='natural')
                y_smoothed = cs(x)
            except Exception:
                # Fallback to linear interpolation if spline fails
                try:
                    fi = interp.interp1d(np.array(bin_centers), np.array(bin_means), kind='linear', fill_value="extrapolate")
                    y_smoothed = fi(x)
                except Exception:
                    y_smoothed = y
        else:
            y_smoothed = y
    else:
        y_smoothed = y

    # --- Step 2: Chebyshev Regression on Smoothed Data ---
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    
    if x_range < 1e-12:
        mean_y = np.mean(y_smoothed)
        coeff = np.zeros(6)
        coeff[0] = mean_y
        return coeff
    
    x_norm = 2.0 * (x - x_min) / x_range - 1.0

    def chebyshev_basis(x, degree):
        n = len(x)
        basis = np.zeros((n, degree + 1))
        if degree >= 0:
            basis[:, 0] = 1.0
        if degree >= 1:
            basis[:, 1] = x
        for k in range(2, degree + 1):
            basis[:, k] = 2.0 * x * basis[:, k - 1] - basis[:, k - 2]
        return basis

    degree = 5
    T = chebyshev_basis(x_norm, degree)
    
    # Cross-Validation
    n_splits = 5
    indices = np.arange(len(x))
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = [indices[i::n_splits] for i in range(n_splits)]
    
    lambdas = np.logspace(-6, 6, 50)
    
    best_mse = np.inf
    best_lambda = 0.0
    
    train_folds = []
    val_folds = []
    for i in range(n_splits):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        train_folds.append(train_idx)
        val_folds.append(val_idx)
        
    for lam in lambdas:
        if time.time() > deadline - 20.0:
            break
            
        fold_mses = []
        for i in range(n_splits):
            train_idx = train_folds[i]
            val_idx = val_folds[i]
            
            T_train = T[train_idx, :]
            y_train = y_smoothed[train_idx] # Use smoothed y
            T_val = T[val_idx, :]
            y_val = y_smoothed[val_idx]     # Use smoothed y
            
            try:
                A = T_train.T @ T_train
                b = T_train.T @ y_train
                reg_matrix = A + lam * np.eye(degree + 1)
                theta = np.linalg.solve(reg_matrix, b)
                
                y_pred = T_val @ theta
                mse = np.mean((y_val - y_pred) ** 2)
                fold_mses.append(mse)
            except Exception:
                fold_mses.append(np.inf)
        
        mean_mse = np.mean(fold_mses)
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_lambda = lam

    # Final Fit
    if time.time() > deadline - 5.0:
        try:
            theta_cheb = np.linalg.lstsq(T, y_smoothed, rcond=None)[0]
        except Exception:
            theta_cheb = np.zeros(degree + 1)
    else:
        try:
            A = T.T @ T
            b = T.T @ y_smoothed
            reg_matrix = A + best_lambda * np.eye(degree + 1)
            theta_cheb = np.linalg.solve(reg_matrix, b)
        except Exception:
            try:
                theta_cheb = np.linalg.lstsq(T, y_smoothed, rcond=None)[0]
            except Exception:
                theta_cheb = np.zeros(degree + 1)

    # Convert to Power Basis
    try:
        power_u = np.polynomial.chebyshev.cheb2poly(theta_cheb)
        poly_u = np.polynomial.Polynomial(power_u)
        
        A_coeff = 2.0 / x_range
        B_coeff = -1.0 - 2.0 * x_min / x_range
        
        coeff_x = np.zeros(6)
        for k in range(6):
            pk = power_u[k]
            if pk == 0:
                continue
            for j in range(k + 1):
                binom_k_j = 1
                if j > 0:
                    binom_k_j = int(np.prod([k - i + 1 for i in range(1, j+1)])) / int(np.prod([i for i in range(1, j+1)]))
                term_coeff = pk * binom_k_j * (A_coeff ** j) * (B_coeff ** (k - j))
                coeff_x[j] += term_coeff
    except Exception:
        try:
            V = np.vander(x, N=6, increasing=True)
            coeff_x = np.linalg.lstsq(V, y_smoothed, rcond=None)[0]
        except Exception:
            coeff_x = np.zeros(6)

    if not np.all(np.isfinite(coeff_x)):
        coeff_x = np.where(np.isfinite(coeff_x), coeff_x, 0.0)

    return coeff_x
