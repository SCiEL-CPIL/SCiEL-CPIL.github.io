import numpy as np
import time
from scipy.interpolate import CubicSpline

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 285  # Safety margin before 330s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # ---- Fallback for d > 1 (unlikely per contract) ----
    if d != 1:
        try:
            X_aug = np.column_stack([np.ones(n), X])
            w, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
            def predict(X_new):
                X_new = np.asarray(X_new, dtype=np.float64)
                X_aug = np.column_stack([np.ones(X_new.shape[0]), X_new])
                return X_aug @ w
            return predict
        except Exception:
            y_mean = float(np.mean(y))
            def predict(X_new):
                return np.full(len(np.asarray(X_new).reshape(-1, 1)), y_mean)
            return predict

    x = X.flatten()
    y = y.flatten()

    # =========================================================
    # Model A: RBF Interpolation (Primary)
    # Exact interpolation for noiseless data, robust to non-smoothness
    # =========================================================
    def fit_rbf_interpolation(x, y):
        try:
            # Estimate length scale
            x_sorted = np.sort(x)
            gaps = np.diff(x_sorted)
            gaps = gaps[gaps > 1e-12]
            if len(gaps) > 0:
                med_gap = float(np.median(gaps))
            else:
                med_gap = 1.0
            if not np.isfinite(med_gap) or med_gap < 1e-8:
                med_gap = 1.0
            
            # Gamma = 1 / (2 * sigma^2). 
            # If sigma = med_gap, gamma = 1 / (2 * med_gap^2)
            sigma = med_gap
            gamma = 1.0 / (2.0 * sigma * sigma)
            
            # Build Kernel Matrix
            diff = x[:, None] - x[None, :]
            K = np.exp(-gamma * diff * diff)
            
            # Add jitter for numerical stability
            K += 1e-10 * np.eye(n)
            
            try:
                # Solve K @ w = y
                w = np.linalg.solve(K, y)
            except np.linalg.LinAlgError:
                # Fallback to lstsq if singular
                w, _, _, _ = np.linalg.lstsq(K, y, rcond=None)
            
            def predict(X_new):
                xn = np.asarray(X_new, dtype=np.float64).reshape(-1)
                diff = xn[:, None] - x[None, :]
                Ks = np.exp(-gamma * diff * diff)
                return Ks @ w
            return predict
        except Exception:
            return None

    # =========================================================
    # Model B: Cubic Spline (Secondary)
    # =========================================================
    def create_spline_model(x, y):
        try:
            idx = np.argsort(x)
            xs = x[idx]
            ys = y[idx]
            # Handle duplicates
            ux, inv = np.unique(xs, return_inverse=True)
            if len(ux) < len(xs):
                agg = np.zeros(len(ux))
                cnt = np.zeros(len(ux))
                np.add.at(agg, inv, ys)
                np.add.at(cnt, inv, 1)
                ys = agg / cnt
                xs = ux
            cs = CubicSpline(xs, ys)
            def predict(X_new):
                xn = np.asarray(X_new, dtype=np.float64).reshape(-1)
                return cs(xn)
            return predict
        except Exception:
            return None

    # =========================================================
    # Model C: Polynomial Regression (Tertiary)
    # =========================================================
    def fit_poly(x, y, deg=5):
        try:
            # Use scaled x for better conditioning
            x_min = np.min(x)
            x_max = np.max(x)
            if x_max == x_min:
                x_scaled = np.zeros_like(x)
            else:
                x_scaled = (x - x_min) / (x_max - x_min) * 2 - 1 # [-1, 1]
            
            coeffs = np.polyfit(x_scaled, y, deg)
            
            def predict(X_new):
                xn = np.asarray(X_new, dtype=np.float64).reshape(-1)
                if x_max == x_min:
                    xn_scaled = np.zeros_like(xn)
                else:
                    xn_scaled = (xn - x_min) / (x_max - x_min) * 2 - 1
                return np.polyval(coeffs, xn_scaled)
            return predict
        except Exception:
            return None

    # =========================================================
    # Selection
    # =========================================================
    candidates = []
    
    rbf_model = fit_rbf_interpolation(x, y)
    if rbf_model is not None:
        candidates.append(('rbf', rbf_model))
        
    sp_model = create_spline_model(x, y)
    if sp_model is not None:
        candidates.append(('spline', sp_model))
        
    # Try a few polynomial degrees
    for deg in [2, 4, 6]:
        p_model = fit_poly(x, y, deg)
        if p_model is not None:
            candidates.append((f'poly{deg}', p_model))

    if not candidates:
        y_mean = float(np.mean(y))
        def predict(X_new):
            return np.full(len(np.asarray(X_new).reshape(-1, 1)), y_mean)
        return predict

    # Evaluate on a small validation set (or just training error since noiseless)
    # Since data is noiseless, training error is 0 for interpolants.
    # We prefer RBF or Spline for smoothness.
    # Let's just pick RBF as it is globally smooth and interpolating.
    # If RBF fails, Spline is a good fallback.
    
    final_model = None
    for name, model in candidates:
        if name == 'rbf':
            final_model = model
            break
    if final_model is None:
        final_model = candidates[0][1]

    def predict(X_new):
        X_new = np.asarray(X_new, dtype=np.float64)
        try:
            out = final_model(X_new)
            out = np.asarray(out, dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(out)):
                out = np.where(np.isfinite(out), out, float(np.mean(y)))
            return out
        except Exception:
            return np.full(X_new.shape[0], float(np.mean(y)))

    return predict
