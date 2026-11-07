"""Symbolic regression over 4 hidden equations using a portfolio of basis functions."""
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0


def fit(X_train, y_train):
    t_start = time.time()
    X = X_train
    y = y_train
    n, d = X.shape

    # Helper to check time
    def time_left():
        return TIME_LIMIT - SAFETY_MARGIN - (time.time() - t_start)

    # Generate candidate feature matrices
    candidate_features = []
    candidate_names = []

    # 1. Polynomial features (degree 1-6)
    for deg in range(1, 7):
        if d == 1:
            x = X[:, 0]
            cols = [np.ones(n)]
            for k in range(1, deg + 1):
                cols.append(x ** k)
            feat = np.column_stack(cols)
            candidate_features.append(feat)
            candidate_names.append(f"poly_deg{deg}")
        else:
            # For d>1, generate polynomial features
            cols = [np.ones(n)]
            for k in range(1, deg + 1):
                for i in range(d):
                    for j in range(i, d):
                        if k == 1:
                            cols.append(X[:, i] * (X[:, j] if i != j else 1.0))
                        else:
                            # Simple polynomial terms
                            cols.append((X[:, i] ** k) if i == j else (X[:, i] ** (k // 2) * X[:, j] ** (k - k // 2)))
            feat = np.column_stack(cols)
            candidate_features.append(feat)
            candidate_names.append(f"poly_deg{deg}_multi")

    # 2. Trigonometric features
    if d == 1:
        x = X[:, 0]
        trig_cols = [np.ones(n)]
        for k in range(1, 5):
            trig_cols.append(np.sin(k * x))
            trig_cols.append(np.cos(k * x))
        candidate_features.append(np.column_stack(trig_cols))
        candidate_names.append("trig")

    # 3. Exponential / Log features (with safe handling)
    if d == 1:
        x = X[:, 0]
        exp_cols = [np.ones(n)]
        exp_cols.append(np.exp(x))
        exp_cols.append(np.exp(-x))
        # Safe log: only if x > 0
        pos_mask = x > 0
        if np.any(pos_mask):
            safe_x = np.where(pos_mask, x, 1.0)
            exp_cols.append(np.log(safe_x))
            exp_cols.append(np.log1p(np.abs(x)))
        else:
            exp_cols.append(np.ones(n))
            exp_cols.append(np.ones(n))
        candidate_features.append(np.column_stack(exp_cols))
        candidate_names.append("exp_log")

    # 4. Rational-like features (x, 1/x, x^2, etc.)
    if d == 1:
        x = X[:, 0]
        rat_cols = [np.ones(n)]
        rat_cols.append(x)
        safe_x = np.where(np.abs(x) > 1e-6, x, 1e-6)
        rat_cols.append(1.0 / safe_x)
        rat_cols.append(x ** 2)
        candidate_features.append(np.column_stack(rat_cols))
        candidate_names.append("rational")

    # 5. Sigmoid / Tanh features
    if d == 1:
        x = X[:, 0]
        sig_cols = [np.ones(n)]
        sig_cols.append(np.tanh(x))
        sig_cols.append(1.0 / (1.0 + np.exp(-x)))
        sig_cols.append(np.tanh(2 * x))
        candidate_features.append(np.column_stack(sig_cols))
        candidate_names.append("tanh")

    # 6. Piecewise linear / hinge features
    if d == 1:
        x = X[:, 0]
        hinge_cols = [np.ones(n)]
        for threshold in [-2, -1, 0, 1, 2]:
            hinge_cols.append(np.maximum(0, x - threshold))
            hinge_cols.append(np.maximum(0, threshold - x))
        candidate_features.append(np.column_stack(hinge_cols))
        candidate_names.append("hinge")

    # 7. Combined polynomial + trig
    if d == 1:
        x = X[:, 0]
        comb_cols = [np.ones(n), x, x**2, np.sin(x), np.cos(x), np.sin(2*x), np.cos(2*x)]
        candidate_features.append(np.column_stack(comb_cols))
        candidate_names.append("poly_trig_comb")

    # 8. Pure linear
    if d == 1:
        x = X[:, 0]
        candidate_features.append(np.column_stack([np.ones(n), x]))
        candidate_names.append("linear")

    # Now fit each candidate and evaluate on training data
    best_mse = np.inf
    best_predictor = None

    for i, feat in enumerate(candidate_features):
        if time_left() < 5.0:
            break

        try:
            # Use ridge regression for stability
            # Solve (X^T X + lambda I) w = X^T y
            Xf = feat
            XtX = Xf.T @ Xf
            Xty = Xf.T @ y
            n_features = Xf.shape[1]

            # Try multiple regularization strengths
            for lam in [0.0, 1e-6, 1e-4, 1e-2, 1e0, 1e2]:
                if time_left() < 2.0:
                    break
                A = XtX + lam * np.eye(n_features)
                try:
                    w = np.linalg.solve(A, Xty)
                except np.linalg.LinAlgError:
                    w = np.linalg.lstsq(A, Xty, rcond=None)[0]

                y_pred_train = Xf @ w
                mse_train = np.mean((y_pred_train - y) ** 2)

                if mse_train < best_mse:
                    best_mse = mse_train
                    best_w = w.copy()
                    best_feat = feat
                    best_feat_name = candidate_names[i]
                    best_lam = lam
        except Exception:
            continue

    # If no candidate worked, fall back to simple linear
    if best_predictor is None:
        if d == 1:
            x = X[:, 0]
            A = np.column_stack([np.ones(n), x])
            w = np.linalg.lstsq(A, y, rcond=None)[0]
            return lambda X_test: np.column_stack([np.ones(len(X_test)), X_test[:, 0]]) @ w
        else:
            A = np.column_stack([np.ones(n)] + [X[:, j] for j in range(d)])
            w = np.linalg.lstsq(A, y, rcond=None)[0]
            return lambda X_test: np.column_stack([np.ones(len(X_test))] + [X_test[:, j] for j in range(d)]) @ w

    # Define the predictor using the best feature set
    def predict(X_test):
        n_test = X_test.shape[0]
        # Reconstruct features for test data
        if best_feat_name.startswith("poly_deg") and d == 1:
            deg = int(best_feat_name.split("deg")[1])
            x = X_test[:, 0]
            cols = [np.ones(n_test)]
            for k in range(1, deg + 1):
                cols.append(x ** k)
            feat_test = np.column_stack(cols)
        elif best_feat_name == "trig":
            x = X_test[:, 0]
            cols = [np.ones(n_test)]
            for k in range(1, 5):
                cols.append(np.sin(k * x))
                cols.append(np.cos(k * x))
            feat_test = np.column_stack(cols)
        elif best_feat_name == "exp_log":
            x = X_test[:, 0]
            cols = [np.ones(n_test), np.exp(x), np.exp(-x)]
            pos_mask = x > 0
            if np.any(pos_mask):
                safe_x = np.where(pos_mask, x, 1.0)
                cols.append(np.log(safe_x))
                cols.append(np.log1p(np.abs(x)))
            else:
                cols.append(np.ones(n_test))
                cols.append(np.ones(n_test))
            feat_test = np.column_stack(cols)
        elif best_feat_name == "rational":
            x = X_test[:, 0]
            safe_x = np.where(np.abs(x) > 1e-6, x, 1e-6)
            feat_test = np.column_stack([np.ones(n_test), x, 1.0/safe_x, x**2])
        elif best_feat_name == "tanh":
            x = X_test[:, 0]
            feat_test = np.column_stack([np.ones(n_test), np.tanh(x), 1.0/(1.0+np.exp(-x)), np.tanh(2*x)])
        elif best_feat_name == "hinge":
            x = X_test[:, 0]
            cols = [np.ones(n_test)]
            for threshold in [-2, -1, 0, 1, 2]:
                cols.append(np.maximum(0, x - threshold))
                cols.append(np.maximum(0, threshold - x))
            feat_test = np.column_stack(cols)
        elif best_feat_name == "poly_trig_comb":
            x = X_test[:, 0]
            feat_test = np.column_stack([np.ones(n_test), x, x**2, np.sin(x), np.cos(x), np.sin(2*x), np.cos(2*x)])
        elif best_feat_name == "linear":
            feat_test = np.column_stack([np.ones(n_test), X_test[:, 0]])
        else:
            # Fallback: use the stored feature matrix structure (not ideal but safe)
            # For multi-dim, we need to rebuild
            if d > 1:
                # Rebuild polynomial features for multi-dim
                deg = 2  # default
                cols = [np.ones(n_test)]
                for k in range(1, deg + 1):
                    for i in range(d):
                        for j in range(i, d):
                            if k == 1:
                                cols.append(X_test[:, i] * (X_test[:, j] if i != j else 1.0))
                            else:
                                cols.append((X_test[:, i] ** k) if i == j else (X_test[:, i] ** (k // 2) * X_test[:, j] ** (k - k // 2)))
                feat_test = np.column_stack(cols)
            else:
                feat_test = np.column_stack([np.ones(n_test), X_test[:, 0]])

        return feat_test @ best_w

    return predict
