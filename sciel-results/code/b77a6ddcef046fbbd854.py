"""Symbolic regression over 10 hidden equations.
Uses a small neural network trained with Adam, combined with a rich feature basis
and ensemble of models for robustness."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 310  # safety margin before 330s wall-clock kill


def _elapsed():
    return time.time() - _DEADLINE


def _build_rich_basis(X):
    """Build a rich feature basis for regression."""
    d = X.shape[1]
    n = X.shape[0]
    cols = [np.ones(n)]

    # Linear terms
    for j in range(d):
        cols.append(X[:, j])

    # Quadratic terms
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])

    # Cubic terms
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append(X[:, i] * X[:, j] * X[:, k])

    # Trigonometric terms
    for j in range(d):
        cols.append(np.sin(X[:, j]))
        cols.append(np.cos(X[:, j]))

    # Scaled exponential terms
    for j in range(d):
        x_scaled = X[:, j] / max(5.0, X[:, j].max() + 1e-8)
        cols.append(np.exp(x_scaled))
        cols.append(np.exp(-x_scaled))

    # Log terms
    for j in range(d):
        cols.append(np.log(X[:, j] + 1e-10))

    # Tanh terms
    for j in range(d):
        cols.append(np.tanh(X[:, j]))

    # Sqrt terms
    for j in range(d):
        cols.append(np.sqrt(np.maximum(X[:, j], 0)))

    # Reciprocal terms
    for j in range(d):
        cols.append(1.0 / (X[:, j] + 1e-8))

    # Interactions with trig
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * np.sin(X[:, j]))
            cols.append(X[:, i] * np.cos(X[:, j]))

    return np.column_stack(cols)


def _fit_ridge(F, y, alpha):
    """Fit ridge regression."""
    n_features = F.shape[1]
    A = F.T @ F + alpha * np.eye(n_features)
    b = F.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    pred = F @ w
    mse = np.mean((pred - y) ** 2)
    return w, mse


def _forward_selection(F, y, max_features=20, alpha=1e-4, time_budget=15):
    """Forward feature selection."""
    n_samples, n_features = F.shape
    selected = [0]  # start with intercept
    best_mse = np.mean(y ** 2)
    start_time = time.time()

    while len(selected) < max_features:
        if time.time() - start_time > time_budget:
            break

        best_idx = -1
        best_mse_iter = best_mse

        for idx in range(n_features):
            if idx in selected:
                continue
            trial_selected = selected + [idx]
            F_trial = F[:, trial_selected]
            w, mse = _fit_ridge(F_trial, y, alpha)
            if mse < best_mse_iter:
                best_mse_iter = mse
                best_idx = idx

        if best_idx == -1 or best_mse_iter >= best_mse * 0.999:
            break

        selected.append(best_idx)
        best_mse = best_mse_iter

    F_final = F[:, selected]
    w_final, mse_final = _fit_ridge(F_final, y, alpha)
    return selected, w_final, mse_final


def _train_neural_net(X, y, hidden_sizes=[32, 32], lr=0.01, epochs=500, time_budget=20):
    """Train a small neural network with Adam optimizer."""
    n_samples, n_features = X.shape
    rng = np.random.default_rng(42)

    # Initialize weights
    layers = [n_features] + list(hidden_sizes) + [1]
    weights = []
    biases = []
    for i in range(len(layers) - 1):
        fan_in = layers[i]
        fan_out = layers[i + 1]
        w = rng.normal(0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))
        b = np.zeros(fan_out)
        weights.append(w)
        biases.append(b)

    # Adam optimizer state
    m_w = [np.zeros_like(w) for w in weights]
    v_w = [np.zeros_like(w) for w in weights]
    m_b = [np.zeros_like(b) for b in biases]
    v_b = [np.zeros_like(b) for b in biases]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    start_time = time.time()

    # Normalize inputs
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    y_mean = y.mean()
    y_std = y.std() + 1e-8
    y_norm = (y - y_mean) / y_std

    for epoch in range(epochs):
        if time.time() - start_time > time_budget:
            break

        # Forward pass
        a = X_norm
        activations = [a]
        for i in range(len(weights)):
            z = a @ weights[i] + biases[i]
            if i < len(weights) - 1:
                a = np.tanh(z)
            else:
                a = z  # linear output
            activations.append(a)

        pred = a.flatten()
        loss = np.mean((pred - y_norm) ** 2)

        # Backward pass
        delta = (pred - y_norm) / n_samples

        for i in range(len(weights) - 1, -1, -1):
            if i < len(weights) - 1:
                delta = delta * (1 - activations[i + 1] ** 2)  # tanh derivative

            grad_w = activations[i].T @ delta
            grad_b = delta.sum(axis=0)

            # Adam update
            t += 1
            m_w[i] = beta1 * m_w[i] + (1 - beta1) * grad_w
            v_w[i] = beta2 * v_w[i] + (1 - beta2) * (grad_w ** 2)
            m_hat = m_w[i] / (1 - beta1 ** t)
            v_hat = v_w[i] / (1 - beta2 ** t)
            weights[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

            m_b[i] = beta1 * m_b[i] + (1 - beta1) * grad_b
            v_b[i] = beta2 * v_b[i] + (1 - beta2) * (grad_b ** 2)
            m_hat_b = m_b[i] / (1 - beta1 ** t)
            v_hat_b = v_b[i] / (1 - beta2 ** t)
            biases[i] -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

    def predict_nn(X_test):
        X_test_norm = (X_test - X_mean) / X_std
        a = X_test_norm
        for i in range(len(weights)):
            z = a @ weights[i] + biases[i]
            if i < len(weights) - 1:
                a = np.tanh(z)
            else:
                a = z
        return a.flatten() * y_std + y_mean

    # Compute training MSE
    pred_train = predict_nn(X)
    mse = np.mean((pred_train - y) ** 2)
    return predict_nn, mse


def fit(X_train, y_train):
    d = X_train.shape[1]

    # Build rich basis
    F_train = _build_rich_basis(X_train)

    models = []

    # Model 1: Forward selection with ridge
    try:
        if _elapsed() < 10:
            selected_idx, w_ridge, mse_fs = _forward_selection(
                F_train, y_train, max_features=20, alpha=1e-4, time_budget=15
            )

            def predict_fs(X):
                F = _build_rich_basis(X)
                F_sel = F[:, selected_idx]
                return F_sel @ w_ridge

            models.append((predict_fs, mse_fs))
    except Exception:
        pass

    # Model 2: Simple polynomial ridge
    try:
        if _elapsed() < 20:
            n_linear = 1 + d  # intercept + linear terms
            F_poly = F_train[:, :n_linear]
            w_poly, mse_poly = _fit_ridge(F_poly, y_train, 1e-6)

            def predict_poly(X):
                F = _build_rich_basis(X)
                F_sel = F[:, :n_linear]
                return F_sel @ w_poly

            models.append((predict_poly, mse_poly))
    except Exception:
        pass

    # Model 3: Neural network
    try:
        if _elapsed() < 25:
            predict_nn, mse_nn = _train_neural_net(
                X_train, y_train, hidden_sizes=[32, 32], lr=0.01, epochs=500, time_budget=20
            )
            models.append((predict_nn, mse_nn))
    except Exception:
        pass

    # Model 4: Deeper neural network
    try:
        if _elapsed() < 50:
            predict_nn2, mse_nn2 = _train_neural_net(
                X_train, y_train, hidden_sizes=[64, 32], lr=0.005, epochs=800, time_budget=30
            )
            models.append((predict_nn2, mse_nn2))
    except Exception:
        pass

    # Model 5: Ensemble of best models
    if len(models) >= 2:
        # Sort by MSE and take top models
        models_sorted = sorted(models, key=lambda x: x[1])
        top_models = [m[0] for m in models_sorted[:min(3, len(models_sorted))]]
        weights_ens = [1.0 / m[1] for m in models_sorted[:min(3, len(models_sorted))]]
        total_w = sum(weights_ens)
        weights_ens = [w / total_w for w in weights_ens]

        def predict_ensemble(X):
            preds = [model(X) for model in top_models]
            result = np.zeros(len(preds[0]))
            for pred, w in zip(preds, weights_ens):
                result += w * pred
            return result

        # Compute ensemble MSE
        pred_ens = predict_ensemble(X_train)
        mse_ens = np.mean((pred_ens - y_train) ** 2)
        models.append((predict_ensemble, mse_ens))

    # Choose best model
    if not models:
        # Fallback: mean predictor
        mean_y = np.mean(y_train)
        def predict_fallback(X):
            return np.full(X.shape[0], mean_y)
        return predict_fallback

    best_model = min(models, key=lambda x: x[1])
    return best_model[0]
# EVOLVE-BLOCK-END
