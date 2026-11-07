"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import comb
from itertools import product
import warnings
warnings.filterwarnings("ignore")


def fit(X_train, y_train):
    d = X_train.shape[1]
    n = len(X_train)
    
    # Generate candidate basis functions
    # We'll include:
    # 1. Constant
    # 2. Linear terms: x_i
    # 3. Pairwise products: x_i * x_j (i <= j)
    # 4. Cubic terms: x_i^3, x_i^2 * x_j, x_i * x_j * x_k (for small d)
    # 5. Sine/cosine for potential periodicity (only if d is small and range suggests)
    # 6. Exp/log for potential exponential/logarithmic behavior
    
    basis_funcs = []
    basis_names = []
    
    # Constant
    basis_funcs.append(lambda X: np.ones(len(X)))
    basis_names.append("const")
    
    # Linear
    for i in range(d):
        basis_funcs.append(lambda X, i=i: X[:, i])
        basis_names.append(f"x{i}")
    
    # Quadratic
    for i in range(d):
        for j in range(i, d):
            if i == j:
                basis_funcs.append(lambda X, i=i: X[:, i] ** 2)
                basis_names.append(f"x{i}^2")
            else:
                basis_funcs.append(lambda X, i=i, j=j: X[:, i] * X[:, j])
                basis_names.append(f"x{i}*x{j}")
    
    # Cubic (only if d <= 3 to keep size manageable)
    if d <= 3:
        for i in range(d):
            basis_funcs.append(lambda X, i=i: X[:, i] ** 3)
            basis_names.append(f"x{i}^3")
        for i in range(d):
            for j in range(i+1, d):
                basis_funcs.append(lambda X, i=i, j=j: X[:, i] ** 2 * X[:, j])
                basis_names.append(f"x{i}^2*x{j}")
                basis_funcs.append(lambda X, i=i, j=j: X[:, i] * X[:, j] ** 2)
                basis_names.append(f"x{i}*x{j}^2")
        if d == 3:
            basis_funcs.append(lambda X: X[:, 0] * X[:, 1] * X[:, 2])
            basis_names.append("x0*x1*x2")
    
    # Trigonometric (only if input range suggests periodicity, e.g., range <= 2*pi)
    # Check ranges from the task description
    # For 1 input in [1,3] or [1,5], maybe sin/cos
    # For 2-3 inputs in [1,5] or [1,10], maybe sin/cos of linear combos
    # Let's add a few common trig functions
    if d == 1:
        basis_funcs.append(lambda X: np.sin(X[:, 0]))
        basis_names.append("sin(x0)")
        basis_funcs.append(lambda X: np.cos(X[:, 0]))
        basis_names.append("cos(x0)")
        basis_funcs.append(lambda X: np.sin(2*X[:, 0]))
        basis_names.append("sin(2x0)")
        basis_funcs.append(lambda X: np.cos(2*X[:, 0]))
        basis_names.append("cos(2x0)")
    elif d == 2:
        basis_funcs.append(lambda X: np.sin(X[:, 0] + X[:, 1]))
        basis_names.append("sin(x0+x1)")
        basis_funcs.append(lambda X: np.cos(X[:, 0] + X[:, 1]))
        basis_names.append("cos(x0+x1)")
        basis_funcs.append(lambda X: np.sin(X[:, 0] - X[:, 1]))
        basis_names.append("sin(x0-x1)")
        basis_funcs.append(lambda X: np.cos(X[:, 0] - X[:, 1]))
        basis_names.append("cos(x0-x1)")
        basis_funcs.append(lambda X: np.sin(X[:, 0]))
        basis_names.append("sin(x0)")
        basis_funcs.append(lambda X: np.cos(X[:, 0]))
        basis_names.append("cos(x0)")
        basis_funcs.append(lambda X: np.sin(X[:, 1]))
        basis_names.append("sin(x1)")
        basis_funcs.append(lambda X: np.cos(X[:, 1]))
        basis_names.append("cos(x1)")
    elif d == 3:
        basis_funcs.append(lambda X: np.sin(X[:, 0] + X[:, 1] + X[:, 2]))
        basis_names.append("sin(x0+x1+x2)")
        basis_funcs.append(lambda X: np.cos(X[:, 0] + X[:, 1] + X[:, 2]))
        basis_names.append("cos(x0+x1+x2)")
        for i in range(d):
            basis_funcs.append(lambda X, i=i: np.sin(X[:, i]))
            basis_names.append(f"sin(x{i})")
            basis_funcs.append(lambda X, i=i: np.cos(X[:, i]))
            basis_names.append(f"cos(x{i})")
    
    # Exponential and log (careful with domains)
    if d == 1:
        basis_funcs.append(lambda X: np.exp(X[:, 0]))
        basis_names.append("exp(x0)")
        basis_funcs.append(lambda X: np.log(X[:, 0]))
        basis_names.append("log(x0)")
    elif d == 2:
        basis_funcs.append(lambda X: np.exp(X[:, 0] + X[:, 1]))
        basis_names.append("exp(x0+x1)")
        basis_funcs.append(lambda X: np.exp(X[:, 0] - X[:, 1]))
        basis_names.append("exp(x0-x1)")
    
    # Build the feature matrix
    # Evaluate all basis functions on training data
    try:
        F_train = np.column_stack([f(X_train) for f in basis_funcs])
        # Replace any non-finite values with 0
        F_train[~np.isfinite(F_train)] = 0
    except Exception:
        F_train = None
    
    if F_train is None or F_train.shape[1] == 0:
        # Fallback to simple linear regression
        F_train = np.column_stack([np.ones(n), X_train])
        w, *_ = np.linalg.lstsq(F_train, y_train, rcond=None)
        def predict(X):
            F = np.column_stack([np.ones(len(X)), X])
            return F @ w
        return predict
    
    # Use Ridge regression for stability
    # Try different alpha values
    best_w = None
    best_mse = np.inf
    
    # Use ridge regression with a few alpha values
    for alpha in [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]:
        try:
            # Ridge: minimize ||F w - y||^2 + alpha * ||w||^2
            # (F^T F + alpha I) w = F^T y
            FTF = F_train.T @ F_train + alpha * np.eye(F_train.shape[1])
            Fty = F_train.T @ y_train
            w = np.linalg.solve(FTF, Fty)
            
            # Calculate training MSE
            pred_train = F_train @ w
            mse = np.mean((pred_train - y_train) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_w = w
        except Exception:
            continue
    
    if best_w is None:
        # Fallback to least squares
        w, *_ = np.linalg.lstsq(F_train, y_train, rcond=None)
        best_w = w
    
    def predict(X):
        try:
            F = np.column_stack([f(X) for f in basis_funcs])
            F[~np.isfinite(F)] = 0
            return F @ best_w
        except Exception:
            # Fallback to linear
            F = np.column_stack([np.ones(len(X)), X])
            w_fallback, *_ = np.linalg.lstsq(F, y_train, rcond=None)
            return F @ w_fallback
    
    return predict
# EVOLVE-BLOCK-END
