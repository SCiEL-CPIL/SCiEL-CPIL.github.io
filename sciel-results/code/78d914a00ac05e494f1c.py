"""Symbolic regression via Differential Evolution over Expression Trees."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import operator

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 325.0  # Safety margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # 1. Prepare Basis Functions
    # We define a fixed set of primitive functions to be used as leaves.
    # This allows the DE to select which primitives are relevant.
    # For d inputs, we have:
    # 0: 1 (constant)
    # 1..d: x_i
    # d+1..2d: sin(pi * x_i)
    # 2d+1..3d: cos(pi * x_i)
    # 3d+1..4d: x_i^2
    # 4d+1..5d: x_i^3
    # 5d+1..6d: exp(-x_i)
    # 6d+1..7d: 1/(1+x_i)
    
    def compute_basis(X_in):
        X_in = np.asarray(X_in, dtype=np.float64)
        n_in = X_in.shape[0]
        # Constant
        basis = [np.ones(n_in)]
        # Linear
        for j in range(d):
            basis.append(X_in[:, j])
        # Trig
        for j in range(d):
            basis.append(np.sin(np.pi * X_in[:, j]))
        for j in range(d):
            basis.append(np.cos(np.pi * X_in[:, j]))
        # Powers
        for j in range(d):
            basis.append(X_in[:, j] ** 2)
        for j in range(d):
            basis.append(X_in[:, j] ** 3)
        # Exp & Rational
        for j in range(d):
            basis.append(np.exp(-X_in[:, j]))
        for j in range(d):
            basis.append(1.0 / (1.0 + X_in[:, j]))
        
        return np.column_stack(basis)

    B_train = compute_basis(X)
    n_features = B_train.shape[1]
    
    # 2. Define Operator Set
    # 0: Add, 1: Sub, 2: Mul, 3: Div (safe), 4: Pow (safe)
    def safe_div(a, b):
        # Avoid division by zero or near-zero
        mask = np.abs(b) < 1e-6
        b_safe = np.where(mask, 1e-6, b)
        res = a / b_safe
        res[mask] = 0.0
        return res

    def safe_pow(a, b):
        # Avoid negative base with fractional power or huge values
        # Since inputs are positive (ranges [1, ...]), bases are usually positive.
        # But combinations might be negative.
        # Clip base to small positive for safety if negative?
        # Better: use sign-preserving power if possible, or just clip.
        # For simplicity and stability, we clip base to [1e-6, 1000] and exponent to [-3, 3]
        a_safe = np.clip(a, 1e-6, 1000.0)
        b_safe = np.clip(b, -3.0, 3.0)
        return np.power(a_safe, b_safe)

    def apply_op(op_idx, a, b):
        if op_idx == 0:
            return a + b
        elif op_idx == 1:
            return a - b
        elif op_idx == 2:
            return a * b
        elif op_idx == 3:
            return safe_div(a, b)
        else: # 4
            return safe_pow(a, b)

    # 3. Expression Tree Evaluation
    # Tree structure:
    # Depth 2:
    #   Root: Op0( Left, Right )
    #   Left: Op1( LeafL, LeafM )
    #   Right: Op2( LeafN, LeafR )
    # Leaves are indices into B_train.
    # Total variables: 3 ops + 4 leaf indices = 7 variables.
    
    def eval_tree(params):
        """
        params: [op0, op1, op2, leafL, leafM, leafN, leafR]
        Returns vector of predicted values before linear scaling.
        """
        op0 = int(np.round(params[0])) % 5
        op1 = int(np.round(params[1])) % 5
        op2 = int(np.round(params[2])) % 5
        
        leafL = int(np.round(params[3])) % n_features
        leafM = int(np.round(params[4])) % n_features
        leafN = int(np.round(params[5])) % n_features
        leafR = int(np.round(params[6])) % n_features
        
        vL = B_train[:, leafL]
        vM = B_train[:, leafM]
        vN = B_train[:, leafN]
        vR = B_train[:, leafR]
        
        left = apply_op(op1, vL, vM)
        right = apply_op(op2, vN, vR)
        root = apply_op(op0, left, right)
        
        return root

    # 4. Objective Function for DE
    # We want to find the tree structure that, when optimally scaled (a*tree + b), fits y best.
    # Since scaling is linear, we can solve for optimal a, b via Least Squares for any given tree.
    # Objective: Minimize Residual Sum of Squares (RSS) after optimal scaling.
    
    def objective(params):
        try:
            tree_vals = eval_tree(params)
            
            # Check for non-finite values
            if not np.all(np.isfinite(tree_vals)):
                return 1e10
                
            # Fit y = a * tree_vals + b
            # Matrix form: [tree_vals, 1] @ [a, b]^T = y
            A = np.column_stack([tree_vals, np.ones(n)])
            
            # Solve least squares
            try:
                sol, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
                a, b = sol
                preds = A @ sol
                rss = np.sum((y - preds) ** 2)
                return rss
            except Exception:
                return 1e10
        except Exception:
            return 1e10

    # 5. Differential Evolution Search
    # Bounds: Ops [0, 4.99], Leaves [0, n_features-0.01]
    bounds = [
        (0, 4.99), (0, 4.99), (0, 4.99),
        (0, n_features - 0.01), (0, n_features - 0.01), 
        (0, n_features - 0.01), (0, n_features - 0.01)
    ]
    
    best_params = None
    best_rss = np.inf
    
    # Run DE with a time limit
    # We use a callback to check time
    def callback(xk, convergence=True):
        if time.time() > DEADLINE - 10:
            raise TimeoutError("Time limit reached")

    try:
        # maxiter and popsize tuned for speed
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=20, 
            popsize=10, 
            tol=0.01,
            callback=callback,
            seed=42,
            polish=False # Polish is slow, we do LSQ ourselves
        )
        if res.success or res.fun < best_rss:
            best_params = res.x
            best_rss = res.fun
    except TimeoutError:
        pass
    except Exception:
        pass

    # 6. Fallback & Selection
    # If DE didn't find anything good, or failed, we need a robust fallback.
    # We also compare against a simple Linear Model on the basis to ensure we don't do worse than linear.
    
    # Fit Linear Model on Basis
    A_lin = B_train
    try:
        w_lin, *_ = np.linalg.lstsq(A_lin, y, rcond=None)
        preds_lin = A_lin @ w_lin
        rss_lin = np.sum((y - preds_lin) ** 2)
    except Exception:
        w_lin = np.zeros(n_features)
        preds_lin = np.zeros(n)
        rss_lin = np.inf

    # Evaluate DE result
    final_params = best_params
    if final_params is None:
        # Use a default simple tree: x_0 (index 1)
        final_params = np.array([0, 0, 0, 1, 1, 1, 1]) # x0 + x0 + x0 (effectively linear-ish)
        
    tree_vals = eval_tree(final_params)
    if not np.all(np.isfinite(tree_vals)):
        tree_vals = np.ones(n)
        
    A_tree = np.column_stack([tree_vals, np.ones(n)])
    try:
        sol_tree, *_ = np.linalg.lstsq(A_tree, y, rcond=None)
        a_tree, b_tree = sol_tree
        preds_tree = A_tree @ sol_tree
        rss_tree = np.sum((y - preds_tree) ** 2)
    except Exception:
        a_tree, b_tree = 0.0, np.mean(y)
        preds_tree = np.full(n, b_tree)
        rss_tree = np.sum((y - preds_tree) ** 2)

    # Decide which model to use
    # If DE found a significantly better fit (lower RSS), use it.
    # Otherwise, use the Linear Basis model which is more robust.
    
    use_tree = (rss_tree < 0.9 * rss_lin) and np.isfinite(rss_tree)
    
    if use_tree:
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            B_test = compute_basis(X_test)
            # Re-evaluate tree on test basis
            # We need to map the leaf indices from train basis to test basis.
            # Since compute_basis is deterministic and same length, indices are valid.
            
            op0 = int(np.round(final_params[0])) % 5
            op1 = int(np.round(final_params[1])) % 5
            op2 = int(np.round(final_params[2])) % 5
            
            leafL = int(np.round(final_params[3])) % n_features
            leafM = int(np.round(final_params[4])) % n_features
            leafN = int(np.round(final_params[5])) % n_features
            leafR = int(np.round(final_params[6])) % n_features
            
            vL = B_test[:, leafL]
            vM = B_test[:, leafM]
            vN = B_test[:, leafN]
            vR = B_test[:, leafR]
            
            left = apply_op(op1, vL, vM)
            right = apply_op(op2, vN, vR)
            root = apply_op(op0, left, right)
            
            # Ensure finite
            if not np.all(np.isfinite(root)):
                root = np.where(np.isfinite(root), root, 0.0)
            
            preds = a_tree * root + b_tree
            # Final safety check
            if not np.all(np.isfinite(preds)):
                preds = np.where(np.isfinite(preds), preds, b_tree)
            return preds
    else:
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            B_test = compute_basis(X_test)
            preds = B_test @ w_lin
            if not np.all(np.isfinite(preds)):
                preds = np.where(np.isfinite(preds), preds, np.mean(y))
            return preds

    return predict
# EVOLVE-BLOCK-END
