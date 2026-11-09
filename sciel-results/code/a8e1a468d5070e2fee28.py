import numpy as np
import time
from itertools import combinations

def fit(X_train, y_train):
    t_start = time.time()
    deadline = t_start + 320.0  # Safety margin for 330s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Handle constant y
    if np.var(y) < 1e-12:
        y_mean = np.mean(y)
        return lambda X: np.full(len(X), y_mean)
        
    # Standardize X for stability in basis functions
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-10] = 1.0
    X_std = (X - mu) / sigma
    
    def standardize(X_new):
        return (np.asarray(X_new, dtype=np.float64) - mu) / sigma
        
    # --- Basis Function Definitions ---
    # We define a pool of diverse basis functions.
    # Each basis maps X_std (n, d) -> (n, m)
    
    basis_pool = []
    
    # 1. Linear (including bias)
    def basis_linear(X_std):
        return np.hstack([np.ones((len(X_std), 1)), X_std])
    basis_pool.append(('linear', basis_linear))
    
    # 2. Quadratic (interactions + squares)
    def basis_quad(X_std):
        n = len(X_std)
        # Squares
        sq = X_std ** 2
        # Interactions
        inter = []
        for i in range(d):
            for j in range(i + 1, d):
                inter.append(X_std[:, i:i+1] * X_std[:, j:j+1])
        if inter:
            inter = np.hstack(inter)
        else:
            inter = np.zeros((n, 0))
        return np.hstack([np.ones((n, 1)), X_std, sq, inter])
    basis_pool.append(('quad', basis_quad))
    
    # 3. RBFs (Gaussian kernels) with various lengthscales
    def make_rbf(l):
        def rbf_basis(X_std):
            # Use a grid of centers? No, use all training points as centers for simplicity in LOO?
            # That's O(N^2). With N=200, it's 40k features. Solving 200x40k is fine.
            # But LOO on 40k features is heavy. 
            # Better: Use a subset of centers or a grid.
            # Let's use a grid of centers for efficiency.
            # Grid size depends on d.
            # For d=1: 20 centers. d=2: 10x10=100. d=3: 7x7x7=343.
            # This keeps feature count manageable.
            
            if d == 1:
                centers = np.linspace(-2, 2, 20).reshape(-1, 1)
            elif d == 2:
                grid = np.linspace(-2, 2, 8)
                mesh = np.meshgrid(grid, grid)
                centers = np.column_stack([mesh[0].flatten(), mesh[1].flatten()])
            else: # d=3
                grid = np.linspace(-2, 2, 5)
                mesh = np.meshgrid(grid, grid, grid)
                centers = np.column_stack([mesh[0].flatten(), mesh[1].flatten(), mesh[2].flatten()])
                
            # Compute distances from X_std to centers
            sq_dists = np.sum(X_std**2, axis=1)[:, None] + np.sum(centers**2, axis=1)[None, :] - 2 * X_std @ centers.T
            np.maximum(sq_dists, 0, out=sq_dists)
            r = np.sqrt(sq_dists) / l
            return np.exp(-0.5 * r**2)
        return rbf_basis

    for l in [0.5, 1.0, 2.0]:
        basis_pool.append((f'rbf_{l}', make_rbf(l)))
        
    # 4. Fourier features (for periodic or oscillatory functions)
    def make_fourier(max_freq):
        def fourier_basis(X_std):
            # Apply to each dimension
            feats = []
            for i in range(d):
                x = X_std[:, i:i+1]
                for k in range(1, max_freq + 1):
                    feats.append(np.sin(k * np.pi * x))
                    feats.append(np.cos(k * np.pi * x))
            if feats:
                return np.hstack(feats)
            return np.zeros((len(X_std), 0))
        return fourier_basis

    for f in [2, 4]:
        basis_pool.append((f'fourier_{f}', make_fourier(f)))

    # --- LOO Cross-Validation for Ridge Regression ---
    # For a basis matrix Phi (n x m), the LOO error for Ridge regression with lambda is:
    # e_i = (y_i - h_i) / (1 - H_ii)
    # where H = Phi (Phi^T Phi + lambda I)^-1 Phi^T
    # H_ii = Phi_i (Phi^T Phi + lambda I)^-1 Phi_i^T
    
    def compute_loo_errors(Phi, y, lambda_val):
        n, m = Phi.shape
        # Efficient computation of H_ii using Woodbury or direct solve
        # A = Phi.T @ Phi + lambda * I
        A = Phi.T @ Phi + lambda_val * np.eye(m)
        # We need diag(Phi @ inv(A) @ Phi.T)
        # inv_A = np.linalg.inv(A)
        # H_ii = np.sum((Phi @ inv_A) * Phi, axis=1)
        # To save time, we can solve A X = Phi.T to get X = A^-1 Phi.T
        # Then H_ii = sum(Phi * X.T, axis=1)
        try:
            X_inv = np.linalg.solve(A, Phi.T)
            h_ii = np.sum(Phi * X_inv.T, axis=1)
        except np.linalg.LinAlgError:
            # Fallback if singular
            X_inv, *_ = np.linalg.lstsq(A, Phi.T, rcond=None)
            h_ii = np.sum(Phi * X_inv.T, axis=1)
            
        # Prevent division by zero
        denom = 1 - h_ii
        denom[np.abs(denom) < 1e-10] = 1e-10
        
        # Predictions h_i = Phi @ w
        # w = A^-1 Phi^T y
        w = X_inv @ y
        h = Phi @ w
        
        errors = (y - h) / denom
        mse = np.mean(errors**2)
        return mse

    # --- Search Strategy ---
    # We will test each basis individually with a range of lambdas.
    # Then we will try combining the top few bases.
    
    candidate_models = [] # List of (name, basis_func, best_lambda, loo_mse)
    
    lambdas = [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    
    # Pre-compute basis matrices for all bases in pool
    # This is expensive if we do it repeatedly, so we store them.
    basis_matrices = {}
    for name, func in basis_pool:
        if time.time() > deadline - 20: break
        try:
            basis_matrices[name] = func(X_std)
        except Exception:
            pass
            
    # Evaluate individual bases
    for name, Phi in basis_matrices.items():
        if time.time() > deadline - 20: break
        best_lam = None
        best_mse = np.inf
        for lam in lambdas:
            mse = compute_loo_errors(Phi, y, lam)
            if mse < best_mse:
                best_mse = mse
                best_lam = lam
        candidate_models.append((name, Phi, best_lam, best_mse))
        
    # Sort by LOO MSE
    candidate_models.sort(key=lambda x: x[3])
    
    # If time permits, try ensembles of top 2 and top 3
    # Ensemble: Stack basis matrices horizontally, solve Ridge on combined matrix
    best_ensemble_mse = np.inf
    best_ensemble_indices = []
    
    if time.time() < deadline - 30:
        top_k = min(5, len(candidate_models))
        # Try pairs
        for i in range(top_k):
            for j in range(i + 1, top_k):
                if time.time() > deadline - 20: break
                name1, Phi1, lam1, mse1 = candidate_models[i]
                name2, Phi2, lam2, mse2 = candidate_models[j]
                # Combine
                Phi_comb = np.hstack([Phi1, Phi2])
                # We need to optimize lambda for the combined matrix.
                # For speed, just try a few lambdas or use the individual ones?
                # Using individual lambdas is tricky because the regularization is on the combined vector.
                # Let's try a small set of lambdas for the combined matrix.
                for lam in [1e-4, 1e-2, 1e-1]:
                    mse = compute_loo_errors(Phi_comb, y, lam)
                    if mse < best_ensemble_mse:
                        best_ensemble_mse = mse
                        best_ensemble_indices = [i, j]
                        best_ensemble_lam = lam
                        best_ensemble_name = f"ens_{name1}_{name2}"
                        
        # Try triples if time is very generous (unlikely with 300s limit and 200 points, but possible)
        # Actually, solving 200x(M1+M2+M3) is fast.
        # Let's stick to pairs to be safe on time.

    # Decide which model to use
    if candidate_models:
        if best_ensemble_indices and best_ensemble_mse < candidate_models[0][3] * 0.9:
            # Use ensemble
            # Reconstruct Phi and solve final
            idxs = best_ensemble_indices
            Phis = [candidate_models[i][1] for i in idxs]
            Phi_final = np.hstack(Phis)
            lam_final = best_ensemble_lam
        else:
            # Use best single
            name, Phi, lam, mse = candidate_models[0]
            Phi_final = Phi
            lam_final = lam
    else:
        # Fallback: Linear regression on standardized X
        Phi_final = np.hstack([np.ones((n, 1)), X_std])
        lam_final = 1e-6

    # Final Fit
    A = Phi_final.T @ Phi_final + lam_final * np.eye(Phi_final.shape[1])
    try:
        w = np.linalg.solve(A, Phi_final.T @ y)
    except:
        w, *_ = np.linalg.lstsq(A, Phi_final.T @ y, rcond=None)

    # Store basis function for prediction
    # We need to replicate the basis generation for new data
    # This is tricky because we stored the matrix, not the function.
    # We must map the chosen name back to the function.
    
    # Re-implement basis generation for prediction
    def predict(X_test):
        X_test_std = standardize(X_test)
        
        # Reconstruct basis
        # We need to know which basis was used.
        # If ensemble, we need to hstack them.
        
        # This logic is a bit fragile. Let's ensure we keep the function references.
        # Better approach: Store the function names in candidate_models.
        
        if best_ensemble_indices and best_ensemble_mse < candidate_models[0][3] * 0.9:
            idxs = best_ensemble_indices
            # Get the functions
            # candidate_models[i] is (name, Phi, lam, mse). We lost the function.
            # We need to look up the function from basis_pool by name.
            funcs = []
            for i in idxs:
                name = candidate_models[i][0]
                # Find func in basis_pool
                for nm, fn in basis_pool:
                    if nm == name:
                        funcs.append(fn)
                        break
            Phis = [fn(X_test_std) for fn in funcs]
            Phi_test = np.hstack(Phis)
        else:
            name = candidate_models[0][0]
            for nm, fn in basis_pool:
                if nm == name:
                    Phi_test = fn(X_test_std)
                    break
            
        return Phi_test @ w

    return predict
