"""
Strategy: Spectral Solution via Separable FFT Laplacian
Instead of computing the dense matrix exponential expm(A*dt) which is O(N^3) for N=100,
we exploit the fact that the 2D discrete Laplacian with Neumann BC is separable.
We can diagonalize the 1D Laplacian matrix L_1d using its known eigenvectors (which can
be generated via cosine transforms or specific FFT-based constructions) or simply
compute the eigenvalues analytically and apply the exponential decay in the spectral domain.
For a 10x10 grid, N=100.
The operator A = (1/h^2) * (L_x \otimes I + I \otimes L_y).
Eigenvalues of A are lambda_ij = (1/h^2) * (lambda_i^1d + lambda_j^1d).
We can compute the solution y_final by:
1. Reshaping y0 to 10x10.
2. Transforming to the eigenbasis of L_1d (using the known eigenvectors of the 1D Neumann Laplacian).
3. Multiplying by exp(-lambda_ij * dt).
4. Transforming back.

To find the scaling factor 1/h^2 (let's call it kappa), we use 1-2 calls to f.
We assume the structure is the standard 5-point stencil.
We verify the structure with a random probe. If it fails, we fall back to general probing.

Key improvement over current:
1. Faster computation (though expm on 100x100 is already fast, this is cleaner).
2. More importantly, we can potentially reduce calls. The current program uses 4 calls
   (ones, linear, quad, quad2, random). We can likely get away with 2 calls:
   - One to determine if it's homogeneous (f(0) == 0).
   - One to determine the scaling kappa using a test vector with known non-zero Laplacian.
   - One verification call.
   Total: 3 calls.
   Current uses 5 calls in the structured path.
   Score difference: -0.5 * log10(3/5) = 0.5 * log10(5/3) ≈ 0.5 * 0.22 = 0.11.
   This is a meaningful improvement.
"""

import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 320

    if n != 100:
        return _general_solve(f, t_span, y0, deadline)
    
    # 1. Check for homogeneity and get a sample for scaling
    # Call 1: f(t0, zeros)
    y_zero = np.zeros(100)
    f_zero = f(t0, y_zero)
    
    # If f_zero is not zero, it's non-homogeneous. 
    # The heat equation y' = A*y + b is linear.
    # Solution: y(t) = expm(A*t) y0 + int_0^t expm(A*(t-s)) b ds
    # If b is constant (likely for heat eq with source), we can augment the matrix.
    # However, the standard "heat equation IVP" usually implies homogeneous or that f(t,y) is linear.
    # The task says "fixed 10x10 two-dimensional heat equation initial-value problem".
    # Usually y' = D^2 u. So f(t, y) = A y.
    # Let's assume homogeneous first. If f_zero is large, we might need to handle b.
    # But the standard heat equation solver benchmark likely uses homogeneous.
    # Let's check if f is linear by probing.
    
    # Strategy for minimal calls:
    # We need to determine A.
    # Assume A is the 5-point Laplacian scaled by kappa.
    # A = kappa * L_struct
    
    # Call 2: Use a test vector to estimate kappa.
    # Use a vector that is an eigenvector of the 1D Laplacian?
    # Or just a random vector?
    # Let's use a vector with a simple pattern to avoid cancellation issues.
    # y_test[i, j] = i * j ?
    # Let's construct the structural Laplacian L_struct first.
    
    def build_laplacian_neumann(n_grid):
        N = n_grid * n_grid
        L = np.zeros((N, N))
        for i in range(n_grid):
            for j in range(n_grid):
                idx = i * n_grid + j
                n_neighbors = 0
                if i > 0:
                    L[idx, (i-1)*n_grid + j] = 1.0
                    n_neighbors += 1
                if i < n_grid - 1:
                    L[idx, (i+1)*n_grid + j] = 1.0
                    n_neighbors += 1
                if j > 0:
                    L[idx, i*n_grid + (j-1)] = 1.0
                    n_neighbors += 1
                if j < n_grid - 1:
                    L[idx, i*n_grid + (j+1)] = 1.0
                    n_neighbors += 1
                L[idx, idx] = -n_neighbors
        return L

    L_struct = build_laplacian_neumann(10)
    
    # Test vector: 
    # We want y such that L_struct @ y is not zero and not too small.
    # Let y be a "checkerboard" or linear ramp.
    # Linear ramp i: L_struct @ y is 0 in interior, non-zero at boundaries?
    # For Neumann BC, L @ (linear) is 0 everywhere if the grid is symmetric?
    # No, for Neumann, the discrete Laplacian of a linear function is 0 in the interior.
    # At the boundary, the "missing" neighbor is handled by the BC.
    # Standard 5-point with Neumann: 
    # If we define the grid such that boundaries are the first/last points,
    # and we use the "mirror" or "one-sided" difference?
    # The `build_laplacian_neumann` above uses the "graph Laplacian" definition
    # where degree is the number of neighbors.
    # For a linear function u(i) = i:
    # Interior: (i+1 - 2i + i-1) = 0.
    # Boundary i=0: Neighbors are 1. Diag is -1.
    # L[0,0]*u[0] + L[0,1]*u[1] = -1*0 + 1*1 = 1.
    # So L @ linear is non-zero at boundaries.
    
    y_test = np.zeros(100)
    for i in range(10):
        for j in range(10):
            y_test[i*10+j] = float(i + j)
            
    f_test = f(t0, y_test)
    
    # Predicted f_test = kappa * (L_struct @ y_test)
    L_y_test = L_struct @ y_test
    
    # Estimate kappa
    # kappa = dot(f_test, L_y_test) / dot(L_y_test, L_y_test)
    denom = np.dot(L_y_test, L_y_test)
    if denom < 1e-10:
        # Degenerate test vector, try another
        y_test2 = np.zeros(100)
        y_test2[0] = 1.0
        f_test2 = f(t0, y_test2)
        L_y_test2 = L_struct @ y_test2
        denom = np.dot(L_y_test2, L_y_test2)
        if denom < 1e-10:
            return _general_solve(f, t_span, y0, deadline)
        kappa = np.dot(f_test2, L_y_test2) / denom
        # Note: we used 3 calls so far (zero, test, test2). 
        # We still need to verify.
        # But if we assume the structure is correct, we can proceed.
        # Let's do a verification call.
    else:
        kappa = np.dot(f_test, L_y_test) / denom
        
    # Call 3: Verification
    rng = np.random.default_rng(42)
    y_verify = rng.random(100)
    f_verify = f(t0, y_verify)
    
    predicted_verify = kappa * (L_struct @ y_verify)
    residual = np.linalg.norm(f_verify - predicted_verify) / (np.linalg.norm(f_verify) + 1e-30)
    
    # Also check if f_zero was significant (non-homogeneous)
    # If f_zero is not close to 0, our model y' = A y is wrong.
    # It would be y' = A y + b.
    # If b is constant, we can solve the augmented system.
    # However, the "Heat Equation" benchmark usually implies homogeneous.
    # If residual is small but f_zero is large, we have a constant source.
    
    has_source = np.linalg.norm(f_zero) > 1e-10
    
    if residual > 1e-6:
        # Structure assumption failed. Fall back to general.
        # We have used 3-4 calls. General solve uses 100+.
        # It's better to fail fast if structure is wrong?
        # Or try to fit a general sparse matrix?
        # For now, fallback to general probing.
        return _general_solve(f, t_span, y0, deadline)
        
    # Build A
    A = kappa * L_struct
    dt = t1 - t0
    
    # Spectral Solution
    # We need to solve y' = A y (+ b)
    
    # Method: Diagonalize A.
    # Since A is symmetric (L_struct is symmetric, kappa is scalar), A is symmetric.
    # We can use eigendecomposition.
    # However, the directive suggests FFT/Spectral.
    # For 10x10, `np.linalg.eigh` is very fast and robust.
    # The "FFT" approach is essentially constructing the eigenvectors analytically.
    # For Neumann BC on a uniform grid, the eigenvectors are discrete cosines.
    # We can use `scipy.fft` or just `np.linalg.eigh`.
    # Given N=100, `eigh` is O(N^3) ~ 10^6 ops, which is negligible (<1ms).
    # The benefit of FFT is asymptotic, but here robustness of `eigh` is higher.
    # Let's stick to `eigh` for simplicity and accuracy, which is effectively the same
    # "spectral" family but implemented via robust linear algebra.
    
    try:
        if not has_source:
            # Homogeneous: y(t) = V exp(-Lambda t) V^T y0
            eigvals, eigvecs = np.linalg.eigh(A)
            # eigvals are in ascending order. For heat eq, they should be <= 0.
            # exp(-eigvals * dt)
            decay = np.exp(-eigvals * dt)
            # y_final = eigvecs @ (decay * (eigvecs.T @ y0))
            y_coeff = eigvecs.T @ y0
            y_final_coeff = decay * y_coeff
            y_final = eigvecs @ y_final_coeff
        else:
            # Non-homogeneous: y' = A y + b
            # Augmented system:
            # [y']   = [A  b] [y]
            # [0']   = [0  0] [1]
            # Actually, the constant 1 is preserved.
            # Let z = [y; 1]. z' = [A b; 0 0] z.
            A_aug = np.zeros((101, 101))
            A_aug[:100, :100] = A
            A_aug[:100, 100] = f_zero
            # A_aug[100, :] is 0
            
            eigvals_aug, eigvecs_aug = np.linalg.eigh(A_aug)
            # Note: A_aug is not symmetric if b is not orthogonal to eigenvectors?
            # Wait, A is symmetric. b is a vector.
            # The augmented matrix [[A, b], [0, 0]] is NOT symmetric.
            # So `eigh` (for Hermitian) will fail or give wrong results.
            # We must use `eig` for non-symmetric.
            eigvals_aug, eigvecs_aug = np.linalg.eig(A_aug)
            
            z0 = np.zeros(101)
            z0[:100] = y0
            z0[100] = 1.0
            
            # z(t) = V exp(Delta t) V^{-1} z0
            # This is numerically less stable than eigh, but necessary for non-symmetric.
            # Alternatively, we can use the formula:
            # y(t) = expm(A t) y0 + int_0^t expm(A (t-s)) b ds
            # = expm(A t) y0 + (expm(A t) - I) A^{-1} b  (if A invertible)
            # Or simply compute expm(A*dt) and use the integral form.
            # Since N=100, expm is fast.
            from scipy.linalg import expm
            M = expm(A * dt)
            # Integral of expm(A tau) b dtau from 0 to dt
            # If A is invertible: A^{-1} (expm(A dt) - I) b
            try:
                A_inv = np.linalg.inv(A)
                integral_part = A_inv @ (M - np.eye(100)) @ f_zero
            except np.linalg.LinAlgError:
                # A is singular (eigenvalue 0). 
                # Use series or augmented eig.
                # Let's stick to augmented eig for robustness in this branch.
                # But we already computed eig for aug? No, we only computed if has_source.
                # Let's re-evaluate.
                # If A is singular, we can use the augmented eigen decomposition.
                eigvals_aug, eigvecs_aug = np.linalg.eig(A_aug)
                z0 = np.zeros(101)
                z0[:100] = y0
                z0[100] = 1.0
                # z(t) = V diag(exp(lam t)) V^{-1} z0
                D = np.diag(np.exp(eigvals_aug * dt))
                z_final = eigvecs_aug @ (D @ (np.linalg.inv(eigvecs_aug) @ z0))
                y_final = z_final[:100]
                return y_final
                
            y_final = M @ y0 + integral_part
            
    except Exception:
        return _general_solve(f, t_span, y0, deadline)

    return y_final

def _general_solve(f, t_span, y0, deadline):
    """Fallback: probe A with basis vectors and use expm."""
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Check homogeneity
    y_zero = np.zeros_like(y0)
    f_zero = f(t0, y_zero)
    
    if np.linalg.norm(f_zero) > 1e-12:
        b = f_zero
    else:
        b = None
    
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros_like(y0)
        e_j[j] = 1.0
        f_ej = f(t0, e_j)
        if b is not None:
            A[:, j] = f_ej - b
        else:
            A[:, j] = f_ej
    
    dt = t1 - t0
    
    try:
        from scipy.linalg import expm
        if b is None:
            M = expm(A * dt)
            y_final = M @ y0
        else:
            # Use augmented system for non-homogeneous
            A_aug = np.zeros((n+1, n+1))
            A_aug[:n, :n] = A
            A_aug[:n, n] = b
            y_aug_0 = np.zeros(n+1)
            y_aug_0[:n] = y0
            y_aug_0[n] = 1.0
            
            # A_aug is not symmetric generally.
            eigvals, eigvecs = np.linalg.eig(A_aug)
            D = np.diag(np.exp(eigvals * dt))
            y_aug_final = eigvecs @ (D @ (np.linalg.inv(eigvecs) @ y_aug_0))
            y_final = y_aug_final[:n]
    except Exception:
        # Last resort: simple RK4
        steps = 1000
        h = dt / steps
        t, y = t0, y0.copy()
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        y_final = y
    
    return y_final
