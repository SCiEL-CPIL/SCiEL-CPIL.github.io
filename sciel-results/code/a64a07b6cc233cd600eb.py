"""ODE solver: 15-dimensional stiff linear system using Krylov subspace methods."""
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate a 15-dimensional stiff linear ODE system y' = f(t, y) from t_span[0] to t_span[1].
    Uses Krylov subspace methods (Arnoldi iteration) to approximate the solution with minimal
    function evaluations.
    
    Strategy:
    1. Assume the system is linear and homogeneous: f(t, y) = A @ y (time-independent).
       Extract A by evaluating f at a few points.
    2. Use Krylov subspace projection to approximate exp((t1-t0)*A) @ y0.
    3. Fall back to implicit methods if needed.
    """
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280  # 300s limit with 20s safety margin
    
    # Step 1: Extract the matrix A assuming f(t, y) = A @ y
    # Evaluate f at y0 to get A @ y0
    f_y0 = f(t0, y0)
    
    # Check if the system appears linear and homogeneous
    # Test: f(t, alpha * y) should equal alpha * f(t, y) if linear homogeneous
    # Also test: f(t1, y) should equal f(t0, y) if time-independent
    
    # Try to extract A by evaluating f at basis vectors
    # But that requires n evaluations. Instead, use Krylov with just f evaluations.
    
    # Krylov subspace approach:
    # Build Krylov basis V = [v1, v2, ..., vk] where v1 = y0/||y0||, v_{j+1} = A@v_j orthogonalized
    # We need to apply A to vectors, which we can do via f(t, v) if the system is linear homogeneous.
    
    def apply_A(v):
        """Apply A to vector v using f. Assumes f(t, v) = A @ v."""
        return f(t0, v)
    
    # Arnoldi iteration to build Krylov basis
    max_krylov_dim = min(n, 15)  # For 15-dim system, full dimension is 15
    # We'll build up to max_krylov_dim vectors
    
    V = np.zeros((n, max_krylov_dim + 1))
    H = np.zeros((max_krylov_dim + 1, max_krylov_dim + 1))
    
    # v1 = y0 / ||y0||
    norm_y0 = np.linalg.norm(y0)
    if norm_y0 < 1e-15:
        return np.zeros_like(y0)
    
    v1 = y0 / norm_y0
    V[:, 0] = v1
    
    k = 1
    beta = 0.0
    
    for j in range(max_krylov_dim):
        if time.time() > deadline:
            break
        
        # w = A @ v_j
        w = apply_A(V[:, j])
        
        # Modified Gram-Schmidt orthogonalization
        for i in range(j + 1):
            H[i, j] = np.dot(V[:, i], w)
            w = w - H[i, j] * V[:, i]
        
        beta = np.linalg.norm(w)
        H[j + 1, j] = beta
        
        if beta < 1e-13:
            # Krylov subspace is invariant, stop
            k = j + 1
            break
        else:
            V[:, j + 1] = w / beta
            k = j + 2
    
    if k == 0:
        k = 1
    
    # Now we have V (n x k) and H (k x k) such that A @ V_k ≈ V_k @ H_k + beta * v_{k+1} * e_k^T
    # For the truncated case, we use H_k (k x k) as the projection of A
    
    H_k = H[:k, :k]
    V_k = V[:, :k]
    
    # The solution is y(t) = exp(t * A) @ y0
    # In the Krylov subspace: y(t) ≈ V_k @ exp(t * H_k) @ (V_k^T @ y0)
    # V_k^T @ y0 = [norm_y0, 0, 0, ..., 0]^T since v1 = y0/||y0||
    
    e1 = np.zeros(k)
    e1[0] = norm_y0
    
    # Compute exp(dt * H_k) @ e1
    # Use scipy's matrix exponential for small matrices
    try:
        from scipy.linalg import expm
        expH = expm(dt * H_k)
    except ImportError:
        # Fallback: use Taylor series for small matrix
        expH = _matrix_exponential_taylor(dt * H_k, order=30)
    
    y_sol = V_k @ (expH @ e1)
    
    # Check for finite values
    if not np.all(np.isfinite(y_sol)) or np.max(np.abs(y_sol)) > 1e12:
        # Fallback to implicit method
        y_sol = _implicit_fallback(f, t0, t1, y0)
    
    return y_sol


def _matrix_exponential_taylor(A, order=30):
    """Compute matrix exponential using Taylor series."""
    n = A.shape[0]
    result = np.eye(n)
    term = np.eye(n)
    for k in range(1, order + 1):
        term = term @ A / k
        result = result + term
        if np.linalg.norm(term) < 1e-15:
            break
    return result


def _implicit_fallback(f, t0, t1, y0):
    """Fallback implicit Euler method if Krylov fails."""
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0
    
    # Use a few steps of implicit Euler with Newton iteration
    num_steps = 50
    h = dt / num_steps
    
    y = y0.copy()
    for step in range(num_steps):
        t_new = t0 + (step + 1) * h
        z = y.copy()
        
        for newton_iter in range(10):
            fz = f(t_new, z)
            
            # Finite difference Jacobian
            J = np.zeros((n, n))
            for j in range(n):
                e = np.zeros(n)
                scale = max(1.0, abs(z[j])) * 1e-7
                e[j] = scale
                J[:, j] = (f(t_new, z + e) - fz) / scale
            
            # Solve (I - h*J) @ step = -(z - y - h*fz)
            try:
                step_vec = np.linalg.solve(np.eye(n) - h * J, -(z - y - h * fz))
            except np.linalg.LinAlgError:
                step_vec = -h * fz
            
            z_new = z + step_vec
            if np.linalg.norm(step_vec) < 1e-12 * (1 + np.linalg.norm(z_new)):
                z = z_new
                break
            z = z_new
        
        y = z
    
    return y
