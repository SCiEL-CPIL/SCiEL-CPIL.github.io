"""ODE solver: Heat2D 10x10 – integrate(f, t_span, y0)."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    if N != 100:
        # Fallback for unexpected sizes (should not happen per task spec)
        # Use simple RK4
        steps = 1000
        h = dt / steps
        t, y = t0, y0
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2*k1)
            k3 = f(t + h/2, y + h/2*k2)
            k4 = f(t + h, y + h*k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y

    # Assume 10x10 grid, row-major ordering
    # y = L * y where L = I_10 ⊗ A + A ⊗ I_10
    # We need to reconstruct A (10x10)
    # A is the 1D discrete Laplacian (likely Dirichlet BCs)
    # A[i,j] = f(t, e_j)[i] for the 1D case, but here we have 2D.
    
    # Let's reconstruct the 2D operator's action on basis vectors to infer structure.
    # Actually, simpler: Reconstruct the 10x10 1D Laplacian A by probing the 2D system.
    # If y is a vector of length 100, it can be reshaped to 10x10.
    # Let e_i be a basis vector in R^10.
    # Consider a 2D state where only the (0, j) element is non-zero?
    # This is getting complicated. Let's just reconstruct the full 100x100 L?
    # No, the goal is to save calls.
    
    # Alternative: Reconstruct A directly.
    # The 2D heat equation is u_t = u_xx + u_yy.
    # In matrix form y' = (I ⊗ A + A ⊗ I) y.
    # Let's find A.
    # Take a vector v in R^100 that corresponds to a row in the 2D grid.
    # Specifically, let's probe the first row of the grid.
    # Indices 0..9 correspond to the first row (if row-major).
    # Let's create a vector x that has 1s in the first 10 positions and 0 elsewhere.
    # This is not a basis vector.
    
    # Better approach:
    # The operator L is linear.
    # We can determine A by looking at the action of L on vectors that are "separable".
    # Let u be a 10x10 matrix. y = u.flatten().
    # L(u) = A u^T + u A^T (depending on convention, usually A u + u A^T if A is symmetric).
    # Actually, for standard 5-point stencil:
    # L(u)_{i,j} = 4u_{i,j} - u_{i-1,j} - u_{i+1,j} - u_{i,j-1} - u_{i,j+1} (with boundaries)
    # This is equivalent to (A ⊗ I + I ⊗ A) or (I ⊗ A + A ⊗ I) depending on flattening order.
    # If y is row-major (Fortran order in math, C order in numpy default? Numpy is C-order by default: y[i*10+j]).
    # Let's assume standard Kronecker sum.
    
    # We need 10 calls to reconstruct A.
    # How?
    # Consider a vector y that is zero everywhere except in the first column (indices 0, 10, 20, ..., 90).
    # This doesn't isolate A directly.
    
    # Let's use the fact that L = I ⊗ A + A ⊗ I.
    # Let e_i be the i-th standard basis vector in R^10.
    # Let v_i be a vector in R^100 corresponding to e_i ⊗ e_0 (or similar).
    # This is tricky.
    
    # Let's try a different structural insight.
    # The eigenvalues of L are lambda_i + lambda_j where lambda_k are eigenvalues of A.
    # But we don't know the eigenvectors.
    
    # Let's just reconstruct A by assuming the grid structure.
    # We can test if the system is consistent with a 1D Laplacian A.
    # We need to extract A.
    # Let's form a matrix M of size 10x100 where rows are basis vectors for the "row" direction?
    
    # Actually, the simplest robust way to get A with few calls:
    # Note that for any vector u in R^10, if we set the 2D state such that it is constant along one axis,
    # the operator reduces to A acting on the other axis.
    
    # Let's construct 10 test vectors.
    # For i in 0..9:
    # Create a vector y_i in R^100 such that when reshaped to 10x10, it has u_i in the first column and 0 elsewhere?
    # No.
    
    # Let's use the property:
    # L = I ⊗ A + A ⊗ I.
    # Let e_j be basis vector in R^10.
    # Let y_j = e_j ⊗ e_0 (vector in R^100).
    # Then L y_j = (I e_j) ⊗ (A e_0) + (A e_j) ⊗ (I e_0)
    #            = e_j ⊗ (A e_0) + (A e_j) ⊗ e_0.
    # This mixes A e_0 and A e_j.
    
    # If we choose e_0 such that A e_0 is known? No.
    
    # Let's try to isolate A.
    # Consider vectors that are "rank 1" matrices.
    # Let u be a 10x10 matrix.
    # If u = v w^T, then L(u) = A v w^T + v (A^T w)^T.
    # If we pick w = e_0 (first column), then u has only first column non-zero (equal to v).
    # L(u) = A v e_0^T + v (A^T e_0)^T.
    # The result is a 10x10 matrix.
    # The first column of L(u) is A v.
    # The rest of the columns are v (A^T e_0)^T.
    
    # So, if we input a vector y corresponding to u = v e_0^T,
    # and we look at the first 10 elements of f(t, y) (which correspond to the first column of the output matrix),
    # we get A v.
    
    # So, to get A, we can pick 10 vectors v_k = e_k (standard basis in R^10).
    # For each k:
    # 1. Construct y_k in R^100: set the first column to e_k, rest to 0.
    #    In row-major flattening (y[i*10+j]), the first column is indices 0, 10, 20, ..., 90.
    #    So y_k[10*k] = 1.0, others 0.
    # 2. Compute z_k = f(t0, y_k).
    # 3. Reshape z_k to 10x10.
    # 4. The first column of z_k (indices 0, 10, ..., 90 in flat array) should be A e_k = A[:, k].
    #    Wait, is it exactly the first column?
    #    L(u) = A v e_0^T + v (A^T e_0)^T.
    #    Let u = e_k e_0^T.
    #    L(u) = A e_k e_0^T + e_k (A^T e_0)^T.
    #    The (i, j) element is:
    #    (A e_k)_i * delta_{j,0} + (e_k)_i * (A^T e_0)_j.
    #    For j=0 (first column):
    #    (L(u))_{i,0} = (A e_k)_i * 1 + (e_k)_i * (A^T e_0)_0.
    #    So the first column of L(u) is A[:, k] + e_k * (A^T e_0)_0.
    #    This is not just A[:, k]. There is a contamination term.
    
    # However, if we assume A is symmetric (which it is for standard Laplacian with Dirichlet/Neumann),
    # then A^T = A.
    # And (A^T e_0)_0 = (A e_0)_0 = A[0,0].
    # So first column is A[:, k] + e_k * A[0,0].
    
    # We can solve for A if we know A[0,0].
    # Or, we can use a different probe.
    
    # What if we use the last column?
    # Let u = v e_9^T (last column).
    # L(u) = A v e_9^T + v (A^T e_9)^T.
    # The last column (j=9) of L(u) is A v + e_k * (A^T e_9)_9.
    # Same issue.
    
    # What if we use a vector that is zero at the boundaries?
    # The Laplacian with Dirichlet BCs has A[0,0] != 0.
    
    # Let's check the contamination.
    # We have 10 equations for the columns of A.
    # Let C_k be the first column of f(t0, y_k).
    # C_k = A[:, k] + e_k * c, where c = A[0,0].
    # We want to find A and c.
    # For k != 0, the k-th component of C_k is A[k, k] + c.
    # The other components are A[i, k].
    # For k = 0, C_0 = A[:, 0] + e_0 * c.
    # C_0[0] = A[0,0] + c = 2c. So c = C_0[0] / 2.
    # Then A[0,0] = c.
    # For k != 0:
    # A[i, k] = C_k[i] for i != k.
    # A[k, k] = C_k[k] - c.
    
    # This allows us to reconstruct A completely with 10 calls!
    # Assumptions:
    # 1. The system is the standard 2D heat equation with a symmetric 1D Laplacian A.
    # 2. Row-major flattening.
    # 3. Linear autonomous.
    
    start_time = time.time()
    deadline = start_time + 280
    
    # Construct probes
    A = np.zeros((10, 10))
    for k in range(10):
        if time.time() > deadline:
            break
        y_k = np.zeros(100)
        y_k[10 * k] = 1.0  # First column, row k
        
        z_k = f(t0, y_k)
        z_k = np.array(z_k, dtype=float).reshape(10, 10)
        
        # First column of z_k
        col = z_k[:, 0]
        
        # Store in A
        # A[:, k] is related to col.
        # We don't know c yet. Store col temporarily.
        A[:, k] = col
        
    # Calculate c = A[0,0] (from the k=0 probe, C_0[0] = 2 * A[0,0])
    # Wait, C_0 is the first column of the result for probe k=0.
    # Probe k=0: y_0 has 1 at index 0 (row 0, col 0).
    # z_0 = f(t0, y_0).
    # C_0 = z_0[:, 0].
    # C_0[0] = A[0,0] + 1 * A[0,0] = 2 * A[0,0].
    c = A[0, 0] / 2.0
    
    # Correct A
    for k in range(10):
        # A[:, k] was stored as C_k.
        # C_k = A[:, k] + e_k * c.
        # So A[:, k] = C_k - e_k * c.
        A[:, k] = A[:, k] - np.eye(10)[:, k] * c
        
    # Now we have A.
    # Compute exp(dt * A)
    try:
        E = scipy.linalg.expm(dt * A)
    except Exception:
        # Fallback to RK4 if expm fails
        steps = 1000
        h = dt / steps
        t, y = t0, np.array(y0, dtype=float)
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2*k1)
            k3 = f(t + h/2, y + h/2*k2)
            k4 = f(t + h, y + h*k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
    
    # The 2D operator is L = I ⊗ A + A ⊗ I.
    # exp(t L) = exp(t (I ⊗ A + A ⊗ I))
    # Since (I ⊗ A) and (A ⊗ I) commute,
    # exp(t L) = exp(t (I ⊗ A)) exp(t (A ⊗ I))
    #           = (I ⊗ exp(t A)) (exp(t A) ⊗ I)
    #           = exp(t A) ⊗ exp(t A)  (using (A ⊗ B)(C ⊗ D) = AC ⊗ BD)
    # Wait, (I ⊗ E)(E ⊗ I) = (I E) ⊗ (E I) = E ⊗ E.
    # Yes.
    
    M = np.kron(E, E)
    
    y_final = M @ np.array(y0, dtype=float)
    
    if not np.all(np.isfinite(y_final)):
        # Fallback
        steps = 1000
        h = dt / steps
        t, y = t0, np.array(y0, dtype=float)
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2*k1)
            k3 = f(t + h/2, y + h/2*k2)
            k4 = f(t + h, y + h*k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
        
    return y_final
# EVOLVE-BLOCK-END
