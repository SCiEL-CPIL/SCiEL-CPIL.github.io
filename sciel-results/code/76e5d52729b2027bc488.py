import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    # Build the matmul tensor T in R^{(n*m) x (m*p) x (p*n)}
    # T[a, b, c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some (i,j,k)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition: rank = n*m*p = 48
    rank0 = n * m * p
    U0 = np.zeros((n * m, rank0))
    V0 = np.zeros((m * p, rank0))
    W0 = np.zeros((p * n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-8):
        """Verify tensor decomposition for matmul tensor."""
        r = U.shape[1]
        # Check ones
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Check zeros
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for j2 in range(m):
                    for k2 in range(p):
                        b = j2 * p + k2
                        for k3 in range(p):
                            for i3 in range(n):
                                c = k3 * n + i3
                                if j == j2 and i == i3 and k == k2:
                                    continue
                                val = np.dot(U[a, :], V[b, :] * W[c, :])
                                if abs(val) > tol:
                                    return False
        return True

    # Verify schoolbook
    if not verify(U0, V0, W0):
        pass  # Should not happen
    
    best_rank = rank0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Attempt to find a rank 47 decomposition
    # We use Alternating Least Squares (ALS) to approximate T with rank 47
    target_rank = rank0 - 1
    
    if time.time() < deadline - 20:
        # Initialize with random values
        np.random.seed(42)
        U_alt = np.random.randn(n * m, target_rank)
        V_alt = np.random.randn(m * p, target_rank)
        W_alt = np.random.randn(p * n, target_rank)
        
        # Scale initialization to match magnitude
        U_alt /= np.linalg.norm(U_alt)
        V_alt /= np.linalg.norm(V_alt)
        W_alt /= np.linalg.norm(W_alt)
        
        for iteration in range(100):
            if time.time() > deadline - 20:
                break
            
            # Solve for W
            # For each c, we want to minimize sum_a,b (T[a,b,c] - sum_s U[a,s]*V[b,s]*W[c,s])^2
            # This is a linear system: (U^T U) W[:, c] = U^T T[:, :, c] (reshaped)
            # Actually, let's use the standard ALS update:
            # W[:, c] = (U^T U)^{-1} U^T (T[:, :, c] * V^T) ? No.
            
            # Let's use a simpler vectorized approach for ALS
            # Update W:
            # W[:, c] = argmin_w sum_a,b (T[a,b,c] - sum_s U[a,s] V[b,s] w_s)^2
            # This is equivalent to solving:
            # (U^T U) w = U^T (T[:, :, c] * V^T) ? No, the term is sum_s U[a,s] V[b,s] w_s
            # Let's reshape T to (a, b, c).
            # For a fixed c, let T_c be the matrix T[:, :, c].
            # We want to approximate T_c by U * diag(w_c) * V^T.
            # This is a Procrustes-like problem or least squares.
            # Minimizing || T_c - U W_c V^T ||_F^2 with respect to W_c (vector).
            # This is not standard least squares because of the V^T on the right.
            # Let's use the gradient descent or a specific solver.
            
            # A common ALS update for rank-1 slices:
            # W[:, c] = (U^T U)^{-1} U^T (T[:, :, c] * V^T) is incorrect dimensions.
            
            # Let's use the update rule:
            # W[:, c] = (V^T V)^{-1} V^T (T[:, :, c] * U^T) ? No.
            
            # Correct ALS update for W:
            # Fix U, V.
            # For each c, solve:
            # min_w || T[:, :, c] - U * diag(w) * V^T ||_F^2
            # This can be solved by:
            # (U^T U) w = U^T (T[:, :, c] * V^T) is still wrong.
            
            # Let's use the property:
            # vec(T[:, :, c]) = (V \otimes U) w_c
            # So w_c = ((V \otimes U)^T (V \otimes U))^{-1} (V \otimes U)^T vec(T[:, :, c])
            # Kronecker product might be slow but dimensions are small.
            
            kron_VU = np.kron(V_alt, U_alt) # shape (m*p*n*m, target_rank)
            T_flat = T.reshape(-1, p * n) # shape (n*m*m*p, p*n)
            
            # Solve for W
            # W.T = (kron_VU^T kron_VU)^{-1} kron_VU^T T_flat^T
            try:
                A = kron_VU.T @ kron_VU
                b = kron_VU.T @ T_flat.T
                W_alt.T = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                W_alt.T = np.linalg.lstsq(kron_VU, T_flat.T, rcond=None)[0]
            
            # Solve for V
            # Fix U, W.
            # For each b, solve:
            # min_v || T[:, b, :] - U * diag(v) * W^T ||_F^2
            # vec(T[:, b, :]) = (W \otimes U) v_b
            kron_WU = np.kron(W_alt, U_alt) # shape (p*n*n*m, target_rank)
            T_flat_V = T.reshape(n * m, -1).T # shape (m*p, n*m*p*n) -> wait
            # T is (n*m, m*p, p*n).
            # For fixed b, T[:, b, :] is (n*m, p*n).
            # vec(T[:, b, :]) is (n*m*p*n,).
            # We want to solve for V[:, b].
            # (W \otimes U) is (p*n*n*m, target_rank).
            # T_flat_V should be (m*p, n*m*p*n).
            T_flat_V = T.transpose(0, 2, 1).reshape(p * n * n * m, -1).T # shape (m*p, p*n*n*m)
            # Wait, T is (a, b, c).
            # For fixed b, we have T[a, b, c].
            # vec(T[:, b, :]) stacks a and c.
            # The operator is (W \otimes U).
            # Let's just use a loop for clarity and safety, dimensions are small.
            
            for b in range(m * p):
                try:
                    # Solve for V[:, b]
                    # min_v || T[:, b, :] - U * diag(v) * W^T ||_F^2
                    # This is equivalent to:
                    # (W^T W) v = W^T (T[:, b, :] * U^T) ? No.
                    # Let's use the Kronecker form.
                    # vec(T[:, b, :]) = (W \otimes U) v_b
                    T_b_vec = T[:, b, :].reshape(-1)
                    kron_WU_b = np.kron(W_alt, U_alt)
                    V_alt[:, b] = np.linalg.lstsq(kron_WU_b, T_b_vec, rcond=None)[0]
                except:
                    pass
                    
            for a in range(n * m):
                try:
                    # Solve for U[:, a]
                    # min_u || T[a, :, :] - u * V^T * W^T ||_F^2 ? No.
                    # T[a, b, c] = sum_s U[a,s] V[b,s] W[c,s]
                    # For fixed a, T[a, :, :] is (m*p, p*n).
                    # vec(T[a, :, :]) = (W \otimes V) u_a
                    T_a_vec = T[a, :, :].reshape(-1)
                    kron_WV = np.kron(W_alt, V_alt)
                    U_alt[:, a] = np.linalg.lstsq(kron_WV, T_a_vec, rcond=None)[0]
                except:
                    pass
            
            # Check error
            T_approx = np.einsum('as,bs,cs->abc', U_alt, V_alt, W_alt)
            error = np.linalg.norm(T - T_approx)
            
            if error < 1e-8:
                # Try to round to half-integers
                U_rounded = np.round(U_alt * 2) / 2
                V_rounded = np.round(V_alt * 2) / 2
                W_rounded = np.round(W_alt * 2) / 2
                
                if verify(U_rounded, V_rounded, W_rounded):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    print(f"Found rank {best_rank} decomposition!")
                    break

    if best_rank < rank0:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": U0.tolist(),
            "V": V0.tolist(),
            "W": W0.tolist()
        }
