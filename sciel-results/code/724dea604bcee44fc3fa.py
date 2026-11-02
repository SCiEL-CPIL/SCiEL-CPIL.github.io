import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 5, 5
    rank = 25
    
    # Target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We will work with flattened indices for U, V, W
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    def compute_tensor(U, V, W):
        """Compute the resulting tensor C from U, V, W.
        Returns a dict mapping (i, k) to sum, or a matrix C of shape (n, p).
        C[i, k] = sum_{j, r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        """
        C = np.zeros((n, p), dtype=np.float64)
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    # U[i*m+j, :] dot V[j*p+k, :] weighted by W?
                    # No, sum over r.
                    # C[i, k] += sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # This is not a simple dot product of U and V because of W.
                    # It's a trilinear form.
                    # We can compute:
                    # term_j = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # This is element-wise product of U row, V row, W row, then sum.
                    # We can vectorize over r.
                    pass
                # Let's just loop over r for simplicity in the inner loop, 
                # or vectorize.
                # Vectorized:
                # For fixed i, k:
                # We need sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # Let W_row = W[k*n+i, :] (shape rank)
                # For each j, compute dot(U[i*m+j, :], V[j*p+k, :] * W_row)
                # Sum over j.
                w_row = W[k*n+i, :]
                s = 0.0
                for j in range(m):
                    s += np.dot(U[i*m+j, :], V[j*p+k, :] * w_row)
                C[i, k] = s
        return C

    def error_and_grad(U, V, W):
        """Compute error and gradients for U, V, W."""
        C = compute_tensor(U, V, W)
        # Target C is identity-like?
        # No, the tensor T has 1s at specific positions.
        # The "diagonal" of the tensor T_{ijk} corresponds to C_{ki} = delta_{?}
        # Wait, the tensor decomposition defines the multiplication.
        # The condition is that the computed multiplication matches the standard one.
        # Standard multiplication: C[i, k] = sum_j A[i, j] B[j, k].
        # This is linear in A and B.
        # The tensor T satisfies T(A, B) = A B.
        # So we need the decomposition to represent the identity map on the space of multiplications?
        # No, we need the decomposition to BE the multiplication tensor.
        # The "error" is not just on C, because C depends on A and B.
        # We need the coefficients of the polynomial to match.
        # The coefficient of A[i, j] B[j, k] in C[i, k] must be 1.
        # The coefficient of A[i, j] B[j', k] in C[i, k] (where j != j') must be 0?
        # No, C[i, k] only depends on A[i, :] and B[:, k].
        # So A[i, j] and B[j, k] are the only terms.
        # So for each (i, k), the sum over j of (coeff of A[i, j] B[j, k]) must be 1?
        # No, for a fixed (i, k), the term A[i, j] B[j, k] appears in C[i, k].
        # In the decomposition:
        # C[i, k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * ... wait.
        # Let's check the expansion.
        # C[i, k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * ?)
        # The V factor depends on j and k.
        # So C[i, k] = sum_r W[k*n+i, r] * sum_j U[i*m+j, r] * V[j*p+k, r].
        # We want this to equal sum_j A[i, j] B[j, k].
        # So for each i, k:
        # sum_r W[k*n+i, r] * sum_j U[i*m+j, r] * V[j*p+k, r] = sum_j A[i, j] B[j, k].
        # This must hold for all A, B.
        # So the coefficient of A[i, j] B[j', k] must be:
        # 1 if j == j', 0 if j != j'.
        # Coefficient of A[i, j] B[j', k] in the LHS:
        # If j == j': sum_r W[k*n+i, r] U[i*m+j, r] V[j*p+k, r].
        # If j != j': sum_r W[k*n+i, r] U[i*m+j, r] V[j'*p+k, r].
        # So we need:
        # 1. For all i, j, k: sum_r W[k*n+i, r] U[i*m+j, r] V[j*p+k, r] = 1
        # 2. For all i, j, j', k with j != j': sum_r W[k*n+i, r] U[i*m+j, r] V[j'*p+k, r] = 0
        
        # This is a set of linear constraints on the products of U, V, W?
        # No, it's a set of equations where the variables are the entries of U, V, W.
        # The equations are cubic? No, they are linear in W if U, V are fixed?
        # No, they are trilinear.
        
        # Let's define the error as the sum of squares of these constraints.
        # Constraints:
        # D[i, j, k] = sum_r W[k*n+i, r] U[i*m+j, r] V[j*p+k, r] - 1 = 0
        # O[i, j, j', k] = sum_r W[k*n+i, r] U[i*m+j, r] V[j'*p+k, r] = 0
        
        # Number of D constraints: n * m * p = 50.
        # Number of O constraints: n * (m*(m-1)) * p = 2 * 20 * 5 = 200.
        # Total 250 constraints.
        
        err = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Compute D and O terms
        # To save time, we can accumulate gradients.
        
        for i in range(n):
            for k in range(p):
                w_row = W[k*n+i, :]
                for j in range(m):
                    # D constraint
                    v_row = V[j*p+k, :]
                    u_row = U[i*m+j, :]
                    val = np.dot(u_row * v_row, w_row)
                    err += (val - 1.0)**2
                    # Gradient wrt U[i*m+j, r]: 2(val-1) * V[j*p+k, r] * W[k*n+i, r]
                    grad_U[i*m+j, :] += 2.0 * (val - 1.0) * v_row * w_row
                    # Gradient wrt V[j*p+k, r]: 2(val-1) * U[i*m+j, r] * W[k*n+i, r]
                    grad_V[j*p+k, :] += 2.0 * (val - 1.0) * u_row * w_row
                    # Gradient wrt W[k*n+i, r]: 2(val-1) * U[i*m+j, r] * V[j*p+k, r]
                    grad_W[k*n+i, :] += 2.0 * (val - 1.0) * u_row * v_row
                    
                    for j2 in range(m):
                        if j == j2:
                            continue
                        # O constraint
                        v_row2 = V[j2*p+k, :]
                        val_o = np.dot(u_row * v_row2, w_row)
                        err += val_o**2
                        # Gradient wrt U[i*m+j, r]: 2 val_o * V[j2*p+k, r] * W[k*n+i, r]
                        grad_U[i*m+j, :] += 2.0 * val_o * v_row2 * w_row
                        # Gradient wrt V[j2*p+k, r]: 2 val_o * U[i*m+j, r] * W[k*n+i, r]
                        grad_V[j2*p+k, :] += 2.0 * val_o * u_row * w_row
                        # Gradient wrt W[k*n+i, r]: 2 val_o * U[i*m+j, r] * V[j2*p+k, r]
                        grad_W[k*n+i, :] += 2.0 * val_o * u_row * v_row2
                        
        return err, grad_U, grad_V, grad_W

    def verify(U, V, W):
        # Check if entries are close to integers or half-integers
        # and if constraints are satisfied
        tol = 1e-6
        for i in range(n):
            for k in range(p):
                w_row = W[k*n+i, :]
                for j in range(m):
                    v_row = V[j*p+k, :]
                    u_row = U[i*m+j, :]
                    val = np.dot(u_row * v_row, w_row)
                    if abs(val - 1.0) > tol:
                        return False
                    for j2 in range(m):
                        if j == j2: continue
                        v_row2 = V[j2*p+k, :]
                        val_o = np.dot(u_row * v_row2, w_row)
                        if abs(val_o) > tol:
                            return False
        return True

    # Optimization Loop
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for seed in range(10):
        if time.time() > deadline:
            break
        np.random.seed(seed)
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        lr = 0.01
        for step in range(1000):
            if time.time() > deadline:
                break
            err, gU, gV, gW = error_and_grad(U, V, W)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-9:
                break
            # Update
            U -= lr * gU
            V -= lr * gV
            W -= lr * gW
            # Decay learning rate
            lr *= 0.999
            
    if best_err < 1e-6:
        # Round to nearest 0.5
        U_int = np.round(best_U * 2) / 2
        V_int = np.round(best_V * 2) / 2
        W_int = np.round(best_W * 2) / 2
        if verify(U_int, V_int, W_int):
            return {
                "U": U_int.tolist(),
                "V": V_int.tolist(),
                "W": W_int.tolist()
            }
            
    # Fallback to schoolbook
    U, V, W = schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
