"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    def schoolbook():
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
        return {"U": U, "V": V, "W": W}

    def verify_numpy(U, V, W, n, m, p, tol=1e-9):
        """Fast verification using NumPy."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        
        # Build the tensor from U, V, W
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We need T[a,b,c] = 1 if (i_a,j_a) matches (j_b,k_b) matches (k_c,i_c)
        # a = i*m+j, b = j'*p+k, c = k'*n+i'
        # Condition: j=j' and k=k' and i=i'
        
        # For each (i,j,k), the valid (a,b,c) is (i*m+j, j*p+k, k*n+i)
        # All other combinations should give 0.
        
        # Check valid positions
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U_np[a] * V_np[b] * W_np[c])
                    if abs(val - 1) > tol:
                        return False
        
        # Check a sample of invalid positions for speed
        # We'll check all invalid positions since the tensor is small
        # Total positions: (n*m) * (m*p) * (p*n) = 6*21*14 = 1764
        # Valid: 42, Invalid: 1722
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        continue  # already checked
                    val = np.sum(U_np[a] * V_np[b] * W_np[c])
                    if abs(val) > tol:
                        return False
        return True

    def verify_exact(U, V, W, n, m, p):
        """Verify with exact Fraction arithmetic."""
        rank = len(U[0])
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        s = Fraction(0)
                        for r in range(rank):
                            s += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                        if s != 1:
                            return False
                    else:
                        s = Fraction(0)
                        for r in range(rank):
                            s += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                        if s != 0:
                            return False
        return True

    # Start with schoolbook
    best = schoolbook()
    best_rank = n * m * p

    # Strategy: Use a structured parametric approach.
    # The tensor is T[a,b,c] = sum_j delta_{j, j_a} delta_{j, j_b} delta_{k_b, k_c} delta_{i_a, i_c}
    # Actually T[i*m+j, j'*p+k, k'*n+i'] = 1 iff j=j', k=k', i=i'
    
    # Let's try to find a decomposition with rank < 42.
    # We'll use a local search / optimization approach.
    # Represent U, V, W as flat arrays and use gradient-free optimization.
    
    def newton_search(rank, max_iters=50, seed=42):
        """
        Attempt to find a rank-`rank` decomposition using a local search approach.
        We parameterize U, V, W and try to minimize the error.
        Since the problem is multilinear, we can use alternating least squares (ALS)
        or a custom Newton-like step.
        
        For simplicity and robustness, we'll use a random restart + local refinement.
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize with random small values
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Build target tensor indices
        valid_abc = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    valid_abc.append((a, b, c))
        
        # Invalid indices (sample for speed, or use all if small)
        all_abc = [(a, b, c) for a in range(n*m) for b in range(m*p) for c in range(p*n)]
        invalid_abc = [t for t in all_abc if t not in set(valid_abc)]
        
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                return None
            
            # Alternating Least Squares
            # Fix V, W -> solve for U
            # For each valid (a,b,c): U[a] . (V[b] * W[c]) = 1
            # For each invalid (a,b,c): U[a] . (V[b] * W[c]) = 0
            
            # This is a linear system for U given V, W.
            # Let's set up the system for U.
            # Rows correspond to all (b,c) pairs? No, the constraints are on specific (a,b,c).
            # Actually, for a fixed a, the constraints are:
            # sum_b,c T_target[a,b,c] U[a] . (V[b] * W[c]) = ...
            # This is not a standard linear system in U[a] alone because the "coefficients" depend on b,c.
            
            # Let's use a simpler gradient descent on the loss function.
            # Loss = sum_valid (val - 1)^2 + sum_invalid (val)^2
            
            # Compute gradients
            # dLoss/dU[a] = 2 * sum_b,c (val - target) * V[b] * W[c]
            
            lr = 0.01
            loss_prev = float('inf')
            loss = 0.0
            
            # Compute current loss
            for (a, b, c) in valid_abc:
                val = np.dot(U[a], V[b] * W[c])
                loss += (val - 1)**2
            for (a, b, c) in invalid_abc:
                val = np.dot(U[a], V[b] * W[c])
                loss += val**2
            
            # Gradient step
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for (a, b, c) in valid_abc:
                val = np.dot(U[a], V[b] * W[c])
                err = val - 1
                grad_U[a] += 2 * err * (V[b] * W[c])
                grad_V[b] += 2 * err * (U[a] * W[c])
                grad_W[c] += 2 * err * (U[a] * V[b])
            
            for (a, b, c) in invalid_abc:
                val = np.dot(U[a], V[b] * W[c])
                err = val
                grad_U[a] += 2 * err * (V[b] * W[c])
                grad_V[b] += 2 * err * (U[a] * W[c])
                grad_W[c] += 2 * err * (U[a] * V[b])
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            loss_curr = 0.0
            for (a, b, c) in valid_abc:
                val = np.dot(U[a], V[b] * W[c])
                loss_curr += (val - 1)**2
            for (a, b, c) in invalid_abc:
                val = np.dot(U[a], V[b] * W[c])
                loss_curr += val**2
                
            if loss_curr < 1e-10:
                # Convert to list of Fractions for exact verification
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                if verify_numpy(U_list, V_list, W_list, n, m, p, tol=1e-6):
                    # Try to find exact rational approximation
                    # For now, return the float solution and let the caller verify
                    return {"U": U_list, "V": V_list, "W": W_list}
            
            loss_prev = loss_curr
        
        return None

    # Try Newton search for ranks below schoolbook
    for r in range(best_rank - 1, 30, -1):
        if time.time() > deadline - 15.0:
            break
        result = newton_search(r, max_iters=100, seed=r * 7919)
        if result is not None:
            # Verify with exact arithmetic if possible, or accept with high tolerance
            # Since we might have float errors, let's try to round to nearby rationals
            U, V, W = result["U"], result["V"], result["W"]
            # Try rounding to halves or integers
            for scale in [1, 2, 4]:
                U_r = [[round(x * scale) / scale for x in row] for row in U]
                V_r = [[round(x * scale) / scale for x in row] for row in V]
                W_r = [[round(x * scale) / scale for x in row] for row in W]
                if verify_numpy(U_r, V_r, W_r, n, m, p, tol=1e-9):
                    best = {"U": U_r, "V": V_r, "W": W_r}
                    best_rank = r
                    break
            if best_rank == r:
                break

    return best
# EVOLVE-BLOCK-END
