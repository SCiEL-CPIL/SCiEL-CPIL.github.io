import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = DIMENSIONS
    num_u = n * m  # 9
    num_v = m * p  # 15
    num_w = p * n  # 15

    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(num_u)]
        V = [[0] * rank for _ in range(num_v)]
        W = [[0] * rank for _ in range(num_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly computes matrix multiplication.
        Returns True if valid, False otherwise."""
        rank = len(U[0])
        
        # Check all (i, j, k) combinations
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # The term U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] should sum to 1
                    total = 0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if total != 1:
                        return False
        
        # Check that no spurious terms exist
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            if i2 == i and k2 == k:
                                continue
                            total = 0
                            for r in range(rank):
                                total += U[i * m + j][r] * V[j * p + k][r] * W[k2 * n + i2][r]
                            if total != 0:
                                return False
        return True

    def snap_to_rational(values, max_denom=8):
        """Snap values to nearest fraction with small denominator."""
        snapped = []
        for v in values:
            best_frac = Fraction(0)
            best_dist = float('inf')
            for denom in range(1, max_denom + 1):
                num = round(v * denom)
                frac = Fraction(num, denom)
                dist = abs(v - float(frac))
                if dist < best_dist:
                    best_dist = dist
                    best_frac = frac
            snapped.append(best_frac)
        return snapped

    def compute_error(U, V, W):
        """Compute the total error of the decomposition."""
        rank = len(U[0])
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                    error += (total - 1) ** 2
                    # Check spurious terms
                    for i2 in range(n):
                        for k2 in range(p):
                            if i2 == i and k2 == k:
                                continue
                            total_spurious = sum(U[i*m+j][r] * V[j*p+k][r] * W[k2*n+i2][r] for r in range(rank))
                            error += total_spurious ** 2
        return error

    def compute_gradients(U, V, W):
        """Compute gradients for U, V, W."""
        rank = len(U[0])
        grad_U = [[0.0] * rank for _ in range(num_u)]
        grad_V = [[0.0] * rank for _ in range(num_v)]
        grad_W = [[0.0] * rank for _ in range(num_w)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        u = U[i*m+j][r]
                        v = V[j*p+k][r]
                        w = W[k*n+i][r]
                        
                        # Gradient from the main term
                        main_grad = 2 * (u * v * w - 1)
                        
                        # Gradient for U
                        grad_U[i*m+j][r] += main_grad * v * w
                        
                        # Gradient for V
                        grad_V[j*p+k][r] += main_grad * u * w
                        
                        # Gradient for W
                        grad_W[k*n+i][r] += main_grad * u * v
                        
                        # Gradients from spurious terms
                        for i2 in range(n):
                            for k2 in range(p):
                                if i2 == i and k2 == k:
                                    continue
                                w_spurious = W[k2*n+i2][r]
                                spurious_val = u * v * w_spurious
                                spurious_grad = 2 * spurious_val
                                
                                grad_U[i*m+j][r] += spurious_grad * v * w_spurious
                                grad_V[j*p+k][r] += spurious_grad * u * w_spurious
                                grad_W[k2*n+i2][r] += spurious_grad * u * v
        return grad_U, grad_V, grad_W

    def search_rank(R, max_iterations=3000):
        """Try to find a rank-R decomposition using gradient descent with restarts."""
        if time.time() > deadline - 10:
            return None
        
        best_result = None
        best_error = float('inf')
        
        # Multiple random restarts
        num_restarts = max(1, int((deadline - 10 - time.time()) / 5))
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with some structure
            # Use a mix of small integers and fractions
            U = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(R)] for _ in range(num_u)]
            V = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(R)] for _ in range(num_v)]
            W = [[random.choice([0, 1, -1, 0.5, -0.5]) for _ in range(R)] for _ in range(num_w)]
            
            # Ensure at least some non-zero entries
            for _ in range(R * 3):
                U[random.randint(0, num_u-1)][random.randint(0, R-1)] = random.uniform(-1, 1)
                V[random.randint(0, num_v-1)][random.randint(0, R-1)] = random.uniform(-1, 1)
                W[random.randint(0, num_w-1)][random.randint(0, R-1)] = random.uniform(-1, 1)
            
            current_error = compute_error(U, V, W)
            
            for iteration in range(max_iterations):
                if time.time() > deadline - 10:
                    break
                
                # Compute gradients
                grad_U, grad_V, grad_W = compute_gradients(U, V, W)
                
                # Adaptive learning rate
                lr = 0.05 / (1 + iteration * 0.005)
                
                # Update with momentum-like behavior
                for i in range(num_u):
                    for r in range(R):
                        U[i][r] -= lr * grad_U[i][r]
                
                for j in range(num_v):
                    for r in range(R):
                        V[j][r] -= lr * grad_V[j][r]
                
                for k in range(num_w):
                    for r in range(R):
                        W[k][r] -= lr * grad_W[k][r]
                
                # Compute new error
                new_error = compute_error(U, V, W)
                
                if new_error < current_error:
                    current_error = new_error
                    if current_error < best_error:
                        best_error = current_error
                        if current_error < 1e-10:
                            # Found a good solution, try to snap to rationals
                            U_snapped = []
                            for row in U:
                                U_snapped.append([float(x) for x in snap_to_rational(row)])
                            V_snapped = []
                            for row in V:
                                V_snapped.append([float(x) for x in snap_to_rational(row)])
                            W_snapped = []
                            for row in W:
                                W_snapped.append([float(x) for x in snap_to_rational(row)])
                            
                            # Verify with exact arithmetic
                            U_exact = [[Fraction(str(x)) for x in row] for row in U_snapped]
                            V_exact = [[Fraction(str(x)) for x in row] for row in V_snapped]
                            W_exact = [[Fraction(str(x)) for x in row] for row in W_snapped]
                            
                            if verify_decomposition(U_exact, V_exact, W_exact):
                                return {"U": U_snapped, "V": V_snapped, "W": W_snapped}
                else:
                    # Revert if error increased
                    # (In a full implementation, we'd keep the old values)
                    pass
                
                # Occasionally add noise to escape local minima
                if iteration % 200 == 0 and iteration > 0:
                    for _ in range(3):
                        U[random.randint(0, num_u-1)][random.randint(0, R-1)] += random.gauss(0, 0.2)
                        V[random.randint(0, num_v-1)][random.randint(0, R-1)] += random.gauss(0, 0.2)
                        W[random.randint(0, num_w-1)][random.randint(0, R-1)] += random.gauss(0, 0.2)
        
        return None

    # Try ranks from low to high
    best_result = None
    
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # Upper bound: schoolbook rank = 45
    # Try to find better than schoolbook
    for R in range(15, 46):
        if time.time() > deadline - 15:
            break
        result = search_rank(R, max_iterations=2000)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return get_schoolbook()
