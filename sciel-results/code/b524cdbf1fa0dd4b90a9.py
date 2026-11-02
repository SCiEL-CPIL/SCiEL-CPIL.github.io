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
        # For any (i, j, k) and (i', k') where (i', k') != (i, k),
        # the sum should be 0
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
    
    def snap_to_rational(values, max_denom=4):
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
    
    def search_rank(R, max_iterations=5000):
        """Try to find a rank-R decomposition using randomized local search."""
        if time.time() > deadline - 10:
            return None
        
        # Initialize with random values
        for attempt in range(20):
            if time.time() > deadline - 10:
                return None
            
            # Random initialization
            U = [[random.uniform(-2, 2) for _ in range(R)] for _ in range(num_u)]
            V = [[random.uniform(-2, 2) for _ in range(R)] for _ in range(num_v)]
            W = [[random.uniform(-2, 2) for _ in range(R)] for _ in range(num_w)]
            
            best_error = float('inf')
            
            for iteration in range(max_iterations):
                if time.time() > deadline - 10:
                    break
                
                # Compute current error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(R))
                            error += (total - 1) ** 2
                            # Check spurious terms
                            for i2 in range(n):
                                for k2 in range(p):
                                    if i2 == i and k2 == k:
                                        continue
                                    total_spurious = sum(U[i*m+j][r] * V[j*p+k][r] * W[k2*n+i2][r] for r in range(R))
                                    error += total_spurious ** 2
                
                if error < best_error:
                    best_error = error
                    if error < 1e-12:
                        # Found a solution, snap to rationals
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
                
                # Gradient-based update (simplified)
                lr = 0.01 / (1 + iteration * 0.001)
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        for r in range(R):
                            grad = 0
                            for k in range(p):
                                grad += 2 * (U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] - 1) * V[j*p+k][r] * W[k*n+i][r]
                                for i2 in range(n):
                                    for k2 in range(p):
                                        if i2 == i and k2 == k:
                                            continue
                                        grad += 2 * U[i*m+j][r] * V[j*p+k][r] * W[k2*n+i2][r] * V[j*p+k][r] * W[k2*n+i2][r]
                            U[i*m+j][r] -= lr * grad
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        for r in range(R):
                            grad = 0
                            for i in range(n):
                                grad += 2 * (U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] - 1) * U[i*m+j][r] * W[k*n+i][r]
                                for i2 in range(n):
                                    for k2 in range(p):
                                        if i2 == i and k2 == k:
                                            continue
                                        grad += 2 * U[i*m+j][r] * V[j*p+k][r] * W[k2*n+i2][r] * U[i*m+j][r] * W[k2*n+i2][r]
                            V[j*p+k][r] -= lr * grad
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        for r in range(R):
                            grad = 0
                            for j in range(m):
                                grad += 2 * (U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] - 1) * U[i*m+j][r] * V[j*p+k][r]
                                for i2 in range(n):
                                    for k2 in range(p):
                                        if i2 == i and k2 == k:
                                            continue
                                        grad += 2 * U[i*m+j][r] * V[j*p+k][r] * W[k2*n+i2][r] * U[i*m+j][r] * V[j*p+k][r]
                            W[k*n+i][r] -= lr * grad
                
                # Occasionally add noise to escape local minima
                if iteration % 100 == 0:
                    for r in range(R):
                        U[random.randint(0, num_u-1)][r] += random.gauss(0, 0.1)
                        V[random.randint(0, num_v-1)][r] += random.gauss(0, 0.1)
                        W[random.randint(0, num_w-1)][r] += random.gauss(0, 0.1)
        
        return None
    
    # Try ranks from low to high
    best_result = None
    for R in range(20, 46):
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

# EVOLVE-BLOCK-END
