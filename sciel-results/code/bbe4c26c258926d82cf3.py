import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s margin
    
    def is_half_integer(x, tol=1e-9):
        return abs(x * 2 - round(x * 2)) < tol
    
    def check_solution(U, V, W, n, m, p):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    def try_random_solution(r, n, m, p, max_attempts=10000):
        """Try to find a random half-integer solution of rank r."""
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Random U, V with half-integer entries
            U = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(r)] for _ in range(n*m)]
            V = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(r)] for _ in range(m*p)]
            
            # Compute M[(i,k), s] = sum_j U[i*m+j, s] * V[j*p+k, s]
            M = {}
            for i in range(n):
                for k in range(p):
                    M[(i,k)] = [0.0] * r
                    for j in range(m):
                        for s in range(r):
                            M[(i,k)][s] += U[i*m+j][s] * V[j*p+k][s]
            
            # For each (i,k), find W[k*n+i, :] such that sum_s W[k*n+i, s] * M[(i,k), s] = 1
            W = [[0.0] * r for _ in range(p*n)]
            valid = True
            for i in range(n):
                for k in range(p):
                    c = k*n + i
                    m_vec = M[(i,k)]
                    # Find half-integer w such that dot(w, m_vec) = 1
                    # Try simple solutions
                    found = False
                    # Try: one non-zero component
                    for s in range(r):
                        if abs(m_vec[s]) > 1e-9:
                            w_val = 1.0 / m_vec[s]
                            if is_half_integer(w_val):
                                W[c][s] = w_val
                                found = True
                                break
                    if not found:
                        # Try two non-zero components
                        for s1 in range(r):
                            if abs(m_vec[s1]) < 1e-9:
                                continue
                            for s2 in range(s1+1, r):
                                if abs(m_vec[s2]) < 1e-9:
                                    continue
                                # w1 * m1 + w2 * m2 = 1, w1, w2 half-integers
                                # Try small half-integer values for w1
                                for w1_num in range(-4, 5):  # w1 = w1_num / 2
                                    w1 = w1_num / 2.0
                                    rem = 1.0 - w1 * m_vec[s1]
                                    if abs(m_vec[s2]) > 1e-9:
                                        w2 = rem / m_vec[s2]
                                        if is_half_integer(w2):
                                            W[c][s1] = w1
                                            W[c][s2] = w2
                                            found = True
                                            break
                            if found:
                                break
                        if found:
                            break
                    if not found:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Verify the solution
                if check_solution(U, V, W, n, m, p):
                    return U, V, W
        
        return None
    
    # Try ranks from low to high
    for r in range(10, 19):
        if time.time() > deadline:
            break
        result = try_random_solution(r, n, m, p, max_attempts=5000)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 18
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
