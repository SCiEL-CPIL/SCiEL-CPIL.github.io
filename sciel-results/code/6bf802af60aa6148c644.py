import time
import numpy as np
import random
from math import gcd
from functools import reduce

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook fallback
    def get_schoolbook():
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

    def validate(U, V, W):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i*m+j]
                    v_val = V_arr[j*p+k]
                    w_val = W_arr[k*n+i]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def egcd(a, b):
        if b == 0:
            return (a, 1, 0)
        else:
            g, x, y = egcd(b, a % b)
            return (g, y, x - (a // b) * y)

    def solve_diophantine_target4(coeffs):
        """
        Find integer x such that dot(coeffs, x) = 4.
        Assumes gcd(coeffs) divides 4.
        Returns None if no solution found (should not happen if gcd check passed).
        """
        # Try to find a solution with 1 non-zero entry
        for r in range(len(coeffs)):
            if coeffs[r] != 0 and 4 % abs(coeffs[r]) == 0:
                x_vec = [0] * len(coeffs)
                x_vec[r] = 4 // coeffs[r]
                return x_vec
        
        # Try to find a solution with 2 non-zero entries
        for r1 in range(len(coeffs)):
            if coeffs[r1] == 0: continue
            for r2 in range(r1 + 1, len(coeffs)):
                if coeffs[r2] == 0: continue
                g = gcd(abs(coeffs[r1]), abs(coeffs[r2]))
                if 4 % g == 0:
                    # Find a, b such that a*c1 + b*c2 = 4
                    g, x, y = egcd(abs(coeffs[r1]), abs(coeffs[r2]))
                    scale = 4 // g
                    a = x * scale
                    if coeffs[r1] < 0: a = -a
                    b = y * scale
                    if coeffs[r2] < 0: b = -b
                    
                    x_vec = [0] * len(coeffs)
                    x_vec[r1] = a
                    x_vec[r2] = b
                    if sum(x_vec[i] * coeffs[i] for i in range(len(coeffs))) == 4:
                        return x_vec
        return None

    def try_structured_search(target_rank, max_attempts=20000):
        # Pre-compute indices for faster access
        um_indices = [(i, j) for i in range(n) for j in range(m)]
        pm_indices = [(j, k) for j in range(m) for k in range(p)]
        pk_indices = [(k, i) for k in range(p) for i in range(n)]
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Generate random U', V' with small integers
            # Using a range of -2 to 2 to keep numbers small and increase GCD divisibility chances
            # The current range -4 to 4 produces larger C values, making gcd(C) less likely to divide 4
            U_prime = np.random.randint(-2, 3, size=(n*m, target_rank))
            V_prime = np.random.randint(-2, 3, size=(m*p, target_rank))
            
            # Calculate C[i, k, r] = sum_j U_prime[i*m+j, r] * V_prime[j*p+k, r]
            C = np.zeros((n, p, target_rank), dtype=int)
            for i in range(n):
                for k in range(p):
                    for r in range(target_rank):
                        s = 0
                        for j in range(m):
                            s += U_prime[i*m+j, r] * V_prime[j*p+k, r]
                        C[i, k, r] = s
            
            # Check GCD condition: gcd(C[i,k,:]) must divide 4
            valid = True
            W_prime = np.zeros((p*n, target_rank), dtype=int)
            
            for i in range(n):
                for k in range(p):
                    coeffs = C[i, k, :]
                    non_zero = [abs(c) for c in coeffs if c != 0]
                    if not non_zero:
                        valid = False
                        break
                    g = reduce(gcd, non_zero)
                    if 4 % g != 0:
                        valid = False
                        break
                    
                    # Solve for W_prime
                    w_vec = solve_diophantine_target4(coeffs)
                    if w_vec is None:
                        valid = False
                        break
                    
                    for r in range(target_rank):
                        W_prime[k*n+i, r] = w_vec[r]
                if not valid:
                    break
            
            if valid:
                # Convert back to U, V, W (divide by 2)
                U_list = (U_prime / 2.0).tolist()
                V_list = (V_prime / 2.0).tolist()
                W_list = (W_prime / 2.0).tolist()
                
                if validate(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try ranks from lower bound upwards
    # Schoolbook is 30. Let's try 29 down to 25 first.
    # If we find a solution, we return it.
    # Allocate more time to lower ranks (harder to find but better score)
    # Use a time-based loop instead of fixed attempts to maximize coverage
    for rank in range(29, 24, -1):
        if time.time() > deadline - 5:
            break
        
        # Give more time to rank 29 (easiest sub-schoolbook), less to higher ranks?
        # Actually, lower rank is better. But rank 29 is the first one to try.
        # Let's distribute time proportionally to remaining time.
        # Or just try each rank for a fixed duration.
        
        # Let's try rank 29 for 10s, 28 for 10s, etc.
        # But we have 140s total.
        # Let's just run the search for a fixed number of attempts that fits in ~20s per rank.
        # The current max_attempts=10000 might be too few or too many depending on speed.
        
        # Let's increase attempts significantly for the first few ranks.
        attempts = 50000 if rank >= 27 else 20000
        result = try_structured_search(rank, max_attempts=attempts)
        if result is not None:
            return result

    # Fallback to schoolbook
    return get_schoolbook()
