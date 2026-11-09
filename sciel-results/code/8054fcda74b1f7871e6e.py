import numpy as np
import random
import time
import sys

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115
    
    def verify(U, V, W):
        """Verify if U, V, W form a valid decomposition."""
        # Check tensor identity: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
        # for all i in [0, n), j in [0, m), k in [0, p)
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m+j, :]
                    v_val = V[j*p+k, :]
                    w_val = W[k*n+i, :]
                    val = np.dot(u_val, v_val * w_val)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def get_violations(U, V, W):
        """Get list of (i,j,k) tuples that are violated."""
        violations = []
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m+j, :]
                    v_val = V[j*p+k, :]
                    w_val = W[k*n+i, :]
                    val = np.dot(u_val, v_val * w_val)
                    if abs(val - 1.0) > 1e-9:
                        violations.append((i, j, k))
        return violations
    
    def local_search(U, V, W, max_steps=200):
        """Perform local search to fix violations."""
        for step in range(max_steps):
            if verify(U, V, W):
                return True
            
            violations = get_violations(U, V, W)
            if not violations:
                return True
            
            # Pick a random violation
            i, j, k = random.choice(violations)
            
            # Try to fix by modifying entries involved in this (i,j,k) term
            candidates = []
            
            # Entries in U[i*m+j, :]
            for col in range(U.shape[1]):
                if U[i*m+j, col] != 0:
                    candidates.append(('U', i*m+j, col))
            
            # Entries in V[j*p+k, :]
            for col in range(V.shape[1]):
                if V[j*p+k, col] != 0:
                    candidates.append(('V', j*p+k, col))
            
            # Entries in W[k*n+i, :]
            for col in range(W.shape[1]):
                if W[k*n+i, col] != 0:
                    candidates.append(('W', k*n+i, col))
            
            if not candidates:
                # No non-zero entries, try adding a new one
                mat_name = random.choice(['U', 'V', 'W'])
                if mat_name == 'U':
                    row = i*m+j
                    col = random.randint(0, U.shape[1]-1)
                    U[row, col] += 1
                elif mat_name == 'V':
                    row = j*p+k
                    col = random.randint(0, V.shape[1]-1)
                    V[row, col] += 1
                else:
                    row = k*n+i
                    col = random.randint(0, W.shape[1]-1)
                    W[row, col] += 1
                continue
            
            # Try a random candidate
            mat_name, row, col = random.choice(candidates)
            if mat_name == 'U':
                U[row, col] += random.choice([-2, 2])
            elif mat_name == 'V':
                V[row, col] += random.choice([-2, 2])
            else:
                W[row, col] += random.choice([-2, 2])
        
        return verify(U, V, W)
    
    def init_structured(r):
        """Initialize with structured pattern based on schoolbook."""
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        # Use a subset of schoolbook terms
        all_terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        random.shuffle(all_terms)
        
        for idx, (i, j, k) in enumerate(all_terms[:r]):
            U[i*m+j, idx] = 1
            V[j*p+k, idx] = 1
            W[k*n+i, idx] = 1
        
        return U, V, W
    
    def init_random_sparse(r, density=0.3):
        """Initialize with sparse random values."""
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        for mat in [U, V, W]:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if random.random() < density:
                        mat[i, j] = random.choice([-1, 1])
        
        return U, V, W
    
    # Try to find a low rank decomposition starting from 14
    best_result = None
    
    for r in range(14, 31):
        if time.time() > deadline - 10:
            break
        
        max_attempts = 500 if r <= 15 else 200
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Choose initialization strategy
            if attempt % 3 == 0:
                U, V, W = init_structured(r)
            elif attempt % 3 == 1:
                U, V, W = init_random_sparse(r, density=0.25)
            else:
                # Hybrid: start structured, then add random noise
                U, V, W = init_structured(r)
                for _ in range(r):
                    U[random.randint(0, n*m-1), random.randint(0, r-1)] += random.choice([-1, 1])
                    V[random.randint(0, m*p-1), random.randint(0, r-1)] += random.choice([-1, 1])
                    W[random.randint(0, p*n-1), random.randint(0, r-1)] += random.choice([-1, 1])
            
            # Perform local search
            if local_search(U, V, W, max_steps=100):
                if verify(U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If no low rank found, return schoolbook
    rank = rank_schoolbook
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
