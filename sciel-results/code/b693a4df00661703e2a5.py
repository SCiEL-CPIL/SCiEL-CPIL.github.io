import numpy as np
import random
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 3]
GLOBAL_START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s wall-clock kill

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 9
    pn = p * n  # 6

    def verify_full(U, V, W):
        """Full verification: check all entries of the tensor."""
        R = U.shape[1]
        # Check valid entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = 0.0
                    for r in range(R):
                        val += U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check off-diagonal entries
        for u in range(nm):
            for v in range(mp):
                for w in range(pn):
                    i = u // m
                    u_j = u % m
                    v_j = v // p
                    k_v = v % p
                    k_w = w // n
                    i_w = w % n
                    if i == i_w and u_j == v_j and k_v == k_w:
                        continue
                    val = 0.0
                    for r in range(R):
                        val += U[u, r] * V[v, r] * W[w, r]
                    if abs(val) > 1e-9:
                        return False
        return True

    def try_find_decomposition(R, time_budget=30.0):
        """Try to find a rank-R decomposition using random U,V and solving for W."""
        start_time = time.time()
        max_attempts = 50000
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_budget or time.time() - GLOBAL_START_TIME > TIME_LIMIT:
                break
            
            # Generate random U and V with small integer/half-integer entries
            # Use sparse patterns to make the system more likely to be solvable
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            
            for r in range(R):
                # Random pattern for U: each row gets 1-2 non-zero entries
                for idx in range(nm):
                    if random.random() < 0.4:
                        U[idx, r] = random.choice([-2.0, -1.0, 1.0, 2.0])
                # Random pattern for V
                for idx in range(mp):
                    if random.random() < 0.4:
                        V[idx, r] = random.choice([-2.0, -1.0, 1.0, 2.0])
            
            # Solve for W using the valid equations
            num_eq = n * m * p  # 18
            num_unk = pn * R  # 6*R
            
            A = np.zeros((num_eq, num_unk))
            b = np.ones(num_eq)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for r in range(R):
                            coeff = U[i * m + j, r] * V[j * p + k, r]
                            if coeff != 0:
                                w_row = k * n + i
                                A[eq_idx, r * pn + w_row] = coeff
            
            try:
                w_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            if len(residuals) > 0 and residuals[0] > 1e-8:
                continue
            
            W = w_flat.reshape(pn, R)
            
            # Try rounding to half-integers
            for mult in [1.0, 2.0, 4.0]:
                W_rounded = np.round(W * mult) / mult
                
                # Quick check: verify valid entries
                valid_ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            val = 0.0
                            for r in range(R):
                                val += U[idx_u, r] * V[idx_v, r] * W_rounded[idx_w, r]
                            if abs(val - 1.0) > 1e-9:
                                valid_ok = False
                                break
                        if not valid_ok:
                            break
                    if not valid_ok:
                        break
                
                if not valid_ok:
                    continue
                
                # Full verification
                if verify_full(U, V, W_rounded):
                    return U.tolist(), V.tolist(), W_rounded.tolist()
            
            # Also try without rounding
            if verify_full(U, V, W):
                return U.tolist(), V.tolist(), W.tolist()
        
        return None

    def try_structured_search(R, time_budget=30.0):
        """Try structured patterns for rank-R decomposition."""
        start_time = time.time()
        max_attempts = 20000
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_budget or time.time() - GLOBAL_START_TIME > TIME_LIMIT:
                break
            
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            
            # Structured approach: each term covers a specific pattern
            for r in range(R):
                term_type = random.randint(0, 4)
                
                if term_type == 0:
                    # Single (i,j,k) with sign
                    i = random.randint(0, n-1)
                    j = random.randint(0, m-1)
                    k = random.randint(0, p-1)
                    s = random.choice([-1.0, 1.0])
                    U[i*m+j, r] = s
                    V[j*p+k, r] = s
                elif term_type == 1:
                    # Two U entries for same j, different i
                    j = random.randint(0, m-1)
                    k = random.randint(0, p-1)
                    s1 = random.choice([-1.0, 1.0])
                    s2 = random.choice([-1.0, 1.0])
                    U[0*m+j, r] = s1
                    U[1*m+j, r] = s2
                    V[j*p+k, r] = 1.0
                elif term_type == 2:
                    # One U, two V entries for same j, different k
                    i = random.randint(0, n-1)
                    j = random.randint(0, m-1)
                    s1 = random.choice([-1.0, 1.0])
                    s2 = random.choice([-1.0, 1.0])
                    s3 = random.choice([-1.0, 1.0])
                    U[i*m+j, r] = s1
                    V[j*p+0, r] = s2
                    V[j*p+1, r] = s3
                elif term_type == 3:
                    # Two U, two V
                    j = random.randint(0, m-1)
                    s1 = random.choice([-1.0, 1.0])
                    s2 = random.choice([-1.0, 1.0])
                    s3 = random.choice([-1.0, 1.0])
                    s4 = random.choice([-1.0, 1.0])
                    U[0*m+j, r] = s1
                    U[1*m+j, r] = s2
                    V[j*p+0, r] = s3
                    V[j*p+1, r] = s4
                else:
                    # Three V entries
                    i = random.randint(0, n-1)
                    j = random.randint(0, m-1)
                    s1 = random.choice([-1.0, 1.0])
                    s2 = random.choice([-1.0, 1.0])
                    s3 = random.choice([-1.0, 1.0])
                    s4 = random.choice([-1.0, 1.0])
                    U[i*m+j, r] = s1
                    V[j*p+0, r] = s2
                    V[j*p+1, r] = s3
                    V[j*p+2, r] = s4
            
            # Solve for W
            num_eq = n * m * p
            num_unk = pn * R
            
            A = np.zeros((num_eq, num_unk))
            b = np.ones(num_eq)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for r in range(R):
                            coeff = U[i * m + j, r] * V[j * p + k, r]
                            if coeff != 0:
                                w_row = k * n + i
                                A[eq_idx, r * pn + w_row] = coeff
            
            try:
                w_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            if len(residuals) > 0 and residuals[0] > 1e-8:
                continue
            
            W = w_flat.reshape(pn, R)
            
            for mult in [1.0, 2.0]:
                W_rounded = np.round(W * mult) / mult
                if verify_full(U, V, W_rounded):
                    return U.tolist(), V.tolist(), W_rounded.tolist()
            
            if verify_full(U, V, W):
                return U.tolist(), V.tolist(), W.tolist()
        
        return None

    # Try ranks from 13 to 17
    for R in [13, 14, 15, 16, 17]:
        remaining = TIME_LIMIT - (time.time() - GLOBAL_START_TIME)
        if remaining < 10.0:
            break
        
        time_budget = min(20.0, remaining - 5.0)
        
        # Try structured search first
        result = try_structured_search(R, time_budget=time_budget)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        # Try random search
        remaining = TIME_LIMIT - (time.time() - GLOBAL_START_TIME)
        if remaining < 10.0:
            break
        time_budget = min(15.0, remaining - 5.0)
        result = try_find_decomposition(R, time_budget=time_budget)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank-18
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
