import numpy as np
import random
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 3]
GLOBAL_START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s wall-clock kill

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    def verify_full(U, V, W):
        """Full verification: check all entries of the tensor."""
        R = U.shape[1]
        # Check all valid (i,j,k) entries give 1
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
        # Check all off-diagonal entries give 0
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

    def verify_partial(U, V, W):
        """Fast partial verification: check valid entries + random off-diagonal sample."""
        R = U.shape[1]
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
        # Sample 300 random off-diagonal entries
        for _ in range(300):
            u = random.randint(0, nm - 1)
            v = random.randint(0, mp - 1)
            w = random.randint(0, pn - 1)
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

    def solve_for_W(U, V, R):
        """Given U and V, solve the linear system for W exactly."""
        # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k)
        # This is a linear system in the W variables.
        # There are n*m*p equations and pn*R unknowns.
        
        # Build the system: for each (i,j,k), equation is:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
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
            return None, None
        
        if len(residuals) > 0 and residuals[0] > 1e-8:
            return None, None
        
        W = w_flat.reshape(pn, R)
        return W, residuals[0] if len(residuals) > 0 else 0.0

    def try_rank_structured(R, time_budget=20.0):
        """Try to find a rank-R decomposition using structured random U,V."""
        start_time = time.time()
        max_attempts = 5000
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_budget or time.time() - GLOBAL_START_TIME > TIME_LIMIT:
                break
            
            # Strategy: generate sparse U and V with small integer values
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            
            for r in range(R):
                # Each column of U and V should be sparse
                # Try different sparsity patterns
                pattern = random.randint(0, 4)
                
                if pattern == 0:
                    # Very sparse: 1 nonzero each
                    u_idx = random.randint(0, nm - 1)
                    v_idx = random.randint(0, mp - 1)
                    U[u_idx, r] = random.choice([-1.0, 1.0])
                    V[v_idx, r] = random.choice([-1.0, 1.0])
                elif pattern == 1:
                    # 2 nonzeros in U, 1 in V
                    u_indices = random.sample(range(nm), 2)
                    v_idx = random.randint(0, mp - 1)
                    U[u_indices[0], r] = random.choice([-1.0, 1.0])
                    U[u_indices[1], r] = random.choice([-1.0, 1.0])
                    V[v_idx, r] = random.choice([-1.0, 1.0])
                elif pattern == 2:
                    # 1 nonzero in U, 2 in V
                    u_idx = random.randint(0, nm - 1)
                    v_indices = random.sample(range(mp), 2)
                    U[u_idx, r] = random.choice([-1.0, 1.0])
                    V[v_indices[0], r] = random.choice([-1.0, 1.0])
                    V[v_indices[1], r] = random.choice([-1.0, 1.0])
                elif pattern == 3:
                    # 2 nonzeros each
                    u_indices = random.sample(range(nm), 2)
                    v_indices = random.sample(range(mp), 2)
                    for idx in u_indices:
                        U[idx, r] = random.choice([-1.0, 1.0])
                    for idx in v_indices:
                        V[idx, r] = random.choice([-1.0, 1.0])
                else:
                    # 3 nonzeros each
                    u_indices = random.sample(range(nm), 3)
                    v_indices = random.sample(range(mp), 3)
                    for idx in u_indices:
                        U[idx, r] = random.choice([-1.0, 1.0])
                    for idx in v_indices:
                        V[idx, r] = random.choice([-1.0, 1.0])
            
            W, residual = solve_for_W(U, V, R)
            if W is None:
                continue
            
            # Try rounding to different precision levels
            for mult in [2.0, 4.0, 8.0]:
                W_rounded = np.round(W * mult) / mult
                
                if verify_partial(U, V, W_rounded):
                    if verify_full(U, V, W_rounded):
                        return U.tolist(), V.tolist(), W_rounded.tolist()
            
            # Also try without rounding (if values are already close to half-integers)
            if verify_partial(U, V, W):
                if verify_full(U, V, W):
                    return U.tolist(), V.tolist(), W.tolist()
        
        return None

    def try_rank_with_fixed_structure(R, time_budget=15.0):
        """Try with more structured patterns based on known constructions."""
        start_time = time.time()
        max_attempts = 3000
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_budget or time.time() - GLOBAL_START_TIME > TIME_LIMIT:
                break
            
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            
            for r in range(R):
                # Use a mix of structured and random
                if random.random() < 0.3:
                    # Structured: based on row/col patterns
                    u_row = random.randint(0, n - 1)
                    v_col = random.randint(0, p - 1)
                    U[u_row * m + random.randint(0, m - 1), r] = random.choice([-1.0, 1.0])
                    V[random.randint(0, m - 1) * p + v_col, r] = random.choice([-1.0, 1.0])
                else:
                    # Random sparse
                    n_u = random.randint(1, 3)
                    n_v = random.randint(1, 3)
                    u_indices = random.sample(range(nm), n_u)
                    v_indices = random.sample(range(mp), n_v)
                    for idx in u_indices:
                        U[idx, r] = random.choice([-1.0, 1.0])
                    for idx in v_indices:
                        V[idx, r] = random.choice([-1.0, 1.0])
            
            W, residual = solve_for_W(U, V, R)
            if W is None:
                continue
            
            for mult in [2.0, 4.0, 8.0]:
                W_rounded = np.round(W * mult) / mult
                if verify_partial(U, V, W_rounded):
                    if verify_full(U, V, W_rounded):
                        return U.tolist(), V.tolist(), W_rounded.tolist()
            
            if verify_partial(U, V, W):
                if verify_full(U, V, W):
                    return U.tolist(), V.tolist(), W.tolist()
        
        return None

    # Try ranks from 13 to 17, prioritizing lower ranks
    for R in range(13, 18):
        remaining = TIME_LIMIT - (time.time() - GLOBAL_START_TIME)
        if remaining < 5.0:
            break
        
        # Allocate time based on rank priority
        if R <= 15:
            time_budget = min(25.0, remaining - 5.0)
        else:
            time_budget = min(15.0, remaining - 5.0)
        
        result = try_rank_structured(R, time_budget=time_budget)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        remaining = TIME_LIMIT - (time.time() - GLOBAL_START_TIME)
        if remaining < 5.0:
            break
        
        time_budget = min(10.0, remaining - 5.0)
        result = try_rank_with_fixed_structure(R, time_budget=time_budget)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
        
        # Check time before next rank
        remaining = TIME_LIMIT - (time.time() - GLOBAL_START_TIME)
        if remaining < 5.0:
            break

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
