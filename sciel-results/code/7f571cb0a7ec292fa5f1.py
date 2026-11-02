"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
import itertools

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

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

    def verify_numpy(U, V, W, n, m, p, tol=1e-8):
        """Fast verification using NumPy - checks all entries."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        
        # Check valid entries = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U_np[a] * V_np[b] * W_np[c])
                    if abs(val - 1) > tol:
                        return False
        
        # Check invalid entries = 0
        for a in range(n * m):
            i_a = a // m
            j_a = a % m
            for b in range(m * p):
                j_b = b // p
                k_b = b % p
                if j_a != j_b:
                    continue
                for c in range(p * n):
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        continue
                    val = np.sum(U_np[a] * V_np[b] * W_np[c])
                    if abs(val) > tol:
                        return False
        return True

    def verify_exact(U, V, W, n, m, p):
        """Verify with exact Fraction arithmetic - only for final confirmation."""
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

    def try_rational_approx(U_float, V_float, W_float, n, m, p, max_denom=16):
        """Try to find rational approximations with small denominators."""
        def approx_rational(x, max_d):
            if abs(x) < 1e-10:
                return Fraction(0)
            for d in range(1, max_d + 1):
                num = round(x * d)
                if abs(x - num / d) < 1e-9:
                    return Fraction(num, d)
            return Fraction(x).limit_denominator(max_d * max_d)
        
        U_r = [[approx_rational(float(x), max_denom) for x in row] for row in U_float]
        V_r = [[approx_rational(float(x), max_denom) for x in row] for row in V_float]
        W_r = [[approx_rational(float(x), max_denom) for x in row] for row in W_float]
        if verify_exact(U_r, V_r, W_r, n, m, p):
            return {"U": [[float(f) for f in row] for row in U_r],
                    "V": [[float(f) for f in row] for row in V_r],
                    "W": [[float(f) for f in row] for row in W_r]}
        return None

    # === Structured parametric search exploiting block structure ===
    # Key insight: The tensor T[a,b,c] = 1 iff a=(i,j), b=(j,k), c=(k,i)
    # We can parameterize with shared parameters across blocks.
    
    # Ansatz: Use a "Kronecker-like" structure where:
    # U[i*m+j, r] = f_i(j, r) for some parameterization
    # V[j*p+k, r] = g_j(k, r) 
    # W[k*n+i, r] = h_k(i, r)
    
    # More concretely, try: rank = m*p + n*p + n*m - overlap
    # Or use the fact that we can write:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # as a sum of rank-1 outer products with shared intermediate values.
    
    def parametric_search(rank, seed=42):
        """
        Try to find a rank decomposition using a parametric approach.
        We parameterize U, V, W with a smaller set of free parameters
        and solve the resulting system.
        """
        if time.time() > deadline - 10.0:
            return None
        
        np.random.seed(seed)
        
        # Strategy: Random structured initialization + ALS with small constraints
        # But faster: use the fact that each row of U only interacts with
        # specific rows of V and W.
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Initialize with small random values
        for i in range(n):
            for j in range(m):
                U[i*m+j] = np.random.randn(rank) * 0.3
        for j in range(m):
            for k in range(p):
                V[j*p+k] = np.random.randn(rank) * 0.3
        for k in range(p):
            for i in range(n):
                W[k*n+i] = np.random.randn(rank) * 0.3
        
        # Precompute valid triples
        valid_abc = [(i*m+j, j*p+k, k*n+i) for i in range(n) for j in range(m) for k in range(p)]
        valid_set = set(valid_abc)
        
        # Fast ALS: for each row, solve a small linear system
        for it in range(100):
            if time.time() > deadline - 10.0:
                return None
            
            # Update U rows
            for a in range(n * m):
                i_a = a // m
                j_a = a % m
                # Constraints: for each (b,c) where (a,b,c) is valid or invalid
                # sum_r U[a,r] * V[b,r] * W[c,r] = target
                # This is: U[a] @ (V[b] * W[c]) = target
                # Stack all constraints for this a
                
                A_rows = []
                y_vals = []
                for b in range(m * p):
                    for c in range(p * n):
                        A_rows.append(V[b] * W[c])
                        if (a, b, c) in valid_set:
                            y_vals.append(1.0)
                        else:
                            y_vals.append(0.0)
                
                if A_rows:
                    A_mat = np.array(A_rows)
                    y_vec = np.array(y_vals)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_mat, y_vec, rcond=None)
                        U[a] = sol
                    except:
                        pass
            
            # Update V rows
            for b in range(m * p):
                A_rows = []
                y_vals = []
                for a in range(n * m):
                    for c in range(p * n):
                        A_rows.append(U[a] * W[c])
                        if (a, b, c) in valid_set:
                            y_vals.append(1.0)
                        else:
                            y_vals.append(0.0)
                
                if A_rows:
                    A_mat = np.array(A_rows)
                    y_vec = np.array(y_vals)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_mat, y_vec, rcond=None)
                        V[b] = sol
                    except:
                        pass
            
            # Update W rows
            for c in range(p * n):
                A_rows = []
                y_vals = []
                for a in range(n * m):
                    for b in range(m * p):
                        A_rows.append(U[a] * V[b])
                        if (a, b, c) in valid_set:
                            y_vals.append(1.0)
                        else:
                            y_vals.append(0.0)
                
                if A_rows:
                    A_mat = np.array(A_rows)
                    y_vec = np.array(y_vals)
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_mat, y_vec, rcond=None)
                        W[c] = sol
                    except:
                        pass
            
            # Check loss (fast)
            loss = 0.0
            for (a, b, c) in valid_abc:
                val = np.dot(U[a], V[b] * W[c])
                loss += (val - 1)**2
            # Sample some invalid entries for speed
            invalid_sample = []
            for _ in range(50):
                a = random.randint(0, n*m-1)
                b = random.randint(0, m*p-1)
                c = random.randint(0, p*n-1)
                if (a, b, c) not in valid_set:
                    invalid_sample.append((a, b, c))
            for (a, b, c) in invalid_sample:
                val = np.dot(U[a], V[b] * W[c])
                loss += val**2
            
            if loss < 1e-10:
                result = try_rational_approx(U.tolist(), V.tolist(), W.tolist(), n, m, p, max_denom=32)
                if result:
                    return result
        
        return None

    # === Direct constructive approach for specific ranks ===
    # For 2x3 x 3x7, try to construct explicit low-rank decompositions.
    
    def try_construction(rank):
        """Try to construct a decomposition with the given rank using known patterns."""
        if time.time() > deadline - 10.0:
            return None
        
        # Pattern: For each j (0..m-1), we have n*p terms.
        # If we can share across j values, we reduce rank.
        # 
        # Idea: C[i,k] = sum_j A[i,j]*B[j,k]
        # Write A[i,j]*B[j,k] as part of a larger product.
        # 
        # For m=3, n=2, p=7:
        # Total terms = 42. Lower bound = max(6, 21, 14) = 21.
        # 
        # Try rank = 21, 22, ..., 41
        
        # Use random search with structured initialization
        for seed in range(20):
            if time.time() > deadline - 10.0:
                return None
            result = parametric_search(rank, seed=seed * 1000 + rank)
            if result is not None:
                return result
        return None

    # Try ranks from lower bound upward
    best = schoolbook()
    best_rank = n * m * p
    
    # Lower bound: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    lb = max(n*m, m*p, p*n)
    
    for r in range(lb, best_rank):
        if time.time() > deadline - 15.0:
            break
        result = try_construction(r)
        if result is not None:
            best = result
            best_rank = r
            break
    
    return best
# EVOLVE-BLOCK-END
