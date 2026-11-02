import time
import random
import numpy as np
from fractions import Fraction
from itertools import product

def _verify_exact(U, V, W, n, m, p):
    rank = len(U[0])
    nm, mp, pn = n * m, m * p, p * n
    for u in range(nm):
        i, j = u // m, u % m
        for v in range(mp):
            j_v, k = v // p, v % p
            for w in range(pn):
                i_w, k_w = w % n, w // n
                expected = 1 if (j == j_v and i == i_w and k == k_w) else 0
                val = Fraction(0)
                for c in range(rank):
                    val += U[u][c] * V[v][c] * W[w][c]
                if val != expected:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U); V = np.array(V); W = np.array(W)
    nm, mp, pn = n*m, m*p, p*n
    for u in range(nm):
        i, j = u // m, u % m
        for v in range(mp):
            j_v, k = v // p, v % p
            for w in range(pn):
                i_w, k_w = w % n, w // n
                expected = 1.0 if (j == j_v and i == i_w and k == k_w) else 0.0
                if abs(np.dot(U[u] * V[v], W[w]) - expected) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115
    n, m, p = 3, 5, 7
    nm, mp, pn = n*m, m*p, p*n
    
    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(nm)]
        V = [[0.0] * rank for _ in range(mp)]
        W = [[0.0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Discrete Local Search
    # Values allowed: {1, -1, 0.5, -0.5}
    # We search for a decomposition of rank r < 105.
    # We use a random restart local search on the discrete variables.
    
    def try_discrete_search(r, time_limit=5.0):
        if time.time() > deadline - 2.0:
            return None
            
        # Initialize with random sparse half-integer vectors
        # We represent U, V, W as lists of lists of Fractions for exactness in evaluation,
        # but for speed in search, we use floats and check exactness at the end.
        
        # To make it faster, we can restrict the support.
        # A term (c) is a triplet of vectors.
        # We will maintain U, V, W as numpy arrays of floats.
        
        best_error = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for restart in range(20):
            if time.time() > deadline - 2.0:
                break
                
            np.random.seed(42 + r * 100 + restart)
            
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            
            # Random sparse initialization
            for c in range(r):
                # Pick a few non-zeros
                nz_u = np.random.choice(nm, size=min(2, nm), replace=False)
                nz_v = np.random.choice(mp, size=min(2, mp), replace=False)
                nz_w = np.random.choice(pn, size=min(2, pn), replace=False)
                
                for idx in nz_u:
                    U[idx, c] = np.random.choice([0.5, 1.0, -0.5, -1.0])
                for idx in nz_v:
                    V[idx, c] = np.random.choice([0.5, 1.0, -0.5, -1.0])
                for idx in nz_w:
                    W[idx, c] = np.random.choice([0.5, 1.0, -0.5, -1.0])
            
            # Compute initial error
            # S = sum_c U[:,c] outer V[:,c] outer W[:,c]
            # This is expensive to compute from scratch every time.
            # We can update it incrementally if we change one vector.
            
            S = np.zeros((nm, mp, pn))
            for c in range(r):
                S += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            
            error = np.sum((T - S) ** 2)
            
            # Local Search: Randomly pick a term c, and a vector (U, V, or W),
            # and change one of its non-zero entries to a different half-integer value
            # or change its support.
            
            for step in range(1000):
                if time.time() > deadline - 2.0:
                    break
                
                c = np.random.randint(r)
                vec_type = np.random.randint(3) # 0: U, 1: V, 2: W
                
                if vec_type == 0:
                    vec = U[:, c]
                elif vec_type == 1:
                    vec = V[:, c]
                else:
                    vec = W[:, c]
                
                # Pick an index to change
                idx = np.random.randint(len(vec))
                old_val = vec[idx]
                
                # Pick a new value
                new_val = np.random.choice([0.0, 0.5, 1.0, -0.5, -1.0])
                if new_val == old_val:
                    continue
                
                # Calculate delta in S
                # Delta S = (new_val - old_val) * (other vectors outer product)
                
                if vec_type == 0:
                    # U changed
                    delta = (new_val - old_val) * np.einsum('j,k->ijk', V[:, c], W[:, c])
                elif vec_type == 1:
                    # V changed
                    delta = (new_val - old_val) * np.einsum('i,k->ijk', U[:, c], W[:, c])
                else:
                    # W changed
                    delta = (new_val - old_val) * np.einsum('i,j->ijk', U[:, c], V[:, c])
                
                new_S = S + delta
                new_error = np.sum((T - new_S) ** 2)
                
                if new_error < error:
                    # Accept
                    vec[idx] = new_val
                    S = new_S
                    error = new_error
                    if error < 1e-9:
                        # Check exactness
                        U_r = np.round(U * 2.0) / 2.0
                        V_r = np.round(V * 2.0) / 2.0
                        W_r = np.round(W * 2.0) / 2.0
                        
                        if _verify_np(U_r, V_r, W_r, n, m, p):
                            U_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in U_r]
                            V_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in V_r]
                            W_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in W_r]
                            if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
                                return {
                                    "U": [[float(x) for x in row] for row in U_frac],
                                    "V": [[float(x) for x in row] for row in V_frac],
                                    "W": [[float(x) for x in row] for row in W_frac]
                                }
                else:
                    # Reject (or accept with prob exp(-delta/T) if SA, but let's stick to Hill Climbing for simplicity)
                    pass
            
            if error < best_error:
                best_error = error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            # If best error is very small, try to polish
            if best_error < 1e-6:
                U_r = np.round(best_U * 2.0) / 2.0
                V_r = np.round(best_V * 2.0) / 2.0
                W_r = np.round(best_W * 2.0) / 2.0
                if _verify_np(U_r, V_r, W_r, n, m, p):
                    U_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in U_r]
                    V_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in V_r]
                    W_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in W_r]
                    if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
                        return {
                            "U": [[float(x) for x in row] for row in U_frac],
                            "V": [[float(x) for x in row] for row in V_frac],
                            "W": [[float(x) for x in row] for row in W_frac]
                        }

        return None

    # Try ranks from 104 down to 95
    for r in range(104, 94, -1):
        if time.time() > deadline:
            break
        result = try_discrete_search(r)
        if result is not None:
            return result

    return schoolbook()
