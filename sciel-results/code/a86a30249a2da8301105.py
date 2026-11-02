import numpy as np
from fractions import Fraction
import time
import random

def construct():
    n, m, p = 2, 3, 7
    rank_school = n * m * p  # 42
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin

    # Precompute valid indices
    valid_indices = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_indices.add((a, b, c))

    def verify_float(U, V, W, rank):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        T = np.zeros((n*m, m*p, p*n), dtype=float)
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            T += np.einsum('a,b,c->abc', u, v, w)
        
        # Check valid
        for (a, b, c) in valid_indices:
            if abs(T[a, b, c] - 1.0) > 1e-6:
                return False
        
        # Check invalid
        mask = np.zeros((n*m, m*p, p*n), dtype=bool)
        for (a, b, c) in valid_indices:
            mask[a, b, c] = True
        invalid_vals = T[~mask]
        if np.any(np.abs(invalid_vals) > 1e-6):
            return False
        return True

    def verify_exact(U, V, W, rank):
        Uf = [[Fraction(U[i][r]) for r in range(rank)] for i in range(n*m)]
        Vf = [[Fraction(V[i][r]) for r in range(rank)] for i in range(m*p)]
        Wf = [[Fraction(W[i][r]) for r in range(rank)] for i in range(p*n)]
        
        for (a, b, c) in valid_indices:
            s = Fraction(0)
            for r in range(rank):
                s += Uf[a][r] * Vf[b][r] * Wf[c][r]
            if s != 1:
                return False
        
        # Check invalids
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    if (a, b, c) in valid_indices:
                        continue
                    s = Fraction(0)
                    for r in range(rank):
                        s += Uf[a][r] * Vf[b][r] * Wf[c][r]
                    if s != 0:
                        return False
        return True

    # Initialize Schoolbook
    U_sb = [[0]*rank_school for _ in range(n*m)]
    V_sb = [[0]*rank_school for _ in range(m*p)]
    W_sb = [[0]*rank_school for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j
                b = j*p+k
                c = k*n+i
                U_sb[a][col] = 1
                V_sb[b][col] = 1
                W_sb[c][col] = 1
                col += 1
                
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_school

    # Strategy: Use a more structured approach based on known matrix multiplication algorithms
    # For 2x3 * 3x7, we can try to use a combination of Strassen-like ideas
    
    # Let's try a different approach: use the fact that we can decompose the problem
    # into smaller subproblems and use known low-rank decompositions
    
    # For now, let's enhance the random search with a better initialization strategy
    # and try to find solutions with ranks below 42
    
    # Try ranks from 41 down to 35
    for target_rank in range(41, 34, -1):
        if time.time() > deadline - 15.0:
            break
            
        # Number of iterations - be conservative
        iters = 2000
        for _ in range(iters):
            if time.time() > deadline - 10.0:
                break
                
            # Use a smarter initialization: start with schoolbook and randomly perturb
            U = [[0]*target_rank for _ in range(n*m)]
            V = [[0]*target_rank for _ in range(m*p)]
            W = [[0]*target_rank for _ in range(p*n)]
            
            # Initialize with a subset of schoolbook terms
            num_init = min(target_rank, 42)
            for r in range(num_init):
                # Pick random (i,j,k) for this term
                i = random.randint(0, n-1)
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                a = i*m+j
                b = j*p+k
                c = k*n+i
                U[a][r] = 1
                V[b][r] = 1
                W[c][r] = 1
            
            # Add some random perturbation to try to merge terms
            for r in range(num_init, target_rank):
                for a in range(n*m):
                    if random.random() < 0.3:
                        U[a][r] = random.choice([-1, -0.5, 0, 0.5, 1])
                for b in range(m*p):
                    if random.random() < 0.3:
                        V[b][r] = random.choice([-1, -0.5, 0, 0.5, 1])
                for c in range(p*n):
                    if random.random() < 0.3:
                        W[c][r] = random.choice([-1, -0.5, 0, 0.5, 1])
            
            if verify_float(U, V, W, target_rank):
                if verify_exact(U, V, W, target_rank):
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W
                    break
        
        # If we found a solution at target_rank, continue to target_rank-1
        if best_rank == target_rank:
            continue
        else:
            break

    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
