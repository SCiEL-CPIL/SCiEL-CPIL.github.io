"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 135.0  # safety margin before 150s kill

    # Schoolbook decomposition of rank n*m*p
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

    best = schoolbook()
    best_rank = n * m * p

    # Nonzero entries of the tensor
    nz = [(i*m+j, j*p+k, k*n+i) for i in range(n) for j in range(m) for k in range(p)]
    nz_set = set(nz)
    A_dim = n * m
    B_dim = m * p
    C_dim = p * n

    # Target tensor as a 3D array
    T = np.zeros((A_dim, B_dim, C_dim))
    for (a, b, c) in nz:
        T[a, b, c] = 1.0

    # Verify a decomposition using exact Fraction arithmetic
    def verify_exact(U, V, W):
        rank = len(U[0])
        # Convert to Fractions for exactness
        Uf = [[Fraction(U[a][t]) for t in range(rank)] for a in range(A_dim)]
        Vf = [[Fraction(V[b][t]) for t in range(rank)] for b in range(B_dim)]
        Wf = [[Fraction(W[c][t]) for t in range(rank)] for c in range(C_dim)]
        
        # Check all required entries equal 1
        for (a, b, c) in nz:
            total = Fraction(0)
            for t in range(rank):
                total += Uf[a][t] * Vf[b][t] * Wf[c][t]
            if total != 1:
                return False
        # Check all other entries equal 0
        for a in range(A_dim):
            for b in range(B_dim):
                for c in range(C_dim):
                    if (a, b, c) not in nz_set:
                        total = Fraction(0)
                        for t in range(rank):
                            total += Uf[a][t] * Vf[b][t] * Wf[c][t]
                        if total != 0:
                            return False
        return True

    # Fast numeric verification
    def verify_numeric(U, V, W, tol=1e-9):
        rank = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        # Compute tensor reconstruction
        # T_recon[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
        # Efficiently: for each t, add outer product
        # But for verification, just check specific entries
        for (a, b, c) in nz:
            total = 0.0
            for t in range(rank):
                total += Ua[a, t] * Va[b, t] * Wa[c, t]
            if abs(total - 1.0) > tol:
                return False
        # Sample check for zeros (full check is expensive)
        # Do a full check for small tensors
        for a in range(A_dim):
            for b in range(B_dim):
                for c in range(C_dim):
                    if (a, b, c) not in nz_set:
                        total = 0.0
                        for t in range(rank):
                            total += Ua[a, t] * Va[b, t] * Wa[c, t]
                        if abs(total) > tol:
                            return False
        return True

    # Greedy construction: iteratively add rank-1 tensors to approximate T
    # Coefficient set for entries
    coeff_set = [0, 1, -1]

    def greedy_construct(max_rank, seed=42):
        """Greedy rank-1 tensor selection to approximate T."""
        if time.time() > deadline - 15:
            return None
        
        rng = np.random.RandomState(seed)
        residual = T.copy()
        U_list = []
        V_list = []
        W_list = []
        
        for step in range(max_rank):
            if time.time() > deadline - 15:
                return None
            
            best_score = -1
            best_u = None
            best_v = None
            best_w = None
            
            # Try all combinations of coefficient vectors (with sparsity)
            # For efficiency, sample random sparse vectors
            n_trials = 2000
            for trial in range(n_trials):
                if time.time() > deadline - 15:
                    break
                
                # Random sparse vectors
                u = np.zeros(A_dim)
                v = np.zeros(B_dim)
                w = np.zeros(C_dim)
                
                # Set a few random entries
                nu = rng.randint(1, 4)
                nv = rng.randint(1, 4)
                nw = rng.randint(1, 4)
                
                for _ in range(nu):
                    idx = rng.randint(0, A_dim)
                    u[idx] = rng.choice(coeff_set)
                for _ in range(nv):
                    idx = rng.randint(0, B_dim)
                    v[idx] = rng.choice(coeff_set)
                for _ in range(nw):
                    idx = rng.randint(0, C_dim)
                    w[idx] = rng.choice(coeff_set)
                
                # Skip if any vector is all zeros
                if np.all(u == 0) or np.all(v == 0) or np.all(w == 0):
                    continue
                
                # Compute score: how well does this rank-1 tensor match the residual?
                # score = <residual, u⊗v⊗w> / ||u⊗v⊗w||^2
                # = sum_{a,b,c} residual[a,b,c]*u[a]*v[b]*w[c] / (||u||^2 * ||v||^2 * ||w||^2)
                
                num = 0.0
                # Efficient computation: sum over nonzero entries
                for a in np.nonzero(u)[0]:
                    for b in np.nonzero(v)[0]:
                        for c in np.nonzero(w)[0]:
                            num += residual[a, b, c] * u[a] * v[b] * w[c]
                
                den = np.sum(u*u) * np.sum(v*v) * np.sum(w*w)
                if den == 0:
                    continue
                
                score = abs(num) / den
                if score > best_score:
                    best_score = score
                    best_u = u.copy()
                    best_v = v.copy()
                    best_w = w.copy()
            
            if best_u is None or best_score < 1e-12:
                break
            
            # Add this rank-1 tensor
            U_list.append(best_u.tolist())
            V_list.append(best_v.tolist())
            W_list.append(best_w.tolist())
            
            # Update residual
            for a in range(A_dim):
                for b in range(B_dim):
                    for c in range(C_dim):
                        residual[a, b, c] -= best_u[a] * best_v[b] * best_w[c]
            
            # Check if residual is (approximately) zero
            if np.max(np.abs(residual)) < 1e-9:
                break
        
        rank = len(U_list)
        if rank == 0:
            return None
        
        # Build U, V, W matrices (transposed: rows are indices, cols are rank)
        U = [[U_list[t][a] for t in range(rank)] for a in range(A_dim)]
        V = [[V_list[t][b] for t in range(rank)] for b in range(B_dim)]
        W = [[W_list[t][c] for t in range(rank)] for c in range(C_dim)]
        
        # Verify
        if verify_numeric(U, V, W):
            return {"U": U, "V": V, "W": W}
        
        return None

    # Try greedy construction with various max ranks
    for max_rank in range(45, 61):
        if time.time() > deadline - 20:
            break
        
        result = greedy_construct(max_rank, seed=42)
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank:
                # Try to verify exactly
                if verify_exact(result["U"], result["V"], result["W"]):
                    best = result
                    best_rank = rank
                else:
                    # Try rounding to integers
                    U_r = [[round(x) for x in row] for row in result["U"]]
                    V_r = [[round(x) for x in row] for row in result["V"]]
                    W_r = [[round(x) for x in row] for row in result["W"]]
                    if verify_exact(U_r, V_r, W_r):
                        best = {"U": U_r, "V": V_r, "W": W_r}
                        best_rank = rank
            break
    
    # If greedy didn't find anything better, try a few more random restarts
    if best_rank == n * m * p:
        for seed in [1, 7, 13, 21, 33]:
            if time.time() > deadline - 10:
                break
            result = greedy_construct(55, seed=seed)
            if result is not None:
                rank = len(result["U"][0])
                if rank < best_rank:
                    if verify_exact(result["U"], result["V"], result["W"]):
                        best = result
                        best_rank = rank
                    else:
                        U_r = [[round(x) for x in row] for row in result["U"]]
                        V_r = [[round(x) for x in row] for row in result["V"]]
                        W_r = [[round(x) for x in row] for row in result["W"]]
                        if verify_exact(U_r, V_r, W_r):
                            best = {"U": U_r, "V": V_r, "W": W_r}
                            best_rank = rank
                    break
    
    return best
# EVOLVE-BLOCK-END
