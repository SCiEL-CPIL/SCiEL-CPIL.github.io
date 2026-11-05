"""Systematic rank-1 peeling for 2x3x5 matrix multiplication tensor using exact arithmetic."""
import numpy as np
from fractions import Fraction
import itertools
import random

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n
    
    # Build the tensor T as a 3D array of Fractions
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[(idx_u, idx_v, idx_w)] = Fraction(1)
    
    # Initialize residual
    R = T.copy()
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to convert a vector of Fractions to a list
    def to_list(vec):
        return [float(x) for x in vec]
    
    # Helper to check if a vector has simple entries (0, +/-1, +/-1/2, +/-3/2, etc.)
    def is_simple(vec, max_denom=2, max_num=4):
        for val in vec:
            if val == 0:
                continue
            if val.denominator > max_denom:
                return False
            if abs(val.numerator) > max_num:
                return False
        return True

    # Helper to subtract a rank-1 tensor from R
    def subtract_rank1(R, u, v, w):
        new_R = {}
        # Get all indices in R and the support of u, v, w
        # u is dict {idx: val}, v is dict {idx: val}, w is dict {idx: val}
        # Support of u⊗v⊗w is { (i,j,k) | u[i]!=0, v[j]!=0, w[k]!=0 }
        u_supp = [i for i, val in u.items() if val != 0]
        v_supp = [j for j, val in v.items() if val != 0]
        w_supp = [k for k, val in w.items() if val != 0]
        
        # Compute new values for all affected indices
        affected = set(R.keys())
        for i in u_supp:
            for j in v_supp:
                for k in w_supp:
                    affected.add((i, j, k))
        
        for idx in affected:
            i, j, k = idx
            val_r = R.get(idx, Fraction(0))
            val_u = u.get(i, Fraction(0))
            val_v = v.get(j, Fraction(0))
            val_w = w.get(k, Fraction(0))
            new_val = val_r - val_u * val_v * val_w
            if new_val != 0:
                new_R[idx] = new_val
        return new_R

    # Helper to solve for u, v, w given a set of constraints (indices and values)
    # We fix the scale of u, and solve for v, w
    # This is a linear system for v and w if we fix u.
    # Actually, it's bilinear.
    # Let's try a specific approach:
    # Pick a non-zero entry (i0, j0, k0) in R.
    # Set u[i0] = 1.
    # Then v[j0]*w[k0] = R[i0, j0, k0].
    # Pick another entry (i1, j1, k1).
    # u[i1] * v[j1] * w[k1] = R[i1, j1, k1].
    # This is still hard to solve directly without guessing.
    
    # Instead, we will use a **Randomized Search** for simple rank-1 terms.
    # We will generate random simple vectors u, v, w and check if they "fit" R well.
    
    max_rank = 30
    rank = 0
    iterations = 0
    max_iterations = 5000 # Safety break
    
    while len(R) > 0 and rank < max_rank and iterations < max_iterations:
        iterations += 1
        found = False
        
        # Try to find a rank-1 term to peel
        # We will try a few random simple vectors
        # Simple vectors have entries in {0, 1, -1, 1/2, -1/2}
        
        # To make it more systematic, we can look at the "core" of R
        # Let's try to match the first non-zero entry
        # But random search with simple vectors might find Strassen-like terms
        
        for _ in range(20): # 20 random trials per step
            # Generate random simple u, v, w
            # Bias towards sparse vectors
            u = {}
            v = {}
            w = {}
            
            # Randomly pick support for u
            # N=6. Let's pick 1-3 non-zeros
            num_u = random.randint(1, 3)
            u_indices = random.sample(range(N), num_u)
            for i in u_indices:
                u[i] = random.choice([Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2)])
            
            num_v = random.randint(1, 4)
            v_indices = random.sample(range(M), num_v)
            for j in v_indices:
                v[j] = random.choice([Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2)])
                
            num_w = random.randint(1, 3)
            w_indices = random.sample(range(P), num_w)
            for k in w_indices:
                w[k] = random.choice([Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2)])
            
            # Check if this rank-1 term can be subtracted
            # Condition: For all (i,j,k) in support(u) x support(v) x support(w),
            # R[i,j,k] - u[i]v[j]w[k] should be "simple" or zero.
            # Ideally, R[i,j,k] is 0 or the term matches it.
            
            valid = True
            new_non_zeros = 0
            for i in u:
                if u[i] == 0: continue
                for j in v:
                    if v[j] == 0: continue
                    for k in w:
                        if w[k] == 0: continue
                        idx = (i, j, k)
                        r_val = R.get(idx, Fraction(0))
                        term_val = u[i] * v[j] * w[k]
                        new_val = r_val - term_val
                        # We want new_val to be 0 or "simple"
                        # If r_val was 0, new_val = -term_val. This adds a new non-zero.
                        # We prefer terms that reduce non-zeros.
                        if new_val != 0:
                            # Check if new_val is simple
                            if not is_simple([new_val]):
                                valid = False
                                break
                            new_non_zeros += 1
                        if not valid:
                            break
                if not valid:
                    break
            if not valid:
                continue
            
            # Calculate change in number of non-zeros
            current_nz = len(R)
            # Estimate new nz
            # This is an approximation. We should actually compute it.
            new_R = subtract_rank1(R, u, v, w)
            new_nz = len(new_R)
            
            # Accept if it reduces the number of non-zeros or keeps it same but reduces magnitude
            if new_nz < current_nz:
                R = new_R
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)
                rank += 1
                found = True
                break
            elif new_nz == current_nz:
                # Check if it simplifies (e.g., values become smaller)
                # For now, accept with low probability to avoid loops
                if random.random() < 0.1:
                    R = new_R
                    U_list.append(u)
                    V_list.append(v)
                    W_list.append(w)
                    rank += 1
                    found = True
                    break
        
        if not found:
            # If we can't find a good term, break and fallback
            break
            
    # If we didn't reach zero, fallback to schoolbook
    if len(R) > 0:
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][column] = 1
                    V[j*p+k][column] = 1
                    W[k*n+i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Build factor matrices from the lists of dicts
    # U_list contains dicts {index: value}
    # We need to convert to dense matrices (N, rank)
    
    U_mat = [[0] * rank for _ in range(N)]
    V_mat = [[0] * rank for _ in range(M)]
    W_mat = [[0] * rank for _ in range(P)]
    
    for c in range(rank):
        u_dict = U_list[c]
        v_dict = V_list[c]
        w_dict = W_list[c]
        
        for i, val in u_dict.items():
            if val != 0:
                U_mat[i][c] = float(val)
        for j, val in v_dict.items():
            if val != 0:
                V_mat[j][c] = float(val)
        for k, val in w_dict.items():
            if val != 0:
                W_mat[k][c] = float(val)
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
