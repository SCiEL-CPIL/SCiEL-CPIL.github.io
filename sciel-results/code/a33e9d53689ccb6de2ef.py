import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor T
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U_sb[u, col] = 1.0
                V_sb[v, col] = 1.0
                W_sb[w, col] = 1.0
                col += 1
                
    # Check schoolbook
    T_recon = np.einsum('ac,bc,dc->abd', U_sb, V_sb, W_sb)
    assert np.allclose(T_recon, T)
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    start_time = time.time()
    deadline = start_time + 140
    
    # Try to find rank 41
    target_rank = 41
    if time.time() < deadline - 5:
        # Strategy: Random search with local refinement
        # We will try to find a set of 41 columns that can reconstruct T.
        # Initialize with schoolbook, remove one random column, and try to adjust others.
        # This is very hard. 
        # Alternative: Random sparse initialization.
        
        # Given the difficulty, we will try a limited number of random restarts
        # with a greedy algorithm that tries to match the largest remaining error.
        
        def check_exact(U, V, W):
            # Use fractions for exact check if values are clean
            # First check float
            T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
            if not np.allclose(T_recon, T, atol=1e-9):
                return False
            
            # Snap to half-integers and verify exactly
            U_frac = np.round(U * 2) / 2.0
            V_frac = np.round(V * 2) / 2.0
            W_frac = np.round(W * 2) / 2.0
            
            if not (np.allclose(U, U_frac, atol=1e-9) and np.allclose(V, V_frac, atol=1e-9) and np.allclose(W, W_frac, atol=1e-9)):
                return False
                
            # Exact check with fractions
            # Convert to Fraction
            U_f = [[Fraction(int(round(U_frac[i, c] * 2)), 2) for c in range(target_rank)] for i in range(nm)]
            V_f = [[Fraction(int(round(V_frac[i, c] * 2)), 2) for c in range(target_rank)] for i in range(mp)]
            W_f = [[Fraction(int(round(W_frac[i, c] * 2)), 2) for c in range(target_rank)] for i in range(pn)]
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = i * m + j
                        v = j * p + k
                        w = k * n + i
                        total = Fraction(0)
                        for c in range(target_rank):
                            total += U_f[u][c] * V_f[v][c] * W_f[w][c]
                        if total != 1:
                            return False
            return True
            
        # Random search
        random.seed(42)
        np.random.seed(42)
        
        # We try a few random initializations
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
                
            # Initialize with random half-integers
            # Values: -1, -0.5, 0, 0.5, 1
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(nm, target_rank))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(mp, target_rank))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(pn, target_rank))
            
            # Local search: flip one entry at a time to reduce error
            # Error = sum (T - T_recon)^2
            T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
            error = np.sum((T_recon - T)**2)
            
            # Try to improve
            improved = True
            while improved and time.time() < deadline - 15:
                improved = False
                # Pick a random entry to flip
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    idx = random.randint(0, nm - 1)
                    col_idx = random.randint(0, target_rank - 1)
                    old_val = U[idx, col_idx]
                    # Try flipping to another half-integer
                    new_val = random.choice([-1, -0.5, 0, 0.5, 1])
                    if new_val == old_val: continue
                    U[idx, col_idx] = new_val
                    T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
                    new_error = np.sum((T_recon - T)**2)
                    if new_error < error:
                        error = new_error
                        improved = True
                    else:
                        U[idx, col_idx] = old_val
                elif mat == 'V':
                    idx = random.randint(0, mp - 1)
                    col_idx = random.randint(0, target_rank - 1)
                    old_val = V[idx, col_idx]
                    new_val = random.choice([-1, -0.5, 0, 0.5, 1])
                    if new_val == old_val: continue
                    V[idx, col_idx] = new_val
                    T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
                    new_error = np.sum((T_recon - T)**2)
                    if new_error < error:
                        error = new_error
                        improved = True
                    else:
                        V[idx, col_idx] = old_val
                else:
                    idx = random.randint(0, pn - 1)
                    col_idx = random.randint(0, target_rank - 1)
                    old_val = W[idx, col_idx]
                    new_val = random.choice([-1, -0.5, 0, 0.5, 1])
                    if new_val == old_val: continue
                    W[idx, col_idx] = new_val
                    T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
                    new_error = np.sum((T_recon - T)**2)
                    if new_error < error:
                        error = new_error
                        improved = True
                    else:
                        W[idx, col_idx] = old_val
                        
            if error < 1e-9:
                if check_exact(U, V, W):
                    best_rank = target_rank
                    best_U = U
                    best_V = V
                    best_W = W
                    break

    # Return best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
