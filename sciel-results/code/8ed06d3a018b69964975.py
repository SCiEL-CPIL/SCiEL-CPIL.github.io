"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # Safety margin before 150s limit
    
    # Build the list of nonzero tensor entries
    # T[a, b, c] = 1 where a = k*n+i, b = i*m+j, c = j*p+k
    entries = []
    entry_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = k * n + i
                b = i * m + j
                c = j * p + k
                entries.append((a, b, c, 1))
                entry_set.add((a, b, c))
    
    # Pre-compute for each index which entries involve it
    entries_by_b = {}
    entries_by_c = {}
    entries_by_a = {}
    for a, b, c, t in entries:
        entries_by_b.setdefault(b, []).append((a, c, t))
        entries_by_c.setdefault(c, []).append((a, b, t))
        entries_by_a.setdefault(a, []).append((b, c, t))
    
    def verify(U, V, W, tol=1e-8):
        """Verify that the decomposition is exact."""
        for a, b, c, t in entries:
            val = 0.0
            for r in range(U.shape[1]):
                val += U[b, r] * V[c, r] * W[a, r]
            if abs(val - t) > tol:
                return False
        # Also verify zero entries are zero
        # Check a sample of non-entry positions
        rng = np.random.RandomState(42)
        for _ in range(200):
            a = rng.randint(0, p * n)
            b = rng.randint(0, n * m)
            c = rng.randint(0, m * p)
            if (a, b, c) in entry_set:
                continue
            val = 0.0
            for r in range(U.shape[1]):
                val += U[b, r] * V[c, r] * W[a, r]
            if abs(val) > tol:
                return False
        return True
    
    def verify_fast(U, V, W, tol=1e-8):
        """Fast verification using vectorized operations."""
        # Check all entries
        for a, b, c, t in entries:
            val = np.dot(U[b, :], V[c, :] * W[a, :])
            if abs(val - t) > tol:
                return False
        # Check some non-entries
        rng = np.random.RandomState(42)
        for _ in range(100):
            a = rng.randint(0, p * n)
            b = rng.randint(0, n * m)
            c = rng.randint(0, m * p)
            if (a, b, c) in entry_set:
                continue
            val = np.dot(U[b, :], V[c, :] * W[a, :])
            if abs(val) > tol:
                return False
        return True
    
    def solve_rank_als(r, max_iter=100, seed=0):
        """Try to find a rank-r decomposition using ALS with multiple restarts."""
        for restart in range(5):
            if time.time() > deadline - 25:
                return None
            
            rng = np.random.RandomState(seed + restart * 1000)
            
            # Initialize with small random values
            U = rng.randn(n * m, r) * 0.1
            V = rng.randn(m * p, r) * 0.1
            W = rng.randn(p * n, r) * 0.1
            
            prev_err = float('inf')
            for iteration in range(max_iter):
                if time.time() > deadline - 20:
                    return None
                
                # Update U
                for b in range(n * m):
                    mask = entries_by_b.get(b, [])
                    if not mask:
                        continue
                    A_mat = np.zeros((len(mask), r))
                    t_vec = np.zeros(len(mask))
                    for idx, (a, c, t) in enumerate(mask):
                        A_mat[idx, :] = V[c, :] * W[a, :]
                        t_vec[idx] = t
                    if len(mask) >= r:
                        try:
                            U[b, :] = np.linalg.lstsq(A_mat, t_vec, rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for c in range(m * p):
                    mask = entries_by_c.get(c, [])
                    if not mask:
                        continue
                    A_mat = np.zeros((len(mask), r))
                    t_vec = np.zeros(len(mask))
                    for idx, (a, b, t) in enumerate(mask):
                        A_mat[idx, :] = U[b, :] * W[a, :]
                        t_vec[idx] = t
                    if len(mask) >= r:
                        try:
                            V[c, :] = np.linalg.lstsq(A_mat, t_vec, rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for a in range(p * n):
                    mask = entries_by_a.get(a, [])
                    if not mask:
                        continue
                    A_mat = np.zeros((len(mask), r))
                    t_vec = np.zeros(len(mask))
                    for idx, (b, c, t) in enumerate(mask):
                        A_mat[idx, :] = U[b, :] * V[c, :]
                        t_vec[idx] = t
                    if len(mask) >= r:
                        try:
                            W[a, :] = np.linalg.lstsq(A_mat, t_vec, rcond=None)[0]
                        except:
                            pass
                
                # Compute error
                err = 0.0
                for a, b, c, t in entries:
                    val = np.dot(U[b, :], V[c, :] * W[a, :])
                    err += (val - t) ** 2
                
                if err < 1e-12:
                    return U, V, W
                
                if err > prev_err * 1.01:
                    break
                prev_err = err
        
        # Final verification
        if verify_fast(U, V, W, tol=1e-8):
            return U, V, W
        
        return None
    
    def solve_rank_structured(r, seed=0):
        """Try structured initialization based on Kronecker products."""
        rng = np.random.RandomState(seed)
        
        # Try different structured initializations
        for init_type in range(3):
            if time.time() > deadline - 25:
                return None
            
            if init_type == 0:
                # Identity-like initialization
                U = np.zeros((n * m, r))
                V = np.zeros((m * p, r))
                W = np.zeros((p * n, r))
                for idx in range(min(r, n * m)):
                    U[idx, idx] = 1
                for idx in range(min(r, m * p)):
                    V[idx, idx] = 1
                for idx in range(min(r, p * n)):
                    W[idx, idx] = 1
            elif init_type == 1:
                # Random sparse initialization
                U = np.zeros((n * m, r))
                V = np.zeros((m * p, r))
                W = np.zeros((p * n, r))
                for col in range(r):
                    row_u = rng.randint(0, n * m)
                    row_v = rng.randint(0, m * p)
                    row_w = rng.randint(0, p * n)
                    U[row_u, col] = 1
                    V[row_v, col] = 1
                    W[row_w, col] = 1
            else:
                # Small random values
                U = rng.randn(n * m, r) * 0.05
                V = rng.randn(m * p, r) * 0.05
                W = rng.randn(p * n, r) * 0.05
            
            # Run ALS refinement
            prev_err = float('inf')
            for iteration in range(50):
                if time.time() > deadline - 20:
                    return None
                
                for b in range(n * m):
                    mask = entries_by_b.get(b, [])
                    if not mask:
                        continue
                    A_mat = np.zeros((len(mask), r))
                    t_vec = np.zeros(len(mask))
                    for idx, (a, c, t) in enumerate(mask):
                        A_mat[idx, :] = V[c, :] * W[a, :]
                        t_vec[idx] = t
                    if len(mask) >= r:
                        try:
                            U[b, :] = np.linalg.lstsq(A_mat, t_vec, rcond=None)[0]
                        except:
                            pass
                
                for c in range(m * p):
                    mask = entries_by_c.get(c, [])
                    if not mask:
                        continue
                    A_mat = np.zeros((len(mask), r))
                    t_vec = np.zeros(len(mask))
                    for idx, (a, b, t) in enumerate(mask):
                        A_mat[idx, :] = U[b, :] * W[a, :]
                        t_vec[idx] = t
                    if len(mask) >= r:
                        try:
                            V[c, :] = np.linalg.lstsq(A_mat, t_vec, rcond=None)[0]
                        except:
                            pass
                
                for a in range(p * n):
                    mask = entries_by_a.get(a, [])
                    if not mask:
                        continue
                    A_mat = np.zeros((len(mask), r))
                    t_vec = np.zeros(len(mask))
                    for idx, (b, c, t) in enumerate(mask):
                        A_mat[idx, :] = U[b, :] * V[c, :]
                        t_vec[idx] = t
                    if len(mask) >= r:
                        try:
                            W[a, :] = np.linalg.lstsq(A_mat, t_vec, rcond=None)[0]
                        except:
                            pass
                
                err = 0.0
                for a, b, c, t in entries:
                    val = np.dot(U[b, :], V[c, :] * W[a, :])
                    err += (val - t) ** 2
                
                if err < 1e-12:
                    if verify_fast(U, V, W, tol=1e-8):
                        return U, V, W
                    break
                
                if err > prev_err * 1.01:
                    break
                prev_err = err
        
        return None
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    best_result = None
    
    # Try ranks from lower to higher
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # Also rank >= max(n, m, p) * max(n, m, p) / min(n, m, p) for some bounds
    # Let's try from 35 up
    for r in range(35, min(rank_school + 1, 110)):
        if time.time() > deadline - 40:
            break
        
        # Try structured search first
        result = solve_rank_structured(r, seed=r)
        if result is not None:
            best_result = result
            break
        
        # Try ALS with multiple restarts
        result = solve_rank_als(r, max_iter=30, seed=r)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        U, V, W = best_result
    else:
        U, V, W = U_school, V_school, W_school
    
    # Convert to lists with integer/half-integer entries
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for x in row:
                # Round to nearest half-integer
                half = round(x * 2) / 2
                if abs(x - half) < 1e-6:
                    row_list.append(int(half) if half == int(half) else half)
                else:
                    row_list.append(float(x))
            result.append(row_list)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
