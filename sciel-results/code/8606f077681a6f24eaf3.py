import numpy as np
import time
from fractions import Fraction
import itertools
import math
import random
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def _verify(decomp, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    
    # Check dimensions
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    rank = U.shape[1]
    # Check the 42 non-zero entries of the tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i * m + j, :]
                v_row = V[j * p + k, :]
                w_row = W[k * n + i, :]
                s = np.dot(u_row, v_row * w_row)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_structured(rank, deadline):
    """Try structured decompositions with specific patterns."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Try various structured patterns
    patterns = [
        'block',      # Block diagonal structure
        'cyclic',     # Cyclic shifts
        'random_sparse',  # Random sparse initialization
    ]
    
    for pattern in patterns:
        if time.time() > deadline - 5:
            break
        
        if pattern == 'random_sparse':
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            
            # Initialize with random ±1 entries, sparse
            for r in range(rank):
                # Pick random non-zero positions
                u_idx = random.randint(0, nm-1)
                v_idx = random.randint(0, mp-1)
                w_idx = random.randint(0, pn-1)
                
                U[u_idx, r] = random.choice([-1, 1])
                V[v_idx, r] = random.choice([-1, 1])
                W[w_idx, r] = random.choice([-1, 1])
            
            decomp = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            if _verify(decomp):
                return decomp
    
    return None

def _local_repair(U, V, W, rank, deadline, max_iters=1000):
    """Local repair: try to fix errors by adjusting entries."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Compute current error
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    errors.append(abs(s - 1.0))
        
        current_error = max(errors) if errors else 0
        if current_error < best_error:
            best_error = current_error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if current_error < 1e-9:
            break
        
        # Try to improve by adjusting one entry
        # Pick the worst error
        worst_idx = errors.index(current_error)
        i = worst_idx // (m*p)
        k = (worst_idx % (m*p)) // m
        j = worst_idx % m
        
        # Try adjusting U[i*m+j, r] for random r
        r = random.randint(0, rank-1)
        old_val = U[i*m+j, r]
        U[i*m+j, r] = old_val + random.uniform(-0.5, 0.5)
        
        # Check if improved
        new_s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
        new_error = abs(new_s - 1.0)
        
        if new_error < current_error:
            best_error = min(best_error, new_error)
            best_U[i*m+j, r] = U[i*m+j, r]
            best_V, best_W = V.copy(), W.copy()
        else:
            U[i*m+j, r] = old_val
    
    return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_error

def construct():
    """
    Returns the best decomposition found within the time limit.
    Starts with schoolbook and tries to find lower-rank solutions.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = len(best["U"][0]) if best["U"] else 0
    
    # Try to find lower-rank decompositions
    # Minimum possible rank: at least max(nm, mp, pn) = 21
    # But likely higher. Try from rank 21 up to schoolbook rank - 1
    for rank in range(21, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try random sparse initialization
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            
            # Initialize with random ±1
            for r in range(rank):
                for _ in range(random.randint(1, 3)):
                    U[random.randint(0, nm-1), r] = random.choice([-1, 1])
                for _ in range(random.randint(1, 3)):
                    V[random.randint(0, mp-1), r] = random.choice([-1, 1])
                for _ in range(random.randint(1, 3)):
                    W[random.randint(0, pn-1), r] = random.choice([-1, 1])
            
            decomp = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            if _verify(decomp):
                return decomp
        
        # Try local repair from random initialization
        if time.time() > deadline - 10:
            break
        
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        U_l, V_l, W_l, err = _local_repair(U.tolist(), V.tolist(), W.tolist(), rank, deadline, max_iters=200)
        if err < 1e-9:
            decomp = {"U": U_l, "V": V_l, "W": W_l}
            if _verify(decomp):
                return decomp
    
    # Return best found (schoolbook if nothing better)
    return best
# EVOLVE-BLOCK-END
