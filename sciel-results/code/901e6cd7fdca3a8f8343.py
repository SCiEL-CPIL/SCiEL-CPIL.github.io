"""Rewritten program: search for rank-minimal tensor decomposition of M_{2,3,4}.
Falls back to schoolbook (rank 24) if no better solution is found in time.
"""
import time
import random
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 4]
N, M, P = DIMENSIONS
TOTAL = N * M * P  # 24

# Build the target tensor T as a dict of (i,j,k) -> 1.0
# Indices: i in [0,1], j in [0,1,2], k in [0,1,2,3]
T_tensor = {}
for i in range(N):
    for j in range(M):
        for k in range(P):
            T_tensor[(i, j, k)] = 1.0

# Index mappings
def u_idx(i, j):
    return i * M + j

def v_idx(j, k):
    return j * P + k

def w_idx(k, i):
    return k * N + i

def schoolbook():
    rank = N * M * P
    U = np.zeros((N * M, rank), dtype=int)
    V = np.zeros((M * P, rank), dtype=int)
    W = np.zeros((P * N, rank), dtype=int)
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[u_idx(i, j), c] = 1
                V[v_idx(j, k), c] = 1
                W[w_idx(k, i), c] = 1
                c += 1
    return U, V, W

def verify(U, V, W):
    """Check if U, V, W (numpy arrays with integer or half-integer entries)
    exactly decompose the matrix multiplication tensor.
    U shape: (N*M, rank)
    V shape: (M*P, rank)
    W shape: (P*N, rank)
    Returns True if exact, False otherwise.
    """
    rank = U.shape[1]
    # Reconstruct tensor
    recon = np.zeros((N, M, P))
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        # uc[i*M+j] * vc[j*P+k] * wc[k*N+i]
        for i in range(N):
            for j in range(M):
                uval = uc[i * M + j]
                if uval == 0:
                    continue
                for k in range(P):
                    recon[i, j, k] += uval * vc[j * P + k] * wc[k * N + i]
    # Compare with target
    for i in range(N):
        for j in range(M):
            for k in range(P):
                if abs(recon[i, j, k] - 1.0) > 1e-9:
                    return False
    return True

def verify_fractional(U, V, W):
    """Verify with Fraction for exactness. U, V, W are lists of Fractions or ints."""
    rank = len(U[0])
    recon = {}
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(N):
            for j in range(M):
                uval = uc[i * M + j]
                if uval == 0:
                    continue
                for k in range(P):
                    idx = (i, j, k)
                    contrib = uval * vc[j * P + k] * wc[k * N + i]
                    recon[idx] = recon.get(idx, Fraction(0)) + contrib
    for i in range(N):
        for j in range(M):
            for k in range(P):
                val = recon.get((i, j, k), Fraction(0))
                if val != 1:
                    return False
    return True

def search_random_integer(r, time_budget=10.0, seed=None):
    """Random search with local improvement for integer entries in {-2,-1,0,1,2}."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    deadline = time.time() + time_budget
    best_err = float('inf')
    best_U = None
    
    for attempt in range(500000):
        if time.time() > deadline - 1.0:
            break
        
        U = np.zeros((N * M, r), dtype=int)
        V = np.zeros((M * P, r), dtype=int)
        W = np.zeros((P * N, r), dtype=int)
        
        for c in range(r):
            U[:, c] = [rng.choice([-2, -1, 0, 0, 0, 1, 1, 2]) for _ in range(N * M)]
            V[:, c] = [rng.choice([-2, -1, 0, 0, 0, 1, 1, 2]) for _ in range(M * P)]
            W[:, c] = [rng.choice([-2, -1, 0, 0, 0, 1, 1, 2]) for _ in range(P * N)]
        
        # Compute error
        recon = np.zeros((N, M, P))
        for c in range(r):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            for i in range(N):
                for j in range(M):
                    uval = uc[i * M + j]
                    if uval == 0:
                        continue
                    for k in range(P):
                        recon[i, j, k] += uval * vc[j * P + k] * wc[k * N + i]
        err = np.sum((recon - 1.0) ** 2)
        
        if err < best_err:
            best_err = err
            best_U = (U.copy(), V.copy(), W.copy())
            if err == 0:
                return best_U
    
    return best_U

def search_structure_exploit(r, time_budget=15.0, seed=None):
    """Exploit structure: for M_{2,3,4}, try to find solutions where
    some entries are fixed to exploit the 2x3 and 3x4 structure.
    Uses a more targeted random search."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    deadline = time.time() + time_budget
    best_err = float('inf')
    best = None
    
    for attempt in range(200000):
        if time.time() > deadline - 1.0:
            break
        
        # Strategy: for each column c, pick a random (i,j,k) pattern
        # and try to make the product work
        U = np.zeros((N * M, r), dtype=int)
        V = np.zeros((M * P, r), dtype=int)
        W = np.zeros((P * N, r), dtype=int)
        
        for c in range(r):
            # Pick a random anchor
            i0 = rng.randint(0, N-1)
            j0 = rng.randint(0, M-1)
            k0 = rng.randint(0, P-1)
            sign = rng.choice([-1, 1])
            
            # Set one entry in each matrix
            U[u_idx(i0, j0), c] = sign
            V[v_idx(j0, k0), c] = sign
            W[w_idx(k0, i0), c] = 1
            
            # Add some noise
            if rng.random() < 0.3:
                i1 = rng.randint(0, N-1)
                j1 = rng.randint(0, M-1)
                k1 = rng.randint(0, P-1)
                s1 = rng.choice([-1, 1])
                U[u_idx(i1, j1), c] += s1
                V[v_idx(j1, k1), c] += s1
                W[w_idx(k1, i1), c] += 1
        
        # Local improvement: try to fix errors
        for iteration in range(200):
            if time.time() > deadline - 0.5:
                break
            
            recon = np.zeros((N, M, P))
            for c in range(r):
                uc = U[:, c]
                vc = V[:, c]
                wc = W[:, c]
                for i in range(N):
                    for j in range(M):
                        uval = uc[i * M + j]
                        if uval == 0:
                            continue
                        for k in range(P):
                            recon[i, j, k] += uval * vc[j * P + k] * wc[k * N + i]
            
            err = np.sum((recon - 1.0) ** 2)
            if err == 0:
                return (U.copy(), V.copy(), W.copy())
            
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())
            
            # Find the worst entry and try to fix it
            worst_idx = np.unravel_index(np.argmax(np.abs(recon - 1.0)), (N, M, P))
            wi, wj, wk = worst_idx
            worst_val = recon[wi, wj, wk]
            target = 1.0
            diff = target - worst_val
            
            if abs(diff) < 1e-9:
                continue
            
            # Try to adjust a random column
            c = rng.randint(0, r-1)
            # Try adjusting U, V, or W at the relevant position
            choice = rng.randint(0, 2)
            if choice == 0:
                old = U[u_idx(wi, wj), c]
                # Try changing by diff (rounded)
                for delta in [-2, -1, 1, 2]:
                    new = old + delta
                    if -3 <= new <= 3:
                        U[u_idx(wi, wj), c] = new
                        # Quick check
                        recon_new = np.zeros((N, M, P))
                        for cc in range(r):
                            uc = U[:, cc]
                            vc = V[:, cc]
                            wc = W[:, cc]
                            for i in range(N):
                                for j in range(M):
                                    uval = uc[i * M + j]
                                    if uval == 0:
                                        continue
                                    for k in range(P):
                                        recon_new[i, j, k] += uval * vc[j * P + k] * wc[k * N + i]
                        err_new = np.sum((recon_new - 1.0) ** 2)
                        if err_new < err:
                            err = err_new
                            break
                        else:
                            U[u_idx(wi, wj), c] = old
            elif choice == 1:
                old = V[v_idx(wj, wk), c]
                for delta in [-2, -1, 1, 2]:
                    new = old + delta
                    if -3 <= new <= 3:
                        V[v_idx(wj, wk), c] = new
                        recon_new = np.zeros((N, M, P))
                        for cc in range(r):
                            uc = U[:, cc]
                            vc = V[:, cc]
                            wc = W[:, cc]
                            for i in range(N):
                                for j in range(M):
                                    uval = uc[i * M + j]
                                    if uval == 0:
                                        continue
                                    for k in range(P):
                                        recon_new[i, j, k] += uval * vc[j * P + k] * wc[k * N + i]
                        err_new = np.sum((recon_new - 1.0) ** 2)
                        if err_new < err:
                            err = err_new
                            break
                        else:
                            V[v_idx(wj, wk), c] = old
            else:
                old = W[w_idx(wk, wi), c]
                for delta in [-2, -1, 1, 2]:
                    new = old + delta
                    if -3 <= new <= 3:
                        W[w_idx(wk, wi), c] = new
                        recon_new = np.zeros((N, M, P))
                        for cc in range(r):
                            uc = U[:, cc]
                            vc = V[:, cc]
                            wc = W[:, cc]
                            for i in range(N):
                                for j in range(M):
                                    uval = uc[i * M + j]
                                    if uval == 0:
                                        continue
                                    for k in range(P):
                                        recon_new[i, j, k] += uval * vc[j * P + k] * wc[k * N + i]
                        err_new = np.sum((recon_new - 1.0) ** 2)
                        if err_new < err:
                            err = err_new
                            break
                        else:
                            W[w_idx(wk, wi), c] = old
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with schoolbook
    U, V, W = schoolbook()
    best_rank = N * M * P
    best_result = (U, V, W)
    
    # Try to find lower rank solutions
    for r in range(14, best_rank):
        if time.time() > deadline - 20.0:
            break
        
        # Try structure-exploiting search
        result = search_structure_exploit(r, time_budget=min(15.0, deadline - time.time() - 5.0))
        if result is not None:
            Ur, Vr, Wr = result
            if verify(Ur, Vr, Wr):
                best_rank = r
                best_result = (Ur, Vr, Wr)
                break
        
        if time.time() > deadline - 15.0:
            break
        
        # Try random integer search
        result = search_random_integer(r, time_budget=min(10.0, deadline - time.time() - 5.0))
        if result is not None:
            Ur, Vr, Wr = result
            if verify(Ur, Vr, Wr):
                best_rank = r
                best_result = (Ur, Vr, Wr)
                break
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
