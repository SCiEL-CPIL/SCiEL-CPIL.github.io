"""Combined program for matrix multiplication 2x3x7 rank minimization."""
import numpy as np
import random
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    deadline = time.time() + 140  # safety margin before 150s kill

    # Schoolbook as baseline
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    def verify_fast(U, V, W, rank):
        """Vectorized verification using numpy."""
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        U3 = U_np.reshape(n, m, rank)
        V3 = V_np.reshape(m, p, rank)
        W3 = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.all(T == 1)

    def verify_slow(U, V, W, rank):
        """Slow but exact verification."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if s != 1:
                        return False
        return True

    def local_search(U, V, W, target_rank, max_iters=50):
        """Local search to refine a candidate."""
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            if verify_fast(U, V, W, target_rank):
                return True
            # Pick a random position to modify
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = random.randint(0, nm - 1)
                col = random.randint(0, target_rank - 1)
                old = U[row][col]
                U[row][col] = random.randint(-2, 2)
                if not verify_fast(U, V, W, target_rank) and time.time() > deadline - 5:
                    U[row][col] = old
            elif mat == 'V':
                row = random.randint(0, mp - 1)
                col = random.randint(0, target_rank - 1)
                old = V[row][col]
                V[row][col] = random.randint(-2, 2)
                if not verify_fast(U, V, W, target_rank) and time.time() > deadline - 5:
                    V[row][col] = old
            else:
                row = random.randint(0, pn - 1)
                col = random.randint(0, target_rank - 1)
                old = W[row][col]
                W[row][col] = random.randint(-2, 2)
                if not verify_fast(U, V, W, target_rank) and time.time() > deadline - 5:
                    W[row][col] = old
        return verify_fast(U, V, W, target_rank)

    def try_structured_rank(target_rank):
        """Try structured decomposition based on block partitioning."""
        # For 2x3x7, try to exploit structure
        # Partition k into groups: e.g., k=0..6
        # Try rank-1 components with specific patterns
        
        # Strategy: Use the fact that T[i,j,k]=1 for all
        # Try to find sparse rank-1 components
        
        # Initialize with zeros
        U = [[0] * target_rank for _ in range(nm)]
        V = [[0] * target_rank for _ in range(mp)]
        W = [[0] * target_rank for _ in range(pn)]
        
        # Try to fill in rank-1 components greedily
        # Each rank-1 component (u_c, v_c, w_c) contributes u_c[i,j] * v_c[j,k] * w_c[k,i]
        # We want sum over c to equal 1 for all (i,j,k)
        
        # Start with random sparse patterns
        for c in range(target_rank):
            # Choose random support for u, v, w
            # u has nm=6 entries, v has mp=21, w has pn=14
            u_vals = [random.randint(-1, 1) for _ in range(nm)]
            v_vals = [random.randint(-1, 1) for _ in range(mp)]
            w_vals = [random.randint(-1, 1) for _ in range(pn)]
            
            # Make them sparse
            for idx in range(nm):
                if random.random() < 0.5:
                    u_vals[idx] = 0
            for idx in range(mp):
                if random.random() < 0.5:
                    v_vals[idx] = 0
            for idx in range(pn):
                if random.random() < 0.5:
                    w_vals[idx] = 0
            
            for idx in range(nm):
                U[idx][c] = u_vals[idx]
            for idx in range(mp):
                V[idx][c] = v_vals[idx]
            for idx in range(pn):
                W[idx][c] = w_vals[idx]
        
        # Try local search
        if local_search(U, V, W, target_rank, max_iters=30):
            return {"U": U, "V": V, "W": W}
        
        return None

    def try_rank_pure_random(target_rank, max_attempts=100, val_range=(-2, 2)):
        """Pure random search without local search."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
            U = [[random.randint(val_range[0], val_range[1]) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.randint(val_range[0], val_range[1]) for _ in range(target_rank)] for _ in range(mp)]
            W = [[random.randint(val_range[0], val_range[1]) for _ in range(target_rank)] for _ in range(pn)]
            if verify_fast(U, V, W, target_rank):
                return {"U": U, "V": V, "W": W}
        return None

    # Try ranks from 21 to 41
    for target_rank in range(21, best_rank):
        if time.time() > deadline - 15:
            break
        
        # First try structured approach
        result = try_structured_rank(target_rank)
        if result:
            best = result
            best_rank = target_rank
            continue
        
        # Try pure random with small values
        result = try_rank_pure_random(target_rank, max_attempts=30, val_range=(-1, 1))
        if result:
            best = result
            best_rank = target_rank
            continue
        
        # Try pure random with larger values
        result = try_rank_pure_random(target_rank, max_attempts=20, val_range=(-2, 2))
        if result:
            best = result
            best_rank = target_rank
            continue
    
    return best
# EVOLVE-BLOCK-END
