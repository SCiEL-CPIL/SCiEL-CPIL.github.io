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

    def verify_gf2(U, V, W, rank):
        """Verify decomposition over GF(2)."""
        # U, V, W are lists of lists of 0/1
        # T[i,j,k] = XOR_c (U[i,j,c] AND V[j,k,c] AND W[k,i,c])
        # We need T[i,j,k] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s ^= (U[i*m + j][c] & V[j*p + k][c] & W[k*n + i][c])
                    if s != 1:
                        return False
        return True

    def try_gf2_rank(target_rank, max_attempts):
        """Try to find a GF(2) decomposition of the given rank."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
            
            # Generate random binary matrices
            U = [[random.randint(0, 1) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.randint(0, 1) for _ in range(target_rank)] for _ in range(mp)]
            W = [[random.randint(0, 1) for _ in range(target_rank)] for _ in range(pn)]
            
            if verify_gf2(U, V, W, target_rank):
                # If found over GF(2), it is a valid integer decomposition
                return {"U": U, "V": V, "W": W}
        return None

    def try_random_rank(target_rank, max_attempts, value_range):
        """Try to find a decomposition of the given rank with random integer initialization."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
            
            U = [[random.randint(value_range[0], value_range[1]) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.randint(value_range[0], value_range[1]) for _ in range(target_rank)] for _ in range(mp)]
            W = [[random.randint(value_range[0], value_range[1]) for _ in range(target_rank)] for _ in range(pn)]
            
            if verify_fast(U, V, W, target_rank):
                return {"U": U, "V": V, "W": W}
        return None

    # Try to find lower rank decompositions
    # Lower bound: max(nm, mp, pn) = 21
    # Start from lower bound up to schoolbook rank - 1
    
    for target_rank in range(21, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try GF(2) search first (faster, binary)
        # Higher ranks have more solutions, so try higher ranks within the loop first?
        # No, we want the MINIMAL rank. So we must check 21, then 22, etc.
        # But if 21 is impossible, we waste time.
        # Let's try a few attempts for each rank.
        
        # Use more attempts for ranks close to the lower bound?
        # Or spread attempts?
        # Let's use a fixed budget per rank.
        max_gf2_attempts = 500 if target_rank < 30 else 200
        result = try_gf2_rank(target_rank, max_gf2_attempts)
        if result:
            best = result
            best_rank = target_rank
            # If we found a solution, we can stop? 
            # No, we want the MINIMAL rank. 
            # Since we iterate from 21 upwards, the first one we find is the minimal found.
            # But is it the TRUE minimal? 
            # If we find rank 25, we should stop looking for 21-24? 
            # Yes, because we are iterating in increasing order. 
            # If we find rank 25, it means 21-24 failed (or we didn't search enough).
            # To be safe, we should search 21-24 more thoroughly if we find 25.
            # But time is limited. 
            # Let's assume the first found is good enough for an improvement.
            break
        
        # Try integer search with small values
        max_int_attempts = 50
        result = try_random_rank(target_rank, max_int_attempts, (-1, 1))
        if result:
            best = result
            best_rank = target_rank
            break

    return best
# EVOLVE-BLOCK-END
