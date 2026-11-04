"""Combined program for matrix multiplication 2x3x7 rank minimization
using Strassen-style block decomposition and randomized search."""
import numpy as np
import random
import time
from fractions import Fraction
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 130  # safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
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

    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    # Try to find lower rank decompositions
    # Theoretical lower bound for 2x3x7 is around 21
    min_rank = 18

    # Strategy 1: Randomized search with small integers
    for target_rank in range(min_rank, best_rank):
        if time.time() > deadline - 15:
            break

        max_attempts = 100
        found = False

        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break

            # Generate random small integer matrices
            U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(mp)]
            W = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(pn)]

            if verify(U, V, W, target_rank):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                found = True
                break

        if found:
            continue

    # Strategy 2: Finite field F2 rank-revealing algorithm
    # Formulate the matrix multiplication tensor over F2 and use
    # iterative Gaussian elimination on tensor flattenings to empirically
    # determine if the tensor rank over F2 is strictly lower than schoolbook rank.
    # If a reduction exists over F2, it may lift to an integer decomposition.
    if time.time() < deadline - 40:
        def f2_gauss_rank(matrix):
            """Compute rank of a binary matrix over F2 using Gaussian elimination."""
            mat = np.array(matrix, dtype=np.uint8) % 2
            rows = mat.shape[0]
            cols = mat.shape[1]
            rank = 0
            for col in range(cols):
                if rank >= rows:
                    break
                # Find pivot row
                pivot = -1
                for row in range(rank, rows):
                    if mat[row, col] == 1:
                        pivot = row
                        break
                if pivot == -1:
                    continue
                # Swap rows
                mat[[rank, pivot]] = mat[[pivot, rank]]
                # Eliminate
                for row in range(rows):
                    if row != rank and mat[row, col] == 1:
                        mat[row] = (mat[row] + mat[rank]) % 2
                rank += 1
            return rank

        # Strategy 2a: Direct F2 rank estimation via flattening
        # The tensor T has entries T[i,j,k] = 1 for all i,j,k (matrix mult tensor)
        # Flatten along different modes and estimate rank
        # Mode-1 flattening: reshape to (n, m*p) - this is just the identity-like structure
        # For matrix multiplication, the tensor is structured; let's try random projections
        
        # Build the tensor as a 3D array over F2
        T = np.ones((n, m, p), dtype=np.uint8)
        
        # Try to find a low-rank decomposition over F2 using iterative approach
        # We'll use a randomized SVD-like approach over F2
        # First, get an upper bound on F2 rank via flattening
        T_flat_1 = T.reshape(n, m * p)  # Mode-1 flattening
        T_flat_2 = T.transpose(1, 0, 2).reshape(m, n * p)  # Mode-2 flattening
        T_flat_3 = T.transpose(2, 0, 1).reshape(p, n * m)  # Mode-3 flattening
        
        rank1 = f2_gauss_rank(T_flat_1)
        rank2 = f2_gauss_rank(T_flat_2)
        rank3 = f2_gauss_rank(T_flat_3)
        
        # The tensor rank is at least max(rank1, rank2, rank3)
        f2_lower_bound = max(rank1, rank2, rank3)
        
        # Try to construct an actual F2 decomposition with rank close to the bound
        # Use iterative randomized search over F2
        for target_rank in range(max(f2_lower_bound, 10), min(best_rank, 35)):
            if time.time() > deadline - 25:
                break
            
            max_f2_attempts = 50
            found_f2 = False
            
            for attempt in range(max_f2_attempts):
                if time.time() > deadline - 25:
                    break
                
                # Generate random binary matrices
                U = [[random.randint(0, 1) for _ in range(target_rank)] for _ in range(nm)]
                V = [[random.randint(0, 1) for _ in range(target_rank)] for _ in range(mp)]
                W = [[random.randint(0, 1) for _ in range(target_rank)] for _ in range(pn)]
                
                # Verify over F2
                valid_f2 = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(target_rank):
                                s = (s + U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]) % 2
                            if s != 1:
                                valid_f2 = False
                                break
                        if not valid_f2:
                            break
                    if not valid_f2:
                        break
                
                if valid_f2:
                    # Found an F2 decomposition; try to lift to integers
                    # The F2 solution may not directly give an integer solution,
                    # but it indicates the structure exists
                    # Try using the F2 pattern as a starting point for integer search
                    found_f2 = True
                    
                    # Now try to find an integer decomposition with the same rank
                    # by perturbing the F2 solution
                    for int_attempt in range(20):
                        if time.time() > deadline - 25:
                            break
                        
                        U_int = []
                        for row in U:
                            U_int.append([x * random.choice([1, -1]) if x == 1 else 0 for x in row])
                        V_int = []
                        for row in V:
                            V_int.append([x * random.choice([1, -1]) if x == 1 else 0 for x in row])
                        W_int = []
                        for row in W:
                            W_int.append([x * random.choice([1, -1]) if x == 1 else 0 for x in row])
                        
                        if verify(U_int, V_int, W_int, target_rank):
                            best = {"U": U_int, "V": V_int, "W": W_int}
                            best_rank = target_rank
                            found_f2 = False  # Don't break outer loop, keep searching
                            break
                    
                    if found_f2:
                        break
            
            if not found_f2 and best_rank <= target_rank:
                break

    # Strategy 3: Enhanced Strassen-inspired structured search
    if time.time() < deadline - 15:
        for target_rank in range(max(18, best_rank - 5), min(28, best_rank)):
            if time.time() > deadline - 10:
                break
            
            # Try structured random search with Strassen-inspired patterns
            for attempt in range(300):
                if time.time() > deadline - 10:
                    break
                
                # Use patterns inspired by Strassen with more variety
                U = []
                for i in range(nm):
                    row = []
                    for c in range(target_rank):
                        r = random.random()
                        if r < 0.2:
                            row.append(0)
                        elif r < 0.5:
                            row.append(1)
                        elif r < 0.8:
                            row.append(-1)
                        else:
                            row.append(random.choice([2, -2]))
                    U.append(row)
                
                V = []
                for j in range(mp):
                    row = []
                    for c in range(target_rank):
                        r = random.random()
                        if r < 0.2:
                            row.append(0)
                        elif r < 0.5:
                            row.append(1)
                        elif r < 0.8:
                            row.append(-1)
                        else:
                            row.append(random.choice([2, -2]))
                    V.append(row)
                
                W = []
                for k in range(pn):
                    row = []
                    for c in range(target_rank):
                        r = random.random()
                        if r < 0.2:
                            row.append(0)
                        elif r < 0.5:
                            row.append(1)
                        elif r < 0.8:
                            row.append(-1)
                        else:
                            row.append(random.choice([2, -2]))
                    W.append(row)
                
                if verify(U, V, W, target_rank):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    break
            
            if best_rank == target_rank:
                break

    return best
# EVOLVE-BLOCK-END
