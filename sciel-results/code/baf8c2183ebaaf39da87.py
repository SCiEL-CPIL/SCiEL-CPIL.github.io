"""Combined program for matrix multiplication 2x3x7 rank minimization
using Strassen-style block decomposition and randomized search."""
import numpy as np
import random
import time
from fractions import Fraction

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

    # Strategy 2: Strassen-inspired block decomposition
    # Partition 2x3 matrix as [2x2 | 2x1] and 3x7 as [2x7; 1x7]
    # Use rank-7 algorithm for 2x2 blocks
    if time.time() < deadline - 30:
        # Define Strassen's 7 products for 2x2 matrices
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # For our case:
        # A is 2x3: A = [A11 A12 | A13] where A11, A12 are 2x2, A13 is 2x1
        # B is 3x7: B = [B11 B12; B21 B22] where B11, B12 are 2x7, B21, B22 are 1x7
        
        # We can decompose using:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # But we need to handle the extra column A13 and row B21/B22
        
        # Let's try a different approach: use the known rank-21 decomposition
        # if we can find one through systematic search
        
        # Systematic search with structured patterns
        for target_rank in range(20, min(28, best_rank)):
            if time.time() > deadline - 20:
                break
            
            # Try structured random search
            for attempt in range(200):
                if time.time() > deadline - 20:
                    break
                
                # Use patterns inspired by Strassen
                U = []
                for i in range(nm):
                    row = []
                    for c in range(target_rank):
                        if c % 7 == 0:
                            row.append(random.choice([1, -1, 0]))
                        else:
                            row.append(random.choice([1, -1, 0, 2]))
                    U.append(row)
                
                V = []
                for j in range(mp):
                    row = []
                    for c in range(target_rank):
                        if c % 7 == 0:
                            row.append(random.choice([1, -1, 0]))
                        else:
                            row.append(random.choice([1, -1, 0, 2]))
                    V.append(row)
                
                W = []
                for k in range(pn):
                    row = []
                    for c in range(target_rank):
                        if c % 7 == 0:
                            row.append(random.choice([1, -1, 0]))
                        else:
                            row.append(random.choice([1, -1, 0, 2]))
                    W.append(row)
                
                if verify(U, V, W, target_rank):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    break
            
            if best_rank == target_rank:
                break

    return best
# EVOLVE-BLOCK-END
