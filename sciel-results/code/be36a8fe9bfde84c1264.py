"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction
import random
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 20
    pn = p * n  # 15
    
    # Schoolbook decomposition as baseline
    rank_school = n * m * p  # 60
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    def verify(U, V, W):
        """Verify exact decomposition with rational entries."""
        r = len(U[0])
        if len(U) != nm or len(V) != mp or len(W) != pn:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                    if s != 1:
                        return False
        return True
    
    def verify_scaled(U2, V2, W2):
        """Verify with scaled integer entries (all entries * 2)."""
        r = len(U2[0])
        if len(U2) != nm or len(V2) != mp or len(W2) != pn:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U2[i * m + j][c] * V2[j * p + k][c] * W2[k * n + i][c]
                    if s != 8:  # 2*2*2 = 8
                        return False
        return True
    
    # Strategy: Try to find lower-rank decompositions
    # Work with scaled integers for exactness
    
    # First, try structured approaches based on known algebraic identities
    # For small matrices, sometimes rank can be reduced using special structures
    
    # Try ranks from 59 down to a reasonable lower bound
    # Lower bound: max(nm, mp, pn) = 20, but realistically much higher
    
    for rank in range(59, 35, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations with small integer values
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with small integers (-2, -1, 0, 1, 2)
            U2 = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(nm)]
            V2 = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(mp)]
            W2 = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(pn)]
            
            if verify_scaled(U2, V2, W2):
                # Convert back to half-integers
                U = [[x / 2 for x in row] for row in U2]
                V = [[x / 2 for x in row] for row in V2]
                W = [[x / 2 for x in row] for row in W2]
                best_rank = rank
                best_result = {"U": U, "V": V, "W": W}
                return best_result
        
        # Also try with entries in {-1, 0, 1} scaled
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            U2 = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(nm)]
            V2 = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(mp)]
            W2 = [[random.randint(-1, 1) for _ in range(rank)] for _ in range(pn)]
            
            if verify_scaled(U2, V2, W2):
                U = [[x / 2 for x in row] for row in U2]
                V = [[x / 2 for x in row] for row in V2]
                W = [[x / 2 for x in row] for row in W2]
                best_rank = rank
                best_result = {"U": U, "V": V, "W": W}
                return best_result
    
    # If no improvement found, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
