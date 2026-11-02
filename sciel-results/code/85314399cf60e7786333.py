"""Rewritten program: search for a lower-rank exact tensor decomposition of 2x3 * 3x4."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]
n, m, p = DIMENSIONS

# Precompute the multiplication tensor T[i, j, k] = 1 for all i,j,k
# The tensor equation is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
# We want to find rank r such that there exist U (n*m x r), V (m*p x r), W (p*n x r)
# with integer/half-integer entries satisfying the equation exactly.

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8

    # Schoolbook rank is 24. We know that for 2x3*3x4, the rank is at most 24.
    # We'll try to find a decomposition of rank < 24.
    
    # Known lower bound: max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # But the true rank for 2x3*3x4 is known to be 24 (it's a full tensor).
    # Actually, let me check: the rank of the 2x3*3x4 multiplication tensor.
    # For matrix multiplication C_{n x p} = A_{n x m} * B_{m x p},
    # the tensor rank is at least max(n*m, m*p, p*n) and at most n*m*p.
    # For 2x3*3x4: lower bound = max(6, 12, 8) = 12, upper bound = 24.
    
    # Let's try to find a decomposition with rank < 24 using a search approach.
    # Since the entries must be integer or half-integer, we can scale by 2 to work with integers.
    
    # Strategy: Try ranks from the lower bound up to 24, and for each rank,
    # try to find a valid decomposition using a constraint satisfaction approach.
    
    # For small ranks, we can use a randomized search with verification.
    
    best_result = None
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
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
    best_result = {"U": U_school, "V": V_school, "W": W_school, "rank": rank_school}
    
    # Try to find lower rank decompositions
    # We'll use a randomized approach: for a given rank r, try random integer matrices
    # and check if they satisfy the tensor equation.
    
    # Actually, let's think about this differently. The tensor equation is:
    # For all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p = 24 equations in r*3*(nm+mp+pn) variables, but with the
    # constraint that the product structure must hold.
    
    # A better approach: use the fact that we can fix one matrix and solve for others.
    # Or use a greedy/iterative approach.
    
    # Let's try a different strategy: use the known structure of matrix multiplication.
    # For 2x3*3x4, we can try to find a decomposition by exploiting the small dimensions.
    
    # Actually, for this specific case, let's just try to verify if rank 24 is indeed minimal.
    # The lower bound is 12, but achieving 12 is very unlikely for this tensor.
    
    # Let's try a search for rank < 24 with a time limit.
    # We'll use a randomized local search approach.
    
    import random
    random.seed(42)
    
    # Try ranks from 12 to 23
    for target_rank in range(12, rank_school):
        if time.time() > deadline - 10:
            break
        
        # For each rank, try a few random initializations
        max_attempts = 50 if target_rank <= 16 else 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Generate random integer matrices with small entries
            U = np.random.randint(-2, 3, size=(nm, target_rank))
            V = np.random.randint(-2, 3, size=(mp, target_rank))
            W = np.random.randint(-2, 3, size=(pn, target_rank))
            
            # Check if the tensor equation is satisfied
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(target_rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to list of lists with Fraction entries
                U_list = [[int(U[i, c]) for c in range(target_rank)] for i in range(nm)]
                V_list = [[int(V[i, c]) for c in range(target_rank)] for i in range(mp)]
                W_list = [[int(W[i, c]) for c in range(target_rank)] for i in range(pn)]
                best_result = {"U": U_list, "V": V_list, "W": W_list, "rank": target_rank}
                return best_result  # Found a valid decomposition, return immediately
    
    # If no lower rank found, return schoolbook
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
