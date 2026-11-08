import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Leave 20s margin for safety
    
    # Theoretical minimum rank is n*m*p = 56 (schoolbook).
    # However, for specific dimensions, the rank might be lower.
    # We attempt to find a decomposition with rank < 56.
    # If not found, we fall back to the schoolbook decomposition (rank 56).
    
    # We try ranks from a reasonable lower bound up to 56.
    # A simple lower bound is max(n*m, m*p, p*n) = max(8, 28, 14) = 28.
    # But usually it's close to n*m*p.
    # Let's try a few ranks below 56 to see if we can beat the schoolbook.
    
    # For 2x4 * 4x7, the tensor is T_{i,j,k} = delta_{i,j,k} (actually it's the multiplication tensor).
    # The rank of the matrix multiplication tensor M_{n,m,p} is known to be n*m*p for many cases,
    # but can be lower for specific small cases (e.g., M_{2,2,2} has rank 7 vs 8).
    # For M_{2,4,7}, let's see if rank < 56 is possible.
    
    # We will try to find a decomposition with rank r < 56.
    # If we find one, we return it. Otherwise, we return the schoolbook one.
    
    best_result = None
    
    # Try a few ranks below 56.
    # Given the complexity, we might not find a lower rank quickly.
    # We'll use a randomized approach with a tight time limit.
    
    # Let's try ranks from 40 to 55.
    # If we find a valid decomposition, we return it.
    
    for r in range(40, 56):
        if time.time() > deadline - 5:
            break
        result = _try_rank_r(n, m, p, r, deadline - 5)
        if result is not None:
            best_result = result
            break
    
    if best_result is None:
        # Fallback to schoolbook
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
    
    return best_result

def _try_rank_r(n, m, p, r, deadline):
    """Try to find a rank-r exact decomposition with rational entries."""
    # Use a structured approach: assume U and V have a certain form
    # Then solve for W linearly
    
    # We'll use a randomized approach with a tight time limit.
    # We initialize U and V with small rationals, then solve for W.
    
    # To increase the chance of finding a valid decomposition,
    # we'll try multiple random initializations.
    
    random.seed(42 + r)
    
    # Try multiple random initializations
    for attempt in range(50):
        if time.time() > deadline:
            return None
        
        # Initialize U, V with small rationals
        U = [[Fraction(random.randint(-2, 2), 1) for _ in range(r)] for _ in range(n * m)]
        V = [[Fraction(random.randint(-2, 2), 1) for _ in range(r)] for _ in range(m * p)]
        
        # For each (i,k), we need: sum_j sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
        # This is linear in W for fixed U, V
        # But it's also coupled across (i,k)
        
        # Actually, for fixed U and V, the equations for W are:
        # For each (i,k): sum_t W[k*n+i, t] * (sum_j U[i*m+j, t] * V[j*p+k, t]) = 1
        # Let M[i,k,t] = sum_j U[i*m+j, t] * V[j*p+k, t]
        # Then: sum_t M[i,k,t] * W[k*n+i, t] = 1 for all i,k
        
        # This is a system of n*p equations in p*n unknowns (W has p*n rows, r columns)
        # But it's not a standard linear system because each equation only involves one row of W
        
        # Actually, each equation sum_t M[i,k,t] * W[k*n+i, t] = 1 is independent for each (i,k)
        # So for each (i,k), we need to solve: M[i,k,:] · W[k*n+i, :] = 1
        
        # This is underdetermined if r > 1. We can pick a solution.
        
        # Let's check if for each (i,k), M[i,k,:] is not the zero vector
        valid = True
        W = [[Fraction(0) for _ in range(r)] for _ in range(p * n)]
        
        for i in range(n):
            for k in range(p):
                # Compute M[i,k,:]
                M = [Fraction(0) for _ in range(r)]
                for j in range(m):
                    for t in range(r):
                        M[t] += U[i * m + j][t] * V[j * p + k][t]
                
                # Check if M is zero
                if all(x == 0 for x in M):
                    valid = False
                    break
                
                # Find W[k*n+i, :] such that M · W[k*n+i, :] = 1
                # Simple solution: if M[0] != 0, set W[k*n+i, 0] = 1/M[0], rest 0
                # But this might not work if M[0] = 0
                # Better: find any solution
                # We can set W[k*n+i, t] = M[t] / (sum_t M[t]^2) if sum M[t]^2 != 0
                # Then M · W = sum M[t]^2 / sum M[t]^2 = 1
                
                sum_sq = sum(x * x for x in M)
                if sum_sq == 0:
                    valid = False
                    break
                
                for t in range(r):
                    W[k * n + i][t] = M[t] / sum_sq
        
        if not valid:
            continue
        
        # Verify the decomposition
        if _verify_decomposition(n, m, p, U, V, W):
            # Convert to float for output, but keep as rational
            # The task says integer/half-integer entries
            # Check if all entries are integer or half-integer
            if _all_int_or_half(U) and _all_int_or_half(V) and _all_int_or_half(W):
                return {"U": U, "V": V, "W": W}
            else:
                # Try to scale to get integer/half-integer
                # Find LCM of denominators
                pass
    
    return None

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition is exact."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for t in range(r):
                    total += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                if total != 1:
                    return False
    return True

def _all_int_or_half(mat):
    """Check if all entries are integer or half-integer."""
    for row in mat:
        for x in row:
            # x is a Fraction
            if x.denominator not in (1, 2):
                return False
    return True
# EVOLVE-BLOCK-END
