import numpy as np
import time
import random
from fractions import Fraction
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 32
    W_dim = p * n  # 16
    
    deadline = time.time() + 145.0  # Safety margin

    # Fallback: Schoolbook (Rank 64)
    best_rank = n * m * p
    best_U = np.zeros((U_dim, best_rank))
    best_V = np.zeros((V_dim, best_rank))
    best_W = np.zeros((W_dim, best_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j, col] = 1
                best_V[j*p+k, col] = 1
                best_W[k*n+i, col] = 1
                col += 1

    def verify_decomp(U, V, W, tol=1e-9):
        # Use high precision check
        # T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
        # We check if all 64 entries are 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for t in range(U.shape[1]):
                        val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def check_integer_or_half(U, V, W):
        # Check if all entries are integers or half-integers
        # The problem statement allows integer/half-integer entries
        # "score is the rank" ... "entries that exactly decompose"
        # Usually implies rational numbers with small denominators.
        # Let's assume integer or half-integer (denominator 2) is sufficient for the evaluator
        # if the values are exact.
        for mat in [U, V, W]:
            if mat.size == 0: continue
            # Check if 2*x is close to an integer
            two_x = 2.0 * mat
            rounded = np.round(two_x)
            if not np.allclose(two_x, rounded, atol=1e-9):
                return False
        return True

    # Strategy: Deterministic Construction via "Block Diagonal" or "Shared Structure"
    # The tensor is T[i,j,k] = 1.
    # We can try to decompose using a smaller rank by exploiting the fact that
    # the 4x4 submatrices are all identical (all ones).
    
    # Let's try to find a solution with rank r where r < 64.
    # A known lower bound for 2x4x8 is related to the rank of 4x4x4 which is 18 or 19?
    # Actually, rank(4x4x4) is 18.
    # For 2x4x8, we can view it as two 4x4x4 tensors? No, dimensions are different.
    
    # Let's try a structured approach:
    # U has shape (8, r), V (32, r), W (16, r).
    
    # Approach: "Greedy Peeling" with Exact Arithmetic
    # We want sum_t u_t v_t w_t = 1 (elementwise)
    
    # Let's try to construct a solution with rank 32, 24, etc.
    # Since random search failed, we try a deterministic pattern.
    
    # Pattern 1: Basis Expansion
    # If we use the standard basis for the "middle" dimension, we get rank 64.
    # Can we do better?
    
    # Let's try to solve for a specific small rank using a linear algebra trick.
    # Fix U and V to be sparse integers. Solve for W.
    # To make W integer, we need the system to be "nice".
    
    # Alternative: Use the fact that T is a product of three all-ones matrices?
    # No, T[i,j,k] = 1.
    
    # Let's try a specific constructive algorithm for rank 16, 20, 24...
    # We will try to fill the factors column by column (term by term).
    # This is a constraint satisfaction problem.
    
    # Given the time limit and the difficulty of finding sub-64 rank integer solutions
    # for this specific tensor via random search, we will attempt a more targeted search.
    
    # Targeted Search:
    # 1. Assume W has entries in {-1, 0, 1}.
    # 2. Assume U and V have entries in {-1, 0, 1}.
    # 3. This restricts the search space significantly.
    
    # However, 5000 random attempts in the previous code failed.
    # Let's try a different initialization: 
    # Use a Hadamard-like structure or simple repeating patterns.
    
    # Let's try to find a rank-32 solution first, then try to reduce.
    # Actually, let's just try to find ANY rank < 64.
    
    # Improved Random Search with Structure:
    # Instead of pure random, we use "structured random":
    # U and V are block-diagonal or have repeating rows.
    
    found = False
    
    # Try ranks from 24 to 63
    # We will use a timeout-aware loop.
    
    for target_rank in range(24, 64):
        if time.time() > deadline - 2:
            break
            
        # Try a few structured patterns for this rank
        for pattern_idx in range(50):
            if time.time() > deadline - 1:
                break
                
            U = np.zeros((U_dim, target_rank))
            V = np.zeros((V_dim, target_rank))
            
            # Generate U and V with small integers {-1, 1}
            # To increase chance of integer W, let's make U and V orthogonal-like or sparse.
            
            # Simple strategy: Random {-1, 1}
            U = np.random.randint(-1, 2, size=(U_dim, target_rank)).astype(float)
            V = np.random.randint(-1, 2, size=(V_dim, target_rank)).astype(float)
            
            # Solve for W
            # System: for each i in 0..1, for each j in 0..3, for each k in 0..7:
            # sum_t U[i,j,t] V[j,k,t] W[k,i,t] = 1
            # This is linear in W.
            # We have 2 * 4 * 8 = 64 equations.
            # Unknowns: W is 16 x target_rank. Total 16 * target_rank unknowns.
            # If target_rank > 16, we have more unknowns than equations (underdetermined).
            # We want a specific solution where W is integer/half-integer.
            
            # Let's stack the equations for all i.
            # Actually, the equations for different i are coupled only through the shared t index.
            # But W[k, i, t] is separate for each i.
            # So for each i, we have a system:
            # Sum_{j,k} (U[i,j,t] V[j,k,t]) W[k,i,t] = 1 for all j,k.
            # Let M_i be a matrix of size (m*p) x (p*rank).
            # M_i[j*p+k, t*p+l] = U[i,j,t] * V[j,k,t] if l==k else 0?
            # No, W index is [k*n+i, t].
            # Let's vectorize W_i (for fixed i) as a vector of length p*rank.
            # W_i[k*rank + t] = W[k*n+i, t].
            
            # Equation for fixed i, j, k:
            # Sum_t U[i,j,t] * V[j,k,t] * W_i[k*rank + t] = 1.
            
            # This is a linear system A_i * W_i = 1.
            # A_i has size (m*p) x (p*rank) = 32 x (8*rank).
            
            W = np.zeros((W_dim, target_rank))
            valid = True
            
            for i in range(n):
                A_i = np.zeros((m*p, p*target_rank))
                rhs = np.ones(m*p)
                
                for j in range(m):
                    for k in range(p):
                        eq_idx = j*p + k
                        for t in range(target_rank):
                            # Coefficient for W[k*rank + t] is U[i,j,t] * V[j,k,t]
                            A_i[eq_idx, k*target_rank + t] = U[i*m+j, t] * V[j*p+k, t]
                
                # Solve least squares
                try:
                    W_i, residuals, rank_sol, sv = np.linalg.lstsq(A_i, rhs, rcond=None)
                except:
                    valid = False
                    break
                
                # Check if W_i is close to integer or half-integer
                # We allow tolerance 1e-6
                W_i_float = W_i
                W_i_int = np.round(W_i_float)
                W_i_half = np.round(W_i_float * 2) / 2
                
                if np.allclose(W_i_float, W_i_int, atol=1e-6):
                    W_i_final = W_i_int
                elif np.allclose(W_i_float, W_i_half, atol=1e-6):
                    W_i_final = W_i_half
                else:
                    valid = False
                    break
                
                # Reshape W_i_final to (p, rank) and assign to W[:, i, :]
                # W_i is indexed by [k*rank + t]
                W_block = W_i_final.reshape(p, target_rank).T
                # W shape is (p*n, rank), indexed as W[k*n + i, t]
                for k in range(p):
                    for t in range(target_rank):
                        W[k*n + i, t] = W_block[t, k]
            
            if not valid:
                continue
            
            if verify_decomp(U, V, W):
                if check_integer_or_half(U, V, W):
                    best_rank = target_rank
                    best_U = U
                    best_V = V
                    best_W = W
                    found = True
                    break
        
        if found:
            break

    # If we didn't find anything better than 64, return the schoolbook solution.
    # Note: The schoolbook solution uses integers, so it is valid.
    
    # Ensure output format is correct
    # "integer/half-integer entries"
    # We round to half-integers to be safe, although integers are better.
    
    U_out = np.round(best_U * 2) / 2
    V_out = np.round(best_V * 2) / 2
    W_out = np.round(best_W * 2) / 2

    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
