"""Tensor decomposition for 3x3 * 3x6 matrix multiplication."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 6]
n, m, p = DIMENSIONS

def verify(U, V, W, rank):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] == 1 for all valid (a,b,c)."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                if s != 1:
                    return False
    return True

def construct():
    deadline = time.time() + 140.0
    
    def try_rank(r):
        """Try to find a decomposition of rank r."""
        num_eq = n * m * p  # 54
        num_unk = p * n * r  # 18 * r
        
        if num_unk < num_eq:
            return None
        
        # Try several random initializations
        for attempt in range(50):
            if time.time() > deadline:
                return None
            
            # Random U and V with small rational entries
            vals = [-2, -1, -0.5, 0.5, 1, 2]
            U = [[random.choice(vals) for _ in range(r)] for _ in range(n * m)]
            V = [[random.choice(vals) for _ in range(r)] for _ in range(m * p)]
            
            # Build linear system for W
            A = [[Fraction(0) for _ in range(num_unk)] for _ in range(num_eq)]
            b = [Fraction(1) for _ in range(num_eq)]
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r_idx in range(r):
                            coeff = Fraction(U[i * m + j][r_idx]) * Fraction(V[j * p + k][r_idx])
                            unk_idx = k * (n * r) + i * r + r_idx
                            A[eq_idx][unk_idx] = coeff
                        eq_idx += 1
            
            # Gaussian elimination
            aug = [row[:] + [b[i]] for i, row in enumerate(A)]
            
            rank_found = 0
            for col in range(num_unk):
                pivot = -1
                for row in range(rank_found, num_eq):
                    if aug[row][col] != 0:
                        pivot = row
                        break
                if pivot == -1:
                    continue
                aug[rank_found], aug[pivot] = aug[pivot], aug[rank_found]
                pv = aug[rank_found][col]
                for c in range(col, num_unk + 1):
                    aug[rank_found][c] = aug[rank_found][c] / pv
                for row in range(num_eq):
                    if row != rank_found and aug[row][col] != 0:
                        factor = aug[row][col]
                        for c in range(col, num_unk + 1):
                            aug[row][c] -= factor * aug[rank_found][c]
                rank_found += 1
            
            # Check consistency
            consistent = True
            for row in range(rank_found, num_eq):
                if all(aug[row][c] == 0 for c in range(num_unk)) and aug[row][num_unk] != 0:
                    consistent = False
                    break
            
            if not consistent:
                continue
            
            # Extract solution
            w = [Fraction(0) for _ in range(num_unk)]
            for row in range(rank_found):
                lead_col = -1
                for c in range(num_unk):
                    if aug[row][c] != 0:
                        lead_col = c
                        break
                if lead_col != -1:
                    w[lead_col] = aug[row][num_unk]
            
            # Reconstruct W
            W = [[Fraction(0) for _ in range(r)] for _ in range(p * n)]
            for k in range(p):
                for i in range(n):
                    for r_idx in range(r):
                        unk_idx = k * (n * r) + i * r + r_idx
                        W[k * n + i][r_idx] = w[unk_idx]
            
            if verify(U, V, W, r):
                return U, V, W
        
        return None
    
    # Search for minimal rank
    best_result = None
    for r in range(18, 55):
        if time.time() > deadline:
            break
        result = try_rank(r)
        if result is not None:
            best_result = (r, result)
            break
    
    if best_result is None:
        # Schoolbook fallback
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    r, (U, V, W) = best_result
    return {"U": U, "V": V, "W": W}
