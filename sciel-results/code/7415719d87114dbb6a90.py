"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 24
    
    # Start with schoolbook decomposition
    U = [[Fraction(0)] * schoolbook_rank for _ in range(n * m)]
    V = [[Fraction(0)] * schoolbook_rank for _ in range(m * p)]
    W = [[Fraction(0)] * schoolbook_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    best_rank = schoolbook_rank
    
    # Helper: verify decomposition
    def verify(U_mat, V_mat, W_mat, rank):
        """Check if decomposition exactly matches matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += U_mat[i * m + j][r] * V_mat[j * p + k][r] * W_mat[k * n + i][r]
                    if total != 1:
                        return False
        return True
    
    # Verify initial
    if not verify(U, V, W, schoolbook_rank):
        raise ValueError("Initial schoolbook decomposition failed verification")
    
    # Structured search using group-theoretic ideas
    # For 2x3x4, try to find decompositions using structured patterns
    
    # Approach: Use random linear combinations of schoolbook terms
    # If we take a,b,c as coefficients and form new terms, check if valid
    
    # Better: Try to find a rank-r decomposition by solving the system
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # Use a greedy/structured approach: try patterns based on modular arithmetic
    # For each rank r from 20 to 23, try structured initializations
    
    def try_structured_rank(target_rank, seed_val):
        """Try structured initialization for given rank."""
        rng = random.Random(seed_val)
        
        # Strategy 1: Use schoolbook terms but try to merge via linear combinations
        # Create target_rank terms as random linear combinations of schoolbook terms
        # Each new term r is a combination: sum_c alpha[c,r] * schoolbook_term[c]
        # This means U_new[i,j,r] = sum_c alpha[c,r] * U_school[i,j,c]
        # But U_school[i,j,c] is 1 only when c corresponds to (i,j,k_c)
        # So U_new[i,j,r] = sum_{c: i_c=i, j_c=j} alpha[c,r]
        
        # Generate random alpha matrix (schoolbook_rank x target_rank)
        # Use small half-integer values
        alpha = [[rng.choice([Fraction(0), Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2), Fraction(3,2), Fraction(-3,2)]) 
                  for _ in range(target_rank)] for _ in range(schoolbook_rank)]
        
        U_try = [[Fraction(0)] * target_rank for _ in range(n * m)]
        V_try = [[Fraction(0)] * target_rank for _ in range(m * p)]
        W_try = [[Fraction(0)] * target_rank for _ in range(p * n)]
        
        for c in range(schoolbook_rank):
            i_c = c // (m * p)
            j_c = (c // p) % m
            k_c = c % p
            for r in range(target_rank):
                a = alpha[c][r]
                if a != 0:
                    U_try[i_c * m + j_c][r] += a
                    V_try[j_c * p + k_c][r] += a
                    W_try[k_c * n + i_c][r] += a
        
        if verify(U_try, V_try, W_try, target_rank):
            return U_try, V_try, W_try
        return None
    
    # Try structured approaches for ranks 20-23
    for target_rank in range(20, schoolbook_rank):
        if time.time() > deadline - 15:
            break
            
        max_attempts = 200 if target_rank <= 21 else 100
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
                
            result = try_structured_rank(target_rank, seed_val=attempt * 1000 + target_rank)
            if result is not None:
                U_try, V_try, W_try = result
                best_U = U_try
                best_V = V_try
                best_W = W_try
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    # If no improvement found, try random search with better patterns
    if best_rank == schoolbook_rank:
        # Try pure random with more diverse value sets
        for target_rank in range(20, schoolbook_rank):
            if time.time() > deadline - 10:
                break
            
            max_attempts = 300 if target_rank <= 21 else 150
            for attempt in range(max_attempts):
                if time.time() > deadline - 5:
                    break
                
                rng = random.Random(attempt * 7919 + target_rank * 31)
                vals = [Fraction(0), Fraction(1), Fraction(-1), Fraction(1,2), Fraction(-1,2), 
                        Fraction(2), Fraction(-2), Fraction(3,2), Fraction(-3,2)]
                
                U_try = [[rng.choice(vals) for _ in range(target_rank)] for _ in range(n * m)]
                V_try = [[rng.choice(vals) for _ in range(target_rank)] for _ in range(m * p)]
                W_try = [[rng.choice(vals) for _ in range(target_rank)] for _ in range(p * n)]
                
                if verify(U_try, V_try, W_try, target_rank):
                    best_U = U_try
                    best_V = V_try
                    best_W = W_try
                    best_rank = target_rank
                    break
            
            if best_rank < schoolbook_rank:
                break
    
    # Try local improvement: from best found, try to reduce rank by 1
    while best_rank > 20 and time.time() < deadline - 5:
        target_rank = best_rank - 1
        improved = False
        
        # Try to project the current decomposition to lower rank
        # Keep first target_rank columns and see if they still work
        # Or try random linear combinations
        
        for attempt in range(50):
            if time.time() > deadline - 3:
                break
            
            rng = random.Random(attempt * 104729 + best_rank)
            
            # Try: new_decomp = old_decomp * random_invertible_matrix (conceptually)
            # In practice: take random linear combinations of existing columns
            # New col r = sum_s M[s,r] * old_col_s where M is target_rank x best_rank
            
            # Generate M
            M = [[rng.choice([Fraction(0), Fraction(1), Fraction(-1)]) for _ in range(best_rank)] 
                 for _ in range(target_rank)]
            
            # Check if M has full rank (at least target_rank)
            # For simplicity, just try
            
            U_try = [[Fraction(0)] * target_rank for _ in range(n * m)]
            V_try = [[Fraction(0)] * target_rank for _ in range(m * p)]
            W_try = [[Fraction(0)] * target_rank for _ in range(p * n)]
            
            for r in range(target_rank):
                for s in range(best_rank):
                    coeff = M[r][s]
                    if coeff != 0:
                        for i in range(n * m):
                            U_try[i][r] += coeff * best_U[i][s]
                        for j in range(m * p):
                            V_try[j][r] += coeff * best_V[j][s]
                        for k in range(p * n):
                            W_try[k][r] += coeff * best_W[k][s]
            
            if verify(U_try, V_try, W_try, target_rank):
                best_U = U_try
                best_V = V_try
                best_W = W_try
                best_rank = target_rank
                improved = True
                break
        
        if not improved:
            break
    
    # Convert to list of lists of floats
    U_out = [[float(x) for x in row] for row in best_U]
    V_out = [[float(x) for x in row] for row in best_V]
    W_out = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
