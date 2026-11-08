import numpy as np
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Schoolbook decomposition (rank 36) as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best = {'U': U_school, 'V': V_school, 'W': W_school, 'rank': rank_school}
    
    def check_decomposition(U, V, W):
        """Check if the decomposition is valid using exact arithmetic."""
        n, m, p = DIMENSIONS
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                    if s != Fraction(1):
                        return False
        return True
    
    def check_decomposition_fast(U, V, W):
        """Quick check using floating point for speed."""
        n, m, p = DIMENSIONS
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    # Strategy: Start from schoolbook and try to find lower rank decompositions
    # by exploiting structure. For 3x3x4, we know rank 34 is achievable.
    
    # Try to construct a rank-34 decomposition using known patterns
    # The key insight: we can sometimes merge terms when they share structure
    
    def try_structured_rank34():
        """
        Try to find a rank-34 decomposition by starting from schoolbook
        and modifying it.
        
        For 3x3x4, one approach:
        - The schoolbook has 36 terms: one for each (i,j,k)
        - We need to eliminate 2 terms by adjusting the remaining 34
        
        This is equivalent to finding U, V, W of rank 34 such that
        sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] = 1 for all (i,j,k)
        """
        # Use a systematic approach: start with schoolbook, zero out 2 columns,
        # then solve for the remaining entries
        
        # Actually, let's try a different approach: random search with structure
        # Initialize with schoolbook and perturb
        
        random.seed(42)
        
        # Try various random initializations
        for attempt in range(5000):
            if time.time() > deadline:
                return None
            
            rank = 34
            
            # Start with a random pattern
            U = [[random.choice([0, 1, -1]) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice([0, 1, -1]) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.choice([0, 1, -1]) for _ in range(rank)] for _ in range(p*n)]
            
            # Quick check
            if not check_decomposition_fast(U, V, W):
                continue
            
            # Exact check
            if check_decomposition(U, V, W):
                return {'U': U, 'V': V, 'W': W, 'rank': rank}
        
        return None
    
    def try_local_search_from_schoolbook(target_rank):
        """
        Start from schoolbook and try to reduce rank by merging terms.
        """
        if time.time() > deadline:
            return None
        
        # Start with schoolbook
        rank = n * m * p
        U = [row[:] for row in U_school]
        V = [row[:] for row in V_school]
        W = [row[:] for row in W_school]
        
        # We want to go from rank 36 to target_rank
        # Strategy: try to set some entries to 0 and adjust others
        
        # For now, try a simple approach: randomly zero out entries and check
        # This is unlikely to work directly, but let's try
        
        random.seed(42)
        
        # Try to find a decomposition of target_rank by random search
        # but guided by the schoolbook structure
        
        for attempt in range(10000):
            if time.time() > deadline:
                return None
            
            # Create a random candidate of target_rank
            U_cand = [[0] * target_rank for _ in range(n*m)]
            V_cand = [[0] * target_rank for _ in range(m*p)]
            W_cand = [[0] * target_rank for _ in range(p*n)]
            
            # Initialize with random values
            for i in range(n*m):
                for c in range(target_rank):
                    U_cand[i][c] = random.choice([0, 1, -1])
            for j in range(m*p):
                for c in range(target_rank):
                    V_cand[j][c] = random.choice([0, 1, -1])
            for k in range(p*n):
                for c in range(target_rank):
                    W_cand[k][c] = random.choice([0, 1, -1])
            
            if check_decomposition_fast(U_cand, V_cand, W_cand):
                if check_decomposition(U_cand, V_cand, W_cand):
                    return {'U': U_cand, 'V': V_cand, 'W': W_cand, 'rank': target_rank}
        
        return None
    
    # Try to find rank 34 decomposition
    result = try_structured_rank34()
    if result:
        best = result
    
    # Try to find rank 33 decomposition
    if best['rank'] > 33 and time.time() < deadline:
        result = try_local_search_from_schoolbook(33)
        if result:
            best = result
    
    return {'U': best['U'], 'V': best['V'], 'W': best['W']}
