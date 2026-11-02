"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    Uses structured enumeration of half-integer lattice points with algebraic constraints.
    """
    n, m, p = 2, 3, 4
    
    start_time = time.time()
    deadline = start_time + 145
    
    # Target tensor T[a, b, c]
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # T[a, b, c] = 1 if indices match, 0 otherwise.
    T_target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def validate_exact(U, V, W):
        """Check if U, V, W exactly decompose the multiplication tensor using Fraction for exactness."""
        U_frac = [[Fraction(x) for x in U[r]] for r in range(len(U))]
        V_frac = [[Fraction(x) for x in V[r]] for r in range(len(V))]
        W_frac = [[Fraction(x) for x in W[r]] for r in range(len(W))]
        
        rank = len(U[0])
        # Build T_calc[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        T_calc = np.zeros((n*m, m*p, p*n), dtype=object)
        for r in range(rank):
            for a in range(n*m):
                ua = U_frac[a][r]
                if ua == 0:
                    continue
                for b in range(m*p):
                    ub = ua * V_frac[b][r]
                    if ub == 0:
                        continue
                    for c in range(p*n):
                        T_calc[a, b, c] += ub * W_frac[c][r]
        
        # Check exact match
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    if T_calc[i,j,k] != T_target[i,j,k]:
                        return False
        return True

    def validate_fast(U, V, W):
        """Fast float validation for screening."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        T_calc = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        if not np.allclose(T_calc, T_target, atol=1e-9):
            return False
        return True

    # Schoolbook solution of rank 24
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_solution = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_schoolbook

    # Structured search: exploit that each rank-1 component should have
    # support on specific index patterns. For matrix multiplication, natural
    # patterns are:
    # - U supported on one (i,j) pair
    # - V supported on one (j,k) pair  
    # - W supported on one (i,k) pair
    # But we can also allow "sum" patterns where U covers multiple (i,j) with same i, etc.
    
    # Define candidate patterns for each factor
    # For U (size n*m = 6): pattern is which (i,j) indices are active and their values
    # We'll use half-integers: {-1, -0.5, 0, 0.5, 1}
    
    half_ints = [Fraction(-1), Fraction(-1,2), Fraction(0), Fraction(1,2), Fraction(1)]
    
    # For efficiency, we'll do a two-phase approach:
    # Phase 1: Try known structured decompositions (like Strassen-style)
    # Phase 2: Systematic search over small ranks with pattern constraints
    
    # Known result: 2x3x4 matrix multiplication has rank at most 18 (from literature)
    # Let's try to find decompositions of rank 16, 17, 18
    
    # Strategy: For each rank r, enumerate possible "pattern assignments" for each rank component
    # A pattern for U is a sparse vector in {0, ±0.5, ±1}^6
    # We'll restrict to patterns with at most 2 nonzero entries to keep search tractable
    
    # Precompute all sparse patterns with at most 2 nonzeros from half_ints\{0}
    nonzero_half_ints = [Fraction(-1), Fraction(-1,2), Fraction(1,2), Fraction(1)]
    
    def gen_sparse_patterns(dim, max_nonzero=2):
        """Generate all sparse patterns of length dim with at most max_nonzero nonzeros."""
        patterns = []
        # All-zero pattern
        patterns.append([Fraction(0)] * dim)
        
        # One nonzero
        for pos in range(dim):
            for val in nonzero_half_ints:
                pat = [Fraction(0)] * dim
                pat[pos] = val
                patterns.append(pat)
        
        # Two nonzeros
        if max_nonzero >= 2:
            for pos1 in range(dim):
                for pos2 in range(pos1+1, dim):
                    for val1 in nonzero_half_ints:
                        for val2 in nonzero_half_ints:
                            pat = [Fraction(0)] * dim
                            pat[pos1] = val1
                            pat[pos2] = val2
                            patterns.append(pat)
        return patterns
    
    U_patterns = gen_sparse_patterns(n*m)
    V_patterns = gen_sparse_patterns(m*p)
    W_patterns = gen_sparse_patterns(p*n)
    
    # Convert to lists of tuples for hashing
    U_patterns_tup = [tuple(p) for p in U_patterns]
    V_patterns_tup = [tuple(p) for p in V_patterns]
    W_patterns_tup = [tuple(p) for p in W_patterns]
    
    # For each target rank, try combinations of patterns
    # This is still huge, so we use a greedy/beam approach
    
    # Alternative: Use the fact that we can permute columns of U,V,W.
    # So we can assume the first pattern of U is a canonical one.
    
    # Let's try a different structural approach: 
    # Use the isomorphism to embed 2x3x4 into 3x3x3 and leverage known rank-19 decomposition
    # Actually, let's just do systematic search with time limits
    
    # Phase 2: Systematic search for ranks 14-18
    # For each rank, we'll do a beam search over pattern assignments
    
    def beam_search_rank(target_rank, beam_width=500, max_steps=10):
        """
        Beam search over pattern assignments for given rank.
        State: list of (U_patterns, V_patterns, W_patterns) assigned so far
        """
        if time.time() > deadline - 10:
            return None
        
        # Start with empty assignments
        # Beam: list of (score, U_assign, V_assign, W_assign)
        # Score: how many target tensor entries are satisfied so far
        
        # Actually, let's use a simpler approach: random restarts with pattern constraints
        # For each rank component, randomly pick a pattern from the pattern set
        
        random.seed(42)
        
        num_attempts = 200
        for attempt in range(num_attempts):
            if time.time() > deadline - 5:
                break
            
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            for r in range(target_rank):
                # Pick random sparse patterns
                u_pat = random.choice(U_patterns)
                v_pat = random.choice(V_patterns)
                w_pat = random.choice(W_patterns)
                
                for idx in range(n*m):
                    U[idx, r] = float(u_pat[idx])
                for idx in range(m*p):
                    V[idx, r] = float(v_pat[idx])
                for idx in range(p*n):
                    W[idx, r] = float(w_pat[idx])
            
            # Fast validation
            if validate_fast(U, V, W):
                # Exact validation
                U_frac = [[Fraction(U[idx, r]) for r in range(target_rank)] for idx in range(n*m)]
                V_frac = [[Fraction(V[idx, r]) for r in range(target_rank)] for idx in range(m*p)]
                W_frac = [[Fraction(W[idx, r]) for r in range(target_rank)] for idx in range(p*n)]
                
                # Check exactness
                exact = True
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            total = Fraction(0)
                            for r in range(target_rank):
                                total += U_frac[a][r] * V_frac[b][r] * W_frac[c][r]
                            if total != T_target[a,b,c]:
                                exact = False
                                break
                        if not exact:
                            break
                    if not exact:
                        break
                
                if exact:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Try ranks from 14 to 23
    for target_rank in range(14, 24):
        if time.time() > deadline - 15:
            break
        
        result = beam_search_rank(target_rank)
        if result is not None:
            best_solution = result
            best_rank = target_rank
            break
    
    return best_solution
# EVOLVE-BLOCK-END
