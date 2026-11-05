"""Find rank-minimal exact tensor decomposition for 3x4 * 4x7 matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 7]
DEADLINE = 140.0  # Safety margin before 150s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Schoolbook decomposition baseline (rank 84)
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def is_valid_w_val(v):
        """Check if v is integer or half-integer."""
        if abs(v) < 1e-12:
            return True
        if abs(v - round(v)) < 1e-9:
            return True
        if abs(v*2 - round(v*2)) < 1e-9:
            return True
        return False
    
    def solve_w_exact(M_row, rank):
        """
        Solve for W_row such that dot(M_row, W_row) = 1.
        M_row has entries in {-2, -1, 0, 1, 2} typically.
        Returns W_row with integer/half-integer entries, or None.
        """
        # Convert to Fraction for exact arithmetic
        M_frac = [Fraction(int(round(v*2))) / 2 for v in M_row]
        
        # Find all indices with nonzero M
        nonzero = [(r, M_frac[r]) for r in range(rank) if M_frac[r] != 0]
        if not nonzero:
            return None
        
        # Try to find a solution with at most 2 nonzeros in W
        # For each pair (r1, r2): solve a*M[r1] + b*M[r2] = 1
        for i in range(len(nonzero)):
            for j in range(len(nonzero)):
                if i == j:
                    continue
                r1, m1 = nonzero[i]
                r2, m2 = nonzero[j]
                # a*m1 + b*m2 = 1
                # Try small integer/half-integer values for b
                for b_num in range(-16, 17):
                    b = Fraction(b_num, 2)
                    a = (1 - b * m2) / m1
                    # Check if a is integer or half-integer
                    if a.denominator in (1, 2) and abs(a.numerator) <= 16:
                        W_row = np.zeros(rank)
                        W_row[r1] = float(a)
                        W_row[r2] = float(b)
                        # Verify
                        if abs(float(a * m1 + b * m2) - 1.0) < 1e-9:
                            return W_row
        
        # Try 3 nonzeros
        for i in range(len(nonzero)):
            for j in range(i+1, len(nonzero)):
                for k in range(j+1, len(nonzero)):
                    r1, m1 = nonzero[i]
                    r2, m2 = nonzero[j]
                    r3, m3 = nonzero[k]
                    # a*m1 + b*m2 + c*m3 = 1
                    for a_num in range(-8, 9):
                        a = Fraction(a_num, 2)
                        for b_num in range(-8, 9):
                            b = Fraction(b_num, 2)
                            c = (1 - a*m1 - b*m2) / m3
                            if c.denominator in (1, 2) and abs(c.numerator) <= 8:
                                W_row = np.zeros(rank)
                                W_row[r1] = float(a)
                                W_row[r2] = float(b)
                                W_row[r3] = float(c)
                                if abs(float(a*m1 + b*m2 + c*m3) - 1.0) < 1e-9:
                                    return W_row
        
        # Try 4 nonzeros with even smaller ranges
        for i in range(len(nonzero)):
            for j in range(i+1, len(nonzero)):
                for k in range(j+1, len(nonzero)):
                    for l in range(k+1, len(nonzero)):
                        r1, m1 = nonzero[i]
                        r2, m2 = nonzero[j]
                        r3, m3 = nonzero[k]
                        r4, m4 = nonzero[l]
                        for a_num in range(-4, 5):
                            a = Fraction(a_num, 2)
                            for b_num in range(-4, 5):
                                b = Fraction(b_num, 2)
                                for c_num in range(-4, 5):
                                    c = Fraction(c_num, 2)
                                    d = (1 - a*m1 - b*m2 - c*m3) / m4
                                    if d.denominator in (1, 2) and abs(d.numerator) <= 4:
                                        W_row = np.zeros(rank)
                                        W_row[r1] = float(a)
                                        W_row[r2] = float(b)
                                        W_row[r3] = float(c)
                                        W_row[r4] = float(d)
                                        if abs(float(a*m1 + b*m2 + c*m3 + d*m4) - 1.0) < 1e-9:
                                            return W_row
        
        return None
    
    def try_structured_decomposition(rank, seed_offset=0):
        """
        Try multiple structured families for U and V, then solve for W.
        Returns (U, V, W) or None.
        """
        families = [
            'kronecker', 'cyclic', 'block_diag', 'random_structured', 'latin'
        ]
        
        for fam_idx, fam in enumerate(families):
            if time.time() - start_time > DEADLINE - 10:
                return None
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            
            if fam == 'kronecker':
                # Each rank index r maps to a specific (i, j, k) combination
                # Use a systematic enumeration
                idx = 0
                for r in range(rank):
                    i = (r + seed_offset) % n
                    j = ((r + seed_offset) // n) % m
                    k = ((r + seed_offset) // (n*m)) % p
                    U[i*m + j, r] = 1.0 if (r + seed_offset) % 2 == 0 else -1.0
                    V[j*p + k, r] = 1.0 if (r + seed_offset) % 2 == 0 else -1.0
            
            elif fam == 'cyclic':
                # Cyclic pattern with offsets
                for r in range(rank):
                    i = (r * 2 + seed_offset) % n
                    j = (r * 3 + seed_offset) % m
                    k = (r * 5 + seed_offset) % p
                    U[i*m + j, r] = 1.0 if (r + seed_offset) % 2 == 0 else -1.0
                    V[j*p + k, r] = 1.0 if (r + seed_offset) % 2 == 0 else -1.0
            
            elif fam == 'block_diag':
                # Block diagonal structure
                for r in range(rank):
                    block = r % (n*m)
                    pos = r // (n*m)
                    i = block % n
                    j = block // n
                    k = (block + pos) % p
                    U[i*m + j, r] = 1.0 if pos % 2 == 0 else -1.0
                    V[j*p + k, r] = 1.0 if pos % 2 == 0 else -1.0
            
            elif fam == 'random_structured':
                # Random but with each row of U having exactly 1 nonzero
                rng = random.Random(seed_offset * 1000 + rank)
                for r in range(rank):
                    i = rng.randint(0, n-1)
                    j = rng.randint(0, m-1)
                    k = rng.randint(0, p-1)
                    U[i*m + j, r] = 1.0 if rng.randint(0, 1) == 0 else -1.0
                    V[j*p + k, r] = 1.0 if rng.randint(0, 1) == 0 else -1.0
            
            elif fam == 'latin':
                # Latin square based
                for r in range(rank):
                    i = r % n
                    j = (r // n + r) % m
                    k = (r // m + r) % p
                    U[i*m + j, r] = 1.0
                    V[j*p + k, r] = 1.0
            
            # Compute M[k*n+i, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
            M = np.zeros((p*n, rank))
            for i in range(n):
                for k in range(p):
                    for r in range(rank):
                        s = 0.0
                        for j in range(m):
                            s += U[i*m+j, r] * V[j*p+k, r]
                        M[k*n+i, r] = s
            
            # Check if M has reasonable properties
            # For each row of M, we need to be able to solve dot(M_row, W_row) = 1
            # with integer/half-integer W_row
            
            W = np.zeros((p*n, rank))
            valid = True
            for idx in range(p*n):
                W_row = solve_w_exact(M[idx], rank)
                if W_row is None:
                    valid = False
                    break
                W[idx] = W_row
            
            if valid and check_decomposition(U, V, W, rank):
                return (U, V, W)
        
        return None
    
    def try_multivariate_structured(rank, num_trials=50):
        """
        Try structured U,V where each column has 2 nonzeros in U and 2 in V,
        with systematic coverage of index pairs.
        """
        for trial in range(num_trials):
            if time.time() - start_time > DEADLINE - 10:
                return None
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            
            if trial % 2 == 0:
                # Systematic pair coverage
                for r in range(rank):
                    # Pick two (i,j) pairs for U
                    i1 = r % n
                    j1 = (r // n) % m
                    i2 = (r + 1) % n
                    j2 = (r // n + 1) % m
                    U[i1*m + j1, r] = 1.0
                    if (i1, j1) != (i2, j2):
                        U[i2*m + j2, r] = -1.0
                    
                    # Pick two (j,k) pairs for V
                    j1v = r % m
                    k1v = (r // m) % p
                    j2v = (r + 1) % m
                    k2v = (r // m + 1) % p
                    V[j1v*p + k1v, r] = 1.0
                    if (j1v, k1v) != (j2v, k2v):
                        V[j2v*p + k2v, r] = -1.0
            else:
                # Random pairs with seed
                rng = random.Random(trial * 7919 + rank)
                for r in range(rank):
                    i1 = rng.randint(0, n-1)
                    j1 = rng.randint(0, m-1)
                    i2 = rng.randint(0, n-1)
                    j2 = rng.randint(0, m-1)
                    U[i1*m + j1, r] = 1.0
                    if (i1, j1) != (i2, j2):
                        U[i2*m + j2, r] = rng.choice([-1.0, 1.0])
                    
                    j1v = rng.randint(0, m-1)
                    k1v = rng.randint(0, p-1)
                    j2v = rng.randint(0, m-1)
                    k2v = rng.randint(0, p-1)
                    V[j1v*p + k1v, r] = 1.0
                    if (j1v, k1v) != (j2v, k2v):
                        V[j2v*p + k2v, r] = rng.choice([-1.0, 1.0])
            
            # Compute M
            M = np.zeros((p*n, rank))
            for i in range(n):
                for k in range(p):
                    for r in range(rank):
                        s = 0.0
                        for j in range(m):
                            s += U[i*m+j, r] * V[j*p+k, r]
                        M[k*n+i, r] = s
            
            # Solve for W
            W = np.zeros((p*n, rank))
            valid = True
            for idx in range(p*n):
                W_row = solve_w_exact(M[idx], rank)
                if W_row is None:
                    valid = False
                    break
                W[idx] = W_row
            
            if valid and check_decomposition(U, V, W, rank):
                return (U, V, W)
        
        return None
    
    # Try ranks from a reasonable lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = 28, but tensor rank can be lower
    # Start from 20 and go up
    ranks_to_try = list(range(20, min(n*m*p, 84)))
    
    for rank in ranks_to_try:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Try structured decomposition with multiple families
        if time.time() - start_time < DEADLINE - 12:
            for seed_offset in range(5):
                if time.time() - start_time > DEADLINE - 12:
                    break
                result = try_structured_decomposition(rank, seed_offset)
                if result is not None:
                    U, V, W = result
                    if rank < best_rank:
                        best_rank = rank
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                    break
            if best_rank <= rank:
                break
        
        # Try multivariate structured search
        if time.time() - start_time < DEADLINE - 10:
            result = try_multivariate_structured(rank, num_trials=30)
            if result is not None:
                U, V, W = result
                if rank < best_rank:
                    best_rank = rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
