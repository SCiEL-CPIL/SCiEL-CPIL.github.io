import time
import sys
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    nm, mp, pn = n * m, m * p, p * n

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Rank-44 fallback (Strassen + trivial)
    def _rank44():
        S_U = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        S_V = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        S_W = [
            [1, 0, 0, 1],
            [0, 1, 0, -1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ]
        rank = 44
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for block_idx in range(4):
            k0 = block_idx * 2
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val != 0:
                            U[li * m + la][col] = u_val
                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val != 0:
                            V[la * p + (k0 + lk)][col] = v_val
                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val != 0:
                            W[(k0 + lk) * n + li][col] = w_val
                col += 1
        for i in range(2):
            for k in range(8):
                U[i * m + 2][col] = 1
                V[2 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        return {"U": U, "V": V, "W": W}

    best_rank = 44
    best_solution = _rank44()
    
    if not _validate(best_solution):
        best_solution = _schoolbook()
        best_rank = 48

    if not _elapsed_ok():
        return best_solution

    # Laser method: search for rank-36 solution using 9 terms per 2-column block
    # For each block of 2 columns in B, we need 9 rank-1 terms that compute
    # A(2x3) * B_block(3x2) exactly
    # 
    # Tensor entries for one block: T[i,a,k] for i in {0,1}, a in {0,1,2}, k in {0,1}
    # 12 entries total, each must equal 1
    #
    # We parameterize 9 terms with integer/half-integer entries
    # Each term c has: u_c (6 entries for i,a), v_c (6 entries for a,k), w_c (4 entries for k,i)
    # Constraint: sum_c u_c[i,a] * v_c[a,k] * w_c[k,i] = 1 for all (i,a,k)
    
    # Use a structured ansatz: each u_c has at most 2 nonzeros, v_c at most 2, w_c at most 2
    # Search over sparse patterns with small integer values
    
    def _try_rank36_block():
        """Try to find a rank-9 decomposition for 2x3x2 block"""
        # 12 constraints: for each (i,a,k), sum over c of u[i,a,c]*v[a,k,c]*w[k,i,c] = 1
        # 9 terms, each with up to 6+6+4=16 variables, but sparse
        
        # Use random search with structured initialization
        random.seed(42)
        
        for attempt in range(5000):
            if not _elapsed_ok():
                return None
            
            # Initialize with random sparse patterns
            # Each term: pick 1-2 nonzeros for u, v, w from {±1, ±1/2}
            u_terms = []
            v_terms = []
            w_terms = []
            
            for c in range(9):
                u = [0]*6
                v = [0]*6
                w = [0]*4
                
                # Pick positions and values
                u_pos = random.sample(range(6), random.randint(1,2))
                for pos in u_pos:
                    u[pos] = random.choice([1, -1, 0.5, -0.5])
                
                v_pos = random.sample(range(6), random.randint(1,2))
                for pos in v_pos:
                    v[pos] = random.choice([1, -1, 0.5, -0.5])
                
                w_pos = random.sample(range(4), random.randint(1,2))
                for pos in w_pos:
                    w[pos] = random.choice([1, -1, 0.5, -0.5])
                
                u_terms.append(u)
                v_terms.append(v)
                w_terms.append(w)
            
            # Check if this satisfies all 12 constraints
            valid = True
            for i in range(2):
                for a in range(3):
                    for k in range(2):
                        acc = Fraction(0)
                        for c in range(9):
                            u_val = u_terms[c][i*3+a]
                            v_val = v_terms[c][a*2+k]
                            w_val = w_terms[c][k*2+i]
                            if u_val != 0 and v_val != 0 and w_val != 0:
                                acc += Fraction(u_val) * Fraction(v_val) * Fraction(w_val)
                        if acc != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return (u_terms, v_terms, w_terms)
        
        return None

    # Try to find rank-36 solution
    if _elapsed_ok():
        block_solution = _try_rank36_block()
        if block_solution is not None:
            u_terms, v_terms, w_terms = block_solution
            rank = 36
            U = [[0] * rank for _ in range(nm)]
            V = [[0] * rank for _ in range(mp)]
            W = [[0] * rank for _ in range(pn)]
            
            for block_idx in range(4):
                k0 = block_idx * 2
                for c in range(9):
                    col = block_idx * 9 + c
                    for li in range(2):
                        for la in range(3):
                            val = u_terms[c][li*3+la]
                            if val != 0:
                                U[li*m+la][col] = val
                    for la in range(3):
                        for lk in range(2):
                            val = v_terms[c][la*2+lk]
                            if val != 0:
                                V[la*p+(k0+lk)][col] = val
                    for lk in range(2):
                        for li in range(2):
                            val = w_terms[c][lk*2+li]
                            if val != 0:
                                W[(k0+lk)*n+li][col] = val
            
            candidate = {"U": U, "V": V, "W": W}
            if _validate(candidate):
                best_rank = 36
                best_solution = candidate
    
    if not _elapsed_ok():
        return best_solution

    # Try rank-40: use 10 terms per block (4 blocks * 10 = 40)
    # This allows more flexibility than rank-36
    if _elapsed_ok():
        def _try_rank10_block():
            random.seed(123)
            for attempt in range(3000):
                if not _elapsed_ok():
                    return None
                
                u_terms = []
                v_terms = []
                w_terms = []
                
                for c in range(10):
                    u = [0]*6
                    v = [0]*6
                    w = [0]*4
                    
                    u_pos = random.sample(range(6), random.randint(1,2))
                    for pos in u_pos:
                        u[pos] = random.choice([1, -1, 0.5, -0.5])
                    
                    v_pos = random.sample(range(6), random.randint(1,2))
                    for pos in v_pos:
                        v[pos] = random.choice([1, -1, 0.5, -0.5])
                    
                    w_pos = random.sample(range(4), random.randint(1,2))
                    for pos in w_pos:
                        w[pos] = random.choice([1, -1, 0.5, -0.5])
                    
                    u_terms.append(u)
                    v_terms.append(v)
                    w_terms.append(w)
                
                valid = True
                for i in range(2):
                    for a in range(3):
                        for k in range(2):
                            acc = Fraction(0)
                            for c in range(10):
                                u_val = u_terms[c][i*3+a]
                                v_val = v_terms[c][a*2+k]
                                w_val = w_terms[c][k*2+i]
                                if u_val != 0 and v_val != 0 and w_val != 0:
                                    acc += Fraction(u_val) * Fraction(v_val) * Fraction(w_val)
                            if acc != 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return (u_terms, v_terms, w_terms)
            
            return None
        
        block_solution = _try_rank10_block()
        if block_solution is not None:
            u_terms, v_terms, w_terms = block_solution
            rank = 40
            U = [[0] * rank for _ in range(nm)]
            V = [[0] * rank for _ in range(mp)]
            W = [[0] * rank for _ in range(pn)]
            
            for block_idx in range(4):
                k0 = block_idx * 2
                for c in range(10):
                    col = block_idx * 10 + c
                    for li in range(2):
                        for la in range(3):
                            val = u_terms[c][li*3+la]
                            if val != 0:
                                U[li*m+la][col] = val
                    for la in range(3):
                        for lk in range(2):
                            val = v_terms[c][la*2+lk]
                            if val != 0:
                                V[la*p+(k0+lk)][col] = val
                    for lk in range(2):
                        for li in range(2):
                            val = w_terms[c][lk*2+li]
                            if val != 0:
                                W[(k0+lk)*n+li][col] = val
            
            candidate = {"U": U, "V": V, "W": W}
            if _validate(candidate):
                if rank < best_rank:
                    best_rank = rank
                    best_solution = candidate

    return best_solution
