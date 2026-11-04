import numpy as np
import time
import random

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    def verify(U, V, W, rank):
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        # Check zero entries (sampled or full if small)
        # Full check is n*m*p*n*m*p which is too big.
        # But for n=2, m=4, p=5, size is 8*20*10 = 1600 entries.
        # 1600 * rank is small.
        # We must check all zeros to be "exact".
        for a in range(n*m):
            i = a // m
            j = a % m
            for b in range(m*p):
                jj = b // p
                k = b % p
                if jj != j:
                    continue # This index pair (a,b) never contributes to any k?
                    # Wait, if jj != j, then for any k, the term is part of a zero entry?
                    # The tensor entry T[a,b,c] is non-zero only if a->j, b->jj match.
                    # So if j != jj, T[a,b,c] = 0 for all c.
                    # We need to ensure sum is 0 for all c.
                    for c in range(p*n):
                        kk = c // n
                        ii = c % n
                        s = 0
                        for r in range(rank):
                            s += U[a][r] * V[b][r] * W[c][r]
                        if s != 0:
                            return False
                else:
                    # j == jj
                    # Non-zero only if i==ii and k==kk?
                    # Wait, T[a,b,c] non-zero if i(a)==i(c) and k(b)==k(c).
                    # If not, it's 0.
                    for c in range(p*n):
                        kk = c // n
                        ii = c % n
                        if i != ii or k != kk:
                            s = 0
                            for r in range(rank):
                                s += U[a][r] * V[b][r] * W[c][r]
                            if s != 0:
                                return False
                        else:
                            # This is the non-zero case, already checked in first loop
                            pass
        return True

    # Schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
                
    best = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find lower rank
    # Random search with -1, 0, 1
    for rank in range(rank_school - 1, 15, -1):
        if time.time() > deadline - 10:
            break
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p*n)]
            if verify(U, V, W, rank):
                best = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
        if best_rank < rank_school - 1:
            break
            
    return best
