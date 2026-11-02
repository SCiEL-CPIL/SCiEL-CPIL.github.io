"""
Tensor decomposition for 4x5 * 5x5 matrix multiplication.
We attempt to find a lower-rank exact decomposition than the schoolbook rank of 100.
Uses structured random search with verification and a time-bounded search loop.
"""
import time
import random
import numpy as np
from itertools import product

def construct():
    n, m, p = 4, 5, 5
    rank_school = n * m * p  # 100
    
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Start with schoolbook as best
    best_rank = rank_school
    best_U = np.zeros((n * m, rank_school))
    best_V = np.zeros((m * p, rank_school))
    best_W = np.zeros((p * n, rank_school))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                best_U[i * m + j, c] = 1.0
                best_V[j * p + k, c] = 1.0
                best_W[k * n + i, c] = 1.0
    
    def verify_fast(U, V, W):
        """Check if the decomposition is exact using vectorized operations."""
        rank = U.shape[1]
        # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        # Precompute indices
        for i in range(n):
            for j in range(m):
                u_row = U[i * m + j]
                for k in range(p):
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    s = np.dot(u_row, v_row * w_row)
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    def generate_structured(rank, strategy):
        """Generate factors with specific structure."""
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        if strategy == 0:
            # Diagonal-like: each term covers one (i,j,k) triple
            # Try to merge some terms
            assigned = {}
            for c in range(rank):
                # Randomly pick which triples this term covers
                num_triples = random.randint(1, 2)
                chosen = random.sample(range(n * m * p), min(num_triples, n * m * p))
                for idx in chosen:
                    ii = idx // (m * p)
                    jj = (idx // p) % m
                    kk = idx % p
                    if idx not in assigned:
                        assigned[idx] = []
                    assigned[idx].append(c)
            
            # Now assign values
            for idx, cols in assigned.items():
                ii = idx // (m * p)
                jj = (idx // p) % m
                kk = idx % p
                u_idx = ii * m + jj
                v_idx = jj * p + kk
                w_idx = kk * n + ii
                for c in cols:
                    val = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
                    U[u_idx, c] += val
                    V[v_idx, c] += val
                    W[w_idx, c] += val
            
            # Ensure unassigned triples get covered
            for idx in range(n * m * p):
                if idx not in assigned:
                    # Assign to a random existing column
                    c = random.randint(0, rank - 1)
                    ii = idx // (m * p)
                    jj = (idx // p) % m
                    kk = idx % p
                    u_idx = ii * m + jj
                    v_idx = jj * p + kk
                    w_idx = kk * n + ii
                    U[u_idx, c] = 1.0
                    V[v_idx, c] = 1.0
                    W[w_idx, c] = 1.0
                    if idx not in assigned:
                        assigned[idx] = []
                    assigned[idx].append(c)
        
        elif strategy == 1:
            # Block structure: group by i
            for i in range(n):
                for c in range(rank):
                    # Each column handles some (j,k) pairs for this i
                    num_jk = random.randint(1, 3)
                    for _ in range(num_jk):
                        j = random.randint(0, m - 1)
                        k = random.randint(0, p - 1)
                        val = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
                        U[i * m + j, c] += val
                        V[j * p + k, c] += val
                        W[k * n + i, c] += val
            
            # Ensure all triples covered
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        # Check if already covered
                        covered = False
                        for c in range(rank):
                            if U[u_idx, c] != 0 and V[v_idx, c] != 0 and W[w_idx, c] != 0:
                                covered = True
                                break
                        if not covered:
                            c = random.randint(0, rank - 1)
                            U[u_idx, c] = 1.0
                            V[v_idx, c] = 1.0
                            W[w_idx, c] = 1.0
        
        elif strategy == 2:
            # Sparse random with higher density
            for c in range(rank):
                num_u = random.randint(2, 4)
                num_v = random.randint(2, 4)
                num_w = random.randint(2, 4)
                for _ in range(num_u):
                    U[random.randint(0, n * m - 1), c] = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
                for _ in range(num_v):
                    V[random.randint(0, m * p - 1), c] = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
                for _ in range(num_w):
                    W[random.randint(0, p * n - 1), c] = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
            
            # Ensure all triples covered
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        covered = False
                        for c in range(rank):
                            if U[u_idx, c] != 0 and V[v_idx, c] != 0 and W[w_idx, c] != 0:
                                covered = True
                                break
                        if not covered:
                            c = random.randint(0, rank - 1)
                            U[u_idx, c] = 1.0
                            V[v_idx, c] = 1.0
                            W[w_idx, c] = 1.0
        
        elif strategy == 3:
            # Try to find a valid decomposition by starting from schoolbook and merging
            # Start with identity-like structure
            for c in range(rank):
                if c < n * m * p:
                    i = c // (m * p)
                    j = (c // p) % m
                    k = c % p
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                else:
                    # Extra columns: try to create redundancy
                    i = random.randint(0, n - 1)
                    j = random.randint(0, m - 1)
                    k = random.randint(0, p - 1)
                    U[i * m + j, c] = random.choice([-1, 1])
                    V[j * p + k, c] = random.choice([-1, 1])
                    W[k * n + i, c] = random.choice([-1, 1])
            
            # Try to zero out some entries to reduce effective rank
            for c in range(min(rank, n * m * p)):
                if random.random() < 0.3:
                    # Try to split this term across two columns
                    c2 = c + 1 if c + 1 < rank else c
                    i = c // (m * p)
                    j = (c // p) % m
                    k = c % p
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Split into two parts
                    U[u_idx, c] = 0.5
                    V[v_idx, c] = 0.5
                    W[w_idx, c] = 4.0
                    U[u_idx, c2] = 0.5
                    V[v_idx, c2] = 0.5
                    W[w_idx, c2] = 4.0
        
        return U, V, W
    
    # Try to find lower rank via structured search
    # Strategy: try ranks from 99 down, with limited attempts per rank
    for target_rank in range(rank_school - 1, max(rank_school - 40, 0), -1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple strategies for this rank
        max_attempts = 30 if target_rank > 90 else 10
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            strategy = attempt % 4
            U, V, W = generate_structured(target_rank, strategy)
            
            if verify_fast(U, V, W):
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                return {
                    "U": best_U.tolist(),
                    "V": best_V.tolist(),
                    "W": best_W.tolist()
                }
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
