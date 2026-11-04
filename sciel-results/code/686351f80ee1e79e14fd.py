"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    # Schoolbook decomposition as baseline (rank 42)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    best_rank = n * m * p
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True
    
    # Border rank / degeneracy approach:
    # Start from rank-42 and try to merge terms by introducing structured
    # off-diagonal entries that create cancellations.
    # 
    # Key insight: For matrix multiplication, the tensor T[i,j,k] = 1 can be
    # decomposed using structured patterns. We look for solutions where:
    # - U, V, W have entries in {0, ±1, ±1/2}
    # - The structure exploits the rectangular dimensions
    #
    # Strategy: Use a "block" approach where we group indices and use
    # Kronecker-like products with small correction terms.
    
    # First, try a structured search with specific patterns
    def gen_structured_U(rows, cols):
        """Generate U with structure: each row has 1-2 non-zero entries."""
        mat = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            # Each row gets 1-2 non-zeros
            num_nz = random.choice([1, 1, 2])
            cols_chosen = random.sample(range(cols), min(num_nz, cols))
            for c in cols_chosen:
                mat[r][c] = random.choice([-1, 1])
        return mat
    
    def gen_structured_V(rows, cols):
        """Generate V with structure: each row has 1-2 non-zero entries."""
        mat = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            num_nz = random.choice([1, 1, 2])
            cols_chosen = random.sample(range(cols), min(num_nz, cols))
            for c in cols_chosen:
                mat[r][c] = random.choice([-1, 1])
        return mat
    
    def gen_structured_W(rows, cols):
        """Generate W with structure: each row has 1-2 non-zero entries."""
        mat = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            num_nz = random.choice([1, 1, 2])
            cols_chosen = random.sample(range(cols), min(num_nz, cols))
            for c in cols_chosen:
                mat[r][c] = random.choice([-1, 1])
        return mat
    
    def local_search_v2(U, V, W, rank, max_steps=2000):
        """
        Enhanced local search with half-integer support and better move selection.
        """
        # Initialize sums
        sums = [[[0 for _ in range(p)] for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    sums[i][j][k] = s
        
        if all(sums[i][j][k] == 1 for i in range(n) for j in range(m) for k in range(p)):
            return True
        
        # Track error count
        def error_count():
            cnt = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if sums[i][j][k] != 1:
                            cnt += 1
            return cnt
        
        # Allowed values: -1, -0.5, 0, 0.5, 1
        allowed_vals = [-1, -0.5, 0, 0.5, 1]
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            # Pick a random matrix and entry
            mat_choice = random.choice([0, 1, 2])
            
            if mat_choice == 0:
                r = random.randint(0, nm - 1)
                c = random.randint(0, rank - 1)
                old_val = U[r][c]
                # Try a new value from allowed set
                new_val = random.choice(allowed_vals)
                if new_val == old_val:
                    continue
                
                i = r // m
                j = r % m
                delta = new_val - old_val
                
                for k in range(p):
                    sums[i][j][k] += delta * V[j * p + k][c] * W[k * n + i][c]
                
                U[r][c] = new_val
                
            elif mat_choice == 1:
                r = random.randint(0, mp - 1)
                c = random.randint(0, rank - 1)
                old_val = V[r][c]
                new_val = random.choice(allowed_vals)
                if new_val == old_val:
                    continue
                
                j = r // p
                k = r % p
                delta = new_val - old_val
                
                for i in range(n):
                    sums[i][j][k] += U[i * m + j][c] * delta * W[k * n + i][c]
                
                V[r][c] = new_val
                
            else:
                r = random.randint(0, pn - 1)
                c = random.randint(0, rank - 1)
                old_val = W[r][c]
                new_val = random.choice(allowed_vals)
                if new_val == old_val:
                    continue
                
                k = r // n
                i = r % n
                delta = new_val - old_val
                
                for j in range(m):
                    sums[i][j][k] += U[i * m + j][c] * V[j * p + k][c] * delta
                
                W[r][c] = new_val
            
            # Check solution periodically
            if step % 20 == 0 or step == max_steps - 1:
                if all(sums[i][j][k] == 1 for i in range(n) for j in range(m) for k in range(p)):
                    return True
        
        return False
    
    # Try ranks from 21 to 41
    min_rank = 21
    max_search_rank = best_rank - 1
    
    for target_rank in range(min_rank, max_search_rank + 1):
        if time.time() > deadline - 5:
            break
        
        # Allocate more attempts to ranks closer to the known feasible range
        if target_rank >= 35:
            attempts = 300
        elif target_rank >= 30:
            attempts = 200
        else:
            attempts = 100
        
        found = False
        
        for attempt in range(attempts):
            if time.time() > deadline - 5:
                break
            
            # Use structured generation
            U = gen_structured_U(nm, target_rank)
            V = gen_structured_V(mp, target_rank)
            W = gen_structured_W(pn, target_rank)
            
            if local_search_v2(U, V, W, target_rank, max_steps=300):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                found = True
                break
        
        if found:
            continue
    
    return best
# EVOLVE-BLOCK-END
