"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    # Precompute active triples: all (a,b,c) where a=i*m+j, b=j*p+k, c=k*n+i
    active_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                active_triples.append((a, b, c))
    
    active_set = set(active_triples)
    
    # All possible triples for verification
    all_triples = list(itertools.product(range(n*m), range(m*p), range(p*n)))
    
    def verify_exact(U, V, W, r):
        """Exact verification using integer/half-integer arithmetic."""
        for a, b, c in all_triples:
            s = 0
            for ci in range(r):
                u_val = U[a][ci]
                v_val = V[b][ci]
                w_val = W[c][ci]
                # Multiply: handle Fraction or int
                s += u_val * v_val * w_val
            expected = 1 if (a, b, c) in active_set else 0
            if s != expected:
                return False
        return True
    
    def verify_fast(U, V, W, r):
        """Fast verification - check only active triples first, then sample inactive."""
        # Check all active triples
        for (a, b, c) in active_triples:
            s = 0
            for ci in range(r):
                s += U[a][ci] * V[b][ci] * W[c][ci]
            if s != 1:
                return False
        # Check a sample of inactive triples
        inactive = [(a,b,c) for a,b,c in all_triples if (a,b,c) not in active_set]
        sample_size = min(len(inactive), 50)
        for idx in random.sample(range(len(inactive)), sample_size):
            a, b, c = inactive[idx]
            s = 0
            for ci in range(r):
                s += U[a][ci] * V[b][ci] * W[c][ci]
            if s != 0:
                return False
        return True
    
    def error_count(U, V, W, r):
        """Count violations for local search."""
        errors = 0
        for a, b, c in all_triples:
            s = 0
            for ci in range(r):
                s += U[a][ci] * V[b][ci] * W[c][ci]
            expected = 1 if (a, b, c) in active_set else 0
            if s != expected:
                errors += 1
        return errors
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Strategy: Try structured constructions from known algebraic patterns
    # Key idea: Use the fact that T[i,j,k] = 1 for all valid triples
    # Try rank from 20 up to best_rank-1
    
    def try_structured_rank(r, max_attempts=100):
        """Try to find a rank-r decomposition using structured search."""
        if time.time() > deadline:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            strategy = attempt % 4
            
            if strategy == 0:
                # Sparse random with ±1 values
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                for a in range(n * m):
                    nz = random.randint(1, min(4, r))
                    cols = random.sample(range(r), nz)
                    for ci in cols:
                        U[a][ci] = random.choice([-1, 1])
                for b in range(m * p):
                    nz = random.randint(1, min(4, r))
                    cols = random.sample(range(r), nz)
                    for ci in cols:
                        V[b][ci] = random.choice([-1, 1])
                for c in range(p * n):
                    nz = random.randint(1, min(4, r))
                    cols = random.sample(range(r), nz)
                    for ci in cols:
                        W[c][ci] = random.choice([-1, 1])
                        
            elif strategy == 1:
                # Block-structured: group by j index
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                # Assign columns in blocks of ~r/m
                block_size = max(1, r // m)
                for j in range(m):
                    start_col = j * block_size
                    end_col = min(start_col + block_size, r)
                    for i in range(n):
                        for k in range(p):
                            if start_col < end_col:
                                col = start_col + (i * p + k) % (end_col - start_col)
                                U[i * m + j][col] = 1
                                V[j * p + k][col] = 1
                                W[k * n + i][col] = 1
                # Fill remaining columns with random sparse
                for ci in range(r):
                    if random.random() < 0.5:
                        for a in range(n * m):
                            if U[a][ci] == 0:
                                U[a][ci] = random.choice([-1, 0, 1])
                        for b in range(m * p):
                            if V[b][ci] == 0:
                                V[b][ci] = random.choice([-1, 0, 1])
                        for c in range(p * n):
                            if W[c][ci] == 0:
                                W[c][ci] = random.choice([-1, 0, 1])
                                
            elif strategy == 2:
                # Strassen-inspired: use 7-term pattern for 3x3x3, extend to 3x3x4
                # Known rank-7 decomposition for 3x3x3 (one variant)
                # Extend by adding terms for the 4th k index
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                
                # Use a systematic pattern: for each column, assign based on (i,j,k) structure
                for ci in range(r):
                    # Generate a random but structured pattern
                    # Each row gets a value based on a hash-like function
                    u_pattern = []
                    for i in range(n):
                        for j in range(m):
                            val = ((i * 3 + j + ci) % 3) - 1  # -1, 0, 1
                            u_pattern.append(val)
                    v_pattern = []
                    for j in range(m):
                        for k in range(p):
                            val = ((j * 4 + k + ci) % 3) - 1
                            v_pattern.append(val)
                    w_pattern = []
                    for k in range(p):
                        for i in range(n):
                            val = ((k * 3 + i + ci) % 3) - 1
                            w_pattern.append(val)
                    
                    for idx, val in enumerate(u_pattern):
                        U[idx][ci] = val
                    for idx, val in enumerate(v_pattern):
                        V[idx][ci] = val
                    for idx, val in enumerate(w_pattern):
                        W[idx][ci] = val
                        
            else:
                # Dense small-integer random
                scale = random.choice([1, 2, 3])
                U = [[random.randint(-scale, scale) for _ in range(r)] for _ in range(n * m)]
                V = [[random.randint(-scale, scale) for _ in range(r)] for _ in range(m * p)]
                W = [[random.randint(-scale, scale) for _ in range(r)] for _ in range(p * n)]
            
            # Quick check
            if not verify_fast(U, V, W, r):
                continue
            
            # Full verification
            if verify_exact(U, V, W, r):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try local search refinement from a good starting point
    def local_search(U, V, W, r, max_iter=500):
        """Refine a candidate using coordinate descent with exact checks."""
        if time.time() > deadline:
            return None
        
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        
        current_errors = error_count(U, V, W, r)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if current_errors == 0:
                return {"U": U, "V": V, "W": W}
            
            # Randomly pick a matrix and a position to modify
            mat_choice = random.choice(['U', 'V', 'W'])
            if mat_choice == 'U':
                row_idx = random.randint(0, n * m - 1)
                col_idx = random.randint(0, r - 1)
                old_val = U[row_idx][col_idx]
                new_val = old_val + random.choice([-2, -1, 1, 2])
                U[row_idx][col_idx] = new_val
                new_errors = error_count(U, V, W, r)
                if new_errors > current_errors:
                    U[row_idx][col_idx] = old_val
                else:
                    current_errors = new_errors
                    
            elif mat_choice == 'V':
                row_idx = random.randint(0, m * p - 1)
                col_idx = random.randint(0, r - 1)
                old_val = V[row_idx][col_idx]
                new_val = old_val + random.choice([-2, -1, 1, 2])
                V[row_idx][col_idx] = new_val
                new_errors = error_count(U, V, W, r)
                if new_errors > current_errors:
                    V[row_idx][col_idx] = old_val
                else:
                    current_errors = new_errors
                    
            else:
                row_idx = random.randint(0, p * n - 1)
                col_idx = random.randint(0, r - 1)
                old_val = W[row_idx][col_idx]
                new_val = old_val + random.choice([-2, -1, 1, 2])
                W[row_idx][col_idx] = new_val
                new_errors = error_count(U, V, W, r)
                if new_errors > current_errors:
                    W[row_idx][col_idx] = old_val
                else:
                    current_errors = new_errors
        
        if current_errors == 0:
            return {"U": U, "V": V, "W": W}
        return None
    
    # Search from optimistic rank upward
    for r in range(18, best_rank):
        if time.time() > deadline:
            break
        
        time_remaining = deadline - time.time()
        if time_remaining < 3:
            break
        
        # Allocate attempts based on remaining time
        if r <= 22:
            max_attempts = min(200, int(time_remaining * 8))
        elif r <= 28:
            max_attempts = min(100, int(time_remaining * 4))
        else:
            max_attempts = min(50, int(time_remaining * 2))
        
        result = try_structured_rank(r, max_attempts)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    # If we haven't improved, try local search from schoolbook with reduced rank
    if best_rank == n * m * p and time.time() < deadline - 10:
        # Try to reduce schoolbook by merging columns
        for target_rank in range(30, n * m * p):
            if time.time() > deadline:
                break
            # Create a candidate by randomly combining schoolbook columns
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Map original 36 columns to target_rank columns
            col_map = [random.randint(0, target_rank - 1) for _ in range(n * m * p)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        target_col = col_map[col]
                        U[i * m + j][target_col] += 1
                        V[j * p + k][target_col] += 1
                        W[k * n + i][target_col] += 1
                        col += 1
            
            if verify_exact(U, V, W, target_rank):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
    
    return best_result
# EVOLVE-BLOCK-END
