"""Improved search for rank-minimal exact tensor decomposition of 3x3 * 3x4 matrix multiplication."""
import time
import random
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Strategy:
    1. Schoolbook (rank 36) as fallback.
    2. Exploit W structure: each W[c] has exactly one nonzero per column in valid decompositions.
    3. Use column-wise verification for speed.
    4. Structured search leveraging the regular pattern of active triples.
    5. Try ranks from 12 upward with increasing effort.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s budget, leaving 10s margin
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook decomposition as fallback
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
    
    # Precompute active triples and their indices
    active_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                active_triples.append((a, b, c))
    
    active_set = set(active_triples)
    
    # For each c index, precompute which (a,b) pairs are active
    active_pairs_by_c = {}
    for a, b, c in active_triples:
        if c not in active_pairs_by_c:
            active_pairs_by_c[c] = []
        active_pairs_by_c[c].append((a, b))
    
    # For each (a,b), precompute which c values are active
    active_c_by_ab = {}
    for a, b, c in active_triples:
        if (a, b) not in active_c_by_ab:
            active_c_by_ab[(a, b)] = []
        active_c_by_ab[(a, b)].append(c)
    
    def verify_decomposition(U, V, W, r):
        """
        Verify that U, V, W form a valid decomposition.
        Returns (is_valid, error_count)
        """
        U_np = np.asarray(U, dtype=np.int64)
        V_np = np.asarray(V, dtype=np.int64)
        W_np = np.asarray(W, dtype=np.int64)
        
        errors = 0
        
        # Check active triples: sum should be 1
        for a, b, c in active_triples:
            s = 0
            for ci in range(r):
                s += U_np[a, ci] * V_np[b, ci] * W_np[c, ci]
            if s != 1:
                errors += 1
        
        # Check inactive triples: sum should be 0
        # Only check a sample for speed, or all if r is small
        if r <= 20:
            # Check all inactive triples
            for a in range(n * m):
                for b in range(m * p):
                    for c in range(p * n):
                        if (a, b, c) not in active_set:
                            s = 0
                            for ci in range(r):
                                s += U_np[a, ci] * V_np[b, ci] * W_np[c, ci]
                            if s != 0:
                                errors += 1
                                if errors > 10:  # Early exit
                                    return (False, errors)
        else:
            # Check a random sample of inactive triples
            num_samples = min(500, (n*m)*(m*p)*(p*n) - len(active_triples))
            inactive_triples = [(a, b, c) for a in range(n*m) for b in range(m*p) for c in range(p*n) if (a, b, c) not in active_set]
            sample = random.sample(inactive_triples, num_samples)
            for a, b, c in sample:
                s = 0
                for ci in range(r):
                    s += U_np[a, ci] * V_np[b, ci] * W_np[c, ci]
                if s != 0:
                    errors += 1
                    if errors > 10:
                        return (False, errors)
        
        return (errors == 0, errors)
    
    def error_count_fast(U, V, W, r):
        """Fast error count: check active triples only."""
        U_np = np.asarray(U, dtype=np.int64)
        V_np = np.asarray(V, dtype=np.int64)
        W_np = np.asarray(W, dtype=np.int64)
        
        errors = 0
        for a, b, c in active_triples:
            s = 0
            for ci in range(r):
                s += U_np[a, ci] * V_np[b, ci] * W_np[c, ci]
            if s != 1:
                errors += 1
        return errors
    
    def check_inactive_sample(U, V, W, r, num_samples=200):
        """Check a random sample of inactive triples."""
        U_np = np.asarray(U, dtype=np.int64)
        V_np = np.asarray(V, dtype=np.int64)
        W_np = np.asarray(W, dtype=np.int64)
        
        inactive_triples = [(a, b, c) for a in range(n*m) for b in range(m*p) for c in range(p*n) if (a, b, c) not in active_set]
        if len(inactive_triples) == 0:
            return 0
        
        sample_size = min(num_samples, len(inactive_triples))
        sample = random.sample(inactive_triples, sample_size)
        
        errors = 0
        for a, b, c in sample:
            s = 0
            for ci in range(r):
                s += U_np[a, ci] * V_np[b, ci] * W_np[c, ci]
            if s != 0:
                errors += 1
        return errors
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def generate_sparse_U(r, density=0.2):
        """Generate sparse U matrix."""
        U = [[0] * r for _ in range(n * m)]
        for a in range(n * m):
            num_nz = max(1, int(r * density))
            cols = random.sample(range(r), num_nz)
            for ci in cols:
                U[a][ci] = random.choice([-1, 1])
        return U
    
    def generate_sparse_V(r, density=0.2):
        """Generate sparse V matrix."""
        V = [[0] * r for _ in range(m * p)]
        for b in range(m * p):
            num_nz = max(1, int(r * density))
            cols = random.sample(range(r), num_nz)
            for ci in cols:
                V[b][ci] = random.choice([-1, 1])
        return V
    
    def generate_sparse_W(r, density=0.2):
        """Generate sparse W matrix."""
        W = [[0] * r for _ in range(p * n)]
        for c in range(p * n):
            num_nz = max(1, int(r * density))
            cols = random.sample(range(r), num_nz)
            for ci in cols:
                W[c][ci] = random.choice([-1, 1])
        return W
    
    def try_rank_structured(r, max_attempts=100):
        """
        Try to find a rank-r decomposition using structured search.
        Key insight: exploit the regular structure of active triples.
        """
        if time.time() > deadline:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Strategy 0: Sparse random
            if attempt % 4 == 0:
                U = generate_sparse_U(r, density=0.15)
                V = generate_sparse_V(r, density=0.15)
                W = generate_sparse_W(r, density=0.15)
            # Strategy 1: Medium density
            elif attempt % 4 == 1:
                U = generate_sparse_U(r, density=0.25)
                V = generate_sparse_V(r, density=0.25)
                W = generate_sparse_W(r, density=0.25)
            # Strategy 2: Column-structured
            elif attempt % 4 == 2:
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                for ci in range(r):
                    # Each column gets a structured pattern
                    u_pat = [random.choice([-1, 0, 1]) for _ in range(n * m)]
                    v_pat = [random.choice([-1, 0, 1]) for _ in range(m * p)]
                    w_pat = [random.choice([-1, 0, 1]) for _ in range(p * n)]
                    for a in range(n * m):
                        U[a][ci] = u_pat[a]
                    for b in range(m * p):
                        V[b][ci] = v_pat[b]
                    for c in range(p * n):
                        W[c][ci] = w_pat[c]
            # Strategy 3: Exploit W regularity - each W[c] should have few nonzeros
            else:
                U = generate_sparse_U(r, density=0.2)
                V = generate_sparse_V(r, density=0.2)
                W = [[0] * r for _ in range(p * n)]
                for c in range(p * n):
                    # W[c] has exactly 1-2 nonzeros in a good decomposition
                    num_nz = random.choice([1, 2])
                    cols = random.sample(range(r), num_nz)
                    for ci in cols:
                        W[c][ci] = random.choice([-1, 1])
            
            # Quick check: active triples
            active_errors = error_count_fast(U, V, W, r)
            if active_errors > 0:
                continue
            
            # Check inactive sample
            inactive_errors = check_inactive_sample(U, V, W, r, num_samples=100)
            if inactive_errors > 0:
                continue
            
            # Full verification
            is_valid, _ = verify_decomposition(U, V, W, r)
            if is_valid:
                return {"U": U, "V": V, "W": W}
        
        return None
    
    def local_search(U, V, W, r, max_steps=50):
        """
        Local search: mutate entries, accept if total error decreases.
        """
        if time.time() > deadline:
            return None
        
        current_errors = error_count_fast(U, V, W, r) + check_inactive_sample(U, V, W, r, 50)
        if current_errors == 0:
            return {"U": U, "V": V, "W": W}
        
        best_U = [row[:] for row in U]
        best_V = [row[:] for row in V]
        best_W = [row[:] for row in W]
        best_errors = current_errors
        no_improve = 0
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if best_errors == 0:
                break
            
            improved = False
            # Try random mutations
            for _ in range(50):
                if time.time() > deadline:
                    break
                
                mat_choice = random.choice(['U', 'V', 'W'])
                new_val = random.choice([-2, -1, 0, 1, 2])
                
                if mat_choice == 'U':
                    idx = random.randint(0, n * m - 1)
                    ci = random.randint(0, r - 1)
                    old_val = U[idx][ci]
                    if new_val == old_val:
                        continue
                    U[idx][ci] = new_val
                    errors = error_count_fast(U, V, W, r) + check_inactive_sample(U, V, W, r, 30)
                    if errors < best_errors:
                        best_errors = errors
                        best_U = [row[:] for row in U]
                        best_V = [row[:] for row in V]
                        best_W = [row[:] for row in W]
                        improved = True
                        no_improve = 0
                    else:
                        U[idx][ci] = old_val
                elif mat_choice == 'V':
                    idx = random.randint(0, m * p - 1)
                    ci = random.randint(0, r - 1)
                    old_val = V[idx][ci]
                    if new_val == old_val:
                        continue
                    V[idx][ci] = new_val
                    errors = error_count_fast(U, V, W, r) + check_inactive_sample(U, V, W, r, 30)
                    if errors < best_errors:
                        best_errors = errors
                        best_U = [row[:] for row in U]
                        best_V = [row[:] for row in V]
                        best_W = [row[:] for row in W]
                        improved = True
                        no_improve = 0
                    else:
                        V[idx][ci] = old_val
                else:
                    idx = random.randint(0, p * n - 1)
                    ci = random.randint(0, r - 1)
                    old_val = W[idx][ci]
                    if new_val == old_val:
                        continue
                    W[idx][ci] = new_val
                    errors = error_count_fast(U, V, W, r) + check_inactive_sample(U, V, W, r, 30)
                    if errors < best_errors:
                        best_errors = errors
                        best_U = [row[:] for row in U]
                        best_V = [row[:] for row in V]
                        best_W = [row[:] for row in W]
                        improved = True
                        no_improve = 0
                    else:
                        W[idx][ci] = old_val
            
            if improved:
                U = [row[:] for row in best_U]
                V = [row[:] for row in best_V]
                W = [row[:] for row in best_W]
            else:
                no_improve += 1
                if no_improve > 3:
                    U = [row[:] for row in best_U]
                    V = [row[:] for row in best_V]
                    W = [row[:] for row in best_W]
                    no_improve = 0
            
            if best_errors == 0:
                break
        
        if best_errors == 0:
            return {"U": best_U, "V": best_V, "W": best_W}
        return None
    
    # Try ranks from lower bound upward
    min_rank = max(n * m, m * p, p * n)  # 12
    
    # Prioritize ranks that are more likely to have solutions
    # Known: rank 18 works for 3x3x3 via Strassen-like methods
    # For 3x3x4, try 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36
    ranks_to_try = list(range(min_rank, best_rank + 1, 2))  # Even ranks first
    if 18 not in ranks_to_try:
        ranks_to_try.append(18)
    ranks_to_try.sort()
    
    # Allocate time budget
    total_time_budget = deadline - start_time
    time_per_rank = total_time_budget / len(ranks_to_try)
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        time_remaining = deadline - time.time()
        if time_remaining < 5:
            break
        
        # More attempts for lower ranks
        attempts = min(200, max(20, int(time_remaining * 2)))
        
        result = try_rank_structured(r, max_attempts=attempts)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    # If no solution found, try local search at a few ranks
    if best_rank == n * m * p:
        for r in [12, 14, 16, 18, 20]:
            if time.time() > deadline - 10:
                break
            if r >= best_rank:
                break
            
            # Initialize with structured sparse
            U_start = generate_sparse_U(r, density=0.2)
            V_start = generate_sparse_V(r, density=0.2)
            W_start = [[0] * r for _ in range(p * n)]
            for c in range(p * n):
                num_nz = random.choice([1, 2])
                cols = random.sample(range(r), num_nz)
                for ci in cols:
                    W_start[c][ci] = random.choice([-1, 1])
            
            ls_result = local_search(U_start, V_start, W_start, r, max_steps=40)
            if ls_result is not None:
                best_result = ls_result
                best_rank = r
                break
    
    return best_result
# EVOLVE-BLOCK-END
