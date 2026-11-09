import numpy as np
import time
import random

def _count_bad_sets(A, n):
    """Count 4-sets with no common dominator. A[i][j]=1 means i->j."""
    bad = 0
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    # Check if any of a,b,c,d dominates the other 3
                    if (A[a][b] and A[a][c] and A[a][d]):
                        continue
                    if (A[b][a] and A[b][c] and A[b][d]):
                        continue
                    if (A[c][a] and A[c][b] and A[c][d]):
                        continue
                    if (A[d][a] and A[d][b] and A[d][c]):
                        continue
                    bad += 1
    return bad

def _count_bad_sets_fast(A, n):
    """Faster count using bitmasks for n <= 64."""
    if n > 64:
        return _count_bad_sets(A, n)
    # dominators[i] = bitmask of vertices dominated by i
    dom = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        for j in range(n):
            if i != j and A[i][j]:
                dom[i] |= (1 << j)
    
    bad = 0
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                    # a dominates b,c,d if (dom[a] & mask) == mask without a
                    if (dom[a] & ((1<<b)|(1<<c)|(1<<d))) == ((1<<b)|(1<<c)|(1<<d)):
                        continue
                    if (dom[b] & ((1<<a)|(1<<c)|(1<<d))) == ((1<<a)|(1<<c)|(1<<d)):
                        continue
                    if (dom[c] & ((1<<a)|(1<<b)|(1<<d))) == ((1<<a)|(1<<b)|(1<<d)):
                        continue
                    if (dom[d] & ((1<<a)|(1<<b)|(1<<c))) == ((1<<a)|(1<<b)|(1<<c)):
                        continue
                    bad += 1
    return bad

def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill
    
    # Start from known good: Paley on 83
    p = 83
    squares = {x * x % p for x in range(1, p)}
    A0 = np.zeros((p, p), dtype=np.uint8)
    for i in range(p):
        for j in range(i+1, p):
            if (j - i) % p in squares:
                A0[i][j] = 1
            else:
                A0[j][i] = 1
    
    best_A = A0.copy()
    best_n = p
    best_bad = 0  # known valid
    
    # Try to find smaller valid tournaments via local search
    # Strategy: for each candidate n, try random restarts + local search
    rng = np.random.default_rng(42)
    
    # Precompute pairs for efficiency
    def get_pairs(n):
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        return pairs
    
    # Try decreasing n from a reasonable upper bound
    # Lower bound is 31, try from ~50 down
    for target_n in range(50, 30, -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random restarts for this n
        max_restarts = 3 if target_n > 40 else 5
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            A = (rng.random((target_n, target_n)) < 0.5).astype(np.uint8)
            np.fill_diagonal(A, 0)
            # Ensure antisymmetry
            A = ((A + A.T) % 2)
            np.fill_diagonal(A, 0)
            
            pairs = get_pairs(target_n)
            bad = _count_bad_sets_fast(A, target_n)
            
            # Local search: flip edges to reduce bad sets
            T = 1000.0
            T_min = 1.0
            cooling = 0.999
            steps_per_iter = 50
            max_iters = 20000
            
            for it in range(max_iters):
                if time.time() > deadline - 3:
                    break
                if T < T_min:
                    break
                
                # Pick random edge to flip
                idx = rng.integers(0, len(pairs))
                i, j = pairs[idx]
                old_val = A[i][j]
                
                # Flip
                A[i][j] = 1 - old_val
                A[j][i] = old_val  # maintain antisymmetry
                
                new_bad = _count_bad_sets_fast(A, target_n)
                delta = new_bad - bad
                
                if delta <= 0 or rng.random() < np.exp(-delta / T):
                    bad = new_bad
                    # Accept
                else:
                    # Revert
                    A[i][j] = old_val
                    A[j][i] = 1 - old_val
                
                T *= cooling
            
            if bad == 0:
                best_A = A.copy()
                best_n = target_n
                best_bad = 0
                break  # Found valid for this n, try smaller
            
            # If we found a better partial solution, keep it as starting point
            # for next restart (warm start)
    
    # Final: return best found
    n = best_n
    orientation = []
    for i in range(n):
        for j in range(i+1, n):
            orientation.append(int(best_A[i][j]))
    
    return {'n': n, 'orientation': orientation}
