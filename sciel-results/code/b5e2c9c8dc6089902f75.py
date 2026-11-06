import time
import random
import math
import numpy as np

def _score(A):
    """Compute max(|A+A|, |A*A|) for set A using NumPy for speed."""
    A_arr = np.array(A, dtype=np.int64)
    sums = np.add.outer(A_arr, A_arr)
    prods = np.multiply.outer(A_arr, A_arr)
    n_sums = len(np.unique(sums))
    n_prods = len(np.unique(prods))
    return max(n_sums, n_prods)

def _is_valid(A, n=20, maximum=10000):
    """Check if A has n distinct positive integers <= maximum."""
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x <= 0 for x in A):
        return False
    if any(x > maximum for x in A):
        return False
    return True

def _generate_gp_candidate(r, start=1, n=20, max_val=10000):
    """
    Generate a candidate set based on a geometric progression with ratio r.
    """
    A = []
    current = float(start)
    for _ in range(n):
        val = int(round(current))
        if val <= 0:
            val = 1
        if val > max_val:
            break
        if val not in A:
            A.append(val)
        else:
            val += 1
            if val > max_val:
                val = max_val - 1
            if val not in A:
                A.append(val)
            else:
                pass
        current *= r
        
    if len(A) < n:
        existing = set(A)
        last_val = A[-1] if A else 1
        for _ in range(n - len(A)):
            if len(A) > 0 and A[-1] < 10000:
                if len(A) >= 2:
                    r_est = A[-1] / A[-2] if A[-2] != 0 else 1
                    next_est = int(A[-1] * r_est)
                    if next_est > 10000:
                        next_est = 10000
                    if next_est not in existing and next_est > A[-1]:
                        A.append(next_est)
                        existing.add(next_est)
                        continue
            for _ in range(100):
                val = random.randint(last_val + 1, 10000)
                if val not in existing:
                    A.append(val)
                    existing.add(val)
                    break
            else:
                val = A[-1] + 1 if A else 1
                while val in existing and val <= 10000:
                    val += 1
                if val <= 10000:
                    A.append(val)
                    existing.add(val)
    
    A = sorted(A[:n])
    
    if len(set(A)) != n or any(x < 1 or x > 10000 for x in A):
        return None
        
    return A

def _local_search(A, max_iter=2000, deadline=None, seed=None):
    """
    Local search: try single-element changes and accept improvements.
    Uses NumPy for fast scoring.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
        
    current_A = A[:]
    current_score = _score(current_A)
    best_A = current_A[:]
    best_score = current_score
    n = len(current_A)
    
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        # Shuffle indices to avoid bias
        indices = list(range(n))
        rng.shuffle(indices)
        
        for k in indices:
            if deadline is not None and time.time() > deadline:
                break
                
            old_val = current_A[k]
            # Try a few random alternatives
            for _ in range(8):
                if rng.random() < 0.6:
                    # Local perturbation
                    delta = rng.randint(-20, 20)
                    new_val = old_val + delta
                else:
                    # Global jump
                    new_val = rng.randint(1, 10000)
                
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in current_A and new_val != old_val:
                    continue
                    
                current_A[k] = new_val
                new_score = _score(current_A)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_A = current_A[:]
                    break
                else:
                    current_A[k] = old_val
            
            if improved:
                break
        
        if not improved:
            break
            
    return best_A, best_score

def _sa_search(A, start_temp=10.0, end_temp=0.1, steps=10000, seed=None, deadline=None):
    """
    Simulated Annealing search with deadline guard.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
        
    current_A = A[:]
    current_score = _score(current_A)
    best_A = current_A[:]
    best_score = current_score
    
    n = len(current_A)
    
    for step in range(steps):
        if deadline is not None and time.time() > deadline:
            break
            
        T = start_temp * (end_temp / start_temp) ** (step / steps)
        
        k = rng.randint(0, n - 1)
        old_val = current_A[k]
        
        if rng.random() < 0.7:
            delta = rng.randint(-20, 20)
            new_val = old_val + delta
        elif rng.random() < 0.5:
            new_val = rng.randint(1, max(1, old_val // 2)) if old_val > 10 else rng.randint(1, 100)
        else:
            new_val = rng.randint(1, 10000)
            
        if new_val < 1 or new_val > 10000:
            continue
            
        if new_val in current_A and new_val != old_val:
            continue
            
        current_A[k] = new_val
        new_score = _score(current_A)
        
        delta_score = new_score - current_score
        if delta_score < 0 or rng.random() < math.exp(-delta_score / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = current_A[:]
        else:
            current_A[k] = old_val
            
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s limit
    
    best_A = None
    best_score = float('inf')
    
    # Phase 1: Multi-start over Geometric Ratios
    # The product set size is heavily influenced by the multiplicative structure.
    # GPs with small integer-like ratios tend to have small product sets.
    
    # Sample r in [1.05, 1.8]
    r_samples = np.linspace(1.05, 1.8, 100)
    nice_ratios = [1.1, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]
    all_r = list(r_samples) + nice_ratios
    
    for r in all_r:
        if time.time() > deadline:
            break
            
        A = _generate_gp_candidate(r, start=1, n=20, max_val=10000)
        if A is None:
            continue
            
        current_A = A[:]
        current_score = _score(current_A)
        
        # Quick local search to refine the GP structure
        A_ls, score_ls = _local_search(current_A, max_iter=1000, deadline=deadline - 2, seed=42)
        if score_ls < current_score:
            current_A = A_ls
            current_score = score_ls
        
        if current_score < best_score:
            best_score = current_score
            best_A = current_A[:]
            
    # Phase 2: Run SA from a few diverse starting points
    if time.time() < deadline - 10:
        # Start from the best found so far
        if best_A is not None:
            A_res, score_res = _sa_search(best_A, start_temp=5.0, end_temp=0.01, steps=5000, seed=100, deadline=deadline - 2)
            if score_res < best_score:
                best_score = score_res
                best_A = A_res[:]
                
        # Start from a few random points
        for i in range(3):
            if time.time() > deadline - 2:
                break
            A_rand = random.sample(range(1, 10001), 20)
            A_res, score_res = _sa_search(A_rand, start_temp=8.0, end_temp=0.05, steps=3000, seed=200+i, deadline=deadline - 2)
            if score_res < best_score:
                best_score = score_res
                best_A = A_res[:]
                
    # Final polish: local search on best solution
    if best_A is not None and time.time() < deadline - 2:
        A_polish, score_polish = _local_search(best_A, max_iter=2000, deadline=deadline, seed=999)
        if score_polish < best_score:
            best_score = score_polish
            best_A = A_polish
        
    if best_A is None:
        best_A = list(range(1, 21))
        
    return sorted(best_A)
