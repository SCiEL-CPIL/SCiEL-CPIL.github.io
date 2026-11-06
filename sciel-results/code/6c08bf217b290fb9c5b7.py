import time
import random
from itertools import combinations

N = 20
MAX_VAL = 10000
TIME_LIMIT = 85.0  # safety margin before 90s kill

def compute_objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of N distinct positive integers."""
    # Sumset: all a_i + a_j for i <= j (equal summands allowed)
    sums = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            sums.add(ai + A[j])
    
    # Product set: all a_i * a_j for i <= j (equal factors allowed)
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            prods.add(ai * A[j])
    
    return max(len(sums), len(prods))

def geometric_progression():
    """Generate a geometric progression-like set."""
    # Try various bases and starts
    best = None
    best_score = float('inf')
    for base in range(2, 15):
        for start in range(1, 50):
            A = []
            val = start
            for _ in range(N):
                if val > MAX_VAL:
                    break
                A.append(val)
                val = val * base
            if len(A) != N:
                continue
            if len(set(A)) != N:
                continue
            score = compute_objective(A)
            if score < best_score:
                best_score = score
                best = list(A)
    return best, best_score

def arithmetic_progression():
    """Generate an arithmetic progression set."""
    best = None
    best_score = float('inf')
    for a in range(1, 200):
        for d in range(1, 200):
            A = [a + i * d for i in range(N)]
            if A[-1] > MAX_VAL:
                continue
            if len(set(A)) != N:
                continue
            score = compute_objective(A)
            if score < best_score:
                best_score = score
                best = list(A)
    return best, best_score

def mixed_construction():
    """Try a mix: small numbers densely, then spread out."""
    best = None
    best_score = float('inf')
    
    # Strategy 1: First k numbers dense, rest geometric
    for k in range(5, 15):
        for start in range(1, 10):
            for d in range(1, 5):
                for base in range(2, 8):
                    A = []
                    # Dense part
                    for i in range(k):
                        A.append(start + i * d)
                    # Geometric part
                    val = A[-1] * base
                    for i in range(N - k):
                        if val > MAX_VAL:
                            break
                        A.append(val)
                        val = val * base
                    if len(A) != N:
                        continue
                    if len(set(A)) != N:
                        continue
                    if max(A) > MAX_VAL:
                        continue
                    score = compute_objective(A)
                    if score < best_score:
                        best_score = score
                        best = list(A)
    return best, best_score

def random_local_search(A, max_iter=5000, seed=42):
    """Local search: try replacing one element at a time."""
    rng = random.Random(seed)
    A = list(A)
    current_score = compute_objective(A)
    best_A = list(A)
    best_score = current_score
    
    for _ in range(max_iter):
        # Pick a random element to replace
        idx = rng.randint(0, N - 1)
        old_val = A[idx]
        
        # Try a few random replacements
        for _ in range(5):
            new_val = rng.randint(1, MAX_VAL)
            if new_val == old_val:
                continue
            if new_val in A:
                continue
            
            A[idx] = new_val
            score = compute_objective(A)
            if score < best_score:
                best_score = score
                best_A = list(A)
            A[idx] = old_val
        
        # Also try nearby values
        for delta in [-1, 1, -2, 2, -3, 3, -5, 5, -10, 10]:
            new_val = old_val + delta
            if new_val < 1 or new_val > MAX_VAL:
                continue
            if new_val == old_val:
                continue
            if new_val in A:
                continue
            
            A[idx] = new_val
            score = compute_objective(A)
            if score < best_score:
                best_score = score
                best_A = list(A)
            A[idx] = old_val
    
    return best_A, best_score

def simulated_annealing(A, max_iter=10000, seed=42):
    """Simulated annealing for better exploration."""
    rng = random.Random(seed)
    A = list(A)
    current_score = compute_objective(A)
    best_A = list(A)
    best_score = current_score
    
    T = 10.0
    cooling = 0.999
    
    for _ in range(max_iter):
        idx = rng.randint(0, N - 1)
        old_val = A[idx]
        
        # Generate a neighbor
        if rng.random() < 0.5:
            new_val = rng.randint(1, MAX_VAL)
        else:
            new_val = old_val + rng.randint(-20, 20)
        
        if new_val < 1 or new_val > MAX_VAL:
            continue
        if new_val == old_val:
            continue
        if new_val in A:
            continue
        
        A[idx] = new_val
        new_score = compute_objective(A)
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < pow(2.718281828, -delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = list(A)
        else:
            A[idx] = old_val
        
        T *= cooling
        if T < 0.1:
            T = 0.1
    
    return best_A, best_score

def construct():
    start_time = time.time()
    
    # Phase 1: Try structured constructions
    candidates = []
    
    # Geometric progressions
    A_geo, score_geo = geometric_progression()
    if A_geo is not None:
        candidates.append((A_geo, score_geo))
    
    # Arithmetic progressions
    A_arith, score_arith = arithmetic_progression()
    if A_arith is not None:
        candidates.append((A_arith, score_arith))
    
    # Mixed constructions
    A_mixed, score_mixed = mixed_construction()
    if A_mixed is not None:
        candidates.append((A_mixed, score_mixed))
    
    # Simple range
    A_range = list(range(1, N + 1))
    score_range = compute_objective(A_range)
    candidates.append((A_range, score_range))
    
    # Sort by score
    candidates.sort(key=lambda x: x[1])
    
    best_A, best_score = candidates[0]
    
    # Phase 2: Local search from best candidates
    elapsed = time.time() - start_time
    time_budget = TIME_LIMIT - elapsed - 5  # 5s safety margin
    
    if time_budget > 2:
        # Local search from top candidates
        for A_init, score_init in candidates[:3]:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            A_ls, score_ls = random_local_search(A_init, max_iter=3000, seed=123)
            if score_ls < best_score:
                best_score = score_ls
                best_A = A_ls
    
    # Phase 3: Simulated annealing from best found
    elapsed = time.time() - start_time
    time_budget = TIME_LIMIT - elapsed - 5
    
    if time_budget > 2:
        A_sa, score_sa = simulated_annealing(best_A, max_iter=5000, seed=456)
        if score_sa < best_score:
            best_score = score_sa
            best_A = A_sa
    
    # Final check
    if time.time() - start_time > TIME_LIMIT - 3:
        return best_A
    
    # One more round of local search if time permits
    elapsed = time.time() - start_time
    time_budget = TIME_LIMIT - elapsed - 3
    
    if time_budget > 1:
        A_final, score_final = random_local_search(best_A, max_iter=2000, seed=789)
        if score_final < best_score:
            best_score = score_final
            best_A = A_final
    
    return best_A
