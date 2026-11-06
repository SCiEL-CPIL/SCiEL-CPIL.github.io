import time
import random
import math
import numpy as np

def _score(A):
    """Compute max(|A+A|, |A*A|) using numpy for speed."""
    arr = np.array(A, dtype=np.int64)
    sums = np.add.outer(arr, arr)
    prods = np.multiply.outer(arr, arr)
    n_sums = len(np.unique(sums))
    n_prods = len(np.unique(prods))
    return max(n_sums, n_prods)

def _valid(A):
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    if any(x <= 0 or x > 10000 for x in A):
        return False
    return True

def _local_search(A, max_iter=3000, deadline=None):
    """Local search with structured perturbations."""
    current_A = list(A)
    current_score = _score(current_A)
    best_A = list(current_A)
    best_score = current_score
    rng = random.Random(99)
    n = len(current_A)
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        # Try modifying one element
        k = rng.randrange(n)
        old_val = current_A[k]
        
        # Generate candidate replacements: nearby values and structured alternatives
        candidates = []
        # Nearby values
        for delta in [-5, -3, -2, -1, 1, 2, 3, 5, 10, -10]:
            nv = old_val + delta
            if 1 <= nv <= 10000:
                candidates.append(nv)
        # Geometric neighbors
        if old_val > 2:
            candidates.append(max(1, old_val // 2))
            candidates.append(min(10000, old_val * 2))
        # Random
        candidates.append(rng.randint(1, 10000))
        candidates.append(rng.randint(1, max(2, old_val + 10)))
        
        rng.shuffle(candidates)
        
        for new_val in candidates:
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
                    best_A = list(current_A)
                break
            else:
                current_A[k] = old_val
        
        if not improved:
            # Try a more aggressive random move
            k = rng.randrange(n)
            old_val = current_A[k]
            new_val = rng.randint(1, 10000)
            if new_val in current_A and new_val != old_val:
                continue
            current_A[k] = new_val
            new_score = _score(current_A)
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_A = list(current_A)
            else:
                current_A[k] = old_val
    
    return best_A, best_score

def _structural_search(deadline):
    """
    Deterministic search over truncated geometric progressions with rational ratios p/q.
    This explores the trade-off between sumset and product set sizes.
    """
    best_A = None
    best_score = float('inf')
    candidates = []
    
    # Search over rational ratios p/q
    # p, q up to 50
    for p in range(1, 51):
        for q in range(1, 51):
            if p <= q:
                continue # We want r > 1 for growth, or handle r < 1 separately? 
                         # If r < 1, we can just reverse the set. 
                         # So we can restrict to r > 1 and then sort the result.
            
            r = p / q
            
            # Estimate max a such that a * r^19 <= 10000
            max_a = int(10000 / (r ** 19))
            if max_a < 1:
                continue
            
            # We also need distinct integers.
            # If r is close to 1, a must be large enough to ensure distinctness.
            # Gap approx a * r^(i-1) * (r-1).
            # Smallest gap is at i=1: a * (r-1).
            # We need a * (r-1) >= 1 roughly? Or just check distinctness.
            
            # Search a in a logarithmic or linear range?
            # If r is large, a can be small.
            # If r is close to 1, a must be large.
            
            # Let's sample 'a' values.
            # To be efficient, we can try a few strategic 'a's.
            # 1. Smallest valid 'a' (ensures smallest numbers, might help sumset)
            # 2. Largest valid 'a' (ensures largest spread, might help productset? No, productset is relative)
            # 3. Middle values.
            
            # Actually, for a fixed ratio, the structure of the set (relative to itself) 
            # determines the sumset/productset sizes *if* we scale?
            # No, sumset is not scale invariant. $|kA+kA| = |A+A|$. 
            # Wait. $A = \{a, ar, \dots\}$.
            # $A+A = \{a(1+r^i) + a(1+r^j) \dots\} = a \{ (1+r^i) + (1+r^j) \}$.
            # So $|A+A|$ is independent of $a$!
            # $A \cdot A = \{ a^2 r^{i+j} \}$.
            # $|A \cdot A|$ is also independent of $a$!
            
            # SO: For a fixed ratio r (and rounding behavior), the score is independent of a,
            # PROVIDED that the rounding doesn't change the "structure" (collisions).
            # However, rounding *does* affect collisions.
            # $round(a r^i)$ might collide with $round(a r^j)$ if $a$ is small.
            # If $a$ is large enough, the relative positions are preserved better.
            
            # Therefore, we should pick the LARGEST possible 'a' to minimize rounding errors 
            # and preserve the ideal GP structure (minimizing product set) 
            # while keeping the sumset structure (which is scale invariant for exact GPs).
            
            # BUT, if we pick the largest 'a', the numbers are large. 
            # Does that matter? No, only distinctness and <= 10000.
            
            # So, for each (p,q), we should try the largest valid 'a'.
            # And maybe a few smaller ones to see if rounding helps (unlikely) or hurts.
            
            a_start = max(1, int(max_a * 0.5)) # Try middle
            a_end = max_a
            
            # Try a few 'a' values: max, max/2, max/4, 1
            a_values = set([a_end, a_start, max(1, a_end // 4), 1])
            
            for a in a_values:
                A = []
                val = float(a)
                valid = True
                for i in range(20):
                    iv = int(val + 0.5)
                    if iv > 10000 or iv < 1:
                        valid = False
                        break
                    if iv in A:
                        valid = False
                        break
                    A.append(iv)
                    val *= r
                
                if valid and len(A) == 20:
                    A.sort()
                    if _valid(A):
                        s = _score(A)
                        if s < best_score:
                            best_score = s
                            best_A = list(A)
                        candidates.append((s, A))
                
                # Check deadline
                if time.time() > deadline:
                    return best_A, best_score, candidates

    return best_A, best_score, candidates

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # 1. Structural Search
    best_A, best_score, candidates = _structural_search(deadline)
    
    if best_A is None:
        # Fallback to simple AP
        best_A = list(range(1, 21))
        best_score = _score(best_A)
        candidates = [(best_score, best_A)]

    # Sort candidates by score
    candidates.sort(key=lambda x: x[0])
    
    # 2. Local Search on top candidates
    # We have limited time. Pick top 5 diverse candidates?
    # Just pick top 5 by score.
    for s_c, cand in candidates[:5]:
        if time.time() > deadline - 10:
            break
        A_ls, s_ls = _local_search(cand, max_iter=2000, deadline=deadline)
        if s_ls < best_score:
            best_score = s_ls
            best_A = A_ls

    # 3. Final Polish
    if time.time() < deadline - 2:
        A_polish, s_polish = _local_search(best_A, max_iter=3000, deadline=deadline)
        if s_polish < best_score:
            best_score = s_polish
            best_A = A_polish

    return best_A
