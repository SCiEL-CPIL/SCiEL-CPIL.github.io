"""
Strategy: Explicit Sidon Set Construction and Constrained Local Search.
Instead of generic random search, we generate high-quality initial candidates 
using known Sidon set constructions (greedy, Bose-Chowla style) and then 
refine them using a local search that specifically tries to maintain or improve 
the Sidon property (distinct sums) while minimizing the maximum element.
"""
# EVOLVE-BLOCK-START
import time
import math
import random

def _compute_score(U):
    """
    Computes the score: 1 + log(|U-U|/|U+U|) / log(2 max U + 1)
    Returns (score, sorted_list)
    """
    U = list(set(int(x) for x in U))
    if 0 not in U:
        U.append(0)
    if not U:
        return 0.0, []
    
    # Ensure non-negative
    U = [x for x in U if x >= 0]
    if not U:
        return 0.0, []
        
    mx = max(U)
    if mx <= 0:
        return 0.0, sorted(U)
        
    n = len(U)
    # For large n, O(n^2) is too slow if we do many iterations, 
    # but for n < 100, it's fine.
    plus = set()
    minus = set()
    for i in range(n):
        a = U[i]
        for j in range(i, n):
            plus.add(a + U[j])
        for j in range(n):
            minus.add(a - U[j])
            
    n_plus = len(plus)
    n_minus = len(minus)
    
    if n_plus == 0:
        return 0.0, sorted(U)
        
    ratio = n_minus / n_plus
    if ratio <= 0:
        return 0.0, sorted(U)
        
    log_den = math.log(2 * mx + 1)
    if log_den <= 0:
        return 0.0, sorted(U)
        
    val = 1.0 + math.log(ratio) / log_den
    return val, sorted(U)

def _is_sidon(U):
    """Check if U is a Sidon set (distinct sums for unordered pairs)."""
    U = sorted(U)
    n = len(U)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = U[i] + U[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_sidon(start, target_size, max_val=100000):
    """
    Generate a Sidon set greedily.
    Starts with 'start' and adds the smallest integer that preserves the Sidon property.
    """
    U = list(start)
    U.sort()
    # Precompute existing sums
    sums = set()
    n = len(U)
    for i in range(n):
        for j in range(i, n):
            sums.add(U[i] + U[j])
            
    while len(U) < target_size:
        candidate = U[-1] + 1
        found = False
        while candidate <= max_val:
            # Check if adding candidate creates duplicate sums
            valid = True
            for i in range(len(U)):
                s = U[i] + candidate
                if s in sums:
                    valid = False
                    break
            if valid:
                U.append(candidate)
                # Update sums
                for i in range(len(U) - 1):
                    sums.add(U[i] + candidate)
                found = True
                break
            candidate += 1
        if not found:
            break
            
    return U

def _bose_chowla(n, p):
    """
    Bose-Chowla construction for Sidon sets in Z_p.
    Requires p to be prime and n | p-1.
    Returns a set of integers in [0, p-1].
    """
    if n <= 0:
        return [0]
    # Find primitive root
    def primitive_root(p):
        if p == 2: return 1
        for g in range(2, p):
            if all(pow(g, (p-1)//q, p) != 1 for q in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]):
                return g
        return None
        
    g = primitive_root(p)
    if g is None:
        return []
        
    elements = []
    # g^(k*n) for k=0..n-1? No, standard construction uses g^i.
    # Actually, a simple Sidon set is {g^i mod p} for i=0..n-1 if n is small enough relative to p.
    # Let's just use the powers.
    for i in range(n):
        elements.append(pow(g, i, p))
        
    # Ensure 0 is included? The task requires 0 in U.
    # If 0 is not in the set, we can add it if it doesn't break Sidon property?
    # 0 + x = x. If x is already a sum of two other elements, it breaks.
    # In a Sidon set from Z_p, 0 is often not included in the multiplicative group.
    # We will handle 0 in the main construct function.
    return sorted(set(elements))

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin
    rng = random.Random(123)
    
    best_val = -1e9
    best_set = [0, 1, 3]
    
    def consider(U):
        nonlocal best_val, best_set
        if time.time() > deadline:
            return
        try:
            v, s = _compute_score(U)
        except Exception:
            return
        if v > best_val:
            best_val = v
            best_set = s

    # Phase 1: Generate diverse high-quality initial candidates
    
    # 1. Greedy Sidon sets of various sizes
    # Small sets are often optimal for the exponent because max U grows quadratically with n for Sidon sets.
    for target_size in range(3, 30):
        # Start with different seeds
        for seed_val in [0, 1, 2, 5, 10]:
            U = _greedy_sidon([0, seed_val], target_size)
            if len(U) == target_size:
                consider(U)
                
    # 2. Bose-Chowla / Primitive Root based Sidon sets
    primes = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in primes:
        for n in range(3, min(15, p)):
            # Simple powers of primitive root
            # Find primitive root
            g = None
            for gg in range(2, p):
                is_gen = True
                for q in [2, 3, 5, 7, 11, 13]:
                    if (p-1) % q == 0:
                        if pow(gg, (p-1)//q, p) == 1:
                            is_gen = False
                            break
                if is_gen:
                    g = gg
                    break
            if g:
                elems = [pow(g, i, p) for i in range(n)]
                # Add 0
                U = [0] + elems
                U = sorted(set(U))
                consider(U)
                
    # 3. Arithmetic Progressions (Baseline)
    for n in range(3, 20):
        for step in [1, 2, 3, 5]:
            U = [i * step for i in range(n)]
            consider(U)
            
    # 4. Geometric Progressions
    for n in range(3, 20):
        for base in [2, 3, 4, 5]:
            U = [base**i for i in range(n)]
            consider(U)

    # Phase 2: Local Search Refinement
    # We will take the best candidate found so far and refine it.
    # Strategy: Try to replace elements with smaller values or add new elements 
    # that preserve the "Sidon-like" property (minimizing sum collisions).
    
    current_best = list(best_set)
    
    # We will perform multiple rounds of local search from different starting points
    # The top 5 candidates from Phase 1 are good starting points.
    
    # Let's collect top candidates
    candidates = []
    
    # Re-evaluate some specific interesting sets
    # Sidon sets are the key. Let's generate a few more specific ones.
    for n in range(5, 25):
        U = _greedy_sidon([0, 1], n)
        if len(U) == n:
            candidates.append(U)
            
    # Shuffle and take top few by score
    scored_candidates = []
    for U in candidates:
        v, s = _compute_score(U)
        scored_candidates.append((v, s))
        
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Take top 3 distinct starting points for local search
    start_points = []
    seen = set()
    for v, s in scored_candidates:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            start_points.append(s)
        if len(start_points) >= 3:
            break
            
    if not start_points:
        start_points = [list(best_set)]
        
    for start_set in start_points:
        cur = list(start_set)
        cur_val, _ = _compute_score(cur)
        
        improved = True
        it = 0
        max_iters = 5000
        
        while improved and time.time() < deadline - 5.0 and it < max_iters:
            improved = False
            it += 1
            mx = max(cur)
            n_cur = len(cur)
            
            # Generate candidates for modification
            # 1. Try adding a new element
            # Look for smallest number > mx that doesn't create too many sum collisions
            # Or just try random small numbers
            
            # 2. Try removing an element (if n is large)
            # 3. Try swapping an element with a smaller one
            
            # Heuristic: Try to reduce max(cur) by swapping large elements with smaller ones
            # that maintain the Sidon property.
            
            # Try swaps
            if n_cur > 3:
                # Pick a random index to replace
                idx = rng.randint(1, n_cur - 1) # Don't replace 0
                old_val = cur[idx]
                
                # Try replacing with a smaller value
                # We want to find a new_val < old_val such that the set remains "good"
                # We can't easily check Sidon property in O(1), so we just try a few and score them.
                
                # Candidates for new_val:
                # - Random small integers
                # - Gaps in the current set
                cands = set()
                
                # Fill gaps
                sorted_cur = sorted(cur)
                for i in range(len(sorted_cur) - 1):
                    gap = sorted_cur[i+1] - sorted_cur[i]
                    if gap > 1:
                        cands.add(sorted_cur[i] + 1)
                        cands.add(sorted_cur[i] + gap // 2)
                        
                # Random small values
                for _ in range(20):
                    cands.add(rng.randint(1, max(2, mx // 2)))
                    
                # Values near existing ones
                for c in cur:
                    if c > 0:
                        cands.add(c - 1)
                        cands.add(c + 1)
                        
                # Filter invalid candidates (must be in set, >= 1, not already in cur)
                valid_cands = [c for c in cands if c >= 1 and c not in cur and c != old_val]
                
                if valid_cands:
                    # Sort by value (prefer smaller)
                    valid_cands.sort()
                    
                    for new_val in valid_cands[:10]: # Only try top 10 smallest
                        new_set = cur.copy()
                        new_set[idx] = new_val
                        new_set = sorted(set(new_set))
                        if 0 not in new_set:
                            new_set.append(0)
                            
                        v, s = _compute_score(new_set)
                        if v > cur_val + 1e-12:
                            cur_val = v
                            cur = s
                            improved = True
                            break
                            
            # Try adding an element if the set is small
            if n_cur < 20 and not improved:
                # Try adding the smallest possible number > mx
                # Or a number that fits in a gap
                cands_add = set()
                sorted_cur = sorted(cur)
                for i in range(len(sorted_cur) - 1):
                    gap = sorted_cur[i+1] - sorted_cur[i]
                    if gap > 1:
                        cands_add.add(sorted_cur[i] + 1)
                cands_add.add(mx + 1)
                cands_add.add(mx + 2)
                
                for new_val in sorted(cands_add):
                    new_set = cur + [new_val]
                    new_set = sorted(set(new_set))
                    v, s = _compute_score(new_set)
                    if v > cur_val + 1e-12:
                        cur_val = v
                        cur = s
                        improved = True
                        break
                        
            # Update global best
            if cur_val > best_val:
                best_val = cur_val
                best_set = cur.copy()

    # Final check
    v, s = _compute_score(best_set)
    if v > best_val:
        best_val = v
        best_set = s
        
    return best_set

# EVOLVE-BLOCK-END
