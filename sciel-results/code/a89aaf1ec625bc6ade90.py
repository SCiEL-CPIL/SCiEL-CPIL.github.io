"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def _prime_factors(n):
    """Return set of prime factors of n."""
    factors = set()
    if n <= 1:
        return factors
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _score(nums):
    """Count distinct prime factors across all pairwise sums."""
    primes = set()
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            primes |= _prime_factors(s)
    return len(primes)

def _is_smooth(n, prime_set):
    """Check if n has no prime factors outside prime_set."""
    if n <= 1:
        return True
    for p in prime_set:
        while n % p == 0:
            n //= p
    return n == 1

def _find_clique(candidates, adj, target_size, deadline):
    """
    Find a clique of size target_size in the graph defined by adj.
    candidates: sorted list of vertex values
    adj: dict mapping value -> set of values it's connected to
    Uses greedy + backtracking.
    """
    n = len(candidates)
    if n < target_size:
        return None
    
    # Build adjacency as bitmasks for speed
    # Map each candidate to an index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    # adj_mask[i] is a bitmask of which vertices are adjacent to i
    adj_mask = [0] * n
    for i in range(n):
        v = candidates[i]
        for w in adj.get(v, ()):
            j = val_to_idx.get(w)
            if j is not None:
                adj_mask[i] |= (1 << j)
    
    # Greedy: try to build a clique by picking vertices with high degree within candidates
    # Sort by degree descending
    degrees = [bin(adj_mask[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    
    # Try multiple starting points
    best_clique = None
    
    for start_idx in order[:min(50, n)]:
        if time.time() > deadline:
            break
        clique = [start_idx]
        # Candidates that are adjacent to all current clique members
        candidates_mask = adj_mask[start_idx]
        # Remove start_idx from candidates
        candidates_mask &= ~(1 << start_idx)
        
        # Greedy expansion
        while len(clique) < target_size:
            if time.time() > deadline:
                break
            # Find candidates adjacent to all clique members
            valid = candidates_mask
            for c in clique:
                valid &= adj_mask[c]
            
            if valid == 0:
                break
            
            # Pick the vertex with highest degree among valid
            best_v = -1
            best_deg = -1
            m = valid
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                deg = bin(adj_mask[v] & valid).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                m &= m - 1
            
            if best_v == -1:
                break
            
            clique.append(best_v)
            candidates_mask &= ~(1 << best_v)
        
        if len(clique) >= target_size:
            best_clique = [candidates[i] for i in clique[:target_size]]
            break
    
    # If greedy failed, try backtracking from a few starting points
    if best_clique is None:
        for start_idx in order[:min(20, n)]:
            if time.time() > deadline:
                break
            clique = [start_idx]
            candidates_mask = adj_mask[start_idx] & ~(1 << start_idx)
            
            def backtrack(clique, candidates_mask):
                if len(clique) >= target_size:
                    return True
                if time.time() > deadline:
                    return False
                
                # Find valid candidates
                valid = candidates_mask
                for c in clique:
                    valid &= adj_mask[c]
                
                if valid == 0:
                    return False
                
                # Try candidates in order of degree
                m = valid
                order_list = []
                while m:
                    lsb = m & (-m)
                    v = lsb.bit_length() - 1
                    deg = bin(adj_mask[v] & valid).count('1')
                    order_list.append((deg, v))
                    m &= m - 1
                order_list.sort(key=lambda x: -x[0])
                
                for _, v in order_list[:30]:  # Limit branching
                    new_clique = clique + [v]
                    new_mask = candidates_mask & ~(1 << v)
                    if backtrack(new_clique, new_mask):
                        clique[:] = new_clique
                        return True
                
                return False
            
            if backtrack(clique, candidates_mask):
                best_clique = [candidates[i] for i in clique[:target_size]]
                break
    
    return best_clique

def construct():
    start = time.time()
    deadline = start + 85.0
    
    # Initial best from simple heuristics
    best = list(range(1, 13))
    best_score = _score(best)
    
    # Core strategy: Fix a small prime set S, build compatibility graph, find clique of 12
    # Try various small prime sets in order of increasing size
    prime_sets_to_try = [
        {2},
        {2, 3},
        {2, 5},
        {2, 7},
        {3, 5},
        {2, 3, 5},
        {2, 3, 7},
        {2, 5, 7},
        {3, 5, 7},
        {2, 3, 11},
        {2, 5, 11},
        {2, 3, 5, 7},
        {2, 3, 5, 11},
        {2, 3, 7, 11},
        {2, 5, 7, 11},
        {3, 5, 7, 11},
        {2, 3, 5, 7, 11},
        {2, 3, 5, 7, 13},
        {2, 3, 5, 7, 17},
        {2, 3, 5, 7, 19},
        {2, 3, 5, 7, 23},
    ]
    
    # Candidate pool: focus on numbers that are "smooth" with respect to small primes
    # Generate candidates: numbers up to 10000 that are 2^a * 3^b * 5^c * 7^d
    # But also include other numbers that might work
    # Strategy: use numbers that are multiples of small combinations
    
    # Generate a focused candidate pool
    # Include: smooth numbers (2^a*3^b*5^c*7^d), and numbers near them
    candidate_pool = set()
    
    # Smooth numbers
    for a in range(0, 14):
        for b in range(0, 10):
            for c in range(0, 8):
                for d in range(0, 6):
                    val = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d)
                    if 1 <= val <= 10000:
                        candidate_pool.add(val)
    
    # Also add multiples of small numbers
    for base in range(1, 50):
        for k in range(1, 250):
            val = base * k
            if 1 <= val <= 10000:
                candidate_pool.add(val)
    
    # Add some structured sets
    for g in range(1, 30):
        for offset in range(0, 20):
            for i in range(12):
                val = g * (i + offset + 1)
                if 1 <= val <= 10000:
                    candidate_pool.add(val)
    
    candidate_pool = sorted(candidate_pool)
    
    # For each prime set, build graph and find clique
    for prime_set in prime_sets_to_try:
        if time.time() > deadline - 10:
            break
        
        # Determine candidate subset: prefer numbers whose sums with others are smooth
        # Use all candidates but limit to a manageable size
        # Focus on smaller numbers first (sums will be smaller, more likely smooth)
        candidates = [v for v in candidate_pool if v <= 5000]
        
        if len(candidates) < 12:
            continue
        
        # Build adjacency: edge if sum is smooth w.r.t. prime_set
        # For efficiency, limit candidate size
        if len(candidates) > 2000:
            # Prioritize smooth numbers
            smooth_cands = []
            for v in candidates:
                if _is_smooth(v, prime_set):
                    smooth_cands.append(v)
            if len(smooth_cands) >= 12:
                candidates = smooth_cands[:2000]
            else:
                candidates = candidates[:2000]
        
        n_cands = len(candidates)
        
        # Build adjacency
        adj = {}
        for i in range(n_cands):
            vi = candidates[i]
            neighbors = set()
            for j in range(i + 1, n_cands):
                vj = candidates[j]
                s = vi + vj
                if _is_smooth(s, prime_set):
                    neighbors.add(vj)
                    adj.setdefault(vj, set()).add(vi)
            adj[vi] = neighbors
        
        # Find clique of size 12
        clique = _find_clique(candidates, adj, 12, deadline - 5)
        
        if clique is not None:
            sc = _score(clique)
            if sc < best_score:
                best_score = sc
                best = clique[:]
                # If we found a very good solution, we can stop
                if best_score <= len(prime_set):
                    break
    
    if time.time() > deadline - 5:
        return best
    
    # Local search refinement on the best solution
    random.seed(42)
    current = best[:]
    current_score = best_score
    
    no_improve = 0
    max_no_improve = 200
    while time.time() < deadline - 3 and no_improve < max_no_improve:
        i = random.randint(0, 11)
        old_val = current[i]
        
        best_new_val = None
        best_new_score = current_score
        
        # Try a range of candidate values
        # Focus on values that might produce smooth sums
        test_vals = set()
        # Small perturbations
        for delta in range(-50, 51, 1):
            nv = old_val + delta
            if 1 <= nv <= 10000 and nv not in current:
                test_vals.add(nv)
        # Smooth-ish values
        for a in range(0, 10):
            for b in range(0, 8):
                nv = (2 ** a) * (3 ** b)
                if 1 <= nv <= 10000 and nv not in current:
                    test_vals.add(nv)
        # Random
        for _ in range(20):
            nv = random.randint(1, 10000)
            if nv not in current:
                test_vals.add(nv)
        
        for new_val in test_vals:
            if new_val in current:
                continue
            current[i] = new_val
            sc = _score(current)
            if sc < best_new_score:
                best_new_score = sc
                best_new_val = new_val
            current[i] = old_val
        
        if best_new_val is not None:
            current[i] = best_new_val
            current_score = best_new_score
            no_improve = 0
        else:
            no_improve += 1
    
    if current_score < best_score:
        best_score = current_score
        best = current[:]
    
    return best
# EVOLVE-BLOCK-END
