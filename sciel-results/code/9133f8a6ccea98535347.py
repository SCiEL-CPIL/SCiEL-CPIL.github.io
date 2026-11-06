import time
import random
import itertools
from collections import defaultdict

def _is_linear(triples):
    """Check if any two triples share at most one point."""
    pair_count = defaultdict(int)
    for t in triples:
        a, b, c = t
        if pair_count[(a, b)] > 0 or pair_count[(a, c)] > 0 or pair_count[(b, c)] > 0:
            return False
        pair_count[(a, b)] += 1
        pair_count[(a, c)] += 1
        pair_count[(b, c)] += 1
    return True

def _calc_alpha(triples, n_points=16):
    """Calculate independence number using hitting set search."""
    if not triples:
        return n_points
    
    # Convert to tuples for hashing and speed
    t_list = [tuple(t) for t in triples]
    m = len(t_list)
    
    # Memoization dictionary
    memo = {}
    
    def solve(mask):
        if mask in memo:
            return memo[mask]
        
        if mask == 0:
            memo[mask] = 0
            return 0
            
        # Find the first triple in the mask
        idx = 0
        while not (mask & (1 << idx)):
            idx += 1
            
        t = t_list[idx]
        
        best = float('inf')
        
        # Try each vertex in the triple
        for v in t:
            # Build new mask: remove all triples containing v
            new_mask = 0
            # Optimization: Precompute for each vertex v, a bitmask of triples containing v?
            # Let's do it on the fly for simplicity, but it's O(m).
            for j in range(m):
                if mask & (1 << j):
                    if v not in t_list[j]:
                        new_mask |= (1 << j)
                        
            res = 1 + solve(new_mask)
            if res < best:
                best = res
                
        memo[mask] = best
        return best

    # Precompute vertex masks for faster new_mask calculation?
    # Let's try to optimize the inner loop.
    # vertex_mask[v] = bitmask of triples containing v
    
    vertex_masks = [0] * n_points
    for i, t in enumerate(t_list):
        for v in t:
            vertex_masks[v] |= (1 << i)
            
    def solve_opt(mask):
        if mask in memo:
            return memo[mask]
        if mask == 0:
            memo[mask] = 0
            return 0
            
        # Find first triple
        idx = 0
        while not (mask & (1 << idx)):
            idx += 1
        t = t_list[idx]
        
        best = float('inf')
        for v in t:
            # Remove all triples in mask that contain v
            # new_mask = mask & ~vertex_masks[v]
            new_mask = mask & ~vertex_masks[v]
            res = 1 + solve_opt(new_mask)
            if res < best:
                best = res
                
        memo[mask] = best
        return best

    init_mask = (1 << m) - 1
    tau = solve_opt(init_mask)
    return n_points - tau

def construct():
    deadline = time.time() + 145
    n_points = 16
    
    # 1. Initial Greedy Construction
    triples = []
    pair_count = defaultdict(int)
    
    # Shuffle points to break symmetry?
    # Use a fixed seed for reproducibility in the "seed" part, but SA will vary.
    rng = random.Random(42)
    
    candidates = list(itertools.combinations(range(n_points), 3))
    rng.shuffle(candidates)
    
    for c in candidates:
        a, b, c_val = c
        if pair_count[(a, b)] == 0 and pair_count[(a, c_val)] == 0 and pair_count[(b, c_val)] == 0:
            triples.append([a, b, c_val])
            pair_count[(a, b)] += 1
            pair_count[(a, c_val)] += 1
            pair_count[(b, c_val)] += 1
            
    best_triples = [list(t) for t in triples]
    best_alpha = _calc_alpha(best_triples, n_points)
    
    # 2. Simulated Annealing
    current_triples = best_triples
    current_alpha = best_alpha
    T = 2.0
    
    # Precompute all possible triples for quick lookup?
    # Or just generate neighbors on the fly.
    
    while time.time() < deadline:
        # Generate Neighbor
        if not current_triples:
            # If empty, add a random triple
            a, b, c_val = rng.sample(range(n_points), 3)
            new_triples = [[a, b, c_val]]
        else:
            # Remove a random triple
            idx_rem = rng.randrange(len(current_triples))
            t_rem = current_triples[idx_rem]
            new_triples = [t for i, t in enumerate(current_triples) if i != idx_rem]
            
            # Update pair counts for the remaining triples
            pair_count = defaultdict(int)
            for t in new_triples:
                a, b, c_val = t
                pair_count[(a, b)] += 1
                pair_count[(a, c_val)] += 1
                pair_count[(b, c_val)] += 1
                
            # Find valid triples to add
            # A triple is valid if its 3 pairs are not in pair_count
            # We can iterate all candidates?
            # Or just try to add one back?
            # Better: Try to add a random valid triple.
            
            # To be efficient, we can just check a few random candidates
            added = False
            for _ in range(10): # Try 10 random candidates
                a, b, c_val = rng.sample(range(n_points), 3)
                if pair_count[(a, b)] == 0 and pair_count[(a, c_val)] == 0 and pair_count[(b, c_val)] == 0:
                    new_triples.append([a, b, c_val])
                    added = True
                    break
            
            # If we didn't add anything, the graph is smaller.
            # This is a valid move (shrinking).
            
        # Evaluate
        new_alpha = _calc_alpha(new_triples, n_points)
        
        # Acceptance
        delta = current_alpha - new_alpha
        if delta > 0 or rng.random() < pow(2.718, delta / T):
            current_triples = new_triples
            current_alpha = new_alpha
            
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_triples = [list(t) for t in current_triples]
                
        # Cooling
        T *= 0.999
        
    return best_triples
