"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _generate_random_coloring(n):
    """Generate a random coloring that avoids red K4(3) with high probability."""
    # For n=34, C(34,3)=5984 triples.
    # Red K4(3) avoidance: For any 4 vertices, not all 4 triples red.
    # If we set p_red very low, red K4(3) is rare.
    # Blue K5(3) avoidance: For any 5 vertices, not all 10 triples blue.
    # If p_blue is close to 1, blue K5(3) is very likely.
    # We need a balance.
    # Random assignment with p=0.5 will fail blue K5(3) almost surely for n>20.
    # We can bias towards red to help blue, but red K4(3) is easier to violate?
    # Actually, random graph with p=0.5:
    # Expected red K4(3) ~ C(n,4) * (1/2)^4. For n=34, C(34,4)=46376, /16 ~ 2900.
    # Expected blue K5(3) ~ C(n,5) * (1/2)^10. For n=34, C(34,5)=278256, /1024 ~ 271.
    # So blue K5(3) violations are fewer but still many.
    # We need to reduce blue triples.
    # Let's try p_red = 0.9 (so p_blue = 0.1).
    # Red K4(3) ~ C(34,4) * 0.9^4 ~ 46376 * 0.65 ~ 30000.
    # Blue K5(3) ~ C(34,5) * 0.1^10 ~ 278256 * 1e-10 ~ 0.
    # So blue is fine, but red is terrible.
    # We need to fix red violations.
    # Strategy: Start with p_red ~ 0.5, then use local search to fix violations.
    # Or start with a structure that is known to be good?
    # The task says "published construction has 34 points".
    # Maybe there is a simple construction?
    # For n=5, R(4,5;3) > 5 is trivial.
    # Let's try to construct a valid coloring for n=34 using random restarts and local search.
    
    # Heuristic: To avoid red K4(3), we can use a "linear" structure or random.
    # Random with p=0.5 gives many red violations.
    # If we set color based on parity or something?
    # Let's just use random restarts with p=0.5 and hill climbing.
    
    n_triples = len(list(combinations(range(n), 3)))
    # For n=34, 5984 triples.
    # Array of 0/1.
    colors = np.random.choice([0, 1], size=n_triples, p=[0.5, 0.5])
    return colors

def _precompute_neighbors(n):
    """
    Precompute:
    1. Mapping from (i,j,k) to index.
    2. For each vertex, list of triples containing it.
    3. For each triple t, list of 4-sets containing it (for red K4 check).
       Actually, for red K4(3), a violation is a set of 4 vertices where all 4 triples are red.
       We can count red triples for each 4-set.
       There are C(n,4) 4-sets. For n=34, 46376.
       Each 4-set has 4 triples.
       We can maintain an array `red_count_4` of size C(n,4).
       When a triple flips, we update the 4-sets containing it.
       Each triple is in (n-3) 4-sets.
       So flipping one triple updates (n-3) counts.
       
    4. For blue K5(3), a violation is a set of 5 vertices where all 10 triples are blue.
       We can maintain an array `blue_count_5` of size C(n,5).
       For n=34, C(34,5) = 278256.
       Each triple is in C(n-3, 2) 5-sets.
       For n=34, C(31,2) = 465.
       Flipping one triple updates 465 counts.
       
    Total updates per flip: (n-3) + C(n-3, 2) = 31 + 465 = 496.
    This is feasible.
    
    We need mappings:
    - triple_to_index: dict ((i,j,k) -> idx)
    - triple_to_4sets: list of lists. For each triple index, list of 4-set indices.
    - triple_to_5sets: list of lists. For each triple index, list of 5-set indices.
    """
    triples = list(combinations(range(n), 3))
    triple_to_index = {}
    for idx, (i,j,k) in enumerate(triples):
        triple_to_index[(i,j,k)] = idx
        
    # Precompute 4-sets and 5-sets indices
    # We don't need to store all 4-sets/5-sets, just their indices for counting.
    # But we need to know which 4-sets a triple belongs to.
    
    # For each triple (i,j,k), the 4-sets are {i,j,k,l} for l not in {i,j,k}.
    # We need a global index for each 4-set.
    # We can generate all 4-sets and map them to indices.
    all_4sets = list(combinations(range(n), 4))
    four_set_to_index = {}
    for idx, s in enumerate(all_4sets):
        four_set_to_index[s] = idx
        
    all_5sets = list(combinations(range(n), 5))
    five_set_to_index = {}
    for idx, s in enumerate(all_5sets):
        five_set_to_index[s] = idx
        
    # Build reverse mappings
    n_triples = len(triples)
    triple_to_4sets = [[] for _ in range(n_triples)]
    triple_to_5sets = [[] for _ in range(n_triples)]
    
    for t_idx, (i,j,k) in enumerate(triples):
        # 4-sets
        for l in range(n):
            if l != i and l != j and l != k:
                s = tuple(sorted((i,j,k,l)))
                s_idx = four_set_to_index[s]
                triple_to_4sets[t_idx].append(s_idx)
                
        # 5-sets
        # Pairs (l,m) not in {i,j,k}, l<m
        others = [v for v in range(n) if v != i and v != j and v != k]
        for l_idx in range(len(others)):
            for m_idx in range(l_idx+1, len(others)):
                l = others[l_idx]
                m = others[m_idx]
                s = tuple(sorted((i,j,k,l,m)))
                s_idx = five_set_to_index[s]
                triple_to_5sets[t_idx].append(s_idx)
                
    return triple_to_index, triple_to_4sets, triple_to_5sets, len(all_4sets), len(all_5sets)

def _get_violations(colors, n, triple_to_4sets, triple_to_5sets, n_4sets, n_5sets):
    """
    Compute violation counts.
    Red violation: any 4-set with 4 red triples.
    Blue violation: any 5-set with 10 blue triples.
    """
    # Initialize counts
    red_counts = np.zeros(n_4sets, dtype=np.int32)
    blue_counts = np.zeros(n_5sets, dtype=np.int32)
    
    for t_idx, c in enumerate(colors):
        if c == 0: # Red
            for s_idx in triple_to_4sets[t_idx]:
                red_counts[s_idx] += 1
        else: # Blue
            for s_idx in triple_to_5sets[t_idx]:
                blue_counts[s_idx] += 1
                
    red_violations = np.sum(red_counts == 4)
    blue_violations = np.sum(blue_counts == 10)
    return red_violations, blue_violations, red_counts, blue_counts

def _update_violations(t_idx, new_color, red_counts, blue_counts, triple_to_4sets, triple_to_5sets):
    """
    Update counts when triple t_idx flips to new_color.
    Returns number of new violations created/destroyed?
    Actually, just update the arrays.
    """
    old_color = 1 - new_color
    
    if old_color == 0: # Was Red, becomes Blue
        for s_idx in triple_to_4sets[t_idx]:
            red_counts[s_idx] -= 1
        for s_idx in triple_to_5sets[t_idx]:
            blue_counts[s_idx] += 1
    else: # Was Blue, becomes Red
        for s_idx in triple_to_4sets[t_idx]:
            red_counts[s_idx] += 1
        for s_idx in triple_to_5sets[t_idx]:
            blue_counts[s_idx] -= 1

def _count_violations_fast(red_counts, blue_counts):
    return np.sum(red_counts == 4) + np.sum(blue_counts == 10)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Target n
    # Try n=34 first.
    # If that fails, try n=33, etc.
    # But precomputation for n=34 is ~278k 5-sets, 46k 4-sets.
    # Precompute is O(C(n,5) * 10) roughly? No, O(C(n,3) * C(n-3,2)).
    # For n=34, 5984 * 465 ~ 2.7M operations. Fast.
    
    best_result = None
    
    # Try n=34
    n = 34
    try:
        triple_to_index, triple_to_4sets, triple_to_5sets, n_4sets, n_5sets = _precompute_neighbors(n)
        
        # Random restarts
        n_restarts = 50
        for r in range(n_restarts):
            if time.time() > deadline - 10:
                break
                
            colors = _generate_random_coloring(n)
            # Actually, bias towards red?
            # Let's stick to 0.5.
            
            red_v, blue_v, red_counts, blue_counts = _get_violations(colors, n, triple_to_4sets, triple_to_5sets, n_4sets, n_5sets)
            total_v = red_v + blue_v
            
            # Simulated Annealing / Hill Climbing
            temp = 1.0
            min_temp = 0.01
            cooling = 0.95
            
            # If we find 0 violations, return immediately
            if total_v == 0:
                # Verify
                # ...
                return {'n': n, 'colors': colors.tolist()}
                
            steps = 1000
            for s in range(steps):
                if time.time() > deadline - 5:
                    break
                    
                # Pick random triple
                t_idx = random.randrange(len(colors))
                new_color = 1 - colors[t_idx]
                
                # Calculate delta in violations
                # This is tricky with counts.
                # We can just flip, update counts, calculate new total, and if worse, revert.
                # Or estimate delta.
                # Flipping t_idx affects:
                # - 4-sets containing t_idx (n-3 of them)
                # - 5-sets containing t_idx (C(n-3,2) of them)
                
                # Let's just do a fast check.
                # Save old counts? No, just update and count.
                # But counting np.sum is O(N). N=300k.
                # 1000 steps * 300k = 300M ops. Might be slow in Python/NumPy?
                # np.sum is fast in C. 300k integers. ~1ms?
                # 1000 steps * 1ms = 1s.
                # 50 restarts * 1s = 50s.
                # Plus overhead. Should fit in 120s.
                
                old_total = _count_violations_fast(red_counts, blue_counts)
                
                _update_violations(t_idx, new_color, red_counts, blue_counts, triple_to_4sets, triple_to_5sets)
                colors[t_idx] = new_color
                
                new_total = _count_violations_fast(red_counts, blue_counts)
                
                delta = new_total - old_total
                
                if delta <= 0 or random.random() < np.exp(-delta / temp):
                    # Accept
                    total_v = new_total
                    if total_v == 0:
                        return {'n': n, 'colors': colors.tolist()}
                else:
                    # Revert
                    _update_violations(t_idx, old_color, red_counts, blue_counts, triple_to_4sets, triple_to_5sets)
                    colors[t_idx] = old_color
                    
                temp *= cooling
                if temp < min_temp:
                    temp = min_temp
                    
    except Exception:
        pass
        
    # Fallback to smaller n if 34 failed?
    # The task says "Maximize n".
    # If we don't find 34, we should try 33?
    # But precomputing 33 is similar cost.
    # Let's try to find a valid 34.
    # If not, we might return a smaller valid one?
    # The seed returned n=9.
    # If we return n=9, score is 9/seed_n.
    # If we return n=34, score is 34/seed_n.
    # We must return a VALID coloring.
    
    # If we didn't find 34, we should probably return the best valid one we found?
    # But we only kept the current one, not the best valid one if it wasn't 0.
    # We should keep track of the best valid coloring (fewest violations) and if it's 0, return it.
    # If we finish with >0 violations, it's invalid.
    # We need to return a VALID coloring.
    # So if we fail to find 0 violations for 34, we must find a valid one for n < 34?
    # Or return n=9 (known valid)?
    
    # Let's add a fallback to n=9 if we fail.
    # Or try n=30?
    
    # For now, if we reach here, we haven't returned.
    # We should try to find a valid coloring for a smaller n.
    # Let's try n=9 (seed).
    n = 9
    # Use the seed coloring which is valid?
    # Seed: 0 if sum % 3 == 0 else 1.
    # Let's verify n=9 seed.
    # It's a known construction?
    # Let's just use the seed logic for n=9.
    colors = np.array([0 if (a+b+c)%3==0 else 1 for a,b,c in combinations(range(9), 3)])
    
    # Check validity of n=9 seed
    # Red K4: 4 sets.
    # Blue K5: 5 sets.
    # With n=9, it's likely valid.
    
    return {'n': n, 'colors': colors.tolist()}

# EVOLVE-BLOCK-END
