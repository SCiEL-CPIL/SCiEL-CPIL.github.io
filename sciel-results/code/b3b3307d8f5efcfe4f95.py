"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _count_pairs(family):
    """Count unordered pairs where one is a strict subset of the other."""
    n = len(family)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            a, b = family[i], family[j]
            if a < b or b < a:
                count += 1
    return count

def _construct():
    n = 12
    m = 40
    
    # All subsets of {0,...,11}
    all_subsets = [frozenset(c) for r in range(n+1) for c in combinations(range(n), r)]
    all_subsets_set = set(all_subsets)
    
    best_score = -1
    best_family = None
    
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # Strategy: Build a chain-based solution, then refine with local search
    # A chain of length 13 gives C(13,2)=78 pairs. We need 40 sets.
    # If we take 40 sets from a single chain of 13, max is 78.
    # But we can do better by using multiple chains or a dense poset.
    
    # Initial greedy: pick sets that maximize nesting
    # Start with empty set and full set, then fill in
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break
        
        # Greedy construction
        family = [frozenset(), frozenset(range(n))]
        remaining = all_subsets_set - set(family)
        
        while len(family) < m:
            best_set = None
            best_gain = -1
            # Sample a subset of remaining to speed up
            sample = list(remaining)
            random.shuffle(sample)
            sample = sample[:min(200, len(sample))]
            
            for s in sample:
                gain = 0
                for t in family:
                    if s < t or t < s:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_set = s
            
            if best_set is not None:
                family.append(best_set)
                remaining.discard(best_set)
            else:
                # Pick random
                s = random.choice(list(remaining))
                family.append(s)
                remaining.discard(s)
        
        # Convert to lists for scoring
        family_list = [list(s) for s in family]
        score = _count_pairs(family_list)
        
        if score > best_score:
            best_score = score
            best_family = family_list[:]
        
        # Local search refinement
        for iteration in range(200):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            # Try swapping one set with another
            for i in range(m):
                if time.time() > deadline - 5.0:
                    break
                current_set = family[i]
                # Try replacing with a random other set
                for _ in range(5):
                    candidate = random.choice(all_subsets)
                    if candidate in family:
                        continue
                    
                    new_family = family[:]
                    new_family[i] = candidate
                    new_family_list = [list(s) for s in new_family]
                    new_score = _count_pairs(new_family_list)
                    
                    if new_score > score:
                        score = new_score
                        family = new_family
                        improved = True
                        break
            
            if not improved:
                # Try pairwise swaps
                for _ in range(20):
                    if time.time() > deadline - 5.0:
                        break
                    i, j = random.sample(range(m), 2)
                    cand_i = random.choice(all_subsets)
                    cand_j = random.choice(all_subsets)
                    if cand_i in family or cand_j in family:
                        continue
                    if cand_i == cand_j:
                        continue
                    
                    new_family = family[:]
                    new_family[i] = cand_i
                    new_family[j] = cand_j
                    new_family_list = [list(s) for s in new_family]
                    new_score = _count_pairs(new_family_list)
                    
                    if new_score > score:
                        score = new_score
                        family = new_family
                        improved = True
                        break
            
            if not improved:
                break
        
        if score > best_score:
            best_score = score
            best_family = [list(s) for s in family]
    
    # Final polish: try to improve best_family further
    if best_family is not None:
        family = [frozenset(s) for s in best_family]
        score = _count_pairs(best_family)
        
        for iteration in range(500):
            if time.time() > deadline - 2.0:
                break
            
            improved = False
            for i in range(m):
                if time.time() > deadline - 2.0:
                    break
                for _ in range(3):
                    candidate = random.choice(all_subsets)
                    if candidate in family:
                        continue
                    
                    new_family = family[:]
                    new_family[i] = candidate
                    new_family_list = [list(s) for s in new_family]
                    new_score = _count_pairs(new_family_list)
                    
                    if new_score > score:
                        score = new_score
                        family = new_family
                        improved = True
                        break
            
            if not improved:
                break
        
        best_family = [list(s) for s in family]
    
    return best_family

def construct():
    result = _construct()
    # Ensure all elements are Python ints and sequences are lists
    cleaned = []
    for s in result:
        cleaned.append([int(x) for x in s])
    return cleaned
# EVOLVE-BLOCK-END
