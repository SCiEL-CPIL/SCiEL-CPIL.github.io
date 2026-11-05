import time
import itertools
import numpy as np
import random
from scipy.optimize import milp, LinearConstraint, Bounds

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    m = len(all_triples)
    
    # Precompute pair sets for each triple
    triple_pairs = []
    for t in all_triples:
        ps = [tuple(sorted((t[0], t[1]))), tuple(sorted((t[0], t[2]))), tuple(sorted((t[1], t[2])))]
        triple_pairs.append(ps)
    
    # Precompute triple masks for fast independence checking
    triple_masks = []
    for t in all_triples:
        mask = 0
        for v in t:
            mask |= (1 << v)
        triple_masks.append(mask)
    
    # Build a fast independence checker: for each subset mask, precompute if it's independent
    # 2^16 = 65536 subsets, each check is O(num_triples) but we can optimize
    
    # Strategy: use greedy construction + local search
    # The key insight: we want to maximize the number of triples while minimizing alpha
    
    best_triples = []
    best_alpha = n
    
    # Multiple random restarts with greedy construction
    while time.time() < deadline - 30.0:
        candidates = list(range(m))
        random.shuffle(candidates)
        
        current_triples = []
        used_pairs = set()
        
        for idx in candidates:
            tri = all_triples[idx]
            ps = triple_pairs[idx]
            valid = True
            for p in ps:
                if p in used_pairs:
                    valid = False
                    break
            if valid:
                current_triples.append(list(tri))
                for p in ps:
                    used_pairs.add(p)
        
        alpha = _calc_alpha_fast(current_triples, n, triple_masks)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = [t[:] for t in current_triples]
            
            # If we found a very good solution, do local search
            if time.time() < deadline - 10.0:
                improved_triples, improved_alpha = _local_search_improved(
                    current_triples, alpha, n, deadline, all_triples, triple_pairs, triple_masks
                )
                if improved_alpha < best_alpha:
                    best_alpha = improved_alpha
                    best_triples = improved_triples
        
        # If we've found alpha <= 7, that's quite good, stop searching
        if best_alpha <= 7:
            break
    
    return best_triples

def _calc_alpha_fast(triples, n, triple_masks):
    """Calculate independence number using precomputed triple masks."""
    if not triples:
        return n
    
    # For the given set of triples, find the largest independent set
    # A set S is independent iff it doesn't contain any triple from the given set
    
    # Build a set of triple masks for fast lookup
    tm_set = set()
    for t in triples:
        mask = 0
        for v in t:
            mask |= (1 << v)
        tm_set.add(mask)
    
    # Iterate over all subsets, find the largest independent one
    max_size = 0
    for mask in range(1, 1 << n):
        size = bin(mask).count('1')
        if size <= max_size:
            continue
        
        # Check if mask is independent: no triple is a subset of mask
        is_independent = True
        for tm in tm_set:
            if (mask & tm) == tm:
                is_independent = False
                break
        
        if is_independent:
            max_size = size
    
    return max_size

def _local_search_improved(triples, alpha, n, deadline, all_triples, triple_pairs, triple_masks):
    """
    Improved local search: try to reduce alpha by swapping triples.
    Moves: 
    1. Remove one triple, add another valid triple
    2. Remove two triples, add two valid triples
    """
    current_triples = [t[:] for t in triples]
    current_alpha = alpha
    
    # Build used_pairs set
    used_pairs = set()
    for t in current_triples:
        for i in range(3):
            for j in range(i + 1, 3):
                used_pairs.add(tuple(sorted((t[i], t[j]))))
    
    # Precompute which triples are compatible with current used_pairs
    # This is expensive, so we'll do it on-the-fly but cache results
    
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline - 5.0 and no_improve_count < max_no_improve:
        improved = False
        
        # Try single swap: remove one triple, try to add a different one
        if current_triples:
            idx_remove = random.randint(0, len(current_triples) - 1)
            t_remove = current_triples[idx_remove]
            
            # Remove it
            current_triples.pop(idx_remove)
            ps_rem = []
            for i in range(3):
                for j in range(i + 1, 3):
                    p = tuple(sorted((t_remove[i], t_remove[j])))
                    ps_rem.append(p)
                    used_pairs.discard(p)
            
            # Try to add a random valid triple
            for _ in range(100):
                idx_add = random.randint(0, len(all_triples) - 1)
                t_add = all_triples[idx_add]
                ps_add = triple_pairs[idx_add]
                
                valid = True
                for p in ps_add:
                    if p in used_pairs:
                        valid = False
                        break
                
                if valid:
                    # Check if this is the same triple
                    if set(t_add) == set(t_remove):
                        continue
                    
                    # Add it
                    current_triples.append(list(t_add))
                    for p in ps_add:
                        used_pairs.add(p)
                    
                    new_alpha = _calc_alpha_fast(current_triples, n, triple_masks)
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                        no_improve_count = 0
                        improved = True
                        break
                    else:
                        # Revert
                        current_triples.pop()
                        for p in ps_add:
                            used_pairs.discard(p)
            
            if not improved:
                # Restore the removed triple
                current_triples.append(list(t_remove))
                for p in ps_rem:
                    used_pairs.add(p)
                no_improve_count += 1
        
        # Try double swap: remove two triples, add two
        if len(current_triples) >= 2 and not improved:
            idx1, idx2 = random.sample(range(len(current_triples)), 2)
            t1 = current_triples[idx1]
            t2 = current_triples[idx2]
            
            # Remove both
            ps_rem1 = []
            for i in range(3):
                for j in range(i + 1, 3):
                    p = tuple(sorted((t1[i], t1[j])))
                    ps_rem1.append(p)
                    used_pairs.discard(p)
            current_triples.pop(idx1)
            
            # Adjust idx2 if needed
            if idx2 > idx1:
                idx2 -= 1
            
            ps_rem2 = []
            for i in range(3):
                for j in range(i + 1, 3):
                    p = tuple(sorted((t2[i], t2[j])))
                    ps_rem2.append(p)
                    used_pairs.discard(p)
            current_triples.pop(idx2)
            
            # Try to add two valid triples
            for _ in range(50):
                idx_add1 = random.randint(0, len(all_triples) - 1)
                idx_add2 = random.randint(0, len(all_triples) - 1)
                if idx_add1 == idx_add2:
                    continue
                
                t_add1 = all_triples[idx_add1]
                t_add2 = all_triples[idx_add2]
                ps_add1 = triple_pairs[idx_add1]
                ps_add2 = triple_pairs[idx_add2]
                
                # Check validity of both
                valid1 = True
                for p in ps_add1:
                    if p in used_pairs:
                        valid1 = False
                        break
                if not valid1:
                    continue
                
                # Temporarily add first
                for p in ps_add1:
                    used_pairs.add(p)
                
                valid2 = True
                for p in ps_add2:
                    if p in used_pairs:
                        valid2 = False
                        break
                
                if valid2:
                    for p in ps_add2:
                        used_pairs.add(p)
                    
                    current_triples.append(list(t_add1))
                    current_triples.append(list(t_add2))
                    
                    new_alpha = _calc_alpha_fast(current_triples, n, triple_masks)
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                        no_improve_count = 0
                        improved = True
                        break
                    else:
                        # Revert
                        current_triples.pop()
                        current_triples.pop()
                        for p in ps_add2:
                            used_pairs.discard(p)
                        for p in ps_add1:
                            used_pairs.discard(p)
                else:
                    # Revert first addition
                    for p in ps_add1:
                        used_pairs.discard(p)
            
            if not improved:
                # Restore both removed triples
                current_triples.append(list(t1))
                current_triples.append(list(t2))
                for p in ps_rem1:
                    used_pairs.add(p)
                for p in ps_rem2:
                    used_pairs.add(p)
                no_improve_count += 1
    
    return current_triples, current_alpha
# EVOLVE-BLOCK-END
