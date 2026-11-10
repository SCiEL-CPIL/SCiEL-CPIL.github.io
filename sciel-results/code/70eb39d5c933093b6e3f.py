import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute 7-sets
    all_7_sets = list(combinations(points, 7))
    set_to_idx = {s: i for i, s in enumerate(all_7_sets)}
    
    # Precompute which 7-sets each triple covers
    # triple_idx -> list of 7-set indices
    triple_covers_7 = [[] for _ in all_triples]
    
    for t_idx, t in enumerate(all_triples):
        t_set = set(t)
        for s_idx, s in enumerate(all_7_sets):
            if t_set.issubset(set(s)):
                triple_covers_7[t_idx].append(s_idx)
                
    # State
    current_triples = set()
    used_pairs = set()
    counts_7 = [0] * len(all_7_sets)
    uncovered_7 = len(all_7_sets)
    
    def add_triple(t_idx):
        nonlocal uncovered_7
        t = all_triples[t_idx]
        pairs = [(t[0], t[1]), (t[1], t[0]), (t[0], t[2]), (t[2], t[0]), (t[1], t[2]), (t[2], t[1])]
        for p in pairs:
            if p in used_pairs:
                return False
        
        current_triples.add(t_idx)
        for p in pairs:
            used_pairs.add(p)
        
        for s_idx in triple_covers_7[t_idx]:
            if counts_7[s_idx] == 0:
                uncovered_7 -= 1
            counts_7[s_idx] += 1
        return True
        
    def remove_triple(t_idx):
        nonlocal uncovered_7
        t = all_triples[t_idx]
        pairs = [(t[0], t[1]), (t[1], t[0]), (t[0], t[2]), (t[2], t[0]), (t[1], t[2]), (t[2], t[1])]
        
        for s_idx in triple_covers_7[t_idx]:
            counts_7[s_idx] -= 1
            if counts_7[s_idx] == 0:
                uncovered_7 += 1
                
        current_triples.remove(t_idx)
        for p in pairs:
            used_pairs.remove(p)
            
    # Initialization: Random greedy
    triple_indices = list(range(len(all_triples)))
    random.shuffle(triple_indices)
    
    for t_idx in triple_indices:
        if time.time() > deadline - 20: break
        if add_triple(t_idx):
            pass
            
    best_uncovered = uncovered_7
    best_triples = set(current_triples)
    
    # Local Search
    # We want to minimize uncovered_7.
    # If uncovered_7 == 0, we have alpha <= 6.
    
    while time.time() < deadline - 5:
        improved = False
        
        # Pick a triple to remove
        if not current_triples:
            break
            
        t_remove = random.choice(list(current_triples))
        remove_triple(t_remove)
        
        # Find best triple to add
        best_gain = 0
        best_add_candidate = None
        
        # To speed up, we don't check all 560 triples.
        # We check a random subset.
        candidates = random.sample(range(len(all_triples)), min(50, len(all_triples)))
        
        for t_add in candidates:
            if t_add in current_triples:
                continue
            if t_add == t_remove:
                continue
            
            t = all_triples[t_add]
            pairs = [(t[0], t[1]), (t[1], t[0]), (t[0], t[2]), (t[2], t[0]), (t[1], t[2]), (t[2], t[1])]
            valid = True
            for p in pairs:
                if p in used_pairs:
                    valid = False
                    break
            if not valid:
                continue
                
            # Gain is reduction in uncovered_7
            gain = 0
            for s_idx in triple_covers_7[t_add]:
                if counts_7[s_idx] == 0:
                    gain += 1
                    
            if gain > best_gain:
                best_gain = gain
                best_add_candidate = t_add
                
        if best_add_candidate is not None and best_gain > 0:
            add_triple(best_add_candidate)
            improved = True
            if uncovered_7 < best_uncovered:
                best_uncovered = uncovered_7
                best_triples = set(current_triples)
                if uncovered_7 == 0:
                    break
        else:
            add_triple(t_remove)
            
        # Occasionally restart or perturb?
        # If stuck, we can try removing a random triple and adding a random one even if gain is 0?
        # No, that increases uncovered.
        
    final_triples_idx = best_triples
    
    return [list(all_triples[i]) for i in final_triples_idx]
