import numpy as np
import time
import random

def _independence_number_fast(triples):
    """
    Compute the independence number using vectorized numpy operations.
    Returns the size of the largest subset of {0..15} containing no triple.
    """
    if not triples:
        return 16
        
    n = 16
    # Precompute triple masks
    triple_masks = np.array([(1 << a) | (1 << b) | (1 << c) for a, b, c in triples], dtype=np.int64)
    
    # All non-zero masks
    all_masks = np.arange(1, 1 << n, dtype=np.int64)
    
    # Check if each mask contains any triple
    # Shape: (num_masks, num_triples)
    # We want to know if (mask & tm) == tm for ANY tm
    # Note: broadcasting might be memory heavy if num_triples is large, but 30 is fine.
    # (65535, T) boolean array.
    
    # To save memory/time, we can check in chunks or just do it if T is small.
    # Let's assume T < 100.
    
    # We can optimize by checking if the mask has enough bits first?
    # If popcount(mask) < 3, it can't contain a triple.
    # But we want the MAX popcount.
    
    # Vectorized check:
    # contains = (all_masks[:, None] & triple_masks[None, :]) == triple_masks[None, :]
    # is_ind = np.all(~contains, axis=1)
    
    # This creates a (65535, T) array. If T=50, that's 3M booleans. ~3MB. Fine.
    
    # Optimization: Only check masks with popcount >= current_best?
    # We don't know current_best until we finish.
    # However, we can just compute all and take max.
    
    contains = (all_masks[:, None] & triple_masks[None, :]) == triple_masks[None, :]
    is_independent = np.all(~contains, axis=1)
    
    if not np.any(is_independent):
        return 0 # Should not happen for valid linear systems unless T is huge? 
                 # Actually, if every subset of size k contains a triple, alpha < k.
                 # But empty set is always independent. Wait, mask 0 is not in all_masks.
                 # The problem asks for largest point set. Empty set size 0 is always valid.
                 # But usually alpha >= 1.
    
    sizes = np.bitwise_count(all_masks)
    valid_sizes = sizes[is_independent]
    
    if len(valid_sizes) == 0:
        return 0
        
    return int(np.max(valid_sizes))

def _is_valid_addition(triples, new_triple):
    """Check if new_triple can be added to maintain linearity."""
    a, b, c = new_triple
    new_set = {a, b, c}
    for t in triples:
        if len(new_set & set(t)) >= 2:
            return False
    return True

def _generate_greedy_packing(seed, n=16):
    """
    Generate a packing using a greedy approach.
    We prioritize triples that are 'central' or just random but deterministic.
    """
    random.seed(seed)
    triples = []
    points = list(range(n))
    
    # To make it more structured, we can try to add triples that maximize some heuristic,
    # but random greedy with a fixed seed is a valid "Greedy" family member if we fix the order.
    # Let's just do random greedy, which is a standard baseline.
    
    # We can try to fill up to a certain density.
    max_triples = 100
    attempts = 0
    while len(triples) < max_triples and attempts < 1000:
        attempts += 1
        t = tuple(sorted(random.sample(points, 3)))
        if _is_valid_addition(triples, t):
            triples.append(list(t))
            
    return triples

def construct():
    start_time = time.time()
    time_limit = 115
    
    # 1. Initialization: Try several greedy packings with different seeds
    best_triples = []
    best_score = 16
    
    # We will run a few greedy initializations
    for seed in range(10):
        triples = _generate_greedy_packing(seed)
        score = _independence_number_fast(triples)
        if score < best_score:
            best_score = score
            best_triples = triples[:]
            
    # 2. Local Search: First Improvement
    # We will iterate through possible moves and accept the first one that improves the score.
    # To avoid getting stuck, we can randomize the order of moves or restart.
    
    current_triples = best_triples[:]
    current_score = best_score
    
    n = 16
    points = list(range(n))
    
    # We will use a simple loop. 
    # Moves:
    # 1. Add a random valid triple.
    # 2. Remove a random triple.
    # 3. Swap a random triple with a new random valid triple.
    
    # To make it "First Improvement", we should check multiple candidates and pick the best?
    # Or just accept the first improvement found in a random scan?
    # "First Improvement" usually means: scan neighbors in some order, accept first one that is better.
    # Since the neighborhood is huge (all valid triples), we can sample neighbors.
    
    improvements = 0
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        if no_improve_count > 500:
            break
            
        improved = False
        
        # Try a set of random moves
        # We will try 100 random moves per iteration
        for _ in range(100):
            move_type = random.random()
            original_len = len(current_triples)
            
            if move_type < 0.4:
                # Add
                # Find a random valid triple to add
                candidate = None
                for _ in range(20): # Try a few times to find one
                    t = tuple(sorted(random.sample(points, 3)))
                    if _is_valid_addition(current_triples, t):
                        candidate = list(t)
                        break
                if candidate is not None:
                    current_triples.append(candidate)
                    new_score = _independence_number_fast(current_triples)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        break # First improvement: stop scanning moves for this round
                    else:
                        current_triples.pop()
                        
            elif move_type < 0.7 and current_triples:
                # Remove
                idx = random.randint(0, len(current_triples) - 1)
                removed = current_triples.pop(idx)
                new_score = _independence_number_fast(current_triples)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    break
                else:
                    current_triples.insert(idx, removed)
                    
            else:
                # Swap
                if current_triples:
                    idx = random.randint(0, len(current_triples) - 1)
                    removed = current_triples.pop(idx)
                    
                    candidate = None
                    for _ in range(20):
                        t = tuple(sorted(random.sample(points, 3)))
                        if _is_valid_addition(current_triples, t):
                            candidate = list(t)
                            break
                            
                    if candidate is not None:
                        current_triples.append(candidate)
                        new_score = _independence_number_fast(current_triples)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            break
                        else:
                            current_triples.pop()
                    current_triples.insert(idx, removed)
        
        if improved:
            no_improve_count = 0
            if current_score < best_score:
                best_score = current_score
                best_triples = current_triples[:]
        else:
            no_improve_count += 1
            
    return best_triples
