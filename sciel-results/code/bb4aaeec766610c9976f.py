import random
import itertools
import time

def calculate_independence_number(triples):
    """
    Calculate the size of the largest subset of {0..15} that contains no triple from 'triples'.
    Also returns one such subset.
    """
    triple_masks = []
    for t in triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
    
    # Check subsets from largest size down
    # Points are 0..15
    for k in range(16, 0, -1):
        for subset in itertools.combinations(range(16), k):
            sub_mask = 0
            for p in subset:
                sub_mask |= (1 << p)
            
            is_independent = True
            for tm in triple_masks:
                if (tm & sub_mask) == tm:
                    is_independent = False
                    break
            
            if is_independent:
                return k, subset
    return 0, ()

def greedy_construct_optimized(seed):
    random.seed(seed)
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    
    selected_triples = []
    covered_pairs = set()
    
    # We will iteratively add triples.
    # To make it deterministic based on seed but varied, we can shuffle candidates at each step.
    
    while True:
        # Get current largest independent set
        alpha, s_max = calculate_independence_number(selected_triples)
        
        # If alpha is small, we might be done or close to optimal.
        # Let's say we stop if alpha <= 5? 
        # But we should try to add more triples if possible to reduce it further.
        
        # Find valid triples
        valid_triples = []
        for t in all_triples:
            # Check linearity
            pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
            valid = True
            for p in pairs:
                if p in covered_pairs:
                    valid = False
                    break
            if valid:
                valid_triples.append(t)
        
        if not valid_triples:
            break
        
        # Heuristic: Prefer triples that are subsets of s_max
        s_max_set = set(s_max)
        preferred = [t for t in valid_triples if set(t).issubset(s_max_set)]
        
        if preferred:
            # Pick one from preferred
            t = random.choice(preferred)
        else:
            # Pick one from all valid
            t = random.choice(valid_triples)
        
        selected_triples.append(list(t))
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        for p in pairs:
            covered_pairs.add(p)
            
        # Optional: Stop if alpha is very low? 
        # The task says "Minimize". 
        # Let's run until no more triples can be added.
        
    return selected_triples

def construct():
    best_triples = []
    best_score = 17
    
    # Run multiple attempts with different seeds
    # Since the heuristic is random in choice, different seeds yield different systems.
    for seed in range(50):
        triples = greedy_construct_optimized(seed)
        score, _ = calculate_independence_number(triples)
        if score < best_score:
            best_score = score
            best_triples = triples
            
    return best_triples
