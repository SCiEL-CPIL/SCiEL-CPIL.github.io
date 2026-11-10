# EVOLVE-BLOCK-START
import itertools
import random

def construct():
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    n_triples = len(all_triples)
    
    # Precompute compatibility
    # compat[i] is a bitmask of triples compatible with i
    # Using bitmasks for fast intersection checks
    compat_masks = [0] * n_triples
    for i in range(n_triples):
        ti = all_triples[i]
        si = set(ti)
        mask = 0
        for j in range(n_triples):
            if i == j:
                continue
            tj = all_triples[j]
            sj = set(tj)
            if len(si & sj) <= 1:
                mask |= (1 << j)
        compat_masks[i] = mask
        
    def compute_independence_number(triples):
        if not triples:
            return 16
            
        # Convert to set of masks for fast subset check
        triple_masks = []
        for t in triples:
            m = 0
            for p in t:
                m |= (1 << p)
            triple_masks.append(m)
            
        # Remove duplicates
        triple_masks = list(set(triple_masks))
        
        def is_independent(mask):
            for tm in triple_masks:
                if (tm & mask) == tm:
                    return False
            return True

        # Check from 16 down
        # Optimization: if we find an independent set of size k, we can stop.
        # We can use a recursive search to find the max size, but for N=16,
        # iterating combinations is simpler and likely fast enough.
        
        # To speed up, we can check larger sizes first.
        # We can also prune: if a subset S is independent, any subset of S is independent.
        # But we want the MAX size.
        
        # Heuristic check: 
        # If the number of triples is very high, the independence number is small.
        # Let's just iterate.
        
        for k in range(16, 0, -1):
            for subset in itertools.combinations(points, k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                if is_independent(mask):
                    return k
        return 0

    def greedy_construct(seed):
        rng = random.Random(seed)
        indices = list(range(n_triples))
        rng.shuffle(indices)
        
        selected_indices = []
        
        # Maintain a bitmask of available triples
        # Initially all are available
        available_mask = (1 << n_triples) - 1
        
        for idx in indices:
            if not (available_mask & (1 << idx)):
                continue
                
            # Check if idx is compatible with all selected
            # We can maintain an intersection of compat_masks of selected triples
            # Let's maintain `intersection_mask`
            # Initially, intersection_mask = all 1s
            # When we select a triple i, intersection_mask &= compat_masks[i]
            # Then a new triple j is valid if j is in intersection_mask AND j is in available_mask
            
            # Wait, the logic above:
            # If we select i, any future triple must be compatible with i.
            # So yes, we can maintain a running intersection.
            
            # But we need to update it when we select.
            # Let's restructure.
            
            pass # Placeholder, will implement below

    # Implementing the greedy with intersection mask
    def greedy_construct_v2(seed):
        rng = random.Random(seed)
        indices = list(range(n_triples))
        rng.shuffle(indices)
        
        selected_indices = []
        
        # intersection_mask starts as all 1s (all triples are compatible with empty set)
        intersection_mask = (1 << n_triples) - 1
        
        for idx in indices:
            # Check if idx is in intersection_mask
            if not (intersection_mask & (1 << idx)):
                continue
                
            # Select idx
            selected_indices.append(idx)
            
            # Update intersection_mask
            intersection_mask &= compat_masks[idx]
            
            # Optional: Prune if intersection_mask becomes 0
            if intersection_mask == 0:
                break
                
        return [all_triples[i] for i in selected_indices]

    best_triples = None
    best_score = 17
    
    # Run multiple seeds
    # Use 1000 seeds
    for seed in range(1000):
        triples = greedy_construct_v2(seed)
        if not triples:
            continue
        
        score = compute_independence_number(triples)
        if score < best_score:
            best_score = score
            best_triples = triples
            if best_score <= 5:
                break

    if best_triples is None:
        return []
        
    return best_triples
# EVOLVE-BLOCK-END
