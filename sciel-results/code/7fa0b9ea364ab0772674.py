import random
import time
from itertools import combinations

def construct():
    # Points 0..15
    n = 16
    points = list(range(n))
    
    # Helper to get all pairs of a triple
    def get_pairs(triple):
        a, b, c = triple
        return [(a, b), (a, c), (b, c)]
    
    # Helper to check if a triple is valid (no pair overlap with existing covered pairs)
    def is_valid(triple, covered_pairs):
        for p in get_pairs(triple):
            if p in covered_pairs:
                return False
        return True
        
    # Helper to calculate alpha (independence number)
    def calc_alpha(triples):
        if not triples:
            return n
            
        # Precompute masks for triples
        triple_masks = []
        for t in triples:
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_masks.append(mask)
            
        # Check for independent sets of size k
        # We want the MAX k such that an independent set exists.
        # Since alpha is small, we check from n down to 1.
        # Actually, we can check from 1 up to n, but we want the max.
        # It's faster to check large k first if alpha is small?
        # No, if alpha is 5, we check 16, 15, ..., 6 (fail), 5 (success).
        # If alpha is 10, we check 16..11 (fail), 10 (success).
        # Average checks ~ 16 - alpha.
        
        for k in range(n, 0, -1):
            found = False
            for subset in combinations(points, k):
                subset_mask = 0
                for p in subset:
                    subset_mask |= (1 << p)
                
                # Check if any triple is contained in subset
                independent = True
                for tmask in triple_masks:
                    if (subset_mask & tmask) == tmask:
                        independent = False
                        break
                
                if independent:
                    found = True
                    break
            
            if found:
                return k
        return 0

    # Initialize with STS(15) on points 0..14
    # STS(15) construction:
    # Points are Z_15? No, 15 is not prime power friendly for simple STS.
    # We can use a known STS(15) or generate one.
    # Let's use a greedy approach to build a maximal packing starting from empty,
    # but bias towards covering pairs uniformly.
    
    # Better: Start with a random maximal packing.
    def random_maximal_packing():
        covered_pairs = set()
        triples = []
        all_pairs = list(combinations(points, 2))
        random.shuffle(all_pairs)
        
        # We want to add triples greedily.
        # To make it "linear", we just ensure no pair is reused.
        # This creates a packing.
        
        # To ensure it's maximal, we might need to retry or just accept a large packing.
        # Greedy random packing usually gets close to max size.
        
        # Iterate through possible triples?
        # There are C(16,3) = 560 triples.
        all_triples = list(combinations(points, 3))
        random.shuffle(all_triples)
        
        for t in all_triples:
            if is_valid(t, covered_pairs):
                triples.append(t)
                for p in get_pairs(t):
                    covered_pairs.add(p)
                    
        return triples

    # Start with a few random packings and pick the best alpha
    best_triples = None
    best_alpha = n + 1
    
    start_time = time.time()
    time_limit = 100 # Leave margin for evaluation overhead
    
    # Phase 1: Find a good initial solution
    for i in range(50):
        if time.time() - start_time > time_limit:
            break
        triples = random_maximal_packing()
        alpha = calc_alpha(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]
            
    if best_triples is None:
        best_triples = random_maximal_packing()
        best_alpha = calc_alpha(best_triples)
        
    # Phase 2: Simulated Annealing
    # Moves:
    # 1. Swap: Remove a random triple, try to add a random valid triple.
    # 2. Add: If not maximal, add a random valid triple.
    # 3. Remove: Remove a random triple (rarely useful for minimizing alpha, but allows escape).
    
    current_triples = best_triples[:]
    current_alpha = best_alpha
    
    T = 1.0
    cooling = 0.99
    
    while time.time() - start_time < time_limit:
        if T < 0.01:
            break
            
        # Generate neighbor
        neighbor_triples = current_triples[:]
        
        # Random move
        move_type = random.random()
        
        if move_type < 0.7:
            # Swap
            if len(neighbor_triples) > 0:
                # Remove a random triple
                idx = random.randint(0, len(neighbor_triples) - 1)
                removed_triple = neighbor_triples.pop(idx)
                
                # Rebuild covered pairs for the neighbor
                covered_pairs = set()
                for t in neighbor_triples:
                    for p in get_pairs(t):
                        covered_pairs.add(p)
                        
                # Try to add a new triple
                all_triples = list(combinations(points, 3))
                random.shuffle(all_triples)
                added = False
                for t in all_triples:
                    if is_valid(t, covered_pairs):
                        neighbor_triples.append(t)
                        for p in get_pairs(t):
                            covered_pairs.add(p)
                        added = True
                        break
                
                if not added:
                    # Revert if no triple could be added? Or keep it non-maximal?
                    # Keeping it non-maximal might be good.
                    # But let's try to add. If can't, the state is just smaller.
                    pass
                    
        elif move_type < 0.85:
            # Add
            covered_pairs = set()
            for t in neighbor_triples:
                for p in get_pairs(t):
                    covered_pairs.add(p)
            
            all_triples = list(combinations(points, 3))
            random.shuffle(all_triples)
            for t in all_triples:
                if is_valid(t, covered_pairs):
                    neighbor_triples.append(t)
                    for p in get_pairs(t):
                        covered_pairs.add(p)
                    break
        else:
            # Remove
            if len(neighbor_triples) > 0:
                idx = random.randint(0, len(neighbor_triples) - 1)
                neighbor_triples.pop(idx)
        
        new_alpha = calc_alpha(neighbor_triples)
        
        # Acceptance
        if new_alpha < current_alpha:
            current_triples = neighbor_triples
            current_alpha = new_alpha
        else:
            delta = new_alpha - current_alpha
            # delta is positive (worse)
            if delta == 0:
                current_triples = neighbor_triples
            else:
                # Probability of accepting worse solution
                # Use delta/ T
                prob = 1.0 / (1.0 + delta / T)
                if random.random() < prob:
                    current_triples = neighbor_triples
                    current_alpha = new_alpha
        
        T *= cooling
        
    return [list(t) for t in current_triples]

# Test locally
if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"Num triples: {len(res)}")
    # Check validity
    pairs = set()
    for t in res:
        for i in range(3):
            for j in range(i+1, 3):
                p = tuple(sorted((t[i], t[j])))
                if p in pairs:
                    print(f"Invalid: pair {p} repeated")
                pairs.add(p)
    print(f"Pairs covered: {len(pairs)}")
    
    # Check alpha
    # Re-use logic
    n = 16
    points = list(range(n))
    triple_masks = []
    for t in res:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
        
    for k in range(n, 0, -1):
        found = False
        for subset in combinations(points, k):
            subset_mask = 0
            for p in subset:
                subset_mask |= (1 << p)
            independent = True
            for tmask in triple_masks:
                if (subset_mask & tmask) == tmask:
                    independent = False
                    break
            if independent:
                found = True
                break
        if found:
            print(f"Alpha: {k}")
            break
