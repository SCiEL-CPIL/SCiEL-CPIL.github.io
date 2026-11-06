"""Linear triple systems on 16 points - minimize independence number."""
import time
import random
import itertools
import numpy as np

def construct():
    deadline = time.time() + 145
    n = 16
    rng = random.Random(12345)
    
    # Pre-compute all triples
    all_triples = list(itertools.combinations(range(n), 3))
    
    # For each point, pre-compute which triples contain it
    point_to_triples = {p: [] for p in range(n)}
    for idx, t in enumerate(all_triples):
        for p in t:
            point_to_triples[p].append(idx)
    
    # Pre-compute pair conflicts: for each pair of points, which triples contain both
    # A triple is valid if no pair in it has been used before
    # We'll track used pairs as a set of frozensets or encoded integers
    
    def encode_pair(a, b):
        if a > b:
            a, b = b, a
        return a * n + b
    
    def independence_number(triples):
        """Compute independence number: largest set with no triple fully contained."""
        # A set S is independent if no triple is a subset of S
        # We want the largest such S
        # Try all subsets of size k from n down, check if any triple is contained
        # But that's 2^16 = 65536 subsets, which is fine
        # For each subset, check if any triple is contained in it
        
        # Convert triples to frozensets for fast lookup
        triple_set = set()
        for t in triples:
            triple_set.add(frozenset(t))
        
        # For each subset of {0..15}, check if it contains any triple
        # A subset S contains a triple if any (a,b,c) in triple_set is subset of S
        # We can iterate over all subsets from largest to smallest
        
        # Use bitmask representation
        # For each subset mask, check if any triple is contained
        # Pre-compute triple masks
        triple_masks = []
        for t in triples:
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            triple_masks.append(mask)
        
        # For a subset with mask S, it contains a triple if any triple_mask is subset of S
        # i.e., triple_mask & S == triple_mask
        
        # Iterate subsets by size descending
        best = 0
        # Try sizes from n down to 1
        for size in range(n, 0, -1):
            found = False
            for subset in itertools.combinations(range(n), size):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                # Check if any triple is contained
                contains_triple = False
                for tm in triple_masks:
                    if tm & mask == tm:
                        contains_triple = True
                        break
                if not contains_triple:
                    best = size
                    found = True
                    break
            if found:
                break
        return best
    
    def independence_number_fast(triples):
        """Faster independence number using bitmask DP."""
        if not triples:
            return n
        
        # Build a boolean array: for each subset mask (0..2^16-1), is it independent?
        # A subset is independent if it does NOT contain any triple
        # We can compute this via: independent[mask] = True if no triple is subset of mask
        
        # Pre-compute which masks contain at least one triple
        # For each triple mask tm, all supersets of tm are NOT independent
        # But 2^16 = 65536, and for each triple we'd mark all supersets... that's expensive
        # Alternative: for each mask, check if any triple is subset
        
        # Better approach: compute for each mask whether it's "triple-free"
        # Use the fact that if mask is triple-free, all its subsets are triple-free
        # So we can do a DP: independent[0] = True
        # independent[mask] = independent[mask without lowest bit] AND (no triple has exactly the removed bit plus others in mask)
        # This is tricky. Let's just iterate all masks and check.
        
        num_masks = 1 << n
        is_independent = [True] * num_masks
        
        # For each triple, mark all its supersets as not independent
        for t in triples:
            tm = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            # All supersets of tm: tm | extra, where extra is any subset of complement
            # Complement of tm has n-3 bits
            comp = ((1 << n) - 1) ^ tm
            # Iterate over all subsets of comp
            extra = 0
            while True:
                mask = tm | extra
                is_independent[mask] = False
                if extra == comp:
                    break
                # Next subset of comp
                extra = (extra + 1) & comp
                if extra == 0:
                    # Check if we've exhausted all subsets
                    # Actually, iterating (extra+1)&comp doesn't enumerate all subsets properly
                    break
        
        # The above subset enumeration is buggy. Let's use a different approach.
        # For each mask, check directly if any triple is contained.
        # 65536 masks * num_triples checks... if num_triples is ~20, that's 1.3M checks, fine.
        
        triple_masks = []
        for t in triples:
            tm = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            triple_masks.append(tm)
        
        for mask in range(num_masks):
            if not is_independent[mask]:
                continue
            for tm in triple_masks:
                if tm & mask == tm:
                    is_independent[mask] = False
                    break
        
        # Find largest independent set
        best = 0
        for mask in range(num_masks):
            if is_independent[mask]:
                sz = bin(mask).count('1')
                if sz > best:
                    best = sz
        return best
    
    def is_valid_system(triples):
        """Check that any two triples share at most one point."""
        used_pairs = set()
        for t in triples:
            a, b, c = t
            pairs = [(a,b), (a,c), (b,c)]
            for p in pairs:
                key = encode_pair(p[0], p[1])
                if key in used_pairs:
                    return False
                used_pairs.add(key)
        return True
    
    def greedy_build(triple_order):
        """Build a linear triple system greedily following the given order."""
        used_pairs = set()
        chosen = []
        for t in triple_order:
            a, b, c = t
            pairs = [(a,b), (a,c), (b,c)]
            valid = True
            for p in pairs:
                key = encode_pair(p[0], p[1])
                if key in used_pairs:
                    valid = False
                    break
            if valid:
                chosen.append(list(t))
                for p in pairs:
                    used_pairs.add(encode_pair(p[0], p[1]))
        return chosen
    
    def local_search(triples, time_limit):
        """Improve a triple system via local search."""
        if not triples:
            return triples, n
        
        current_triples = [list(t) for t in triples]
        current_indep = independence_number_fast(current_triples)
        best_triples = [list(t) for t in current_triples]
        best_indep = current_indep
        start_time = time.time()
        
        # Build used pairs
        def get_used_pairs(triples):
            up = set()
            for t in triples:
                a, b, c = t
                up.add(encode_pair(a, b))
                up.add(encode_pair(a, c))
                up.add(encode_pair(b, c))
            return up
        
        def can_add(triple, used_pairs):
            a, b, c = triple
            for p in [(a,b), (a,c), (b,c)]:
                if encode_pair(p[0], p[1]) in used_pairs:
                    return False
            return True
        
        # Try to add triples or replace triples
        all_triples_shuffled = list(all_triples)
        
        while time.time() < start_time + time_limit:
            improved = False
            
            # Try adding a random triple
            for _ in range(100):
                idx = rng.randint(0, len(all_triples) - 1)
                t = all_triples[idx]
                if t in current_triples:
                    continue
                up = get_used_pairs(current_triples)
                if can_add(t, up):
                    new_triples = current_triples + [list(t)]
                    new_indep = independence_number_fast(new_triples)
                    if new_indep < current_indep:
                        current_triples = new_triples
                        current_indep = new_indep
                        if current_indep < best_indep:
                            best_indep = current_indep
                            best_triples = [list(t) for t in current_triples]
                        improved = True
                        break
                    elif new_indep == current_indep:
                        # Accept with some probability for exploration
                        if rng.random() < 0.3:
                            current_triples = new_triples
                            improved = True
                            break
            
            # Try replacing a triple
            if not improved and current_triples:
                for _ in range(50):
                    remove_idx = rng.randint(0, len(current_triples) - 1)
                    up = get_used_pairs(current_triples)
                    # Remove the triple
                    removed = current_triples.pop(remove_idx)
                    # Rebuild used pairs without removed
                    up = get_used_pairs(current_triples)
                    
                    # Try to add a different triple
                    for _ in range(20):
                        idx = rng.randint(0, len(all_triples) - 1)
                        t = all_triples[idx]
                        if t in current_triples:
                            continue
                        if can_add(t, up):
                            new_triples = current_triples + [list(t)]
                            new_indep = independence_number_fast(new_triples)
                            if new_indep <= current_indep:
                                current_triples = new_triples
                                current_indep = new_indep
                                if current_indep < best_indep:
                                    best_indep = current_indep
                                    best_triples = [list(t) for t in current_triples]
                                improved = True
                                break
                            else:
                                # Revert
                                current_triples.append(removed)
                    if not improved:
                        current_triples.append(removed)
            
            if not improved:
                # Try random restart with perturbation
                # Remove a few random triples and rebuild
                if current_triples:
                    for _ in range(min(3, len(current_triples))):
                        idx = rng.randint(0, len(current_triples) - 1)
                        current_triples.pop(idx)
                    # Greedily add back
                    up = get_used_pairs(current_triples)
                    for t in all_triples:
                        if t in current_triples:
                            continue
                        if can_add(t, up):
                            current_triples.append(list(t))
                            a, b, c = t
                            up.add(encode_pair(a, b))
                            up.add(encode_pair(a, c))
                            up.add(encode_pair(b, c))
                    current_indep = independence_number_fast(current_triples)
                    if current_indep < best_indep:
                        best_indep = current_indep
                        best_triples = [list(t) for t in current_triples]
        
        return best_triples, best_indep
    
    # Strategy: 
    # 1. Generate many random linear triple systems
    # 2. Improve them with local search
    # 3. Keep the best
    
    best_system = None
    best_score = n  # independence number, lower is better
    
    # Start with a random greedy construction
    for restart in range(200):
        if time.time() > deadline - 5:
            break
        
        # Create a random ordering of triples
        triple_order = list(all_triples)
        rng.shuffle(triple_order)
        
        # Greedy build
        system = greedy_build(triple_order)
        
        if not system:
            continue
        
        # Evaluate
        indep = independence_number_fast(system)
        
        if indep < best_score:
            best_score = indep
            best_system = [list(t) for t in system]
        
        # Local search to improve
        if time.time() < deadline - 10:
            improved_system, improved_indep = local_search(system, 3.0)
            if improved_indep < best_score:
                best_score = improved_indep
                best_system = [list(t) for t in improved_system]
        
        # Early termination if we found a good solution
        if best_score <= 5:
            break
    
    if best_system is None:
        # Fallback: return a simple valid system
        return [[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14]]
    
    return best_system
