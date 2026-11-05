import itertools
import random
import time

def construct():
    n = 16
    # Precompute all possible triples as masks and lists
    all_triples_list = list(itertools.combinations(range(n), 3))
    triple_masks = {}
    for t in all_triples_list:
        mask = 0
        for v in t:
            mask |= (1 << v)
        triple_masks[t] = mask

    # Helper to get mask of a triple
    def get_mask(t):
        return triple_masks[t]

    def calc_alpha(triples):
        """
        Calculates the independence number of the hypergraph defined by triples.
        Returns the size of the largest subset containing no triple.
        """
        if not triples:
            return n
        
        triple_masks_list = [triple_masks[t] for t in triples]
        
        # Check from largest k down to 1
        # Optimization: If we find an independent set of size k, return k.
        # We check subsets of size k.
        for k in range(n, 0, -1):
            # Iterate over all subsets of size k
            found = False
            for subset in itertools.combinations(range(n), k):
                subset_mask = 0
                for v in subset:
                    subset_mask |= (1 << v)
                
                is_independent = True
                for tm in triple_masks_list:
                    if (tm & subset_mask) == tm:
                        is_independent = False
                        break
                if is_independent:
                    found = True
                    break
            if found:
                return k
        return 0

    def is_valid_add(triples, new_triple):
        """Check if new_triple can be added to triples without violating linearity."""
        new_set = set(new_triple)
        for t in triples:
            if len(new_set & set(t)) > 1:
                return False
        return True

    def random_maximal_packing():
        triples = []
        available = all_triples_list[:]
        random.shuffle(available)
        for t in available:
            if is_valid_add(triples, t):
                triples.append(t)
        return triples

    def sts15_based_packing():
        # Construct an STS(15) on points 0..14
        # This is a known STS(15)
        # Source: Standard construction or lookup
        # We'll generate a random valid packing on 0..14 first, 
        # but a specific STS(15) is better.
        # Here is a valid STS(15) (verified mentally or via standard library if available, 
        # but I will hardcode a known one to be safe and fast).
        
        # One STS(15):
        # Based on the affine plane of order 4? No, order 4 is 16 points.
        # STS(15) is unique up to isomorphism.
        # Let's use a generator or a hardcoded list.
        
        # Hardcoded STS(15) triples:
        sts15 = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11), (12, 13, 14),
            (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 9, 12), (1, 10, 13),
            (2, 11, 14), (0, 4, 10), (1, 5, 11), (2, 3, 9), (0, 7, 13),
            (1, 8, 14), (2, 4, 12), (3, 6, 10), (4, 6, 11), (5, 7, 9),
            (3, 12, 13), (4, 13, 14), (5, 12, 14), (6, 9, 13), (7, 10, 14),
            (8, 11, 12), (9, 12, 13), (10, 12, 14), (11, 13, 14), (0, 11, 15) # Wait, 15 is not in 0..14
        ]
        # The above hardcoded list might be incorrect or incomplete. 
        # Let's generate a random packing on 0..14 and maximize it, 
        # then add 15.
        # Actually, a random maximal packing on 16 points is a good baseline.
        # But STS(15) has specific properties.
        
        # Let's just use random_maximal_packing for 16 points as the primary seed,
        # and maybe one based on a greedy construction.
        
        triples = []
        # Try to build a dense packing
        available = all_triples_list[:]
        random.shuffle(available)
        for t in available:
            if is_valid_add(triples, t):
                triples.append(t)
        return triples

    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    best_triples = None
    best_alpha = n + 1
    
    # Multiple restarts
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Initialize
        if r == 0:
            current_triples = random_maximal_packing()
        elif r == 1:
            # Try a different random seed
            random.seed(r * 1000)
            current_triples = random_maximal_packing()
        else:
            current_triples = random_maximal_packing()
            
        current_alpha = calc_alpha(current_triples)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = current_triples[:]
            
        # Simulated Annealing
        T = 1.0
        T_min = 0.01
        cooling = 0.995
        steps = 0
        max_steps = 5000 # Limit steps per restart to ensure time budget
        
        while steps < max_steps and time.time() - start_time < timeout:
            steps += 1
            move_type = random.random()
            
            new_triples = current_triples[:]
            
            if move_type < 0.3:
                # Remove
                if not new_triples:
                    continue
                idx = random.randrange(len(new_triples))
                new_triples.pop(idx)
            elif move_type < 0.7:
                # Add
                # Try to add a random valid triple
                added = False
                for _ in range(20):
                    t = tuple(random.sample(range(n), 3))
                    if is_valid_add(new_triples, t):
                        new_triples.append(t)
                        added = True
                        break
                if not added:
                    continue
            else:
                # Swap
                if not new_triples:
                    continue
                idx = random.randrange(len(new_triples))
                removed = new_triples.pop(idx)
                
                added = False
                for _ in range(20):
                    t = tuple(random.sample(range(n), 3))
                    if is_valid_add(new_triples, t):
                        new_triples.append(t)
                        added = True
                        break
                if not added:
                    # Restore
                    new_triples.append(removed)
                    continue
            
            new_alpha = calc_alpha(new_triples)
            
            if new_alpha < current_alpha:
                current_triples = new_triples
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = current_triples[:]
            else:
                delta = new_alpha - current_alpha
                if delta == 0:
                    current_triples = new_triples
                else:
                    prob = 2.718281828 ** (-delta / T)
                    if random.random() < prob:
                        current_triples = new_triples
                        current_alpha = new_alpha
            
            T *= cooling
            if T < T_min:
                T = T_min

    if best_triples is None:
        return []
        
    return [list(t) for t in best_triples]
