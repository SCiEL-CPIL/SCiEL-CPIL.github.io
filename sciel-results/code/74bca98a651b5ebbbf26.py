import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill
    
    n = 16
    best_triples = None
    best_alpha = n  # worst case: no triples, alpha = 16
    
    # Helper: check linearity (any two triples share at most 1 point)
    def is_linear(triples):
        for i in range(len(triples)):
            si = set(triples[i])
            for j in range(i + 1, len(triples)):
                if len(si & set(triples[j])) > 1:
                    return False
        return True
    
    # Helper: compute independence number (largest set with no triple)
    def calc_alpha(triples):
        if not triples:
            return n
        # For each point, compute which triples it's in
        # Use a smarter approach: try to find max independent set
        # Since n=16 is small, we can use a branch and bound or
        # check subsets by size
        
        triple_sets = [set(t) for t in triples]
        
        # Try from largest size down
        for size in range(n, 0, -1):
            for subset in combinations(range(n), size):
                s = set(subset)
                covered = False
                for ts in triple_sets:
                    if ts <= s:
                        covered = True
                        break
                if not covered:
                    return size
        return 0
    
    # Faster alpha calculation using bitmask
    def calc_alpha_fast(triples):
        if not triples:
            return n
        # Precompute: for each point, bitmask of triples containing it
        # Actually, let's use a different approach:
        # A set S is independent iff no triple is a subset of S
        # We want the largest such S.
        
        # Use iterative deepening from the top
        # Or use the fact that we can check all subsets of size k
        # For n=16, total subsets = 2^16 = 65536, very fast
        
        triple_masks = []
        for t in triples:
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_masks.append(mask)
        
        # Check all subsets from largest to smallest
        # A subset S (as bitmask) is independent if no triple_mask is a subset of S
        # i.e., for all triple_masks tm: (tm & ~S) != 0
        # Equivalently: (tm & S) != tm for all tm
        
        # We want the largest S. Let's iterate by popcount descending.
        # For n=16, we can just check all 65536 subsets and track max independent size.
        
        max_ind = 0
        for mask in range(1 << n):
            # Check if mask is independent
            independent = True
            for tm in triple_masks:
                if (tm & mask) == tm:
                    independent = False
                    break
            if independent:
                pop = bin(mask).count('1')
                if pop > max_ind:
                    max_ind = pop
        return max_ind
    
    # Even faster: use the property that we only need to check subsets
    # Actually for 16 points, 2^16 = 65536 checks, each with up to ~20 triples
    # That's ~1.3M operations per alpha calc, very fast in Python if optimized
    
    def calc_alpha_opt(triples):
        if not triples:
            return n
        triple_masks = []
        for t in triples:
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_masks.append(mask)
        
        max_ind = 0
        # Iterate all masks, find max popcount among independent ones
        for mask in range(1 << n):
            indep = True
            for tm in triple_masks:
                if (tm & mask) == tm:
                    indep = False
                    break
            if indep:
                p = bin(mask).count('1')
                if p > max_ind:
                    max_ind = p
        return max_ind
    
    # Generate all possible triples
    all_triples = list(combinations(range(n), 3))
    random.shuffle(all_triples)
    
    # Greedy construction: add triples that maintain linearity
    def greedy_build(seed_offset=0):
        random.seed(42 + seed_offset)
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        triples = []
        # Track point-pair usage: a pair can be in at most one triple
        pair_used = set()
        for t in shuffled:
            t = tuple(t)
            pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
            if all(p not in pair_used for p in pairs):
                triples.append(list(t))
                for p in pairs:
                    pair_used.add(p)
        return triples
    
    # Local search: try to reduce alpha by modifying triples
    def local_search(triples, time_limit):
        triples = [list(t) for t in triples]
        alpha = calc_alpha_opt(triples)
        
        # Find the current largest independent set
        def find_max_independent_set(triples):
            triple_masks = []
            for t in triples:
                mask = 0
                for p in t:
                    mask |= (1 << p)
                triple_masks.append(mask)
            
            best_mask = 0
            for mask in range(1 << n):
                indep = True
                for tm in triple_masks:
                    if (tm & mask) == tm:
                        indep = False
                        break
                if indep:
                    p = bin(mask).count('1')
                    bp = bin(best_mask).count('1')
                    if p > bp:
                        best_mask = mask
            return best_mask
        
        while time.time() < time_limit:
            # Find a largest independent set
            ind_mask = find_max_independent_set(triples)
            ind_points = [i for i in range(n) if (ind_mask >> i) & 1]
            
            if len(ind_points) <= alpha - 1:
                break
            
            # Try to break this independent set by modifying a triple
            # Strategy: pick a point from the independent set, try to add/modify
            # a triple that includes it and two other points (not both in ind set
            # ideally, to keep linearity)
            
            improved = False
            random.shuffle(ind_points)
            
            for pt in ind_points:
                if time.time() >= time_limit:
                    break
                
                # Try adding a new triple containing pt
                # Or modifying an existing triple
                candidates = []
                for a in range(n):
                    if a == pt:
                        continue
                    for b in range(n):
                        if b == pt or b == a:
                            continue
                        t = tuple(sorted((pt, a, b)))
                        candidates.append(t)
                
                random.shuffle(candidates)
                
                for t in candidates:
                    if time.time() >= time_limit:
                        break
                    new_triples = triples + [list(t)]
                    if is_linear(new_triples):
                        new_alpha = calc_alpha_opt(new_triples)
                        if new_alpha < alpha:
                            triples = new_triples
                            alpha = new_alpha
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                # Try removing a triple and adding a better one
                # Or try swapping
                for i in range(len(triples)):
                    if time.time() >= time_limit:
                        break
                    old_t = triples[i]
                    # Try replacing old_t with a different triple
                    for a in range(n):
                        if a in old_t and a != pt:  # keep some continuity
                            continue
                        for b in range(n):
                            if b in old_t and b != pt:
                                continue
                            if a == b:
                                continue
                            new_t = tuple(sorted((a, b, pt)))
                            if new_t == tuple(sorted(old_t)):
                                continue
                            new_triples = triples[:i] + [list(new_t)] + triples[i+1:]
                            if is_linear(new_triples):
                                new_alpha = calc_alpha_opt(new_triples)
                                if new_alpha < alpha:
                                    triples = new_triples
                                    alpha = new_alpha
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                # Stuck, try a random perturbation
                if triples:
                    i = random.randint(0, len(triples) - 1)
                    old_t = triples[i]
                    # Replace with a random triple
                    new_t = random.choice(all_triples)
                    new_triples = triples[:i] + [list(new_t)] + triples[i+1:]
                    if is_linear(new_triples):
                        new_alpha = calc_alpha_opt(new_triples)
                        if new_alpha <= alpha:
                            triples = new_triples
                            alpha = new_alpha
    
        return triples, alpha
    
    # Main search loop
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Greedy build
        triples = greedy_build(attempt)
        if not triples:
            continue
        
        alpha = calc_alpha_opt(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = [list(t) for t in triples]
        
        # Local search with time budget
        remaining = deadline - time.time() - 5.0  # 5s safety margin
        if remaining > 1.0:
            triples, alpha = local_search(triples, start + remaining)
            if alpha < best_alpha:
                best_alpha = alpha
                best_triples = [list(t) for t in triples]
        
        if best_alpha <= 5:  # Good enough, stop early
            break
    
    if best_triples is None:
        # Fallback: return the initial 5 disjoint triples
        best_triples = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return best_triples
# EVOLVE-BLOCK-END
