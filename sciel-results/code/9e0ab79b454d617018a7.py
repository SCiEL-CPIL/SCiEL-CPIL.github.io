import random
import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 16
    points = list(range(n))
    
    best_alpha = n
    best_triples = []
    
    # Precompute all possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    triple_to_mask = {t: (1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in all_triples}
    
    # Precompute popcount for all 16-bit masks
    popcount = np.zeros(1 << n, dtype=np.uint8)
    for m in range(1, 1 << n):
        popcount[m] = popcount[m & (m - 1)] + 1
    
    def get_alpha_fast(triples):
        """Fast alpha: mark all supersets of each triple as covered, find max popcount uncovered."""
        if not triples:
            return n
        
        # covered[m] = True if m contains any triple
        covered = np.zeros(1 << n, dtype=bool)
        
        for t in triples:
            tm = triple_to_mask[t]
            # All supersets of tm: for each bit not in tm, it can be 0 or 1
            # Enumerate all supersets
            free_bits = [i for i in range(n) if not (tm >> i) & 1]
            k = len(free_bits)
            for sub in range(1 << k):
                sup = tm
                for j in range(k):
                    if sub >> j & 1:
                        sup |= (1 << free_bits[j])
                covered[sup] = True
        
        # Find max popcount among uncovered masks
        # Iterate from high popcount down
        max_pc = n
        for pc in range(n, 0, -1):
            for m in range(1 << n):
                if popcount[m] == pc and not covered[m]:
                    return pc
        return 0
    
    def get_alpha_superset_table(triples):
        """Alternative: build superset table more efficiently using SOS DP."""
        if not triples:
            return n
        
        # is_triple[m] = True if m is exactly a triple
        is_triple = np.zeros(1 << n, dtype=bool)
        for t in triples:
            is_triple[triple_to_mask[t]] = True
        
        # SOS DP: covered[m] = True if any subset of m is a triple
        # Actually we want: covered[m] = True if m is a SUPERSET of some triple
        # i.e., exists triple t such that t ⊆ m
        # This is: covered[m] = OR over all submasks s of m: is_triple[s]
        
        covered = is_triple.copy()
        # SOS DP to compute OR over all submasks
        for i in range(n):
            for m in range(1 << n):
                if m & (1 << i):
                    covered[m] = covered[m] | covered[m ^ (1 << i)]
        
        # Find max popcount among uncovered masks
        for pc in range(n, 0, -1):
            for m in range(1 << n):
                if popcount[m] == pc and not covered[m]:
                    return pc
        return 0
    
    def try_random_greedy():
        # Weighted greedy: prioritize triples that cover points with fewer options
        triples_order = all_triples[:]
        random.shuffle(triples_order)
        
        used_pairs = set()
        triples = []
        point_deg = [0] * n  # how many pairs each point is involved in (among used)
        
        for t in triples_order:
            a, b, c = t
            pairs = frozenset([(a,b), (a,c), (b,c)])
            if not any(p in used_pairs for p in pairs):
                triples.append(t)
                for p in pairs:
                    used_pairs.add(p)
        return triples
    
    def try_weighted_greedy():
        """Greedy that picks triples to maximize coverage of under-covered points."""
        used_pairs = set()
        triples = []
        remaining_points = set(range(n))
        
        while remaining_points:
            # Find triples that use at least one remaining point and don't conflict
            candidates = []
            for t in all_triples:
                a, b, c = t
                if not (a in remaining_points or b in remaining_points or c in remaining_points):
                    continue
                pairs = frozenset([(a,b), (a,c), (b,c)])
                if any(p in used_pairs for p in pairs):
                    continue
                # Score: prefer triples with more remaining points
                score = sum(1 for p in (a,b,c) if p in remaining_points)
                candidates.append((score, t))
            
            if not candidates:
                break
            
            # Pick best candidate (random tie-break)
            max_score = max(c[0] for c in candidates)
            best = [c for c in candidates if c[0] == max_score]
            _, t = random.choice(best)
            
            a, b, c = t
            pairs = frozenset([(a,b), (a,c), (b,c)])
            for p in pairs:
                used_pairs.add(p)
            triples.append(t)
            remaining_points -= {a, b, c}
        
        return triples
    
    def local_search(triples):
        """Try to improve alpha by swapping triples."""
        triples_set = set(triples)
        used_pairs = set()
        for t in triples:
            a, b, c = t
            for p in [(a,b), (a,c), (b,c)]:
                used_pairs.add(p)
        
        current_alpha = get_alpha_superset_table(triples)
        
        # Try removing each triple and see if alpha changes
        # Then try adding a different triple
        for i, t_remove in enumerate(triples):
            if time.time() > deadline:
                break
            
            a, b, c = t_remove
            remove_pairs = [(a,b), (a,c), (b,c)]
            
            # Remove t_remove
            new_triples = [t for j, t in enumerate(triples) if j != i]
            new_used_pairs = set()
            for t in new_triples:
                x, y, z = t
                for p in [(x,y), (x,z), (y,z)]:
                    new_used_pairs.add(p)
            
            # Try adding a different triple that was previously blocked
            for t_add in all_triples:
                if t_add in triples_set:
                    continue
                x, y, z = t_add
                add_pairs = [(x,y), (x,z), (y,z)]
                if any(p in new_used_pairs for p in add_pairs):
                    continue
                
                candidate = new_triples + [t_add]
                alpha = get_alpha_superset_table(candidate)
                
                if alpha < current_alpha:
                    return candidate, alpha
                elif alpha == current_alpha:
                    # Accept with some probability to explore
                    if random.random() < 0.3:
                        return candidate, alpha
                    break  # Only try one replacement per removal to save time
        
        return triples, current_alpha
    
    trials = 0
    while time.time() < deadline:
        trials += 1
        
        if trials % 3 == 0:
            trs = try_weighted_greedy()
        else:
            trs = try_random_greedy()
        
        alpha = get_alpha_superset_table(trs)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = trs
            if alpha <= 5:
                break
        
        # Local search on the best found so far occasionally
        if trials % 5 == 0 and best_triples:
            improved, new_alpha = local_search(best_triples)
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_triples = improved
                if best_alpha <= 5:
                    break
    
    return best_triples
# EVOLVE-BLOCK-END
