import itertools
import random
import time

def construct():
    n = 16
    deadline = time.time() + 115.0
    
    # Precompute all triples and their properties
    all_triples = list(itertools.combinations(range(n), 3))
    triple_masks = {t: (1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in all_triples}
    
    # Precompute supersets for each triple mask
    # There are only 560 unique triples, so 560 lists of ~8192 ints.
    # 560 * 8192 * 4 bytes ~ 18MB. Acceptable.
    supersets_cache = {}
    for t in all_triples:
        mask = triple_masks[t]
        free = (~mask) & 0xFFFF
        subs = []
        sub = free
        while True:
            subs.append(mask | sub)
            if sub == 0:
                break
            sub = (sub - 1) & free
        supersets_cache[t] = subs

    # Precompute popcount for all 16-bit masks
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)

    def calculate_alpha(cover_count):
        """
        Find max popcount of mask where cover_count[mask] == 0.
        """
        max_alpha = 0
        # Iterate all masks
        for mask in range(1 << n):
            if cover_count[mask] == 0:
                pc = popcount[mask]
                if pc > max_alpha:
                    max_alpha = pc
        return max_alpha

    def generate_random_packing():
        used_pairs = set()
        triples = []
        candidates = all_triples[:]
        random.shuffle(candidates)
        
        for t in candidates:
            a, b, c = t
            ps = frozenset([(a, b), (a, c), (b, c)])
            if not (ps & used_pairs):
                triples.append(t)
                used_pairs.update(ps)
        return triples, used_pairs

    def is_valid_addition(t, triples, used_pairs):
        # Check if t can be added to the current set of triples
        # It must not share a pair with any existing triple
        a, b, c = t
        ps = frozenset([(a, b), (a, c), (b, c)])
        if ps & used_pairs:
            return False
        return True

    best_alpha = 17
    best_triples = []
    rng = random.Random(42)
    
    # Main search loop
    while time.time() < deadline:
        triples, used_pairs = generate_random_packing()
        
        # Initialize cover_count
        cover_count = [0] * (1 << n)
        for t in triples:
            for sup in supersets_cache[t]:
                cover_count[sup] += 1
        
        current_alpha = calculate_alpha(cover_count)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = triples[:]
            
        if best_alpha <= 5:
            break
            
        # Local Search
        # We will perform swaps
        search_deadline = min(time.time() + 1.0, deadline - 2.0)
        
        while time.time() < search_deadline:
            if len(triples) < 2:
                break
                
            idx = rng.randrange(len(triples))
            t_old = triples[idx]
            
            # Remove t_old
            used_pairs.difference_update(frozenset([(t_old[0], t_old[1]), (t_old[0], t_old[2]), (t_old[1], t_old[2])]))
            for sup in supersets_cache[t_old]:
                cover_count[sup] -= 1
                
            # Find a replacement
            # Try random candidates
            found = False
            candidates = all_triples[:]
            rng.shuffle(candidates)
            
            for t_new in candidates:
                if t_new == t_old:
                    continue
                if is_valid_addition(t_new, triples, used_pairs):
                    # Valid
                    triples[idx] = t_new
                    used_pairs.update(frozenset([(t_new[0], t_new[1]), (t_new[0], t_new[2]), (t_new[1], t_new[2])]))
                    
                    for sup in supersets_cache[t_new]:
                        cover_count[sup] += 1
                        
                    new_alpha = calculate_alpha(cover_count)
                    
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_triples = triples[:]
                        
                    # Accept if better or equal, or with some probability if worse (Simulated Annealing style?)
                    # For now, accept if <= current_alpha
                    if new_alpha <= current_alpha:
                        current_alpha = new_alpha
                    else:
                        # Revert
                        used_pairs.difference_update(frozenset([(t_new[0], t_new[1]), (t_new[0], t_new[2]), (t_new[1], t_new[2])]))
                        for sup in supersets_cache[t_new]:
                            cover_count[sup] -= 1
                        triples[idx] = t_old
                        used_pairs.update(frozenset([(t_old[0], t_old[1]), (t_old[0], t_old[2]), (t_old[1], t_old[2])]))
                        for sup in supersets_cache[t_old]:
                            cover_count[sup] += 1
                    found = True
                    break
                    
            if not found:
                # Revert
                used_pairs.update(frozenset([(t_old[0], t_old[1]), (t_old[0], t_old[2]), (t_old[1], t_old[2])]))
                for sup in supersets_cache[t_old]:
                    cover_count[sup] += 1
                    triples[idx] = t_old

    return best_triples
