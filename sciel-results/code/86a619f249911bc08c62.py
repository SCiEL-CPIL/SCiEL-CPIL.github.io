"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s buffer before 120s kill

    n = 16
    points = list(range(n))
    
    # Pre-compute popcount for all 16-bit masks
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)

    def get_alpha_fast(triples):
        if not triples:
            return n
        
        t_masks = []
        for a, b, c in triples:
            t_masks.append((1 << a) | (1 << b) | (1 << c))
        
        max_alpha = 0
        # Iterate all masks
        for m in range(1 << n):
            # Check independence
            independent = True
            for t in t_masks:
                if (m & t) == t:
                    independent = False
                    break
            if independent:
                pc = popcount[m]
                if pc > max_alpha:
                    max_alpha = pc
        return max_alpha

    def try_random_greedy():
        all_triples = list(itertools.combinations(range(n), 3))
        random.shuffle(all_triples)
        
        used_pairs = set()
        triples = []
        
        for t in all_triples:
            a, b, c = t
            p1 = (a, b) if a < b else (b, a)
            p2 = (a, c) if a < c else (c, a)
            p3 = (b, c) if b < c else (c, b)
            
            if p1 in used_pairs or p2 in used_pairs or p3 in used_pairs:
                continue
                
            triples.append(t)
            used_pairs.add(p1)
            used_pairs.add(p2)
            used_pairs.add(p3)
            
        return triples

    def local_search(triples, steps=1000):
        triples = list(triples)
        used_pairs = set()
        for t in triples:
            a, b, c = t
            p1 = (a, b) if a < b else (b, a)
            p2 = (a, c) if a < c else (c, a)
            p3 = (b, c) if b < c else (c, b)
            used_pairs.add(p1)
            used_pairs.add(p2)
            used_pairs.add(p3)
        
        current_alpha = get_alpha_fast(triples)
        
        for _ in range(steps):
            if time.time() > deadline:
                break
            
            # Pick a random triple to remove
            if not triples:
                break
            idx = random.randint(0, len(triples) - 1)
            rem_t = triples[idx]
            
            # Remove it
            a, b, c = rem_t
            p1 = (a, b) if a < b else (b, a)
            p2 = (a, c) if a < c else (c, a)
            p3 = (b, c) if b < c else (c, b)
            used_pairs.remove(p1)
            used_pairs.remove(p2)
            used_pairs.remove(p3)
            triples.pop(idx)
            
            # Try to add a random valid triple
            # Generate a random candidate triple
            # To speed up, we can pick random points and check validity
            # Or iterate through a shuffled list of all triples? 
            # Let's pick 3 random distinct points.
            candidates = []
            # Try a few random candidates
            for _ in range(10):
                cand = random.sample(range(n), 3)
                cand.sort()
                ca, cb, cc = cand
                cp1 = (ca, cb)
                cp2 = (ca, cc)
                cp3 = (cb, cc)
                if cp1 not in used_pairs and cp2 not in used_pairs and cp3 not in used_pairs:
                    candidates.append(cand)
            
            if not candidates:
                # Re-add the removed triple if no candidate found
                triples.append(rem_t)
                used_pairs.add(p1)
                used_pairs.add(p2)
                used_pairs.add(p3)
                continue
            
            new_t = random.choice(candidates)
            na, nb, nc = new_t
            np1 = (na, nb)
            np2 = (na, nc)
            np3 = (nb, nc)
            
            triples.append(new_t)
            used_pairs.add(np1)
            used_pairs.add(np2)
            used_pairs.add(np3)
            
            new_alpha = get_alpha_fast(triples)
            
            # Keep if better or equal (with some probability to allow exploration)
            if new_alpha < current_alpha or (new_alpha == current_alpha and random.random() < 0.1):
                current_alpha = new_alpha
            else:
                # Revert
                triples.pop()
                used_pairs.remove(np1)
                used_pairs.remove(np2)
                used_pairs.remove(np3)
                triples.append(rem_t)
                used_pairs.add(p1)
                used_pairs.add(p2)
                used_pairs.add(p3)
                
        return triples, current_alpha

    best_alpha = n
    best_triples = []
    
    # Initial random greedy runs
    trials = 0
    while time.time() < deadline - 20:  # Leave 20s for local search
        trials += 1
        trs = try_random_greedy()
        alpha = get_alpha_fast(trs)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = trs
            if best_alpha <= 6:
                break
                
        # If we have a decent solution, try local search on it occasionally
        if trials % 5 == 0 and best_triples:
            improved_triples, improved_alpha = local_search(best_triples, steps=50)
            if improved_alpha < best_alpha:
                best_alpha = improved_alpha
                best_triples = improved_triples

    # Final local search on the best found
    if best_triples and time.time() < deadline - 5:
        improved_triples, improved_alpha = local_search(best_triples, steps=1000)
        if improved_alpha < best_alpha:
            best_alpha = improved_alpha
            best_triples = improved_triples

    return [list(t) for t in best_triples]

# EVOLVE-BLOCK-END
