import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_alpha = 12
    best_triples = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    
    # Precompute subsets by size as integer masks
    subsets_by_size = [[] for _ in range(n + 1)]
    for i in range(1 << n):
        subsets_by_size[i.bit_count()].append(i)
        
    def get_masks(triples):
        return [ (1<<a) | (1<<b) | (1<<c) for a,b,c in triples ]

    def calc_alpha_optimized(triples):
        masks = get_masks(triples)
        if not masks:
            return n
        for k in range(n, 0, -1):
            for mask in subsets_by_size[k]:
                independent = True
                for m in masks:
                    if (mask & m) == m:
                        independent = False
                        break
                if independent:
                    return k
        return 0

    def is_linear(triples, new_triple):
        a, b, c = new_triple
        new_set = {a, b, c}
        for t in triples:
            if len(new_set.intersection(t)) > 1:
                return False
        return True

    # Initial evaluation of seed
    seed_alpha = calc_alpha_optimized(best_triples)
    best_alpha = min(best_alpha, seed_alpha)
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        if iterations % 2 == 0:
            current_triples = []
        else:
            current_triples = list(best_triples)
            
        random.shuffle(all_triples)
        
        # Greedy add
        for t in all_triples:
            if is_linear(current_triples, t):
                current_triples.append(t)
                
        alpha = calc_alpha_optimized(current_triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = current_triples

    return best_triples
