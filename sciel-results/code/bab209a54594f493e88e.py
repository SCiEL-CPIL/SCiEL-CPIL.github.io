import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_alpha = 16
    best_edges = []
    
    # Precompute all triples
    all_triples = []
    for i in range(16):
        for j in range(i+1, 16):
            for k in range(j+1, 16):
                all_triples.append((i, j, k))
                
    # Shuffle for randomization? Or just pick random indices.
    
    def calc_alpha(edges):
        # edges is list of tuples
        # Convert to masks
        masks = []
        for t in edges:
            m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            masks.append(m)
            
        points = range(16)
        for k in range(16, 0, -1):
            for subset in itertools.combinations(points, k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                is_ind = True
                for m in masks:
                    if (mask & m) == m:
                        is_ind = False
                        break
                if is_ind:
                    return k
        return 0

    def greedy_construct():
        used_pairs = set()
        edges = []
        # Randomize order of triples
        random.shuffle(all_triples)
        for t in all_triples:
            i, j, k = t
            pairs = ((i,j), (i,k), (j,k))
            if all(p not in used_pairs for p in pairs):
                used_pairs.update(pairs)
                edges.append(t)
        return edges

    # Run multiple restarts
    while time.time() < deadline:
        edges = greedy_construct()
        alpha = calc_alpha(edges)
        if alpha < best_alpha:
            best_alpha = alpha
            best_edges = edges
            if best_alpha <= 5: # Likely near optimal, can stop or continue
                # We can continue to see if we get 4 or lower, but 5 is very good.
                # Let's keep running a bit more to be sure, but break if time is tight.
                pass

    return best_edges

# EVOLVE-BLOCK-END
