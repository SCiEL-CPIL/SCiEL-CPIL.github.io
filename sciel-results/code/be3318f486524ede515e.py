import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 16
    points = list(range(n))
    
    # Precompute all possible triples as tuples of sorted points
    all_triples = list(combinations(points, 3))
    
    best_alpha = 16
    best_hypergraph = []
    
    # Helper to calculate alpha
    def calc_alpha(hypergraph):
        if not hypergraph:
            return n
        # Convert to bitmasks for speed
        triple_masks = []
        for t in hypergraph:
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_masks.append(mask)
        
        # Check from largest k down
        for k in range(n, 0, -1):
            found = False
            for subset in combinations(points, k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                # Check if any triple is contained in mask
                is_independent = True
                for tm in triple_masks:
                    if (mask & tm) == tm:
                        is_independent = False
                        break
                if is_independent:
                    found = True
                    break
            if found:
                return k
        return 1

    # Greedy construction
    def greedy_construct():
        used_pairs = set()
        hypergraph = []
        
        # Create a shuffled list of all triples
        candidates = all_triples[:]
        random.shuffle(candidates)
        
        for t in candidates:
            p0, p1, p2 = t
            pairs = [(p0, p1), (p0, p2), (p1, p2)]
            valid = True
            for p in pairs:
                if p in used_pairs:
                    valid = False
                    break
            
            if valid:
                hypergraph.append(t)
                used_pairs.update(pairs)
                
        return hypergraph

    # Local search to improve alpha
    def local_search(hypergraph, time_limit=2.0):
        h = [list(t) for t in hypergraph]
        used_pairs = set()
        for t in h:
            p0, p1, p2 = t
            used_pairs.add((p0, p1))
            used_pairs.add((p0, p2))
            used_pairs.add((p1, p2))
            
        current_alpha = calc_alpha(h)
        best_alpha = current_alpha
        best_h = [t[:] for t in h]
        
        candidates = all_triples[:]
        random.shuffle(candidates)
        
        start_time = time.time()
        while time.time() < start_time + time_limit:
            if not h:
                break
            idx = random.randint(0, len(h) - 1)
            t_out = h[idx]
            
            # Remove t_out
            h.pop(idx)
            p0, p1, p2 = t_out
            used_pairs.discard((p0, p1))
            used_pairs.discard((p0, p2))
            used_pairs.discard((p1, p2))
            
            # Try to add a random triple
            added = False
            for _ in range(5):
                if not candidates: break
                t_in = random.choice(candidates)
                p_a, p_b, p_c = t_in
                pairs = [(p_a, p_b), (p_a, p_c), (p_b, p_c)]
                if all(p not in used_pairs for p in pairs):
                    # Add it
                    h.append(t_in)
                    used_pairs.update(pairs)
                    new_alpha = calc_alpha(h)
                    
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_h = [t[:] for t in h]
                    else:
                        # Revert
                        h.pop()
                        for p in pairs:
                            used_pairs.discard(p)
                    added = True
                    break
            
            if not added:
                # Revert removal
                h.append(t_out)
                used_pairs.update([(p0, p1), (p0, p2), (p1, p2)])
                
        return best_h, best_alpha

    # Main loop
    iterations = 0
    while time.time() < deadline:
        # Generate a few candidates and pick the best for local search
        # To save time, we do 1 greedy + 1 local search per iteration
        h = greedy_construct()
        a = calc_alpha(h)
        
        if a < best_alpha:
            best_alpha = a
            best_hypergraph = h
        
        # Apply local search if we have time
        if time.time() < deadline - 5.0:
            # Limit local search time to ensure we finish
            ls_time = 1.0
            if time.time() > deadline - 15.0:
                ls_time = 0.5
            
            h_ls, a_ls = local_search(h, ls_time)
            
            if a_ls < best_alpha:
                best_alpha = a_ls
                best_hypergraph = h_ls
        
        iterations += 1

    # Return the best hypergraph found
    return [list(t) for t in best_hypergraph]

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} triples")
