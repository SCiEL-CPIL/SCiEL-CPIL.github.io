import itertools
import random
import time

def construct(q, n, radius):
    # Fallback for general case or if not the specific hard instance
    if not (q == 3 and n == 6 and radius == 1):
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Specific solver for K_3(6, 1)
    N = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute spheres as bitmasks
    spheres = [0] * N
    for i, w in enumerate(all_words):
        mask = 1 << i
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                j = word_to_idx[tuple(new_w)]
                mask |= (1 << j)
        spheres[i] = mask
        
    full_mask = (1 << N) - 1
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_code = None
    best_size = N # worst case
    
    # Run multiple greedy restarts + pruning
    # Greedy is fast, pruning is fast. We can do many iterations.
    
    # Helper to prune a code
    def prune(code_list):
        # code_list is a list of indices
        # We want to remove any codeword that is redundant
        # Iterate until no more removals possible
        current = list(code_list)
        changed = True
        while changed:
            changed = False
            for i in range(len(current)):
                # Check if current[i] is redundant
                # Union of all spheres except current[i]
                union_others = 0
                # Optimization: if len(current) is small, this is fast.
                # We can break early if union_others becomes full_mask
                for j in range(len(current)):
                    if i == j:
                        continue
                    union_others |= spheres[current[j]]
                    if union_others == full_mask:
                        break
                if union_others == full_mask:
                    # Redundant, remove it
                    current.pop(i)
                    changed = True
                    break # Restart the for loop
        return current

    # Initial greedy runs
    # We will do a fixed number of restarts, but break if time is up.
    # Each restart: Greedy + Prune
    
    # To diversify, we can also start from the tetracode product and prune?
    # Tetracode product size is 81. Pruning might get us to 75 or 73.
    
    # Let's try starting from tetracode first, as it's a known good structure.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = tuple(h + list(tail))
            init_code.append(word_to_idx[w])
            
    pruned_init = prune(init_code)
    if len(pruned_init) < best_size:
        best_size = len(pruned_init)
        best_code = [all_words[i] for i in pruned_init]
        
    # Now run greedy restarts
    # Greedy tends to find slightly different structures.
    num_restarts = 500
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.seed(r)
        candidates = list(range(N))
        random.shuffle(candidates)
        
        covered = 0
        chosen = []
        
        while covered != full_mask:
            best_gain = -1
            best_idx = -1
            
            for idx in candidates:
                new_covered = spheres[idx] & ~covered
                gain = new_covered.bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    
            if best_gain == 0:
                break
                
            chosen.append(best_idx)
            covered |= spheres[best_idx]
            
        # Prune the greedy result
        pruned_greedy = prune(chosen)
        
        if len(pruned_greedy) < best_size:
            best_size = len(pruned_greedy)
            best_code = [all_words[i] for i in pruned_greedy]
            
    if best_code is None:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(h) + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in best_code]
