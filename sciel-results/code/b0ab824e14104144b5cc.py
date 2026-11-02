import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # 1. Generate seed code (729 words)
    # Tetracode on first 4 coords, all on last 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Convert to integers for efficient counting
    def word_to_int(w):
        v = 0
        for i, d in enumerate(w):
            v += d * (q ** i)
        return v
        
    def int_to_word(v):
        w = [0] * n
        for i in range(n):
            w[i] = v % q
            v //= q
        return w
        
    # Precompute neighbors for all possible words? 
    # 3^8 = 6561 words. Each has 1 + 8*2 = 17 neighbors.
    # Total ~111k pairs. This is fast.
    all_words = list(range(q ** n))
    neighbors = {}
    for v in all_words:
        w = int_to_word(v)
        nb = {v}
        for i in range(n):
            orig = w[i]
            for d in range(q):
                if d == orig: continue
                w[i] = d
                nv = word_to_int(w)
                nb.add(nv)
            w[i] = orig
        neighbors[v] = list(nb)
        
    # Initialize counts
    counts = np.zeros(q ** n, dtype=np.int32)
    current_codewords = set()
    for w in seed_codewords:
        v = word_to_int(w)
        current_codewords.add(v)
        for nb in neighbors[v]:
            counts[nb] += 1
            
    # Local Search
    # Strategy: 
    # 1. Try to remove any codeword that is redundant (all its neighbors have count > 1).
    # 2. If stuck, try swaps.
    
    codeword_list = list(current_codewords)
    random.shuffle(codeword_list)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Pass 1: Remove redundant codewords
        # We iterate through the list. If a codeword can be removed, we do it.
        # To be efficient, we can just scan the list.
        new_list = []
        for v in codeword_list:
            # Check if v is redundant
            # v is redundant if min(counts[nb] for nb in neighbors[v]) > 1
            # Actually, we just need to check if removing it leaves all counts > 0.
            # This is equivalent to: for all nb in neighbors[v], counts[nb] > 1.
            redundant = True
            for nb in neighbors[v]:
                if counts[nb] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove v
                for nb in neighbors[v]:
                    counts[nb] -= 1
                current_codewords.discard(v)
                improved = True
            else:
                new_list.append(v)
        codeword_list = new_list
        
        if time.time() > deadline: break
        
        # Pass 2: Try swaps to escape local minima
        # This is more complex. We pick a codeword v1 to remove, and a non-codeword v2 to add.
        # We want v2 to cover points that v1 uniquely covers, but also cover other points well.
        # Simplified: Random swaps.
        if len(codeword_list) > 0:
            for _ in range(1000): # Fixed number of swap attempts per pass
                if time.time() > deadline: break
                # Pick random v1 from current set
                v1 = random.choice(codeword_list)
                # Pick random v2 from non-set
                # We can pick from all_words and check if in set
                v2 = random.choice(all_words)
                if v2 in current_codewords: continue
                
                # Check validity of swap
                # Remove v1, Add v2
                # We need to check if counts[nb] - (1 if nb in neighbors[v1] else 0) + (1 if nb in neighbors[v2] else 0) > 0 for all nb
                # This is expensive to check for all 6561 words.
                # Optimization: Only check neighbors of v1 and v2.
                # Points not in N(v1) U N(v2) are unaffected.
                # Points in N(v1) \ N(v2): count decreases by 1. Must be > 1 before removal.
                # Points in N(v2) \ N(v1): count increases by 1. Always safe if count >= 0.
                # Points in N(v1) \cap N(v2): count unchanged.
                
                # So we only need to check:
                # For nb in neighbors[v1]:
                #   if nb not in neighbors[v2]: counts[nb] must be > 1
                
                can_swap = True
                # To speed up, we can use a set for neighbors[v2]
                nb_v2_set = set(neighbors[v2])
                for nb in neighbors[v1]:
                    if nb not in nb_v2_set:
                        if counts[nb] <= 1:
                            can_swap = False
                            break
                            
                if can_swap:
                    # Perform swap
                    for nb in neighbors[v1]:
                        counts[nb] -= 1
                    for nb in neighbors[v2]:
                        counts[nb] += 1
                    current_codewords.discard(v1)
                    current_codewords.add(v2)
                    codeword_list.remove(v1)
                    codeword_list.append(v2)
                    improved = True
                    break # Restart pass after a successful swap? Or continue? 
                    # Breaking restarts the removal pass which is good.
                    
    # Convert back to list of lists
    result = [int_to_word(v) for v in current_codewords]
    return result
