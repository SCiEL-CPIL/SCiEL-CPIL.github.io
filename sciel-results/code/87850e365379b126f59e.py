import time
import random

def construct():
    n = 12
    m = 40
    deadline = time.time() + 55.0
    
    def count_pairs(family):
        c = 0
        for i in range(m):
            fi = family[i]
            for j in range(i+1, m):
                fj = family[j]
                if (fi & fj) == fi or (fi & fj) == fj:
                    c += 1
        return c

    best_family = None
    best_score = -1

    while time.time() < deadline:
        # Choose a random base set of size 8 to create a dense pool
        base_elements = random.sample(range(n), 8)
        base_mask = 0
        for e in base_elements:
            base_mask |= (1 << e)
        
        # Generate pool of all subsets of base_mask
        # There are 2^8 = 256 subsets
        pool = []
        for i in range(1 << 8):
            mask = 0
            for b in range(8):
                if (i >> b) & 1:
                    mask |= (1 << base_elements[b])
            pool.append(mask)
        
        # Precompute comparability for the pool?
        # It's 256x256, which is small.
        # comp[i][j] is 1 if pool[i] and pool[j] are comparable.
        n_pool = len(pool)
        comp = [[0] * n_pool for _ in range(n_pool)]
        for i in range(n_pool):
            mi = pool[i]
            for j in range(i + 1, n_pool):
                mj = pool[j]
                if (mi & mj) == mi or (mi & mj) == mj:
                    comp[i][j] = 1
                    comp[j][i] = 1
        
        # Greedy initialization
        family_indices = []
        remaining = set(range(n_pool))
        
        # Start with a random element or empty set (index of 0)
        # Empty set is comparable to everything, so it's a good start
        start_idx = 0
        family_indices.append(start_idx)
        remaining.remove(start_idx)
        
        for _ in range(m - 1):
            best_idx = -1
            best_gain = -1
            for idx in remaining:
                gain = 0
                for f_idx in family_indices:
                    gain += comp[idx][f_idx]
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            family_indices.append(best_idx)
            remaining.remove(best_idx)
            
        current_family = [pool[i] for i in family_indices]
        current_score = count_pairs(current_family)
        
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try swapping one element in family with one in remaining
            # To speed up, we can just try a few random swaps or the best swap
            # Let's do a few random swaps to escape local optima
            for _ in range(100):
                if time.time() >= deadline:
                    break
                i = random.randint(0, m-1)
                old_idx = family_indices[i]
                new_idx = random.choice(list(remaining))
                
                # Calculate gain
                # Remove old_idx contributions
                delta = 0
                for j in range(m):
                    if j != i:
                        f_idx = family_indices[j]
                        delta -= comp[old_idx][f_idx]
                        delta += comp[new_idx][f_idx]
                
                if delta > 0:
                    family_indices[i] = new_idx
                    remaining.remove(new_idx)
                    remaining.add(old_idx)
                    current_score += delta
                    improved = True
        
        if current_score > best_score:
            best_score = current_score
            best_family = current_family[:]

    # Convert to list of lists of ints
    result = []
    for mask in best_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
        
    return result
