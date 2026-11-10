import numpy as np
import time
import random

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        deadline = time.time() + 118.0
        N = q ** n  # 6561
        powers = [q ** i for i in range(n)]
        
        # Precompute neighbors: neighbors[w] is a list of words within distance 1
        # Using a list of lists for flexibility, or a 2D array if fixed size.
        # Size is 1 + n*(q-1) = 1 + 8*2 = 17.
        neigh_size = 1 + n * (q - 1)
        neighbors = np.zeros((N, neigh_size), dtype=np.int32)
        
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                for delta in range(1, q):
                    new_d = (d + delta) % q
                    diff = new_d - d
                    new_w = w + diff * powers[i]
                    neighbors[w, idx] = new_w
                    idx += 1
        
        def compute_coverage(code_set):
            cov = np.zeros(N, dtype=np.int32)
            for c in code_set:
                cov[neighbors[c, :]] += 1
            return cov

        def greedy_init(rng):
            uncovered = np.ones(N, dtype=np.bool_)
            code = set()
            while np.any(uncovered) and time.time() < deadline - 5:
                # Gain is number of uncovered neighbors
                gains = uncovered[neighbors].sum(axis=1)
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                w = rng.choice(candidates)
                code.add(w)
                uncovered[neighbors[w, :]] = False
            return code

        def local_search(code_set, rng, deadline):
            # Convert to list for easier manipulation, keep set for O(1) lookup
            code_list = list(code_set)
            code_set = set(code_list)
            
            while time.time() < deadline - 2:
                # 1. Compute Coverage
                cov = compute_coverage(code_set)
                
                # 2. Deletion Pass
                # A codeword c is redundant if min(cov[neighbors[c]]) > 1
                # We can find all redundant codewords and remove them.
                redundant = []
                for c in code_set:
                    if np.min(cov[neighbors[c, :]]) > 1:
                        redundant.append(c)
                
                if redundant:
                    for c in redundant:
                        code_set.remove(c)
                    continue # Recompute coverage as it changed significantly
                
                # 3. Swap Pass
                # Try to swap one codeword to enable further deletions
                improved = False
                # Randomize order
                c_list = list(code_set)
                rng.shuffle(c_list)
                
                for c in c_list:
                    if time.time() > deadline - 2:
                        break
                    
                    # Essential words: covered ONLY by c
                    # cov is computed with c included.
                    # So essential words are those w in N(c) where cov[w] == 1
                    c_neighbors = neighbors[c, :]
                    essential_mask = (cov[c_neighbors] == 1)
                    essential_words = c_neighbors[essential_mask]
                    
                    if len(essential_words) == 0:
                        # Should have been deleted, but maybe cov changed?
                        # If cov is up to date, this shouldn't happen.
                        continue
                    
                    # We need a d such that d covers all essential_words
                    # d must be in the intersection of N(w) for all w in essential_words
                    # And d not in code_set
                    
                    # Start with all possible candidates (neighbors of first essential word)
                    if len(essential_words) == 1:
                        ew = essential_words[0]
                        candidates = neighbors[ew, :]
                    else:
                        # Intersect neighbor sets
                        # Since essential_words is small (usually 1-3), this is fast
                        cand_set = set(neighbors[essential_words[0], :].tolist())
                        for ew in essential_words[1:]:
                            cand_set.intersection_update(neighbors[ew, :].tolist())
                        candidates = np.array(list(cand_set), dtype=np.int32)
                    
                    if len(candidates) == 0:
                        continue
                    
                    # Filter out existing codewords
                    valid_candidates = []
                    for d in candidates:
                        if d not in code_set:
                            valid_candidates.append(d)
                    
                    if not valid_candidates:
                        continue
                    
                    # Pick a random candidate
                    d = rng.choice(valid_candidates)
                    
                    # Perform swap
                    code_set.remove(c)
                    code_set.add(d)
                    improved = True
                    break # Restart deletion pass with new set
                
                if not improved:
                    # No swap found. Try random perturbation?
                    # Or break and restart?
                    # Let's try a random "kick": remove a random codeword, 
                    # fix coverage by adding best greedy words, then reduce.
                    if len(code_set) > 0 and rng.random() < 0.5:
                         # Simple restart is often better
                         break
                    else:
                         # Try again? Infinite loop risk.
                         break

            return code_set

        best_code = None
        best_size = float('inf')
        
        # Run multiple restarts
        for i in range(20):
            if time.time() > deadline - 10:
                break
                
            rng = np.random.default_rng(42 + i)
            
            # Greedy Init
            code = greedy_init(rng)
            
            # Local Search
            code = local_search(code, rng, deadline)
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = code
                
            if best_size <= 486:
                break
        
        if best_code is None:
            # Fallback to a known structure if search fails (though search should work)
            # Tetracode based construction
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            best_code = set()
            import itertools
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=n - 4):
                    word = h + list(tail)
                    val = 0
                    for i, d in enumerate(word):
                        val += d * (q ** i)
                    best_code.add(val)

        final_code = []
        for w in best_code:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            final_code.append(word)
        
        return final_code
    
    else:
        N = q ** n
        deadline = time.time() + 110.0
        powers = [q ** i for i in range(n)]
        neigh_size = 1 + n * (q - 1)
        neighbors = np.zeros((N, neigh_size), dtype=np.int32)
        
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                for delta in range(1, q):
                    new_d = (d + delta) % q
                    diff = new_d - d
                    new_w = w + diff * powers[i]
                    neighbors[w, idx] = new_w
                    idx += 1
        
        rng = np.random.default_rng()
        uncovered = np.ones(N, dtype=np.bool_)
        code = set()
        while np.any(uncovered) and time.time() < deadline:
            gains = uncovered[neighbors].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            w = rng.choice(candidates)
            code.add(w)
            uncovered[neighbors[w, :]] = False
        
        final_code = []
        for w in code:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            final_code.append(word)
        return final_code
