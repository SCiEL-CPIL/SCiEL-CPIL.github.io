import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        start_time = time.time()
        deadline = start_time + 118.0  # Safety margin before 120s limit
        N = q ** n  # 6561
        powers = [q ** i for i in range(n)]
        
        # Precompute neighbors
        neigh_size = 1 + n * (q - 1)  # 17
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
            """Compute coverage count for each word given a set of codewords."""
            cov = np.zeros(N, dtype=np.int32)
            for c in code_set:
                cov[neighbors[c, :]] += 1
            return cov
        
        def greedy_init(rng):
            """Greedy initialization: repeatedly pick word covering most uncovered."""
            uncovered = np.ones(N, dtype=np.bool_)
            code = set()
            
            while np.any(uncovered) and time.time() < deadline - 3:
                # Compute gain for each word: number of uncovered neighbors
                gains = np.zeros(N, dtype=np.int32)
                for i in range(neigh_size):
                    gains += uncovered[neighbors[:, i]].astype(np.int32)
                
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                
                # Pick randomly among words with max gain
                candidates = np.where(gains == max_gain)[0]
                w = int(rng.choice(candidates))
                code.add(w)
                uncovered[neighbors[w, :]] = False
            
            return code
        
        def local_search(code_set, rng):
            """Local search: delete redundant, then swap to enable more deletions."""
            code_list = list(code_set)
            
            while time.time() < deadline - 2:
                # Compute coverage
                cov = compute_coverage(code_set)
                
                # Deletion pass: remove words where all neighbors have coverage > 1
                redundant = []
                for c in code_set:
                    if np.min(cov[neighbors[c, :]]) > 1:
                        redundant.append(c)
                
                if redundant:
                    for c in redundant:
                        code_set.remove(c)
                    continue  # Recompute coverage
                
                # Swap pass: try to swap a word to enable further deletion
                improved = False
                c_list = list(code_set)
                rng.shuffle(c_list)
                
                for c in c_list:
                    if time.time() > deadline - 2:
                        break
                    
                    # Find words covered only by c
                    c_neighbors = neighbors[c, :]
                    essential_mask = (cov[c_neighbors] == 1)
                    essential_words = c_neighbors[essential_mask]
                    
                    if len(essential_words) == 0:
                        continue
                    
                    # Find candidates that cover all essential words
                    if len(essential_words) == 1:
                        candidates = neighbors[essential_words[0], :]
                    else:
                        # Intersection of neighbor sets
                        cand_set = set(neighbors[essential_words[0], :].tolist())
                        for ew in essential_words[1:]:
                            cand_set.intersection_update(neighbors[ew, :].tolist())
                        candidates = np.array(list(cand_set), dtype=np.int32)
                    
                    if len(candidates) == 0:
                        continue
                    
                    # Filter out existing codewords
                    valid_candidates = [d for d in candidates if d not in code_set]
                    if not valid_candidates:
                        continue
                    
                    # Try each candidate, prefer those that maximize future deletions
                    best_d = None
                    best_gain = -1
                    
                    for d in valid_candidates[:10]:  # Limit to speed up
                        # Simulate swap
                        test_set = set(code_set)
                        test_set.remove(c)
                        test_set.add(d)
                        
                        test_cov = compute_coverage(test_set)
                        
                        # Count how many words become redundant
                        gain = 0
                        for tc in test_set:
                            if np.min(test_cov[neighbors[tc, :]]) > 1:
                                gain += 1
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_d = d
                    
                    if best_d is not None and best_gain > 0:
                        code_set.remove(c)
                        code_set.add(best_d)
                        improved = True
                        break
                
                if not improved:
                    # Try random perturbation: remove random word, fix with greedy
                    if len(code_set) > 0 and rng.random() < 0.3:
                        # Remove a random word
                        c = int(rng.choice(list(code_set)))
                        code_set.remove(c)
                        
                        # Fix coverage by adding best greedy words
                        uncovered = np.zeros(N, dtype=np.bool_)
                        cov = compute_coverage(code_set)
                        uncovered[cov == 0] = True
                        
                        while np.any(uncovered) and time.time() < deadline - 2:
                            gains = np.zeros(N, dtype=np.int32)
                            for i in range(neigh_size):
                                gains += uncovered[neighbors[:, i]].astype(np.int32)
                            
                            max_gain = np.max(gains)
                            if max_gain == 0:
                                break
                            
                            candidates = np.where(gains == max_gain)[0]
                            w = int(rng.choice(candidates))
                            code_set.add(w)
                            uncovered[neighbors[w, :]] = False
                        
                        # Continue local search
                    else:
                        break  # No improvement, stop this iteration
            
            return code_set
        
        best_code = None
        best_size = float('inf')
        
        # Run multiple restarts with different seeds
        for i in range(30):
            if time.time() > deadline - 15:
                break
            
            rng = np.random.default_rng(1000 + i * 7)
            
            # Greedy initialization
            code = greedy_init(rng)
            current_size = len(code)
            
            # Local search
            code = local_search(code, rng)
            current_size = len(code)
            
            if current_size < best_size:
                best_size = current_size
                best_code = set(code)
                
                # Early termination if we hit known optimum
                if best_size <= 486:
                    break
        
        # Final fallback: if search didn't find good solution, use tetracode-based construction
        if best_code is None or best_size > 500:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            best_code = set()
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=n - 4):
                    word = h + list(tail)
                    val = 0
                    for i, d in enumerate(word):
                        val += d * (q ** i)
                    best_code.add(val)
        
        # Convert to list of lists
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
        # Generic case for other parameters
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
            gains = np.zeros(N, dtype=np.int32)
            for i in range(neigh_size):
                gains += uncovered[neighbors[:, i]].astype(np.int32)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            w = int(rng.choice(candidates))
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
