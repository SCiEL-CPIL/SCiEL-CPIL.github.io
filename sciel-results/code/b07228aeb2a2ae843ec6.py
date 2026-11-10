import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Handle general case fallback if not the specific task
    if not (q == 3 and n == 8 and radius == 1):
        # Simple greedy fallback for robustness
        N = q ** n
        if N > 100000:
            # Too large, return all words (trivial upper bound)
            return [[i % q for i in range(n)] for i in range(N)]
        
        powers = [q ** i for i in range(n)]
        neighbors = []
        for w in range(N):
            ball = {w}
            digits = [0] * n
            temp = w
            for i in range(n):
                digits[i] = temp % q
                temp //= q
            for pos in range(n):
                original = digits[pos]
                for val in range(q):
                    if val == original:
                        continue
                    digits[pos] = val
                    new_w = 0
                    mult = 1
                    for i in range(n):
                        new_w += digits[i] * mult
                        mult *= q
                    ball.add(new_w)
                digits[pos] = original
            neighbors.append(list(ball))
        
        uncovered = set(range(N))
        code = []
        while uncovered and time.time() < deadline:
            best_cand = -1
            best_count = -1
            for c in uncovered:
                count = sum(1 for nb in neighbors[c] if nb in uncovered)
                if count > best_count:
                    best_count = count
                    best_cand = c
            if best_cand == -1:
                break
            code.append(best_cand)
            for nb in neighbors[best_cand]:
                if nb in uncovered:
                    uncovered.remove(nb)
        if uncovered:
            code.extend(uncovered)
            
        result = []
        for w in code:
            digits = []
            temp = w
            for _ in range(n):
                digits.append(temp % q)
                temp //= q
            result.append(digits)
        return result

    # Specific implementation for q=3, n=8, radius=1
    N = q ** n  # 6561
    powers = [q ** i for i in range(n)]
    
    # Precompute neighbors: neighbors[w] is a list of words within distance 1
    # Size is 1 + n*(q-1) = 1 + 8*2 = 17
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
        """Compute coverage count for each word given a set of codewords."""
        cov = np.zeros(N, dtype=np.int32)
        for c in code_set:
            cov[neighbors[c, :]] += 1
        return cov

    def greedy_init(rng):
        """Greedy initialization: pick codeword covering most uncovered words."""
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
        """Local search: delete redundant codewords, then swap to enable further deletions."""
        code_list = list(code_set)
        code_set = set(code_list)
        
        while time.time() < deadline - 2:
            # 1. Compute Coverage
            cov = compute_coverage(code_set)
            
            # 2. Deletion Pass
            # A codeword c is redundant if min(cov[neighbors[c]]) > 1
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
            c_list = list(code_set)
            rng.shuffle(c_list)
            
            for c in c_list:
                if time.time() > deadline - 2:
                    break
                
                # Essential words: covered ONLY by c
                c_neighbors = neighbors[c, :]
                essential_mask = (cov[c_neighbors] == 1)
                essential_words = c_neighbors[essential_mask]
                
                if len(essential_words) == 0:
                    continue
                
                # We need a d such that d covers all essential_words
                # d must be in the intersection of N(w) for all w in essential_words
                if len(essential_words) == 1:
                    ew = essential_words[0]
                    candidates = neighbors[ew, :]
                else:
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
                # No swap found. Try random perturbation or break
                if len(code_set) > 0 and rng.random() < 0.5:
                    break
                else:
                    break

        return code_set

    best_code = None
    best_size = float('inf')
    
    # Run multiple restarts
    for i in range(30):
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
        
        # If we have time, try a second pass of local search on the best code found so far
        if time.time() < deadline - 15 and i > 0:
            code2 = local_search(best_code.copy(), rng, deadline)
            if len(code2) < best_size:
                best_size = len(code2)
                best_code = code2

    if best_code is None:
        # Fallback to a known structure if search fails
        # Tetracode based construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best_code = set()
        for c1 in tetracode:
            for c2 in tetracode:
                best_code.add(int("".join(map(str, c1 + c2)), 3))

    # Convert best_code to list of lists of integers
    result = []
    for w in best_code:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        result.append(digits)
    
    return result
