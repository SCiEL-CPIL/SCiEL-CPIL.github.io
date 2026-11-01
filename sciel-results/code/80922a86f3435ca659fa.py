import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Specific case: q=3, n=8, radius=1
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other cases
        total = q ** n
        if total > 100000:
            return []
        # Simple greedy for small general cases
        all_words = np.array(list(np.ndindex(*([q] * n))), dtype=np.int32)
        uncovered = set(all_words)
        codewords = []
        while uncovered and time.time() < deadline:
            # Pick word that covers most uncovered
            best = None
            best_count = -1
            for w in uncovered:
                count = 0
                for i in range(n):
                    for v in range(q):
                        nw = list(w)
                        if nw[i] != v:
                            nw[i] = v
                            if tuple(nw) in uncovered:
                                count += 1
                if count > best_count:
                    best_count = count
                    best = w
            if best is None:
                break
            codewords.append(list(best))
            # Remove all covered
            to_remove = set()
            for i in range(n):
                for v in range(q):
                    nw = list(best)
                    nw[i] = v
                    to_remove.add(tuple(nw))
            uncovered -= to_remove
        return codewords

    total_words = 3 ** 8
    num_coords = 8
    alphabet = 3

    # Precompute neighbors using vectorized approach
    # Each word has 1 + 8*2 = 17 neighbors (itself + 2 changes per position)
    all_indices = np.arange(total_words, dtype=np.int32)
    
    # Convert index to digits
    def idx_to_digits(idx):
        digits = np.zeros(num_coords, dtype=np.int32)
        for i in range(num_coords - 1, -1, -1):
            digits[i] = idx % alphabet
            idx //= alphabet
        return digits

    # Precompute neighbor lists as numpy arrays
    neighbors = []
    for i in range(total_words):
        digits = idx_to_digits(i)
        nbs = [i]
        for pos in range(num_coords):
            orig_val = digits[pos]
            for val in range(alphabet):
                if val == orig_val:
                    continue
                new_digits = digits.copy()
                new_digits[pos] = val
                nbs.append(int(np.dot(new_digits, np.array([3**7, 3**6, 3**5, 3**4, 3**3, 3**2, 3**1, 3**0])))
                )
        neighbors.append(np.array(nbs, dtype=np.int32))

    # Greedy Construction with randomized tie-breaking
    def greedy_construct():
        uncovered = np.ones(total_words, dtype=np.bool_)
        in_code = np.zeros(total_words, dtype=np.bool_)
        codewords = []
        
        # Initial counts: each word covers 17 others
        counts = np.full(total_words, 17, dtype=np.int32)
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
                
            # Find max count with random tie-breaking
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            best_cw = candidates[random.randint(0, len(candidates) - 1)]
            
            if counts[best_cw] == 0:
                break
            
            in_code[best_cw] = True
            codewords.append(best_cw)
            
            for nb in neighbors[best_cw]:
                if uncovered[nb]:
                    uncovered[nb] = False
                    for cw in neighbors[nb]:
                        counts[cw] -= 1

        return codewords

    # Pruning
    def prune_once(code_set):
        if not code_set:
            return code_set
        
        cov_count = np.zeros(total_words, dtype=np.int32)
        for c in code_set:
            for nb in neighbors[c]:
                cov_count[nb] += 1
                
        removable = []
        for c in code_set:
            is_removable = True
            for nb in neighbors[c]:
                if cov_count[nb] <= 1:
                    is_removable = False
                    break
            if is_removable:
                removable.append(c)
                
        if removable:
            for c in removable:
                code_set.remove(c)
            return prune_once(code_set)
        else:
            return code_set

    # Local Search: 1-swaps and 2-swaps
    def local_search(code_set):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            c_list = list(code_set)
            random.shuffle(c_list)
            
            # Compute coverage counts
            cov_count = np.zeros(total_words, dtype=np.int32)
            for c in code_set:
                for nb in neighbors[c]:
                    cov_count[nb] += 1
            
            # Try 1-swaps: remove c1, add d
            for c1 in c_list:
                if time.time() > deadline:
                    return code_set
                if c1 not in code_set:
                    continue
                    
                # Find words that would become uncovered if c1 removed
                u_words = []
                for w in neighbors[c1]:
                    if cov_count[w] <= 1:
                        u_words.append(w)
                
                if not u_words:
                    # c1 is redundant
                    code_set.remove(c1)
                    improved = True
                    break
                
                # Find d that covers all u_words
                # d must be within distance 1 of each w in u_words
                # So d is in intersection of N(w) for all w in u_words
                # And d != c1, d not in code_set (or we can add it)
                
                # Candidate d's: neighbors of u_words[0]
                candidates = set(neighbors[u_words[0]].tolist())
                candidates.discard(c1)
                
                for d in candidates:
                    # Check if d covers all u_words
                    d_neighbors = set(neighbors[d].tolist())
                    if all(w in d_neighbors for w in u_words):
                        # Check if adding d and removing c1 is valid
                        # We need to ensure no other word becomes uncovered
                        # Words that were only covered by c1 and not by d
                        # Actually, we just need: after removing c1 and adding d, all words still covered
                        # Words covered by c1 but not by d: these need other coverage
                        # Words covered by d but not by c1: no problem
                        # Words covered by both: no problem
                        # Words covered by neither: no problem
                        
                        # The critical words are those in N(c1) \ N(d)
                        # For each such word w, cov_count[w] - 1 must be >= 1, i.e., cov_count[w] >= 2
                        c1_neighbors = set(neighbors[c1].tolist())
                        d_neighbors_set = set(neighbors[d].tolist())
                        
                        valid = True
                        for w in c1_neighbors - d_neighbors_set:
                            if cov_count[w] <= 1:
                                valid = False
                                break
                        
                        if valid:
                            code_set.remove(c1)
                            code_set.add(d)
                            improved = True
                            break
                
                if improved:
                    break
            
            if improved:
                continue
            
            # Try 2-removals: remove c1, c2
            for i in range(len(c_list)):
                if time.time() > deadline:
                    return code_set
                c1 = c_list[i]
                if c1 not in code_set:
                    continue
                
                for j in range(i + 1, len(c_list)):
                    c2 = c_list[j]
                    if c2 not in code_set:
                        continue
                    
                    # Check if removing both c1 and c2 leaves any uncovered words
                    # A word w is uncovered if cov_count[w] - (1 if w in N(c1)) - (1 if w in N(c2)) <= 0
                    # and w was covered (cov_count[w] >= 1)
                    
                    c1_nbs = set(neighbors[c1].tolist())
                    c2_nbs = set(neighbors[c2].tolist())
                    union_nbs = c1_nbs | c2_nbs
                    
                    valid = True
                    for w in union_nbs:
                        rem = cov_count[w]
                        if w in c1_nbs:
                            rem -= 1
                        if w in c2_nbs:
                            rem -= 1
                        if rem <= 0:
                            valid = False
                            break
                    
                    if valid:
                        code_set.remove(c1)
                        code_set.remove(c2)
                        improved = True
                        break
                
                if improved:
                    break
            
            # Try 2-swaps: remove c1, c2, add d1, d2
            # This is expensive, skip for now or do limited search
            if not improved and time.time() < deadline - 10:
                # Try to find a pair of removals and a pair of additions
                # Simplified: try removing one and adding two? Or just do more 1-swaps
                pass

        return code_set

    # Main: try multiple random restarts of greedy + local search
    best_code = None
    best_size = float('inf')
    
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        codewords = greedy_construct()
        code_set = set(codewords)
        code_set = prune_once(code_set)
        code_set = local_search(code_set)
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = code_set
        
        if time.time() > deadline - 30:
            break

    if best_code is None:
        # Fallback: just do one greedy
        codewords = greedy_construct()
        best_code = set(codewords)
        best_code = prune_once(best_code)

    # Final conversion
    result = [idx_to_digits(c).tolist() for c in best_code]
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)
    return unique_result
