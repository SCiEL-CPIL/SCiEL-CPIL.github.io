import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Parameters are fixed by the task: q=3, n=6, radius=1
    # But we handle general case for safety, though optimization is specific to this size.
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute covers: covers[i] is a list of indices of words covered by word i
    # A word w covers v if Hamming distance(w, v) <= radius
    # For q=3, n=6, radius=1, this is fast to compute.
    
    # Optimization: Only compute for the specific instance if it's small enough
    # 729 words, distance check is O(n). 729*729*6 is ~3M ops. Very fast.
    
    covers = []
    for w in all_words:
        covered_indices = []
        for v in all_words:
            # Hamming distance
            dist = sum(1 for a, b in zip(w, v) if a != b)
            if dist <= radius:
                covered_indices.append(word_to_idx[v])
        covers.append(covered_indices)
        
    # Also precompute reverse index: covered_by[v] is list of word indices covering v
    covered_by = [[] for _ in range(num_words)]
    for i, covs in enumerate(covers):
        for v in covs:
            covered_by[v].append(i)
            
    def is_valid_cover(codeword_indices):
        # Check if union of covers covers all words
        # Using a boolean array for speed
        is_covered = [False] * num_words
        for i in codeword_indices:
            for v in covers[i]:
                is_covered[v] = True
        return all(is_covered)
        
    def greedy_cover(seed):
        random.seed(seed)
        uncovered = set(range(num_words))
        codewords = []
        while uncovered:
            best_idx = -1
            best_count = -1
            # Randomize tie-breaking by shuffling candidates or adding noise
            # For speed, we just pick max. To get diversity, we can randomize the order of evaluation or add small random noise to count.
            # Let's use a simple randomization: pick from top 10 candidates?
            # Or just standard greedy with random restarts.
            
            # To make it faster, we can maintain a list of candidates.
            # But num_words is small.
            for i in range(num_words):
                if i in codewords:
                    continue
                # Count how many uncovered words i covers
                count = 0
                for v in covers[i]:
                    if v in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_idx = i
                elif count == best_count and best_idx != -1:
                    # Random tie break
                    if random.random() < 0.5:
                        best_idx = i
                        
            if best_idx == -1:
                break
            codewords.append(best_idx)
            # Update uncovered
            for v in covers[best_idx]:
                if v in uncovered:
                    uncovered.remove(v)
                    
        return codewords

    def local_search(codewords):
        # codewords is a list of indices
        # Try to remove redundant codewords
        # Try 2-for-1 swaps
        
        improved = True
        while improved:
            improved = False
            
            # 1. Remove redundant codewords
            current_set = set(codewords)
            for i in list(codewords):
                # Check if i is redundant
                # i is redundant if for every v in covers[i], there exists j in current_set - {i} such that v in covers[j]
                # Equivalent: for every v in covers[i], covered_by[v] contains some j != i
                redundant = True
                for v in covers[i]:
                    # Check if any other codeword covers v
                    found = False
                    for j in covered_by[v]:
                        if j != i and j in current_set:
                            found = True
                            break
                    if not found:
                        redundant = False
                        break
                if redundant:
                    current_set.remove(i)
                    codewords = list(current_set)
                    improved = True
            
            # 2. Try 2-for-1 moves
            # For each pair (i, j) in codewords, see if there is a k such that covers[i] U covers[j] is subset of covers[k]
            # If so, replace i, j with k.
            # Optimization: 
            # The set of words covered by i and j is S = covers[i] U covers[j].
            # We need k such that S is subset of covers[k].
            # This means k must cover all words in S.
            # So k must be in the intersection of covered_by[v] for all v in S.
            
            # This check is expensive if done for all pairs.
            # We can limit the number of attempts or do it more smartly.
            # Given small size, let's try it but with a time check.
            
            pairs = list(itertools.combinations(codewords, 2))
            random.shuffle(pairs)
            
            for i, j in pairs:
                if time.time() > deadline - 10:
                    return codewords
                
                # Compute S = covers[i] U covers[j]
                # We need to find k such that covers[k] is a superset of S.
                # This is equivalent to: for all v in S, k in covered_by[v].
                # Let's intersect the lists covered_by[v] for v in S.
                
                # To speed up, we can pick the v in S with the smallest covered_by[v] list and filter.
                # But S is small (<= 26).
                
                # Start with the first word in S
                s_list = covers[i] + [v for v in covers[j] if v not in covers[i]]
                if not s_list:
                    continue
                
                # Find intersection of covered_by[v] for v in s_list
                # Initialize with the smallest list
                min_len = float('inf')
                min_idx = -1
                for v in s_list:
                    if len(covered_by[v]) < min_len:
                        min_len = len(covered_by[v])
                        min_idx = v
                        
                candidates = set(covered_by[s_list[min_idx] if min_idx != -1 else s_list[0]])
                
                for v in s_list:
                    if v == (s_list[min_idx] if min_idx != -1 else s_list[0]):
                        continue
                    candidates &= set(covered_by[v])
                    if not candidates:
                        break
                
                if candidates:
                    # Find a candidate k that is not i or j (or even if it is, it doesn't help reduce size)
                    # Actually if k == i, then covers[j] subset covers[i], so j was redundant. We already checked redundancy.
                    # So k should be new.
                    for k in candidates:
                        if k != i and k != j:
                            # Replace i, j with k
                            new_set = set(codewords)
                            new_set.discard(i)
                            new_set.discard(j)
                            new_set.add(k)
                            # Verify validity (should be valid by construction, but safe to check)
                            # Actually, we must ensure that words covered by i and j are still covered.
                            # Since S subset covers[k], yes.
                            # And words not in S were covered by others, and we didn't change others.
                            codewords = list(new_set)
                            improved = True
                            break
                    if improved:
                        break # Restart the pair loop with new set
                
            if time.time() > deadline - 10:
                return codewords

        return codewords

    best_codewords = None
    best_size = float('inf')
    
    # Run multiple restarts
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        cw = greedy_cover(r)
        cw = local_search(cw)
        
        if len(cw) < best_size:
            best_size = len(cw)
            best_codewords = cw
            print(f"Restart {r}: Found size {best_size}")
            
        # If we hit the lower bound or a very good size, we can stop early?
        # Lower bound is 71. If we find 71, we can stop.
        if best_size <= 71:
            break

    if best_codewords is None:
        # Fallback to a simple construction if search fails or takes too long
        # Use the tetracode construction from seed
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tails = list(itertools.product(range(3), repeat=n - 4))
        best_codewords = [h + list(tail) for h in tetracode for tail in tails]
        
    # Convert indices back to word lists
    return [list(all_words[i]) for i in best_codewords]
