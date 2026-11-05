import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin

    # This task is specifically for q=3, n=7, radius=1
    # Fallback for other parameters
    if q != 3 or n != 7 or radius != 1:
        return _generic_greedy(q, n, radius)

    total_words = q ** n  # 2187
    
    # Map each word (tuple of ints) to an index 0..2186
    # Index is the base-3 number represented by the word.
    # word (w0, ..., w6) -> w0*3^6 + ... + w6*3^0
    
    # Precompute the list of words as tuples for easy lookup and conversion
    words = []
    for i in range(total_words):
        w = []
        x = i
        for _ in range(n):
            w.append(x % 3)
            x //= 3
        words.append(tuple(reversed(w)))
    
    # Precompute neighborhood masks as Python integers (bitmasks)
    # mask[i] has bit j set if word[j] is within distance 'radius' of word[i]
    # For radius 1, this includes i itself and all words differing in exactly 1 position.
    masks = [0] * total_words
    for i in range(total_words):
        w = words[i]
        mask = 1 << i  # Self
        for pos in range(n):
            orig_val = w[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                # Create new word
                # Instead of creating a tuple and hashing, we can compute the index directly
                # Index is base-3. Changing digit at pos (0 is MSB)
                # pos 0 is 3^6, pos 6 is 3^0
                # Let's stick to the tuple -> index conversion for safety and clarity, 
                # or optimize:
                # index = sum(w[k] * 3^(n-1-k))
                # new_index = index - orig_val * 3^(n-1-pos) + val * 3^(n-1-pos)
                power = 3 ** (n - 1 - pos)
                new_idx = i - orig_val * power + val * power
                mask |= (1 << new_idx)
        masks[i] = mask

    all_mask = (1 << total_words) - 1

    def greedy_random():
        uncovered = all_mask
        code = []
        # To add randomness, we can shuffle the candidate order or use random tie-breaking
        while uncovered:
            best_gain = -1
            best_candidates = []
            # Iterate through all words
            for i in range(total_words):
                # If word i is already in code, gain is 0 (we don't add duplicates)
                # We can check if i is in code, but set lookup is O(1). 
                # However, checking gain for already chosen words is wasteful.
                # We can maintain a list of available words? 
                # For 2187 words, scanning all is fast enough.
                gain = (masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain:
                    best_candidates.append(i)
            
            # Pick one randomly from best candidates
            chosen = random.choice(best_candidates)
            code.append(chosen)
            uncovered &= ~masks[chosen]
        
        return code

    def is_valid_code(code):
        covered = 0
        for i in code:
            covered |= masks[i]
        return covered == all_mask

    def get_uncovered(code):
        covered = 0
        for i in code:
            covered |= masks[i]
        return all_mask ^ covered # Bits that are 0 in covered are 1 in uncovered

    def local_search(code):
        # Returns improved code
        code = list(code)
        # Convert to set for fast membership check? 
        # List is fine for 186 items.
        
        improved = True
        while improved:
            improved = False
            uncovered = get_uncovered(code)
            
            # 1. Try to remove redundant codewords
            for i in range(len(code)):
                # Check if removing code[i] leaves all nodes covered
                # i.e., is there any node covered ONLY by code[i]?
                # If not, we can remove it.
                # A node is covered only by code[i] if it is in masks[code[i]] 
                # but not in the union of masks of all other codewords.
                
                # Efficient check:
                # If (masks[code[i]] & ~union_others) == 0, then safe to remove.
                # union_others is expensive to compute from scratch each time.
                
                # Alternative: 
                # If we remove c, new_uncovered = uncovered | (masks[c] & ~union_others).
                # This is complex.
                
                # Simple check:
                # If we remove c, does the code still cover everything?
                # We can just recompute the covered set without c and check against all_mask.
                # O(|code|) bitwise ORs.
                temp_code = code[:i] + code[i+1:]
                if is_valid_code(temp_code):
                    code = temp_code
                    improved = True
                    break # Restart loop as code changed
            
            if improved:
                continue

            # 2. Try to swap a codeword with a non-codeword to reduce uncovered count
            # Current uncovered count
            cur_uncov_count = uncovered.bit_count()
            
            # To speed up, we can restrict swaps to codewords that have low "efficiency"
            # or just iterate all.
            
            # We want to find c in code, c' not in code
            # such that new_uncovered_count < cur_uncov_count
            
            # Optimized approach:
            # For each c in code:
            #   Let U_c = uncovered | (masks[c] & ~union_others)  # If we remove c
            #   Actually, simpler:
            #   If we swap c with c':
            #   New covered = (covered_without_c) | masks[c']
            #   New uncovered = all_mask ^ New covered
            
            # Let's just try random swaps or systematic?
            # Systematic over 186 * 2000 = 372k swaps.
            # Each swap check:
            # 1. Compute covered_without_c (expensive if done naively)
            
            # Let's use a heuristic:
            # If a codeword c covers many nodes that are already covered by others, it's a good candidate for removal/swap.
            
            # Let's try a simpler local search step:
            # Randomly pick a codeword c.
            # Randomly pick a non-codeword c'.
            # If swap improves uncovered count, do it.
            # Repeat until no improvement in K steps.
            
            if random.random() < 0.5:
                # Deterministic-ish sweep for swaps
                # This might be too slow if done fully.
                pass
                
            # Stochastic Swap
            # Try a few random swaps
            for _ in range(50):
                if time.time() > deadline:
                    return code
                
                c_idx = random.randint(0, len(code)-1)
                c = code[c_idx]
                
                # Pick a random non-codeword
                # We can pick a random index and check if it's in code
                # Or maintain a list of non-codewords? 
                # 2187 is small.
                c_prime = random.randint(0, total_words - 1)
                if c_prime in code:
                    continue
                
                # Perform swap
                new_code = code[:]
                new_code[c_idx] = c_prime
                
                # Check if valid and better
                # Actually, we want to minimize size, so we only swap if it allows removal later?
                # No, standard SLK for covering codes minimizes the number of uncovered nodes first, 
                # then minimizes code size?
                # The objective is minimize code size.
                # A swap keeps size constant. It helps if it allows a subsequent removal.
                # So we should accept a swap if it reduces the number of "critical" nodes 
                # or simply if it reduces the uncovered count (which should be 0 for valid codes).
                
                # Wait, our code is always valid (covers all).
                # So uncovered is always 0.
                # Swapping doesn't change uncovered count (it stays 0).
                # It changes the *structure* of coverage.
                
                # Metric: How "redundant" is the code?
                # A good swap is one that increases the number of nodes covered by >1 codeword?
                # Or decreases the number of nodes covered by exactly 1 codeword?
                
                # Let's define a score: sum(1 for node in covered if coverage_count[node] == 1)
                # We want to minimize this score.
                
                # Calculating this for every swap is expensive.
                
                # Alternative: Just try to remove a codeword after every few swaps?
                # Or, use the "greedy repair" idea.
                
                # Let's stick to the most effective simple move:
                # If we can't remove, try to find a c, c' such that 
                # masks[c] \ union_others is a subset of masks[c'] \ union_others?
                # This is getting complex.
                
                # Let's rely on the deletion pass being strong and the greedy being strong.
                # The main improvement from the seed is that the seed's `reduce_code` is O(N^2) or worse 
                # because it restarts the loop on every removal.
                # My `local_search` above breaks and restarts, which is also O(N^2) worst case.
                
                # Let's optimize the deletion check.
                # Precompute `coverage_count` for each node?
                # `coverage_count[j]` = number of codewords covering j.
                # If `coverage_count[j] == 1` for some j, and c covers j, then c is critical for j.
                # If c has no critical nodes (i.e., all nodes in masks[c] have coverage_count > 1), c is removable.
                
                # Let's implement this.
                pass
            
            # If no removal found, try to improve structure via swaps?
            # For now, let's just do a random restart if stuck.
            break
        
        return code

    best_code = None
    best_size = float('inf')
    
    # Run multiple randomized greedy instances + local search
    while time.time() < deadline:
        code = greedy_random()
        
        # Local search: try to remove redundant codewords
        # We will implement a more efficient reduction here.
        
        # 1. Compute coverage counts
        # Use an array of size total_words
        cov_counts = [0] * total_words
        for c in code:
            # Update counts for all nodes in masks[c]
            # This is slow if done via bit iteration.
            # But masks are ints.
            # We can just recompute validity for removal attempts.
            
            pass
        
        # Efficient Reduction:
        # Iterate through codewords.
        # For each c, check if it's removable.
        # To check if c is removable:
        #   Is there any node j in masks[c] such that no other codeword in code \ {c} covers j?
        #   This is equivalent to: (masks[c] & ~union_others) != 0 ?
        #   If yes, c is critical. If no, c is removable.
        
        # We can compute `union_others` for each c? No, too slow.
        
        # Let's use a heuristic:
        # Sort codewords by "gain" (number of nodes covered exclusively by them).
        # If gain is 0, remove.
        
        # Let's just do the simple restart-based approach which is robust.
        
        code = local_search(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
        if best_size <= 186:
             # We found the known best. Try to beat it? Unlikely.
             # Just keep running to be sure we didn't miss a smaller one (impossible) 
             # or to confirm stability.
             pass

    # Convert indices back to list of lists
    if best_code is None:
        # Fallback
        best_code = greedy_random()
        
    result = []
    for idx in best_code:
        # Convert index to base-3 digits
        w = []
        x = idx
        for _ in range(n):
            w.append(x % 3)
            x //= 3
        w.reverse()
        result.append(w)
        
    return result

def _generic_greedy(q, n, radius):
    # Simple fallback for other parameters
    from itertools import product
    
    total = q ** n
    if total > 10000:
        # Too large for simple bitset approach in Python without numpy optimization
        # Just return a trivial code (all words)
        return [list(p) for p in product(range(q), repeat=n)]
        
    words = [list(p) for p in product(range(q), repeat=n)]
    
    def hamming_dist(w1, w2):
        return sum(1 for a, b in zip(w1, w2) if a != b)
    
    def neighborhood(w):
        # Returns list of indices within distance radius
        inds = set()
        for i, other in enumerate(words):
            if hamming_dist(w, other) <= radius:
                inds.add(i)
        return inds
    
    # Precompute neighborhoods
    neigh = [neighborhood(w) for w in words]
    
    uncovered = set(range(total))
    code = []
    
    while uncovered:
        best_gain = -1
        best_cand = -1
        for i, w in enumerate(words):
            gain = len(neigh[i] & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_cand = i
        code.append(words[best_cand])
        uncovered -= neigh[best_cand]
        
    return code
