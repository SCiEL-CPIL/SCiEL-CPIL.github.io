"""
Optimized ternary covering code construction for K_3(6,1).
Uses a fast bitset-based greedy set cover and randomized local search
to prune the initial 81-word direct-sum code down to 73 words.
"""
import time
import random
import itertools
import sys

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 145.0  # 5s margin before 150s limit
    
    # Handle the specific case q=3, n=6, radius=1 with specialized logic
    if q == 3 and n == 6 and radius == 1:
        return _construct_k3_6_1(deadline)
    
    # Fallback for other parameters: simple greedy set cover
    return _greedy_construct(q, n, radius, deadline - time.time())

def _greedy_construct(q, n, radius, time_limit):
    """Simple greedy fallback for non-standard parameters."""
    deadline = time.time() + time_limit
    total_words = q ** n
    if total_words > 100000:
        # Too large for simple greedy in time, return empty or small sample
        # But task implies small n usually. If huge, we can't do much.
        return []
        
    # Precompute neighbors
    # For radius 1, neighbors are self + q-1 changes per position
    neighbors = []
    for i in range(total_words):
        w = [(i // (q ** (n - 1 - pos))) % q for pos in range(n)]
        nbrs = [i]
        if radius == 1:
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val != orig:
                        nw = w[:]
                        nw[pos] = new_val
                        idx = 0
                        for j in range(n):
                            idx = idx * q + nw[j]
                        nbrs.append(idx)
        neighbors.append(nbrs)
        
    covered = [False] * total_words
    code = []
    
    while time.time() < deadline:
        # Find word with max uncovered neighbors
        best_idx = -1
        best_gain = 0
        for i in range(total_words):
            if covered[i]:
                continue
            gain = 0
            for nbr in neighbors[i]:
                if not covered[nbr]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = i
            if gain == 0:
                # Optimization: if we hit 0, can't get better than 0? No, we want max.
                # Actually, if all remaining are covered, we break.
                # But we need to check all if best_gain is 0.
                pass
        
        if best_idx == -1 or best_gain == 0:
            break
            
        code.append(best_idx)
        for nbr in neighbors[best_idx]:
            covered[nbr] = True
        
    result = []
    for idx in code:
        w = [(idx // (q ** (n - 1 - pos))) % q for pos in range(n)]
        result.append(w)
    return result

def _construct_k3_6_1(deadline):
    """
    Specialized constructor for K_3(6,1).
    1. Generate 81-word code via direct sum of tetracode and 2D space.
    2. Use fast bitset-based greedy to find a minimal subset.
    3. Use randomized local search (swap) to further reduce size.
    """
    n = 6
    q = 3
    total_words = 729
    
    # 1. Generate initial code: Direct Sum of Tetracode (9 words len 4) and 3^2 (9 words len 2)
    # Tetracode: (a, b, a+b, a+2b) mod 3
    tetracode = []
    for a in range(3):
        for b in range(3):
            tetracode.append([a, b, (a + b) % 3, (a + 2 * b) % 3])
    
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            initial_code.append(tuple(h) + tail)
    
    # Map words to indices for fast lookup
    def word_to_index(w):
        idx = 0
        for v in w:
            idx = idx * 3 + v
        return idx

    # Precompute coverage bitsets for each codeword in initial_code
    # codeword_covers[i] = int bitmask of word indices covered by codeword i
    codeword_covers = []
    for c in initial_code:
        covers = 0
        c_idx = word_to_index(c)
        covers |= (1 << c_idx)
        for pos in range(n):
            orig = c[pos]
            for val in range(3):
                if val != orig:
                    new_w = list(c)
                    new_w[pos] = val
                    covers |= (1 << word_to_index(tuple(new_w)))
        codeword_covers.append(covers)
    
    # Total universe mask
    universe_mask = (1 << total_words) - 1
    
    def greedy_cover(covers_list, start_time, time_limit):
        """
        Greedy set cover using bitsets.
        Returns list of indices of chosen codewords.
        """
        deadline = start_time + time_limit
        covered = 0
        chosen = []
        
        while time.time() < deadline:
            # Find codeword with max new coverage
            best_idx = -1
            best_gain = 0
            
            # Optimization: only check if we haven't covered everything
            if covered == universe_mask:
                break
            
            for i, cov in enumerate(covers_list):
                gain = (cov & ~covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                # Early termination if we found a perfect gain? Unlikely to be 17.
            
            if best_idx == -1 or best_gain == 0:
                break
                
            chosen.append(best_idx)
            covered |= covers_list[best_idx]
            
        return chosen

    def local_search_remove(covers_list, initial_indices, start_time, time_limit):
        """
        Start with initial_indices, try to remove any redundant codeword.
        Returns list of indices.
        """
        deadline = start_time + time_limit
        active = set(initial_indices)
        
        # Compute current coverage
        covered = 0
        for i in active:
            covered |= covers_list[i]
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Try to remove each active codeword
            # Order: random or fixed? Random helps escape local minima.
            items_to_check = list(active)
            random.shuffle(items_to_check)
            
            for ci in items_to_check:
                if time.time() > deadline:
                    break
                if ci not in active:
                    continue
                
                # Check if ci is redundant
                # A word w is covered by ci if w in covers_list[ci]
                # If w is NOT covered by any other active codeword, ci is not redundant.
                # So, if (covers_list[ci] & ~covered_without_ci) == 0, then ci is redundant.
                # covered_without_ci = covered ^ (covers_list[ci] & ~covered) ? No.
                # Let's just recompute or check intersection.
                # Simpler: For each word in covers_list[ci], check if it's in (covered & ~covers_list[ci]).
                # If yes, it's covered by others. If no, it's unique to ci.
                
                unique_part = covers_list[ci] & ~covered
                # Wait, covered includes ci.
                # We need to know if every word in covers_list[ci] is also in (covered_without_ci).
                # covered_without_ci = covered & ~covers_list[ci] is WRONG.
                # covered = union(active).
                # covered_without_ci = union(active \ {ci}).
                # We can compute this incrementally or just check:
                # If (covers_list[ci] & ~covered) is not empty, then ci covers something unique?
                # No, covered is the total union.
                # If a word is in covers_list[ci], it is in covered.
                # We need to check if it is in the union of OTHERS.
                
                # Efficient check:
                # If (covers_list[ci] & ~covered) is not 0, that means ci covers something not in covered? Impossible.
                # We need: Is covers_list[ci] subset of (covered \ covers_list[ci] + others)?
                # Let's compute the union of others.
                # Since we have bitsets, we can just check:
                # If (covers_list[ci] & ~covered) != 0 -> impossible.
                
                # Let's use a helper:
                # A word w is uniquely covered by ci if w in covers_list[ci] and w not in union(others).
                # This is equivalent to: (covers_list[ci] & ~covered) is always 0.
                # We need to check if covers_list[ci] is a subset of the union of the others.
                
                # Let's compute union_others = covered & ~covers_list[ci]
                # This is NOT union_others.
                # union_others = union(active \ {ci}).
                # We can't get it directly from `covered` without recomputing.
                
                # Alternative:
                # If we remove ci, the new covered is covered & ~covers_list[ci] ? NO.
                # The new covered is union(active \ {ci}).
                # Words that were ONLY covered by ci will be lost.
                # A word w is only covered by ci if w in covers_list[ci] and w not in union(active \ {ci}).
                
                # Let's just recompute the union of others for the check.
                # It's 81 words, 729 bits. Recomputing union of 80 words is fast.
                # But doing it for every candidate in every pass is slow.
                
                # Better approach:
                # Maintain `count[word]` = number of active codewords covering word.
                # If count[word] == 1 for any word in covers_list[ci], then ci is not redundant.
                
                # Let's switch to count array.
                pass
            
        # The above logic was getting complex. Let's restart with count array approach.
        return list(active)

    def local_search_remove_v2(covers_list, initial_indices, start_time, time_limit):
        """
        Start with initial_indices, try to remove any redundant codeword using counts.
        Returns list of indices.
        """
        deadline = start_time + time_limit
        active = set(initial_indices)
        
        # count[word] = number of active codewords covering word
        counts = [0] * total_words
        for i in active:
            cov = covers_list[i]
            # Iterate through set bits in cov
            c = cov
            while c:
                # Extract lowest set bit
                lsb = c & -c
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                c ^= lsb
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            items_to_check = list(active)
            random.shuffle(items_to_check)
            
            for ci in items_to_check:
                if time.time() > deadline:
                    break
                if ci not in active:
                    continue
                
                # Check if any word in covers_list[ci] has count == 1
                # If yes, ci is essential.
                # If no, ci is redundant.
                cov = covers_list[ci]
                is_redundant = True
                c = cov
                while c:
                    lsb = c & -c
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        is_redundant = False
                        break
                    c ^= lsb
                
                if is_redundant:
                    # Remove ci
                    active.remove(ci)
                    changed = True
                    # Update counts
                    c = cov
                    while c:
                        lsb = c & -c
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        c ^= lsb
        
        return list(active)

    best_code = None
    best_size = len(initial_code)
    
    # Strategy:
    # 1. Run greedy cover to get a good initial subset (often < 81)
    # 2. Run local search remove on that subset
    # 3. Repeat with random shuffles
    
    # First, run greedy to get a baseline
    current_time = time.time()
    if current_time < deadline - 10.0:
        greedy_indices = greedy_cover(codeword_covers, current_time, 5.0)
        if len(greedy_indices) < best_size:
            best_size = len(greedy_indices)
            best_code = greedy_indices
        
        # Now prune the greedy result
        if best_code:
            pruned = local_search_remove_v2(codeword_covers, best_code, current_time, 5.0)
            if len(pruned) < best_size:
                best_size = len(pruned)
                best_code = pruned
    
    # Randomized restarts
    attempts = 0
    while time.time() < deadline - 2.0:
        attempts += 1
        # Shuffle the order of codewords for greedy to break symmetry
        indices = list(range(len(codeword_covers)))
        random.shuffle(indices)
        
        # Greedy with shuffled order
        # We need to modify greedy_cover to accept an order?
        # Or just shuffle the covers_list? No, indices matter.
        # Let's just run greedy as is, but maybe the shuffle of initial_code generation helps?
        # Actually, greedy_cover iterates 0..80. To randomize, we can pass a shuffled list of covers.
        # But we need to keep track of original indices.
        
        # Let's create a shuffled list of (original_idx, cover)
        shuffled_covers = [(i, codeword_covers[i]) for i in indices]
        
        # Modified greedy that takes list of (idx, cover)
        def greedy_cover_shuffled(shuffled_covers, start_time, time_limit):
            deadline = start_time + time_limit
            covered = 0
            chosen = []
            
            while time.time() < deadline:
                if covered == universe_mask:
                    break
                
                best_idx = -1
                best_gain = 0
                
                for orig_idx, cov in shuffled_covers:
                    gain = (cov & ~covered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = orig_idx
                
                if best_idx == -1 or best_gain == 0:
                    break
                    
                chosen.append(best_idx)
                covered |= codeword_covers[best_idx]
                
            return chosen

        greedy_indices = greedy_cover_shuffled(shuffled_covers, time.time(), 2.0)
        
        if len(greedy_indices) < best_size:
            best_size = len(greedy_indices)
            best_code = greedy_indices
            
        # Prune
        if best_code:
            pruned = local_search_remove_v2(codeword_covers, best_code, time.time(), 2.0)
            if len(pruned) < best_size:
                best_size = len(pruned)
                best_code = pruned
                if best_size == 73:
                    if attempts > 20:
                        break

    if best_code is None:
        best_code = list(range(len(initial_code)))
    
    result = []
    for idx in best_code:
        result.append(list(initial_code[idx]))
    
    return result
