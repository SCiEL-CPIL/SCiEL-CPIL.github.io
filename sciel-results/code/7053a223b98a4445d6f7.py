import random
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    
    # Helper to convert index to tuple of digits
    def index_to_digits(idx):
        digits = [0] * n
        for k in range(n):
            digits[k] = idx % q
            idx //= q
        return digits

    # Helper to convert tuple of digits to index
    def digits_to_index(digits):
        idx = 0
        for k in range(n - 1, -1, -1):
            idx = idx * q + digits[k]
        return idx

    # Precompute all words' digits
    all_digits = [index_to_digits(i) for i in range(total_words)]
    
    # Precompute spheres as bitsets (Python integers)
    # Bit i is set if word i is in the sphere
    spheres = [0] * total_words
    
    for i in range(total_words):
        base_digits = all_digits[i]
        sphere_bits = 0
        # Iterate over all possible words to see if they are in the sphere
        # For small N (7) and Q (3), total_words is 2187.
        # Checking distance for all pairs is 2187 * 2187 * 7 ~ 33M ops.
        # This might be too slow in pure Python for precomputation if done naively.
        # Let's optimize distance check or use a different generation method.
        
        # Alternative: Generate all words within radius explicitly.
        # Number of words in sphere of radius 1 for q=3, n=7 is 1 + 7*2 = 15.
        # Radius is 1. So sphere size is small.
        
        sphere_bits |= (1 << i) # self
        
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == base_digits[pos]:
                        continue
                    new_digits = base_digits[:]
                    new_digits[pos] = val
                    new_idx = digits_to_index(new_digits)
                    sphere_bits |= (1 << new_idx)
                    
        if radius >= 2:
            # Generate all pairs of positions
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    for val1 in range(q):
                        if val1 == base_digits[pos1]:
                            continue
                        for val2 in range(q):
                            if val2 == base_digits[pos2]:
                                continue
                            new_digits = base_digits[:]
                            new_digits[pos1] = val1
                            new_digits[pos2] = val2
                            new_idx = digits_to_index(new_digits)
                            sphere_bits |= (1 << new_idx)
                            
        spheres[i] = sphere_bits

    best_codewords = None
    best_size = float('inf')
    
    # Run multiple greedy instances with different random seeds for tie-breaking
    num_runs = 50 # Bitsets are fast, so we can do more runs
    
    for seed in range(num_runs):
        rng = random.Random(seed)
        uncovered = (1 << total_words) - 1 # All bits set
        codewords = []
        
        while uncovered:
            best_idx = -1
            best_count = -1
            candidates = []
            
            # Find the word with the maximum number of newly covered words
            # We can restrict candidates to uncovered words? No, the center can be any word.
            # But if a word is already covered, it's less likely to be the best center?
            # Actually, the "set" is the sphere. We pick the set with max intersection with uncovered.
            
            for i in range(total_words):
                # If the sphere of i is completely covered, count is 0
                # Intersection: spheres[i] & uncovered
                common = spheres[i] & uncovered
                if common:
                    count = common.bit_count()
                    if count > best_count:
                        best_count = count
                        candidates = [i]
                    elif count == best_count:
                        candidates.append(i)
            
            if best_count == 0:
                break # Should not happen if uncovered is not 0
            
            # Randomly select from candidates with max coverage
            best_idx = rng.choice(candidates)
            
            codewords.append(best_idx)
            # Update uncovered: remove all words in the sphere of best_idx
            uncovered &= ~spheres[best_idx]
            
        # Post-processing: Remove redundant codewords
        # A codeword c is redundant if for all w in spheres[c], w is covered by other codewords.
        # We can check this by removing c and seeing if the union of remaining spheres covers everything.
        # To make it fast, we can maintain a bitset of covered words for the current set.
        # However, re-computing the full coverage from scratch for each removal attempt is O(K * 2^N).
        # K ~ 180, 2^N ~ 2187. 180 * 2187 ~ 400k ops. Very fast.
        
        changed = True
        while changed:
            changed = False
            # Shuffle to randomize removal order
            indices = list(range(len(codewords)))
            rng.shuffle(indices)
            
            for i in indices:
                if i >= len(codewords):
                    continue
                c = codewords[i]
                
                # Try removing c
                # Build coverage bitset for codewords without c
                # Optimization: Keep a running coverage bitset?
                # If we keep a running bitset `current_cover`, removing c is hard because we don't know if c contributed uniquely.
                # Easier: Just check if `spheres[c]` is a subset of the union of other spheres.
                # Let's compute the union of all other spheres.
                
                # To avoid O(K^2) precomputation per removal, we can do this:
                # If we maintain a count of how many times each word is covered, we can check if any word in spheres[c] has count 1.
                # Let's maintain cover_counts.
                
                # We didn't maintain cover_counts in the main loop. Let's add it or recompute.
                # Recomputing cover_counts for the current set is O(K * SphereSize).
                # SphereSize is 15. K is 180. 180 * 15 = 2700 ops.
                # Then for each removal candidate, we check its sphere (15 words) for count 1.
                # This is very fast.
                
                # Let's recompute cover_counts for the current codewords list
                cover_counts = [0] * total_words
                for cw in codewords:
                    # Iterate through bits in spheres[cw]
                    s = spheres[cw]
                    while s:
                        # Get lowest set bit
                        lsb = s & -s
                        idx = lsb.bit_length() - 1
                        cover_counts[idx] += 1
                        s ^= lsb
                
                # Now check if codeword c is redundant
                redundant = True
                s = spheres[c]
                while s:
                    lsb = s & -s
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 1:
                        redundant = False
                        break
                    s ^= lsb
                    
                if redundant:
                    codewords.pop(i)
                    changed = True
                    break # Restart loop with new list
        
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_codewords = codewords[:]
            
    # Convert indices back to list of lists
    result = []
    for idx in best_codewords:
        result.append(index_to_digits(idx))
        
    return result
