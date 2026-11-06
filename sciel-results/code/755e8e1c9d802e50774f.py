import time
import numpy as np
import random

def construct():
    """
    Construct the largest set of 17-bit words with pairwise Hamming distance >= 6.
    Strategy:
    1. Greedy construction to get a large base code.
    2. Local search using 2-for-1 swaps to escape local maxima and reach > 256 words.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N = 17
    MIN_DIST = 6
    TOTAL_WORDS = 1 << N  # 131072
    
    # Precompute offsets with Hamming weight < MIN_DIST
    # These are the "conflict" masks for a given word
    offsets = []
    for mask in range(TOTAL_WORDS):
        if mask.bit_count() < MIN_DIST:
            offsets.append(mask)
    # offsets[0] is 0, so d(w, w) = 0 < 6, which is correct (self-conflict)
    # Note: We only care about distinct words in the code, but the conflict count 
    # logic works by counting how many codewords are "close" to a candidate.
    # If candidate v is in the code, it has a conflict with itself (offset 0).
    # So conflict_count[v] >= 1 if v is in code.
    # We want conflict_count[v] == 0 to add v.
    
    # 1. Greedy Construction
    # Order words by weight, then value. This tends to pack well.
    # We can vectorize the bit extraction and weight calculation.
    all_words = np.arange(TOTAL_WORDS, dtype=np.uint32)
    
    # Calculate weights
    # Using a lookup table for popcount on 16-bit chunks or just numpy bit operations
    # N=17, so we can do:
    bits = np.zeros((TOTAL_WORDS, N), dtype=np.uint8)
    for i in range(N):
        bits[:, i] = (all_words >> i) & 1
    weights = bits.sum(axis=1)
    
    # Sort indices by weight, then by value
    order = np.lexsort((all_words, weights))
    
    code = []
    # conflict_counts tracks for each word w, how many words in 'code' are at distance < 6 from w.
    # If conflict_counts[w] == 0, w can be added.
    conflict_counts = np.zeros(TOTAL_WORDS, dtype=np.int32)
    
    # Initialize with first word
    # Actually, let's just iterate through the sorted order and add if possible.
    
    for idx in order:
        if time.time() > deadline - 10: # Leave 10s for swaps
            break
        w = all_words[idx]
        if conflict_counts[w] == 0:
            code.append(w)
            # Update conflict counts for neighbors of w
            for mask in offsets:
                v = w ^ mask
                conflict_counts[v] += 1
            
    base_size = len(code)
    
    # 2. Local Search: 2-for-1 Swaps
    # We maintain a set of candidates (words with conflict_count == 0)
    # Initially, after greedy, the code is maximal, so candidates should be empty.
    # But let's build the candidate set properly.
    
    # To make swaps fast, we need to efficiently update candidates.
    # We will maintain a list/set of candidates.
    
    # Since Python sets are slow for large updates, and we have 131k words,
    # we can use a boolean array or just scan if candidate count is small.
    # Given the density, candidates are likely few.
    
    # Let's use a list of candidates.
    # When we remove a word, we decrement counts. If count becomes 0, add to candidates.
    # When we add a word, we increment counts. If count becomes 1, remove from candidates.
    
    # To avoid O(N) removal from list, we can just rebuild the candidate list 
    # if it gets messy, or use a set.
    # Let's use a set for candidates.
    
    candidates = set()
    # Initialize candidates: words with conflict_count == 0
    # This is O(2^17) which is fine.
    for w in range(TOTAL_WORDS):
        if conflict_counts[w] == 0:
            candidates.add(w)
            
    # The code is currently maximal, so candidates should be empty.
    # If not, we can add them directly.
    while candidates:
        if time.time() > deadline - 5:
            break
        v = candidates.pop()
        code.append(v)
        for mask in offsets:
            u = v ^ mask
            conflict_counts[u] += 1
            if conflict_counts[u] == 1:
                if u in candidates:
                    candidates.remove(u)

    best_size = len(code)
    
    # Swap Loop
    # We try to remove one word and add two.
    # To speed up "find pair in candidates", we note that candidates are sparse.
    
    random.seed(42)
    
    # Convert code to list for random access
    # We need to be able to remove and add.
    
    while time.time() < deadline:
        if not code:
            break
            
        # Pick a random word to remove
        rem_idx = random.randint(0, len(code) - 1)
        w_rem = code[rem_idx]
        
        # Remove w_rem
        code.pop(rem_idx)
        
        # Update conflict counts
        for mask in offsets:
            v = w_rem ^ mask
            conflict_counts[v] -= 1
            if conflict_counts[v] == 0:
                candidates.add(v)
            # Note: if conflict_counts[v] was 1 and becomes 0, it's a new candidate.
            # It cannot go negative.
            
        # Try to find 2 compatible candidates
        # Candidates set size is usually small (10-50)
        cand_list = list(candidates)
        if len(cand_list) < 2:
            # Revert
            code.append(w_rem)
            for mask in offsets:
                v = w_rem ^ mask
                conflict_counts[v] += 1
                if conflict_counts[v] == 1:
                    if v in candidates:
                        candidates.remove(v)
            continue
            
        # Find pair v1, v2 in cand_list with d(v1, v2) >= 6
        # Shuffle to randomize search
        random.shuffle(cand_list)
        
        found_pair = False
        v1 = -1
        v2 = -1
        
        # Try to find a pair
        # Optimization: Pick v1, check all others.
        for i in range(len(cand_list)):
            v1 = cand_list[i]
            for j in range(i + 1, len(cand_list)):
                v2 = cand_list[j]
                # Check distance
                if (v1 ^ v2).bit_count() >= MIN_DIST:
                    found_pair = True
                    break
            if found_pair:
                break
        
        if found_pair:
            # Add v1 and v2
            code.append(v1)
            code.append(v2)
            
            # Remove v1, v2 from candidates
            candidates.remove(v1)
            candidates.remove(v2)
            
            # Update conflict counts for v1
            for mask in offsets:
                u = v1 ^ mask
                conflict_counts[u] += 1
                if conflict_counts[u] == 1:
                    if u in candidates:
                        candidates.remove(u)
                        
            # Update conflict counts for v2
            for mask in offsets:
                u = v2 ^ mask
                conflict_counts[u] += 1
                if conflict_counts[u] == 1:
                    if u in candidates:
                        candidates.remove(u)
                        
            best_size = len(code)
        else:
            # Revert
            code.append(w_rem)
            for mask in offsets:
                v = w_rem ^ mask
                conflict_counts[v] += 1
                if conflict_counts[v] == 1:
                    if v in candidates:
                        candidates.remove(v)
                        
    return code

