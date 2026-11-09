import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    N = 17
    MIN_D = 6
    MAX_WORDS = 1 << N
    
    # 1. Precompute weights for all 2^N words
    # Weights[i] is the number of 1s in the binary representation of i
    weights = np.zeros(MAX_WORDS, dtype=np.int32)
    for i in range(1, MAX_WORDS):
        weights[i] = weights[i >> 1] + (i & 1)
        
    # 2. Construct a linear code of size 256 (dimension 8)
    # We use a greedy approach to find 8 linearly independent basis vectors
    # such that the minimum weight of the code is at least 6.
    
    # To speed up the search for basis vectors, we can use a known good basis
    # or a fast heuristic. 
    # The [17, 8, 6] code is the shortened [18, 9, 6] quadratic residue code.
    # We can try to construct it greedily.
    
    basis = []
    current_code = np.array([0], dtype=np.int32)
    
    # We want to find 8 basis vectors.
    # We iterate through candidate vectors. 
    # To ensure we find a valid basis quickly, we can sort candidates by weight 
    # or just scan. Scanning 0..2^17 is 131k iterations.
    # For each candidate, we check if it's valid against the current code.
    # Current code size grows as 1, 2, 4, ..., 128.
    # Total checks: sum_{k=0}^7 2^k = 255.
    # Total operations: 131k * 255 ~ 33M. This might be slow in pure Python loops.
    # Let's use NumPy to check validity.
    
    # We can pre-filter candidates: they must have weight >= 6.
    # And they must not be in the span of the current basis (implied by weight check if 0 is in code).
    
    # Let's try a smarter initialization. 
    # The first vector must have weight >= 6.
    # We can just pick the first one found.
    
    # To optimize the greedy loop:
    # For a candidate g, we need min(weights[g ^ c]) >= 6 for all c in current_code.
    
    # Let's iterate g from 1 to MAX_WORDS.
    # We can break early if we find 8 basis vectors.
    
    # Optimization: 
    # If we fail to find 8 vectors, we might end up with a smaller linear code.
    # But [17, 8, 6] exists.
    
    # Let's try to be smart. 
    # Instead of scanning all g, we can generate random g's or use a specific order.
    # A specific order that works well for QR codes is often related to the field elements.
    # But let's stick to a deterministic scan with NumPy acceleration.
    
    # We will scan g in increasing order.
    # To make it faster, we can skip g if weights[g] < 6.
    
    valid_candidates = np.where(weights >= MIN_D)[0]
    
    for g in valid_candidates:
        if time.time() > deadline - 10: # Leave 10s for nonlinear part
            break
            
        # Check if g can be added
        # g is valid if for all c in current_code, weights[g ^ c] >= MIN_D
        # Since 0 is in current_code, weights[g] >= MIN_D is already true.
        
        # Vectorized check:
        # diffs = g ^ current_code
        # min_wt = np.min(weights[diffs])
        # if min_wt >= MIN_D:
        
        # But current_code is a list/array of ints.
        # XOR with scalar g:
        diffs = g ^ current_code
        min_wt = np.min(weights[diffs])
        
        if min_wt >= MIN_D:
            basis.append(g)
            # Update current_code
            # New code = current_code U (current_code + g)
            current_code = np.concatenate([current_code, current_code ^ g])
            
            if len(basis) == 8:
                break
                
    # Now we have a linear code of size 2^len(basis). 
    # Ideally 256.
    
    # 3. Nonlinear Extension
    # We want to add more words.
    # We maintain a boolean array 'forbidden' of size MAX_WORDS.
    # forbidden[v] = True if v is within distance < MIN_D of any codeword in 'code_set'.
    
    forbidden = np.zeros(MAX_WORDS, dtype=bool)
    
    # Precompute neighbors for each weight < MIN_D
    # neighbors[w] is a list of patterns p such that weights[p] == w
    # Actually, we need all p such that weights[p] < MIN_D.
    # Let's create a list of all "bad" patterns (distance < 6).
    bad_patterns = []
    for p in range(MAX_WORDS):
        if weights[p] < MIN_D:
            bad_patterns.append(p)
            
    # bad_patterns has ~9400 elements.
    
    def add_word_to_code(w, code_list, forbidden_arr):
        # Mark w as forbidden (distance 0)
        # And all neighbors within distance < 6
        # We update forbidden_arr in place
        # Note: w is already in code_list, so we don't add it to code_list here?
        # The caller should handle adding to code_list.
        # This function just updates the forbidden mask.
        
        # We iterate over bad_patterns
        # This is a Python loop over 9400 items. 
        # If we add many words, this adds up.
        # 256 words * 9400 = 2.4M operations. Very fast.
        
        # We can vectorize this update using NumPy if we store bad_patterns as an array
        # forbidden_arr[w ^ bad_patterns] = True
        
        # Let's use the vectorized update
        forbidden_arr[w ^ np.array(bad_patterns, dtype=np.int32)] = True

    # Initialize forbidden mask with the linear code
    # We have current_code array.
    # We can loop through current_code and call add_word_to_code?
    # That would be 256 * 9400 operations.
    # Alternatively, we can just initialize it.
    
    # Actually, let's just loop through the existing linear code words to set the mask.
    for c in current_code:
        forbidden[c ^ np.array(bad_patterns, dtype=np.int32)] = True
        
    # Now find new words
    # We need to find indices i where forbidden[i] is False.
    # We can use np.where(~forbidden)[0]
    
    # However, np.where on a 131k array is fast, but doing it repeatedly might be slow if we do it many times.
    # But we only add a few words (258 - 256 = 2, maybe more).
    
    # Let's iterate.
    # We need to be careful: if we pick a word, we mark it and its neighbors.
    # Then we need to find the next unforbidden word.
    
    # To be efficient, we can maintain a list of available words? 
    # Or just re-scan. Re-scanning 131k bools is ~0.1ms.
    # Even 1000 scans is 100ms. So we can afford to re-scan.
    
    code_set = list(current_code)
    
    while time.time() < deadline:
        # Find first unforbidden word
        # We can use np.argmax(~forbidden) but if all are forbidden, it returns 0.
        # Better:
        avail = np.where(~forbidden)[0]
        if len(avail) == 0:
            break
            
        # Pick the first one (or a random one? First is deterministic)
        new_w = avail[0]
        
        # Add it to code set
        code_set.append(new_w)
        
        # Update forbidden mask
        forbidden[new_w ^ np.array(bad_patterns, dtype=np.int32)] = True
        
    return code_set
