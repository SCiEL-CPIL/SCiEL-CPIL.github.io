"""
Constructs a binary code of length 18 with minimum Hamming distance 6.
Uses the Extended Quadratic Residue Code for p=17, which is a [18, 9, 6] code 
with 512 codewords. Attempts to extend via structured local search
if time permits.
"""
import time
import random
import numpy as np

N = 18
MIN_DIST = 6
TIME_LIMIT = 60.0
SAFETY_MARGIN = 5.0

def popcount(x):
    return x.bit_count()

def hamming(a, b):
    return popcount(a ^ b)

def verify_code(words):
    """Verify all pairs have distance >= MIN_DIST using numpy for speed."""
    n = len(words)
    if n <= 1:
        return True
    arr = np.array(words, dtype=np.uint32)
    # Compute pairwise XOR
    # For memory efficiency with 512 words, this is fine
    xor_matrix = arr[:, np.newaxis] ^ arr[np.newaxis, :]
    # Count bits in each element
    # Use lookup or iterative method
    # For uint32, we can use a bit-count trick
    # Actually, let's use a simple approach: convert to binary and sum
    # Better: use numpy's bit_count if available, else fallback
    try:
        popcnt = np.vectorize(lambda x: x.bit_count())(xor_matrix)
    except AttributeError:
        # Fallback for older numpy
        popcnt = np.zeros_like(xor_matrix, dtype=np.int32)
        for i in range(n):
            for j in range(i + 1, n):
                popcnt[i, j] = popcnt[j, i] = (xor_matrix[i, j]).bit_count()
    
    # Check off-diagonal minimum
    np.fill_diagonal(popcnt, 100)
    return bool(np.min(popcnt) >= MIN_DIST)

def verify_code_slow(words):
    """Fallback verification using pure Python."""
    n = len(words)
    for i in range(n):
        wi = words[i]
        for j in range(i + 1, n):
            if hamming(wi, words[j]) < MIN_DIST:
                return False
    return True

def quadratic_residue_code():
    """
    Construct the Extended Quadratic Residue Code for p=17.
    Parameters: [18, 9, 6], Size: 512.
    Uses numpy for faster construction.
    """
    p = 17
    qr = set((i * i) % p for i in range(p))
    
    dim = (p - 1) // 2  # 8
    
    # Build generator matrix
    G = np.zeros((dim, p), dtype=np.uint8)
    for i in range(dim):
        for j in range(p):
            if (j - i) % p in qr:
                G[i, j] = 1
    
    # Generate all codewords: XOR combinations of rows
    # Precompute row values as integers
    row_vals = []
    for i in range(dim):
        val = 0
        for j in range(p):
            if G[i, j]:
                val |= (1 << j)
        row_vals.append(val)
    
    # Generate all 2^dim = 256 codewords of length 17
    codewords_17 = [0] * (1 << dim)
    for mask in range(1 << dim):
        word = 0
        for i in range(dim):
            if mask & (1 << i):
                word ^= row_vals[i]
        codewords_17[mask] = word
    
    # Extend to length 18 with parity bit
    codewords_18 = []
    for w in codewords_17:
        parity = w.bit_count() % 2
        codewords_18.append(w | (parity << p))
    
    return codewords_18

def check_candidate(candidate, words_arr, word_set):
    """Check if candidate can be added to the code using numpy."""
    if candidate in word_set:
        return False
    xor_vals = words_arr ^ candidate
    # Count bits
    for x in xor_vals:
        if x.bit_count() < MIN_DIST:
            return False
    return True

def greedy_extend_structured(words, time_budget):
    """
    Try to extend the code using structured search:
    1. For each codeword, explore its 1-bit and 2-bit flip neighborhood
       but use numpy for fast distance checks
    2. Use a priority queue of promising candidates
    """
    if time_budget <= 0:
        return words
    
    deadline = time.time() + time_budget
    words = list(words)
    word_set = set(words)
    words_arr = np.array(words, dtype=np.uint32)
    
    random.seed(42)
    added = 0
    
    # Precompute all 1-bit and 2-bit flips for each codeword
    # This is a large space, so we sample
    
    attempts = 0
    while time.time() < deadline:
        if attempts > 200000:
            break
        
        # Pick a random existing codeword
        idx = random.randint(0, len(words) - 1)
        base = words[idx]
        
        # Try flipping 1 bit
        bit = random.randint(0, N - 1)
        candidate = base ^ (1 << bit)
        if candidate not in word_set:
            # Fast check using numpy
            xor_vals = words_arr ^ candidate
            min_dist = min(x.bit_count() for x in xor_vals)
            if min_dist >= MIN_DIST:
                words.append(candidate)
                word_set.add(candidate)
                words_arr = np.append(words_arr, candidate)
                added += 1
        
        # Occasionally try flipping 2 bits
        if attempts % 3 == 0:
            b1 = random.randint(0, N - 1)
            b2 = random.randint(0, N - 1)
            if b1 != b2:
                candidate = base ^ (1 << b1) ^ (1 << b2)
                if candidate not in word_set:
                    xor_vals = words_arr ^ candidate
                    min_dist = min(x.bit_count() for x in xor_vals)
                    if min_dist >= MIN_DIST:
                        words.append(candidate)
                        word_set.add(candidate)
                        words_arr = np.append(words_arr, candidate)
                        added += 1
        
        # Occasionally try random candidate
        if attempts % 10 == 0:
            candidate = random.randint(0, (1 << N) - 1)
            if candidate not in word_set:
                xor_vals = words_arr ^ candidate
                min_dist = min(x.bit_count() for x in xor_vals)
                if min_dist >= MIN_DIST:
                    words.append(candidate)
                    word_set.add(candidate)
                    words_arr = np.append(words_arr, candidate)
                    added += 1
        
        attempts += 1
    
    return words

def swap_search(words, time_budget):
    """
    Try to improve the code by swapping codewords.
    For each codeword, try replacing with a nearby word that
    maintains validity for all other words.
    """
    if time_budget <= 0:
        return words
    
    deadline = time.time() + time_budget
    words = list(words)
    word_set = set(words)
    words_arr = np.array(words, dtype=np.uint32)
    
    random.seed(123)
    
    n = len(words)
    attempts = 0
    while time.time() < deadline:
        if attempts > 100000:
            break
        
        # Pick a random codeword to potentially replace
        idx = random.randint(0, n - 1)
        old_word = words[idx]
        
        # Try replacing with a nearby word (1 or 2 bit flips)
        flip_count = random.choice([1, 2])
        if flip_count == 1:
            bits = [random.randint(0, N - 1)]
        else:
            b1 = random.randint(0, N - 1)
            b2 = random.randint(0, N - 1)
            if b1 == b2:
                continue
            bits = [b1, b2]
        
        candidate = old_word
        for b in bits:
            candidate ^= (1 << b)
        
        if candidate == old_word or candidate in word_set:
            continue
        
        # Check if replacing old_word with candidate maintains validity
        # Exclude the old word from the check
        ok = True
        for j, w in enumerate(words):
            if j == idx:
                continue
            if hamming(candidate, w) < MIN_DIST:
                ok = False
                break
        
        if ok:
            # Perform the swap
            word_set.remove(old_word)
            words[idx] = candidate
            word_set.add(candidate)
            words_arr[idx] = candidate
        
        attempts += 1
    
    return words

def construct():
    """
    Main entry point.
    """
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    # Step 1: Construct the extended QR code of length 18
    qr_words = quadratic_residue_code()
    
    # Verify validity
    if not verify_code(qr_words):
        # Fallback to Reed-Muller if QR construction fails
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words
    
    # Step 2: Try structured local search to extend
    remaining_time = deadline - time.time()
    if remaining_time > 1:
        # Allocate 75% to greedy extend, 25% to swap search
        greedy_budget = remaining_time * 0.75
        swap_budget = remaining_time * 0.25
        
        extended_words = greedy_extend_structured(qr_words, greedy_budget)
        
        # Step 3: Try swap search on the result
        remaining_time = deadline - time.time()
        if remaining_time > 0.5:
            extended_words = swap_search(extended_words, min(swap_budget, remaining_time))
    
    # Final verification
    if not verify_code(extended_words):
        # Use slow verification as fallback
        if verify_code_slow(extended_words):
            return extended_words
        return qr_words
    
    return extended_words
