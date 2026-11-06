import time
import random
import numpy as np
from itertools import combinations

N = 18
MIN_DIST = 6
TIME_LIMIT = 60.0

def popcount(x):
    return x.bit_count()

def hamming(a, b):
    return popcount(a ^ b)

def verify_code(words):
    """Verify all pairwise distances using NumPy for speed."""
    if len(words) <= 1:
        return True
    arr = np.array(words, dtype=np.uint32)
    n = len(arr)
    for i in range(n):
        diffs = arr[i+1:] ^ arr[i]
        counts = np.array([d.bit_count() for d in diffs])
        if np.any(counts < MIN_DIST):
            return False
    return True

def construct_extended_qr_512():
    """
    Constructs the 512-word extended QR code for p=17.
    Uses direct bitmask representation of the generator matrix.
    """
    p = 17
    k = (p + 1) // 2  # 9
    
    # Quadratic residues mod 17
    qr = set((i * i) % p for i in range(p))
    
    # Build generator matrix rows as integer bitmasks
    G_rows = []
    for i in range(k):
        row = 0
        for j in range(p):
            if (j - i) % p in qr:
                row |= (1 << j)
        G_rows.append(row)
    
    # Generate all 2^9 = 512 codewords of length 17 via XOR of basis vectors
    codewords_17 = [0] * (1 << k)
    for mask in range(1, 1 << k):
        lsb = mask & (-mask)
        idx = lsb.bit_length() - 1
        codewords_17[mask] = codewords_17[mask ^ lsb] ^ G_rows[idx]
    
    # Extend each word by an overall parity bit to get length 18
    codewords_18 = []
    for w in codewords_17:
        parity = popcount(w) % 2
        codewords_18.append(w | (parity << p))
        
    return codewords_18

def find_coset_offset(L, time_budget=30.0):
    """
    Searches for a vector v such that the minimum distance from v to any word in L is >= MIN_DIST.
    If found, L U (v+L) is a valid code of size 2*len(L).
    Uses systematic + random search with NumPy acceleration.
    """
    n = len(L)
    L_set = set(L)
    L_arr = np.array(L, dtype=np.uint32)
    
    deadline = time.time() + time_budget
    random.seed(42)
    
    # First try systematic: low-weight vectors
    for weight in range(0, 4):
        if time.time() > deadline:
            break
        for combo in combinations(range(N), weight):
            v = 0
            for pos in combo:
                v |= (1 << pos)
            if v in L_set:
                continue
            diffs = L_arr ^ v
            min_d = min(d.bit_count() for d in diffs)
            if min_d >= MIN_DIST:
                return v
    
    # Then random search
    attempts = 0
    while time.time() < deadline:
        v = random.randint(0, (1 << N) - 1)
        if v in L_set:
            continue
        
        diffs = L_arr ^ v
        min_d = min(d.bit_count() for d in diffs)
        if min_d >= MIN_DIST:
            return v
        attempts += 1
    
    return None

def greedy_extend(words, time_budget=0):
    """
    Greedy extension with NumPy-accelerated distance checks.
    Tries random candidates and low-weight perturbations of existing words.
    """
    deadline = time.time() + time_budget
    words = list(words)
    word_set = set(words)
    words_arr = np.array(words, dtype=np.uint32)
    random.seed(123)
    
    while time.time() < deadline:
        candidate = random.randint(0, (1 << N) - 1)
        if candidate in word_set:
            continue
        
        diffs = words_arr ^ candidate
        min_d = min(d.bit_count() for d in diffs)
        
        if min_d >= MIN_DIST:
            words.append(candidate)
            word_set.add(candidate)
            words_arr = np.append(words_arr, candidate)
    
    return words

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 3.0
    
    # Step 1: Construct the 512-word base code
    base_code = construct_extended_qr_512()
    
    if not verify_code(base_code):
        # Fallback to small RM code
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return [w for w in words]
        
    best_words = list(base_code)
    
    # Step 2: Try to find a coset offset to double the size to 1024
    remaining = deadline - time.time()
    if remaining > 5:
        offset = find_coset_offset(base_code, time_budget=remaining - 2.0)
        
        if offset is not None:
            full_code = list(base_code)
            for w in base_code:
                full_code.append(w ^ offset)
            
            if verify_code(full_code):
                return full_code
    
    # Step 3: Greedy extension
    remaining = deadline - time.time()
    if remaining > 2:
        best_words = greedy_extend(best_words, time_budget=remaining)
        
    return best_words
