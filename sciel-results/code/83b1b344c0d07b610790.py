"""Rewritten program: vectorized forbidden-zone greedy packing for A_2(17,6)."""
import itertools
import time
import numpy as np

N = 17
MIN_DIST = 6
UNIVERSE = 1 << N
TIME_LIMIT = 80.0  # safety margin under 90s wall-clock

def _hamming_weight(x):
    return int(np.bitwise_count(x).sum()) if hasattr(np, 'bitwise_count') else int(bin(x).count('1'))

def _build_weight_lut():
    # LUT for popcount of 17-bit integers via 8-bit chunks
    table = np.zeros(256, dtype=np.int8)
    for i in range(256):
        table[i] = bin(i).count('1')
    return table

def construct():
    start = time.time()
    table8 = _build_weight_lut()
    
    # Precompute popcount for all 2^17 values using table lookups (fast)
    # x = low8 | (high9 << 8)
    low = np.arange(UNIVERSE, dtype=np.int32) & 0xFF
    high = (np.arange(UNIVERSE, dtype=np.int32) >> 8) & 0x1FF
    # high is 9 bits, split into 8+1
    high8 = high & 0xFF
    high1 = (high >> 8) & 1
    popcount = table8[low] + table8[high8] + high1
    
    # Forbidden zone: for each candidate c, it is forbidden if there exists
    # an already-selected codeword w such that Hamming(c, w) < MIN_DIST
    # i.e., Hamming(c, w) <= MIN_DIST - 1 = 5
    #
    # Vectorized approach: maintain a boolean array `occupied` of size 2^17.
    # When we add codeword w, we mark all c with Hamming(c,w) <= 5 as forbidden.
    # But doing this naively is expensive. Instead, use the following trick:
    #
    # For a given w, the set {c : Hamming(c,w) <= 5} has size sum_{i=0}^{5} C(17,i) = 1+17+136+680+2380+6188 = 9402.
    # That's too many to mark per codeword if we have ~250+ codewords.
    #
    # Better: use a "forbidden count" array. For each position i, if w has bit i set,
    # flipping bit i in c gives Hamming distance 1 contribution. We can use a multi-pass
    # approach, but that's complex.
    #
    # Alternative: greedy scan with vectorized distance check against all selected words.
    # Keep selected words in a list/array. For each candidate in order, compute its
    # Hamming distance to all selected words and check if min >= MIN_DIST.
    #
    # To make this fast, batch-check candidates. Process candidates in chunks.
    
    selected = []
    selected_arr = np.array([], dtype=np.int32)
    
    # We'll iterate through candidates 0..2^17-1. For efficiency, we can use
    # a "forbidden" boolean array and update it when adding a codeword.
    # Marking all c with Hamming(c,w) <= 5: generate all such c.
    
    # Precompute all masks with weight <= 5
    masks = []
    for weight in range(MIN_DIST):  # 0..5
        # generate all subsets of {0..16} of size `weight`
        def gen_subsets(n, k):
            if k == 0:
                yield 0
                return
            for i in range(n - k + 1):
                for rest in gen_subsets(n - i - 1, k - 1):
                    yield (1 << i) | (rest << (i + 1))
        for m in gen_subsets(N, weight):
            masks.append(m)
    masks = np.array(masks, dtype=np.int32)
    num_masks = len(masks)
    
    # forbidden[c] = True if c is too close to some selected word
    forbidden = np.zeros(UNIVERSE, dtype=bool)
    
    # Greedy: scan candidates, add if not forbidden
    # To speed up, we can use the fact that after adding w, we mark w XOR mask for all masks
    # But marking 9402 entries per codeword * 250 codewords = ~2.3M operations, fine.
    
    for c in range(UNIVERSE):
        if time.time() - start > TIME_LIMIT - 5:
            break
        if forbidden[c]:
            continue
        # Add c to selected
        selected.append(c)
        # Mark forbidden: for each mask m, c XOR m is forbidden
        to_mark = (c ^ masks)
        forbidden[to_mark] = True
    
    if not selected:
        # Fallback: return the 32-word Reed-Muller code
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words
    
    return selected

# Test
if __name__ == "__main__":
    import time
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Found {len(result)} codewords in {t1-t0:.2f}s")
    # Verify
    import itertools
    min_d = 17
    for a, b in itertools.combinations(result, 2):
        d = bin(a ^ b).count('1')
        if d < min_d:
            min_d = d
    print(f"Min distance: {min_d}")
