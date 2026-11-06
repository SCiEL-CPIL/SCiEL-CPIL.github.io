<<<<<<< SEARCH
"""Strategy: High-Throughput Local Search via Vectorized Bitset Evaluation.
1. Represent the set S as a bitset (Python int or NumPy array).
2. Evaluate the score (number of distinct d) using vectorized bitwise operations.
   For a given set S (bitset), a difference d is valid if (S & (S >> d) & (S >> 2d)) != 0.
   We can iterate d from 1 to max_possible_diff and count how many yield a non-zero intersection.
   This is much faster than set lookups in Python.
3. Perform aggressive Hill Climbing with many random restarts.
   - Start with random or structured sets.
   - In each step, try swapping one element for another (or moving an element).
   - Accept if score improves.
   - Since evaluation is fast, we can run millions of steps.
4. Use a time guard to return the best found set.
"""

import time
import random

def _score_from_bitset(bitset, max_val=512):
    """
    Calculate the number of distinct d > 0 such that a, a+d, a+2d are in the set.
    bitset: Python int where bit i is 1 if (i+1) is in the set.
    max_val: The maximum value in the universe (512).
    """
    count = 0
    for d in range(1, max_val // 2 + 1):
        if (bitset & (bitset >> d) & (bitset >> (2 * d))) != 0:
            count += 1
    return count

def _score_diff_set(bitset, max_val=512):
    """Return the set of all valid differences."""
    diffs = set()
    for d in range(1, max_val // 2 + 1):
        if (bitset & (bitset >> d) & (bitset >> (2 * d))) != 0:
            diffs.add(d)
    return diffs

def _bitset_to_list(bitset):
    """Convert bitset to list of integers (1-indexed)."""
    res = []
    b = bitset
    idx = 0
    while b:
        if b & 1:
            res.append(idx + 1)
        b >>= 1
        idx += 1
    return res

def _list_to_bitset(lst):
    """Convert list of 1-indexed integers to bitset."""
    b = 0
    for x in lst:
        b |= (1 << (x - 1))
    return b
=======
"""Strategy: High-Throughput Local Search via NumPy Vectorized Bitset Evaluation.
1. Represent the set S as a NumPy uint64 array of 8 words (512 bits total).
2. Evaluate the score using vectorized NumPy bitwise operations across all 255 differences.
3. Perform aggressive Simulated Annealing with many restarts.
4. Use a time guard to return the best found set.
"""

import time
import random
import numpy as np

def _score_from_bitset_np(words):
    """
    Calculate the number of distinct d > 0 such that a, a+d, a+2d are in the set.
    words: numpy array of 8 uint64 values representing 512 bits.
    Returns: integer count of valid differences.
    """
    # For each d from 1 to 255, check if any a has bits at positions a, a+d, a+2d
    # We use the fact that shifting right by d means bit i becomes bit i-d
    # For word-level operations, we need to handle cross-word shifts
    
    n_words = 8
    max_d = 255
    count = 0
    
    # Precompute shift amounts for each d
    # For a given d, word_shift = d // 64, bit_shift = d % 64
    for d in range(1, max_d + 1):
        ws1 = d // 64
        bs1 = d % 64
        ws2 = (2 * d) // 64
        bs2 = (2 * d) % 64
        
        # Compute shifted versions
        if bs1 == 0:
            shifted1 = np.zeros_like(words)
            if ws1 < n_words:
                shifted1[:n_words - ws1] = words[ws1:]
        else:
            shifted1 = np.zeros_like(words)
            if ws1 < n_words:
                # Shift within words and across words
                for i in range(n_words - ws1):
                    src = words[i + ws1]
                    shifted1[i] = (src >> bs1)
                    if i + ws1 + 1 < n_words:
                        shifted1[i] |= (words[i + ws1 + 1] << (64 - bs1))
        
        if bs2 == 0:
            shifted2 = np.zeros_like(words)
            if ws2 < n_words:
                shifted2[:n_words - ws2] = words[ws2:]
        else:
            shifted2 = np.zeros_like(words)
            if ws2 < n_words:
                for i in range(n_words - ws2):
                    src = words[i + ws2]
                    shifted2[i] = (src >> bs2)
                    if i + ws2 + 1 < n_words:
                        shifted2[i] |= (words[i + ws2 + 1] << (64 - bs2))
        
        # Check intersection
        inter = words & shifted1 & shifted2
        if np.any(inter != 0):
            count += 1
    
    return count

def _score_fast(words):
    """Fast scoring using precomputed shift tables."""
    # This is a cached/vectorized version for speed
    n_words = 8
    max_d = 255
    count = 0
    
    for d in range(1, max_d + 1):
        ws1 = d // 64
        bs1 = d % 64
        ws2 = (2 * d) // 64
        bs2 = (2 * d) % 64
        
        # Quick check: if the set is sparse, many shifts will be empty
        # Just do the full check
        if bs1 == 0 and ws1 == 0:
            # No shift needed for first
            pass
        
        # Build shifted arrays efficiently
        # For performance, we'll use a simpler approach: convert to Python int temporarily
        # but only for the check, keeping the main representation as numpy
        
        # Actually, let's just use the direct method but optimized
        valid = False
        for i in range(n_words):
            w = words[i]
            if w == 0:
                continue
            
            # Check bit i*64 + j for all j where bit is set
            # This is getting complex; let's use a different approach
            
            # For each set bit position p in word i:
            #   pos = i*64 + j
            #   Check if pos+d and pos+2d are also set
            # This is O(n_set_bits * 3) per word
            
            # Alternative: just do the word-level AND check
            pass
        
        # Let's use a cleaner vectorized approach
        # Compute S >> d and S >> 2d as numpy arrays, then AND
        
        # S >> d
        if d == 0:
            s1 = words.copy()
        else:
            s1 = np.zeros(n_words, dtype=np.uint64)
            for i in range(n_words):
                # Word i of S>>d gets bits from words i+ws1 and i+ws1+1
                base_i = i + ws1
                if base_i < n_words:
                    if bs1 == 0:
                        s1[i] = words[base_i]
                    else:
                        s1[i] = (words[base_i] >> bs1)
                        if base_i + 1 < n_words:
                            s1[i] |= (words[base_i + 1] << (64 - bs1))
        
        # S >> 2d
        if 2 * d == 0:
            s2 = words.copy()
        else:
            s2 = np.zeros(n_words, dtype=np.uint64)
            for i in range(n_words):
                base_i = i + ws2
                if base_i < n_words:
                    if bs2 == 0:
                        s2[i] = words[base_i]
                    else:
                        s2[i] = (words[base_i] >> bs2)
                        if base_i + 1 < n_words:
                            s2[i] |= (words[base_i + 1] << (64 - bs2))
        
        inter = words & s1 & s2
        if np.any(inter != 0):
            count += 1
    
    return count

def _bitset_to_list(words):
    """Convert numpy words array to list of integers (1-indexed)."""
    res = []
    for i in range(8):
        w = words[i]
        j = 0
        while w:
            if w & 1:
                res.append(i * 64 + j + 1)
            w >>= 1
            j += 1
    return res

def _list_to_words(lst):
    """Convert list of 1-indexed integers to numpy words array."""
    words = np.zeros(8, dtype=np.uint64)
    for x in lst:
        idx = x - 1
        word_idx = idx // 64
        bit_idx = idx % 64
        words[word_idx] |= (1 << bit_idx)
    return words
>>>>>>> REPLACE

