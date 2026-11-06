import numpy as np
import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    start_time = time.time()
    DEADLINE = 85  # Safety margin before 90s kill

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    # 1. Try Extended QR Code Construction
    try:
        # Quadratic residues mod 17 (including 0)
        residues = [0, 1, 2, 4, 8, 9, 13, 15, 16]
        v = [0] * 17
        for r in residues:
            v[r] = 1
        
        # Precompute int versions of rows
        row_ints = []
        for i in range(9):
            val = 0
            for k in range(17):
                if v[(k + i) % 17]:
                    val |= (1 << k)
            # Parity bit at position 17
            parity = bin(val).count('1') % 2
            if parity:
                val |= (1 << 17)
            row_ints.append(val)
        
        # Generate all 512 codewords
        codes = []
        for m in range(512):
            c = 0
            for j in range(9):
                if (m >> j) & 1:
                    c ^= row_ints[j]
            codes.append(c)
        
        # Verify minimum distance
        min_d = N + 1
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                d = (codes[i] ^ codes[j]).bit_count()
                if d < min_d:
                    min_d = d
                    if min_d < MIN_DIST:
                        break
            if min_d < MIN_DIST:
                break
        
        if min_d >= MIN_DIST:
            return codes
    except Exception:
        pass

    # Fallback: Greedy with local search improvement
    def greedy_construct():
        codes = [0]
        # Precompute all 2^18 candidates? No, too many. Use random + greedy.
        # Start with all-zero word, then greedily add words with max distance
        
        # Use a smarter approach: start with a few known good words
        # Then greedily add more
        candidates = list(range(2**N))
        random.shuffle(candidates)
        
        for cand in candidates:
            if time_remaining() < 5:
                break
            ok = True
            for c in codes:
                if (cand ^ c).bit_count() < MIN_DIST:
                    ok = False
                    break
            if ok:
                codes.append(cand)
        
        return codes

    def local_search_improve(codes, max_time=30):
        """Try to improve code size by local search: remove a word, try to add two."""
        codes = list(codes)
        deadline = time.time() + max_time
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(codes)
            # Try removing each word and adding up to 2 new words
            for i in range(n):
                if time.time() > deadline:
                    break
                # Remove word i
                codes_without_i = codes[:i] + codes[i+1:]
                removed = codes[i]
                
                # Try to find 2 words that can be added
                added = []
                # Use a limited candidate pool for speed
                # Generate candidates near the removed word
                candidates = []
                for delta in range(1, 100):
                    c1 = removed ^ (1 << delta)
                    if c1 < 2**N:
                        candidates.append(c1)
                    c2 = removed ^ ((1 << delta) - 1)
                    if c2 < 2**N:
                        candidates.append(c2)
                
                random.shuffle(candidates)
                
                for cand in candidates[:200]:
                    if len(added) >= 2:
                        break
                    ok = True
                    for c in codes_without_i + added:
                        if (cand ^ c).bit_count() < MIN_DIST:
                            ok = False
                            break
                    if ok:
                        added.append(cand)
                
                if len(added) >= 2:
                    codes = codes_without_i + added
                    improved = True
                    break
        
        return codes

    # Run greedy
    codes = greedy_construct()
    
    if time_remaining() > 10:
        codes = local_search_improve(codes, max_time=max(5, time_remaining() - 10))
    
    return codes
