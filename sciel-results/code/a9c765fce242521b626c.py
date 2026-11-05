"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import permutations
from math import factorial

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    if n <= 3:
        # Standard construction is optimal for n<=3
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    # Build standard construction
    def build_standard(n):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word
    
    # Encode a window tuple as an integer for fast lookup
    def encode_window(w, n):
        # Map symbols 1..n to 0..n-1, then encode as base-n number
        code = 0
        for sym in w:
            code = code * n + (sym - 1)
        return code
    
    # Precompute all valid permutation codes
    def get_all_perm_codes(n):
        codes = set()
        for p in permutations(range(1, n+1)):
            codes.add(encode_window(p, n))
        return codes
    
    # Verify a word contains all permutations
    def verify(word, n, all_codes):
        covered = set()
        L = len(word)
        for i in range(L - n + 1):
            w = tuple(word[i:i+n])
            code = encode_window(w, n)
            if code in all_codes:
                covered.add(code)
        return len(covered) == len(all_codes), covered
    
    if n == 4:
        word = build_standard(n)
        all_codes = get_all_perm_codes(n)
        
        def count_windows(word, n, all_codes):
            counts = {}
            L = len(word)
            for i in range(L - n + 1):
                code = encode_window(tuple(word[i:i+n]), n)
                if code in all_codes:
                    counts[code] = counts.get(code, 0) + 1
            return counts
        
        improved = list(word)
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            counts = count_windows(improved, n, all_codes)
            for pos in range(len(improved) - 1, 0, -1):
                if time.time() > deadline - 5:
                    break
                safe = True
                for i in range(max(0, pos - n + 1), min(pos, len(improved) - n) + 1):
                    code = encode_window(tuple(improved[i:i+n]), n)
                    if code in all_codes:
                        if counts.get(code, 0) <= 1:
                            safe = False
                            break
                if safe:
                    improved = improved[:pos] + improved[pos+1:]
                    changed = True
                    break
        
        valid, _ = verify(improved, n, all_codes)
        if valid:
            word = improved
        return "".join(map(str, word))
    
    # For n >= 5, use standard construction then try to shorten
    word = build_standard(n)
    base_len = len(word)
    
    all_codes = get_all_perm_codes(n)
    
    # Verify standard word is valid
    valid, _ = verify(word, n, all_codes)
    if not valid:
        return "".join(map(str, word))
    
    improved = list(word)
    
    def count_windows_fast(word, n, all_codes):
        counts = {}
        L = len(word)
        for i in range(L - n + 1):
            code = encode_window(tuple(word[i:i+n]), n)
            if code in all_codes:
                counts[code] = counts.get(code, 0) + 1
        return counts
    
    # Multiple passes of removal - try from end first, then interior
    max_passes = 50
    for _ in range(max_passes):
        if time.time() > deadline - 10:
            break
        changed = False
        counts = count_windows_fast(improved, n, all_codes)
        L = len(improved)
        
        # Try removing from the end first (cheaper to verify)
        for pos in range(L - 1, 0, -1):
            if time.time() > deadline - 10:
                break
            safe = True
            for i in range(max(0, pos - n + 1), min(pos, L - n) + 1):
                code = encode_window(tuple(improved[i:i+n]), n)
                if code in all_codes:
                    if counts.get(code, 0) <= 1:
                        safe = False
                        break
            if safe:
                improved = improved[:pos] + improved[pos+1:]
                changed = True
                break
        
        if not changed:
            break
    
    # Verify the improved word
    valid_improved, _ = verify(improved, n, all_codes)
    if valid_improved:
        word = improved
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
