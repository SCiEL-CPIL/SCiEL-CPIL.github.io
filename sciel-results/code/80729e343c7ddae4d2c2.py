import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    N = 17
    MIN_DIST = 6
    MASK = (1 << N) - 1

    # 1. Construct the [17,9,6] shortened extended QR code (256 words)
    
    # GF(256) operations
    # Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
    def gf_mul(a, b, poly=0x11D):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= poly
        return res & 0xFF

    def gf_pow(a, n, poly=0x11D):
        res = 1
        base = a
        while n:
            if n & 1:
                res = gf_mul(res, base, poly)
            base = gf_mul(base, base, poly)
            n >>= 1
        return res

    def get_primitive_17th_root():
        for candidate in range(2, 256):
            if gf_pow(candidate, 17) == 1 and candidate != 1:
                return candidate
        return None

    beta = get_primitive_17th_root()
    
    code_set = set()
    
    if beta:
        # Construct minimal polynomial of beta over GF(2)
        poly_coeffs = [1]
        current_beta = beta
        for _ in range(8):
            new_coeffs = [0] * (len(poly_coeffs) + 1)
            for i in range(len(poly_coeffs)):
                new_coeffs[i+1] ^= poly_coeffs[i]
                new_coeffs[i] ^= gf_mul(poly_coeffs[i], current_beta)
            poly_coeffs = new_coeffs
            current_beta = gf_mul(current_beta, current_beta)
            
        g_bits = []
        valid_poly = True
        for c in poly_coeffs:
            if c == 0:
                g_bits.append(0)
            elif c == 1:
                g_bits.append(1)
            else:
                valid_poly = False
                break
        
        if valid_poly and len(g_bits) == 9 and g_bits[0] == 1:
            codewords_17 = []
            for u in range(512): # 2^9 = 512 words for [17,9]
                c = 0
                for j in range(9):
                    if (u >> j) & 1:
                        for k in range(9):
                            if g_bits[k]:
                                c ^= 1 << (j + k)
                codewords_17.append(c)
            
            # Extend to 18 bits
            extended_codewords = []
            for c in codewords_17:
                parity = bin(c).count('1') % 2
                ec = c | (parity << 17)
                extended_codewords.append(ec)
            
            # Shorten (fix bit 17 to 0)
            shortened_codewords = []
            for ec in extended_codewords:
                if (ec >> 17) & 1 == 0:
                    shortened_codewords.append(ec & MASK)
            
            code_set = set(shortened_codewords)
            
            # Verify min distance
            if len(code_set) > 0:
                lst = list(code_set)
                min_d = 100
                for i in range(len(lst)):
                    for j in range(i+1, len(lst)):
                        d = (lst[i] ^ lst[j]).bit_count()
                        if d < min_d:
                            min_d = d
                if min_d < 6:
                    code_set = set() # Fallback if construction failed

    if not code_set:
        # Fallback: Greedy construction from scratch
        code_set.add(0)
        random.seed(42)
        # Simple greedy to get a base set
        for _ in range(1000):
            if time.time() > deadline - 10.0:
                break
            cand = random.randint(0, MASK)
            valid = True
            for w in code_set:
                if (cand ^ w).bit_count() < MIN_DIST:
                    valid = False
                    break
            if valid:
                code_set.add(cand)
        
        if len(code_set) < 256:
             # If we failed to get a good base, just return what we have
             return list(code_set)

    # 2. Local Search with Swaps
    # Goal: Expand the code beyond 256.
    
    current_code = list(code_set)
    code_set = set(current_code)
    
    # We will maintain a list of "candidates" that are close to being valid.
    # A candidate x is "blocked" by a set of codewords B(x) = {c in C | d(x,c) < 6}.
    # If |B(x)| == 0, we can add x.
    # If |B(x)| == 1, say {c}, we can try to swap c with x.
    # After swapping, we might be able to add another candidate y that was blocked by c.
    
    random.seed(12345)
    
    # Generate a pool of candidates
    # We want candidates that are "dense" in the space.
    # Random sampling is okay, but we can bias towards words that are at distance 5 or 6 from existing words.
    # Actually, just random is fine if we sample enough.
    
    # To speed up conflict checking, we can use numpy if we batch, but for individual swaps, 
    # checking against 255 words is fast enough if we do it smartly.
    
    def is_valid(cand, code_list):
        for w in code_list:
            if (cand ^ w).bit_count() < MIN_DIST:
                return False
        return True

    def find_conflicts(cand, code_list):
        conflicts = []
        for w in code_list:
            if (cand ^ w).bit_count() < MIN_DIST:
                conflicts.append(w)
        return conflicts

    # Strategy:
    # 1. Try to add random words directly.
    # 2. If stuck, perform swaps.
    
    # Phase 1: Direct Addition
    # Try to add words from a large random pool.
    # We'll generate candidates on the fly to save memory.
    
    added_count = 0
    last_improvement = time.time()
    
    # We'll keep a set of "tried" candidates to avoid re-checking, but 2^17 is large.
    # Instead, we'll just sample.
    
    # Pre-compute popcount? No, bit_count is fast.
    
    # We'll alternate between adding and swapping.
    
    # To make swaps effective, we need to know which codewords are "bottlenecks".
    # A codeword c is a bottleneck if it blocks many potential candidates.
    
    # Let's try a simple Simulated Annealing style loop.
    
    best_code = set(code_set)
    
    # Generate a pool of 5000 candidates
    pool = set()
    while len(pool) < 5000:
        pool.add(random.randint(0, MASK))
    pool = list(pool)
    
    # Filter pool: keep only candidates that are "close" to valid (conflicts <= 2)
    # This makes the search space much smaller and more targeted.
    
    candidate_conflicts = {} # cand -> list of conflicting codewords
    
    for cand in pool:
        if cand in code_set:
            continue
        conflicts = find_conflicts(cand, current_code)
        if len(conflicts) <= 2:
            candidate_conflicts[cand] = conflicts
            
    # Now we have a set of candidates that are either:
    # 0 conflicts: Add immediately
    # 1 conflict: Swap candidate
    # 2 conflicts: Double swap candidate (harder)
    
    # Phase 1a: Add all 0-conflict candidates
    for cand, conflicts in list(candidate_conflicts.items()):
        if len(conflicts) == 0:
            code_set.add(cand)
            current_code.append(cand)
            added_count += 1
            del candidate_conflicts[cand]
            
    if added_count > 0:
        # Re-evaluate pool since code changed
        # This might be expensive, but we only do it if we made progress
        pass

    # Phase 2: Swaps
    # We try to replace one codeword with one candidate, then add another.
    
    # We need to be careful with time.
    
    while time.time() < deadline - 5.0:
        
        # Try to find a beneficial swap
        # Pick a random candidate with 1 conflict
        swap_candidates = [c for c, confs in candidate_conflicts.items() if len(confs) == 1]
        
        if not swap_candidates:
            # Generate new pool
            new_pool = set()
            while len(new_pool) < 2000:
                new_pool.add(random.randint(0, MASK))
            for cand in new_pool:
                if cand in code_set:
                    continue
                conflicts = find_conflicts(cand, current_code)
                if len(conflicts) <= 2:
                    candidate_conflicts[cand] = conflicts
            continue
            
        cand = random.choice(swap_candidates)
        conflict_c = candidate_conflicts[cand][0]
        
        # Try swap: remove conflict_c, add cand
        # Check if this allows adding another word
        
        # Temporarily remove
        code_set.remove(conflict_c)
        current_code.remove(conflict_c)
        code_set.add(cand)
        current_code.append(cand)
        
        # Now try to add another word
        # We can check the other candidates in the pool
        found_addition = False
        for other_cand, other_confs in list(candidate_conflicts.items()):
            if other_cand == cand:
                continue
            # Re-check validity of other_cand against the new code
            # It was blocked by some words. If those words are still there, it's still blocked.
            # But if one of the blocking words was conflict_c, it might be free now.
            
            # Quick check: is other_cand valid against current_code?
            valid = True
            for w in current_code:
                if (other_cand ^ w).bit_count() < MIN_DIST:
                    valid = False
                    break
            
            if valid:
                code_set.add(other_cand)
                current_code.append(other_cand)
                found_addition = True
                # Update candidate_conflicts for remaining? 
                # This is getting complex. Let's just restart the pool analysis if we succeed.
                break
        
        if found_addition:
            # Success! We grew the code.
            # Re-initialize pool to find more
            candidate_conflicts.clear()
            new_pool = set()
            while len(new_pool) < 5000:
                new_pool.add(random.randint(0, MASK))
            for cand in new_pool:
                if cand in code_set:
                    continue
                conflicts = find_conflicts(cand, current_code)
                if len(conflicts) <= 2:
                    candidate_conflicts[cand] = conflicts
        else:
            # Swap didn't lead to immediate growth. 
            # We can either revert or keep the swap if it's "neutral".
            # Keeping neutral swaps might help future growth, but it's risky.
            # Let's revert to keep the code "dense" and structured.
            code_set.remove(cand)
            current_code.remove(cand)
            code_set.add(conflict_c)
            current_code.append(conflict_c)
            
            # To avoid infinite loop on the same bad swap, we can remove this candidate from pool
            # or just let random choice pick another.
            
    return list(code_set)
