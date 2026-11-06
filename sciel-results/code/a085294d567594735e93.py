"""Construct a large binary code of length 17, min distance 6."""
import time
import random
from itertools import combinations

N = 17
MIN_DIST = 6
MASK = (1 << N) - 1
KILL_TIME = 87.0  # safety margin before 90s

def _hamming_dist(a, b):
    return (a ^ b).bit_count()

def _build_linear_code():
    """
    Build the 256-word [17,8,6] code from the [18,9,6] extended QR code.
    """
    # QR(17) generator polynomial: g(x) = 1 + x + x^2 + x^3 + x^5 + x^8 + x^9
    gen_poly_bits = [0, 1, 2, 3, 5, 8, 9]
    
    # Build 9 generator vectors for the [17,9,5] QR code
    gen_vectors_17 = []
    for k in range(9):
        v = 0
        for pos in gen_poly_bits:
            shifted_pos = (pos + k) % 17
            v |= (1 << shifted_pos)
        gen_vectors_17.append(v)
    
    # Extend to [18,9,6] by adding parity bit at position 17
    gen_vectors_18 = []
    for v in gen_vectors_17:
        parity = v.bit_count() % 2
        v18 = v | (parity << 17)
        gen_vectors_18.append(v18)
    
    # Generate all 512 codewords, filter for parity=0, take first 17 bits
    linear_code_17 = []
    seen = set()
    for msg in range(512):
        word = 0
        for bit_idx in range(9):
            if (msg >> bit_idx) & 1:
                word ^= gen_vectors_18[bit_idx]
        if (word >> 17) & 1 == 0:
            w17 = word & MASK
            if w17 not in seen:
                seen.add(w17)
                linear_code_17.append(w17)
    
    return linear_code_17

def _build_candidate_pool(base_code, deadline):
    """
    Build candidates at distance >= 6 from all base codewords.
    Focus on weights 7-10 for good coverage.
    """
    candidates = []
    
    for weight in [7, 8, 9, 10]:
        for positions in combinations(range(N), weight):
            if time.time() > deadline:
                break
            word = 0
            for p in positions:
                word |= (1 << p)
            
            compatible = True
            for b in base_code:
                if _hamming_dist(word, b) < MIN_DIST:
                    compatible = False
                    break
            if compatible:
                candidates.append(word)
        if time.time() > deadline:
            break
    
    return candidates

def _greedy_add(candidates, base_code, deadline):
    """
    Greedily add candidates that maintain min distance with all existing words.
    Order by weight (prefer middle weights) then by some heuristic.
    """
    existing = list(base_code)
    
    # Sort candidates: prefer weight 8, then 9, then 7, then 10
    weight_order = {7: 3, 8: 0, 9: 1, 10: 2}
    candidates.sort(key=lambda w: (weight_order.get(w.bit_count(), 5), w))
    
    for cand in candidates:
        if time.time() > deadline:
            break
        # Check against all existing
        ok = True
        for e in existing:
            if _hamming_dist(cand, e) < MIN_DIST:
                ok = False
                break
        if ok:
            existing.append(cand)
    
    return existing

def construct():
    start_time = time.time()
    deadline = start_time + (KILL_TIME - start_time) - 3.0  # 3s safety margin
    if deadline < start_time:
        deadline = start_time + 5.0  # minimum 5s
    
    # Build the 256-word linear base code
    base_code = _build_linear_code()
    
    # Build candidate pool
    candidates = _build_candidate_pool(base_code, deadline)
    
    if not candidates:
        return base_code
    
    # Greedily add compatible candidates
    result = _greedy_add(candidates, base_code, deadline)
    
    return result

# EVOLVE-BLOCK-START
# Entry point is construct()
# EVOLVE-BLOCK-END
