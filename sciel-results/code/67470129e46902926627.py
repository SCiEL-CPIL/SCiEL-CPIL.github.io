import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # GF(256) parameters
    poly = 0x11D
    
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= poly
        return res & 0xFF

    def gf_pow(a, n):
        res = 1
        base = a
        while n:
            if n & 1:
                res = gf_mul(res, base)
            base = gf_mul(base, base)
            n >>= 1
        return res

    def get_primitive_17th_root():
        for candidate in range(2, 256):
            val = gf_pow(candidate, 17)
            if val == 1:
                if candidate != 1:
                    return candidate
        return None

    beta = get_primitive_17th_root()
    if beta is None:
        # Fallback to seed
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # Build generator polynomial
    poly_coeffs = [1]
    current_beta = beta
    for _ in range(8):
        new_coeffs = [0] * (len(poly_coeffs) + 1)
        for i in range(len(poly_coeffs)):
            new_coeffs[i+1] = poly_coeffs[i]
        for i in range(len(poly_coeffs)):
            new_coeffs[i] ^= gf_mul(poly_coeffs[i], current_beta)
        poly_coeffs = new_coeffs
        current_beta = gf_mul(current_beta, current_beta)
        
    g_bits = [0] * 9
    for i, c in enumerate(poly_coeffs):
        if c == 1:
            g_bits[i] = 1
        elif c == 0:
            g_bits[i] = 0
        else:
            g_bits[i] = 1
            
    # Generate all 2^9 codewords of the [17, 9, d] code
    codewords_17 = []
    for u in range(512):
        c = 0
        for k in range(9):
            if (u >> k) & 1:
                for j in range(9):
                    if g_bits[j]:
                        c ^= 1 << (j + k)
        codewords_17.append(c)
        
    # Extend to 18 bits: add parity bit
    extended_codewords = []
    for c in codewords_17:
        parity = bin(c).count('1') % 2
        ec = c | (parity << 17)
        extended_codewords.append(ec)
        
    # Shorten by one coordinate (fix coordinate 17 to 0)
    shortened_codewords = []
    for ec in extended_codewords:
        if (ec >> 17) & 1 == 0:
            shortened_codewords.append(ec & ((1 << 17) - 1))
            
    # Remove duplicates
    shortened_codewords = list(set(shortened_codewords))
    
    if len(shortened_codewords) < 32:
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # Convert to list for mutable operations
    code_list = list(shortened_codewords)
    best_code = code_list.copy()
    best_len = len(best_code)
    
    # Simulated annealing to add more codewords
    # Strategy: try to add a random word, if conflicts exist, flip bits in existing words to resolve
    
    def hamming_dist(a, b):
        return (a ^ b).bit_count()
    
    def check_min_dist(code):
        """Check if all pairs have distance >= 6"""
        n = len(code)
        for i in range(n):
            for j in range(i+1, n):
                if hamming_dist(code[i], code[j]) < 6:
                    return False
        return True
    
    def find_conflicts(word, code):
        """Find indices of codewords that conflict with word (dist < 6)"""
        conflicts = []
        for i, c in enumerate(code):
            if hamming_dist(word, c) < 6:
                conflicts.append(i)
        return conflicts
    
    def try_add_word(word, code):
        """
        Try to add word to code. If conflicts exist, attempt to repair by flipping bits
        in conflicting codewords. Returns (success, new_code)
        """
        conflicts = find_conflicts(word, code)
        if not conflicts:
            return True, code + [word]
        
        # Try to repair by flipping bits in conflicting codewords
        # We want to increase distance from word to each conflicting codeword
        # Strategy: for each conflicting codeword, flip a bit that increases distance
        # But we must maintain distance >= 6 between all other pairs
        
        max_repair_attempts = 50
        for attempt in range(max_repair_attempts):
            new_code = code.copy()
            success = True
            
            for idx in conflicts:
                # Try flipping different bits in new_code[idx] to increase distance from word
                found_fix = False
                for bit_pos in range(17):
                    flipped = new_code[idx] ^ (1 << bit_pos)
                    # Check if flipping this bit helps
                    new_dist_to_word = hamming_dist(flipped, word)
                    if new_dist_to_word >= 6:
                        # Check if this maintains distance to all other codewords
                        valid = True
                        for j, other in enumerate(new_code):
                            if j == idx:
                                continue
                            if hamming_dist(flipped, other) < 6:
                                valid = False
                                break
                        if valid:
                            new_code[idx] = flipped
                            found_fix = True
                            break
                
                if not found_fix:
                    success = False
                    break
            
            if success:
                return True, new_code + [word]
            
            # If repair failed, try a different random word
            break
        
        return False, code
    
    # Main SA loop
    temperature = 1000.0
    cooling = 0.999
    min_temp = 1.0
    
    random.seed(42)
    
    while time.time() < deadline - 5:
        # Generate random candidate word
        candidate = random.randint(0, (1 << 17) - 1)
        
        # Try to add
        success, new_code = try_add_word(candidate, code_list)
        
        if success and len(new_code) > len(code_list):
            code_list = new_code
            if len(code_list) > best_len:
                best_code = code_list.copy()
                best_len = len(best_code)
        
        # Occasionally try to repair existing conflicts by removing and re-adding
        if random.random() < 0.1:
            # Pick a random codeword to remove and try to replace it
            if len(code_list) > 32:
                idx = random.randint(0, len(code_list) - 1)
                removed = code_list[idx]
                code_list.pop(idx)
                
                # Try to add a new word
                for _ in range(20):
                    candidate = random.randint(0, (1 << 17) - 1)
                    success, new_code = try_add_word(candidate, code_list)
                    if success:
                        code_list = new_code
                        break
                
                # If we couldn't add a new word, put the old one back
                if len(code_list) < best_len - 1:
                    code_list.append(removed)
        
        temperature *= cooling
        if temperature < min_temp:
            break
    
    # Use the best code found
    return best_code
