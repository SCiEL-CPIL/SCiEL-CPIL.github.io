import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s safety margin out of 90s
    
    # 1. Construct Extended Quadratic Residue Code [18, 9, 6]
    words = _construct_extended_qr_code()
    
    # 2. Verify
    if words and _verify(words):
        return words
        
    # 3. Fallback: Greedy Search
    # If QR failed (shouldn't happen), try to build a large code greedily
    # Start with a known good small code (e.g., 0) and add randoms
    # Or start with RM code if we have it.
    
    # Let's try a simple greedy with a time limit
    remaining_time = deadline - time.time()
    if remaining_time > 1.0:
        return _greedy_fallback(remaining_time)
    else:
        return [0] if not words else words

def _construct_extended_qr_code():
    p = 17
    # Quadratic residues mod 17
    qr_set = set()
    for i in range(1, p):
        qr_set.add((i * i) % p)
    
    # Include 0 for the generator vector of the [17, 9, 5] code
    # The generator matrix is circulant with first row being indicator of {0} U QR
    base_vec = [0] * p
    base_vec[0] = 1
    for r in qr_set:
        base_vec[r] = 1
        
    # Generate 9x17 generator matrix
    G = []
    for i in range(9):
        row = [0] * p
        for j in range(p):
            # Cyclic shift
            row[j] = base_vec[(j - i) % p]
        G.append(row)
        
    # Generate all 2^9 = 512 codewords
    codewords = []
    for mask in range(512):
        word = 0
        for i in range(9):
            if (mask >> i) & 1:
                for j in range(p):
                    if G[i][j]:
                        word ^= (1 << j)
        # Extend with parity bit
        parity = word.bit_count() % 2
        word_ext = word | (parity << p)
        codewords.append(word_ext)
        
    return codewords

def _verify(words):
    n = len(words)
    seen = set()
    for w in words:
        if w in seen:
            return False
        seen.add(w)
    for i in range(n):
        for j in range(i + 1, n):
            if (words[i] ^ words[j]).bit_count() < 6:
                return False
    return True

def _greedy_fallback(time_budget):
    deadline = time.time() + time_budget
    codewords = [0]
    random.seed(42)
    
    while time.time() < deadline:
        candidate = random.getrandbits(18)
        if candidate in codewords:
            continue
        valid = True
        for w in codewords:
            if (candidate ^ w).bit_count() < 6:
                valid = False
                break
        if valid:
            codewords.append(candidate)
            
    return codewords

if __name__ == "__main__":
    words = construct()
    print(f"Generated {len(words)} codewords")
