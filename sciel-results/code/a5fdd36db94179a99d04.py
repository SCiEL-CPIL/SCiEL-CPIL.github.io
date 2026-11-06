import time
import random

def construct():
    N = 17
    MIN_DIST = 6
    deadline = time.time() + 85.0  # Safety margin for 90s limit
    
    # 1. Generate the 256-word [17, 8, 6] code from the [18, 9, 6] QR code
    def generate_qr_shortened():
        p = 19
        # Quadratic residues mod 19
        residues = set((i * i) % p for i in range(1, p))
        
        n = 18
        k = 9
        basis = []
        for i in range(k):
            row = 0
            for j in range(n):
                pos = (j + i) % n
                if pos in residues:
                    row |= (1 << pos)
            basis.append(row)
        
        words = []
        for mask in range(1 << k):
            word = 0
            for i in range(k):
                if (mask >> i) & 1:
                    word ^= basis[i]
            words.append(word)
            
        # Shorten: keep words with 0 at position 0, remove position 0
        shortened = []
        for w in words:
            if (w & 1) == 0:
                shortened.append(w >> 1)
        return shortened

    best_words = generate_qr_shortened()
    
    # 2. Randomized Greedy Search to try to find a larger code
    all_candidates = list(range(1 << N))
    
    while time.time() < deadline:
        random.shuffle(all_candidates)
        current_words = []
        
        for w in all_candidates:
            # Check distance to all current words
            is_valid = True
            for c in current_words:
                if (w ^ c).bit_count() < MIN_DIST:
                    is_valid = False
                    break
            if is_valid:
                current_words.append(w)
                
        if len(current_words) > len(best_words):
            best_words = current_words
            # If we find a very large code, we can stop early to save time
            # or continue to see if we can find even larger
            if len(best_words) > 320:
                break

    return best_words
