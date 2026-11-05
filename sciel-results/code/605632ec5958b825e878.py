import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial set: 1..32
    S = 0
    for i in range(1, 33):
        S |= (1 << i)
    
    best_S = S
    best_score = 0
    
    def calc_score(mask):
        score = 0
        # Max d is such that 1 + 2d <= 512 => 2d <= 511 => d <= 255
        for d in range(1, 256):
            # Check if there exists a such that a, a+d, a+2d are in mask
            # mask & (mask >> d) & (mask >> 2*d) != 0
            if mask & (mask >> d) & (mask >> 2 * d):
                score += 1
        return score
    
    best_score = calc_score(best_S)
    
    # Local Search
    # We will try to improve by moving one element
    while time.time() < deadline:
        # Get list of set bits
        bits = []
        tmp = S
        while tmp:
            lsb = tmp & -tmp
            idx = lsb.bit_length() - 1
            bits.append(idx)
            tmp ^= lsb
        
        if not bits:
            break
            
        x = random.choice(bits)
        
        # Pick a random new position y
        valid_y = False
        y = 0
        for _ in range(10):
            y = random.randint(1, 512)
            if not (S & (1 << y)):
                valid_y = True
                break
        
        if not valid_y:
            continue
            
        # New mask
        new_S = S & ~(1 << x)
        new_S |= (1 << y)
        
        new_score = calc_score(new_S)
        
        if new_score > best_score:
            best_score = new_score
            best_S = new_S
            S = new_S
        elif new_score == best_score:
            # Accept equal score to explore
            if random.random() < 0.5:
                S = new_S
        # If new_score < best_score, reject (S remains unchanged)

    return [i for i in range(1, 513) if (best_S >> i) & 1]
