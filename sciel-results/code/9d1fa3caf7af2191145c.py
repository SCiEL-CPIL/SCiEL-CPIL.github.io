import time
import random
import numpy as np

BALLS = []
for mask in range(1 << 17):
    if mask.bit_count() <= 5:
        BALLS.append(mask)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_code = []
    best_size = 0
    
    def get_random_linear_code():
        while True:
            G = np.random.randint(0, 2, size=(8, 17), dtype=np.uint8)
            row_ints = []
            for k in range(8):
                r = 0
                for j in range(17):
                    if G[k][j]:
                        r |= (1 << j)
                row_ints.append(r)
            
            words = []
            for i in range(256):
                w = 0
                for k in range(8):
                    if (i >> k) & 1:
                        w ^= row_ints[k]
                words.append(w)
            
            min_d = 18
            for i in range(256):
                wi = words[i]
                for j in range(i+1, 256):
                    d = (wi ^ words[j]).bit_count()
                    if d < min_d:
                        min_d = d
            if min_d >= 6:
                return words

    num_trials = 0
    while time.time() < deadline:
        num_trials += 1
        lin_code = get_random_linear_code()
        
        forbidden = set(lin_code)
        for w in lin_code:
            for b in BALLS:
                forbidden.add(w ^ b)
        
        current_code = list(lin_code)
        
        indices = list(range(1 << 17))
        random.shuffle(indices)
        
        for w in indices:
            if time.time() > deadline - 5.0:
                break
                
            if w not in forbidden:
                forbidden.add(w)
                for b in BALLS:
                    forbidden.add(w ^ b)
                current_code.append(w)
                
        if len(current_code) > best_size:
            best_size = len(current_code)
            best_code = current_code[:]
            
    return best_code
