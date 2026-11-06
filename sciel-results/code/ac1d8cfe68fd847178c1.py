import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 17
    min_dist = 6
    max_size = 340
    
    # Precompute popcounts
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Construct the [17,8,6] quadratic residue code
    gen_poly = (1 << 9) | (1 << 7) | (1 << 6) | (1 << 3) | (1 << 0)
    
    def encode(msg):
        cw = 0
        for bit_pos in range(9):
            if (msg >> bit_pos) & 1:
                cw ^= (gen_poly << bit_pos)
        return cw
    
    qr_codewords = []
    seen = set()
    for msg in range(256):
        cw = encode(msg)
        if cw not in seen:
            qr_codewords.append(cw)
            seen.add(cw)
    
    valid_qr = (len(qr_codewords) == 256)
    if valid_qr:
        for i in range(len(qr_codewords)):
            for j in range(i + 1, len(qr_codewords)):
                if popcount[qr_codewords[i] ^ qr_codewords[j]] < min_dist:
                    valid_qr = False
                    break
            if not valid_qr:
                break
    
    if valid_qr:
        best_codewords = qr_codewords
        best_size = 256
    else:
        best_codewords = [0]
        best_size = 1
        for candidate in range(1, 1 << n):
            if time.time() > deadline - 30.0:
                break
            valid = True
            for cw in best_codewords:
                if popcount[candidate ^ cw] < min_dist:
                    valid = False
                    break
            if valid:
                best_codewords.append(candidate)
                best_size += 1
    
    if best_size >= max_size:
        return best_codewords
    
    def run_greedy_pass(order, current_codewords):
        codewords = list(current_codewords)
        codeword_set = set(codewords)
        
        for candidate in order:
            if time.time() > deadline - 2.0:
                break
            if candidate in codeword_set:
                continue
            valid = True
            for cw in codewords:
                if popcount[candidate ^ cw] < min_dist:
                    valid = False
                    break
            if valid:
                codewords.append(candidate)
                codeword_set.add(candidate)
        return codewords

    candidates = list(range(1, 1 << n))
    
    while time.time() < deadline - 2.0:
        random.shuffle(candidates)
        current_cw = run_greedy_pass(candidates, best_codewords)
        
        if len(current_cw) > best_size:
            best_size = len(current_cw)
            best_codewords = current_cw
            
            if best_size >= max_size:
                break

    return best_codewords
