import time
import random

N = 18
MIN_DISTANCE = 6

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    # Build RM(1,4)
    rm_basis = []
    b0 = 0
    for i in range(16):
        b0 |= 1 << i
    rm_basis.append(b0)
    for j in range(4):
        bj = 0
        for i in range(16):
            if (i >> j) & 1:
                bj |= 1 << i
        rm_basis.append(bj)
    
    rm_words = []
    for mask in range(32):
        word = 0
        for bit in range(5):
            if (mask >> bit) & 1:
                word ^= rm_basis[bit]
        rm_words.append(word)
    rm_words = list(set(rm_words))
    
    # Verify RM code
    valid_rm = True
    for i in range(len(rm_words)):
        for j in range(i+1, len(rm_words)):
            if (rm_words[i] ^ rm_words[j]).bit_count() < MIN_DISTANCE:
                valid_rm = False
                break
        if not valid_rm:
            break
    
    # Greedy extension with adaptive stuck threshold
    def greedy_extend(code, deadline):
        code = list(code)
        code_set = set(code)
        stuck_count = 0
        max_stuck = 50000
        added_recently = 0
        last_add_time = time.time()
        
        while time.time() < deadline:
            # Adaptive weight selection based on what's been added
            if stuck_count < 1000:
                wt = random.choice([7, 8, 9, 10, 11])
            elif stuck_count < 10000:
                wt = random.randint(6, 12)
            else:
                wt = random.randint(5, 13)
            
            if wt < 1:
                cand = 0
            elif wt >= N:
                cand = (1 << N) - 1
            else:
                positions = random.sample(range(N), wt)
                cand = 0
                for p in positions:
                    cand |= 1 << p
            
            if cand in code_set:
                stuck_count += 1
                if stuck_count > max_stuck:
                    break
                continue
            
            # Check distance to all existing codewords
            valid = True
            for w in code:
                if (cand ^ w).bit_count() < MIN_DISTANCE:
                    valid = False
                    break
            
            if valid:
                code.append(cand)
                code_set.add(cand)
                stuck_count = 0
                added_recently += 1
                last_add_time = time.time()
            else:
                stuck_count += 1
                if stuck_count > max_stuck:
                    break
        
        return code
    
    best_code = rm_words if valid_rm else [0]
    best_size = len(best_code)
    
    # Extend RM code
    if valid_rm:
        code1 = greedy_extend(rm_words, deadline)
        if len(code1) > best_size:
            best_size = len(code1)
            best_code = code1
    
    # Try multiple restarts
    if time.time() < deadline:
        for restart in range(100):
            if time.time() > deadline:
                break
            
            # Create a random initial valid code
            init = [0]
            for _ in range(8):
                if time.time() > deadline:
                    break
                for _try in range(30000):
                    wt = random.randint(6, 12)
                    positions = random.sample(range(N), wt)
                    cand = 0
                    for p in positions:
                        cand |= 1 << p
                    if cand in init:
                        continue
                    valid = True
                    for w in init:
                        if (cand ^ w).bit_count() < MIN_DISTANCE:
                            valid = False
                            break
                    if valid:
                        init.append(cand)
                        break
            
            code2 = greedy_extend(init, deadline)
            if len(code2) > best_size:
                best_size = len(code2)
                best_code = code2
    
    # Final verification
    for i in range(len(best_code)):
        for j in range(i+1, len(best_code)):
            if (best_code[i] ^ best_code[j]).bit_count() < MIN_DISTANCE:
                if valid_rm:
                    return rm_words
                return [0]
    
    return best_code

random.seed(12345)
