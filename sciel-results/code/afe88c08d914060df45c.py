import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe
    
    best_score = -1
    best_mask = 0
    
    # Precompute valid d range
    max_d = 255
    
    def get_score(mask):
        count = 0
        for d in range(1, max_d + 1):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def mask_to_list(mask):
        return [i + 1 for i in range(512) if (mask >> i) & 1]

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << (x - 1))
        return m

    # Initializations
    inits = []
    
    # 1. Random
    for _ in range(5):
        s = random.sample(range(1, 513), 32)
        inits.append(list_to_mask(s))
        
    # 2. Structured: Small numbers
    s = list(range(1, 33))
    inits.append(list_to_mask(s))
    
    # 3. Structured: Geometric-ish
    s = set()
    while len(s) < 32:
        s.add(random.randint(1, 512))
    inits.append(list_to_mask(list(s)))
    
    # 4. Grid-like
    s = set()
    step = 16
    for i in range(0, 512, step):
        if len(s) >= 32: break
        s.add(i + 1)
    # Fill rest randomly
    while len(s) < 32:
        s.add(random.randint(1, 512))
    inits.append(list_to_mask(list(s)))

    for mask in inits:
        if time.time() > deadline:
            break
            
        current_mask = mask
        current_score = get_score(current_mask)
        
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask
            
        # Simulated Annealing
        temp = 10.0
        cooling = 0.999
        min_temp = 0.1
        
        # Track current list for easy manipulation
        current_list = mask_to_list(current_mask)
        
        while temp > min_temp:
            if time.time() > deadline:
                break
                
            # Generate a move: replace one or two elements
            # 2-swap kick
            if random.random() < 0.3:
                # 2-swap
                idx1, idx2 = random.sample(range(32), 2)
                x1 = current_list[idx1]
                x2 = current_list[idx2]
                
                # Pick new values
                new_x1 = random.randint(1, 512)
                new_x2 = random.randint(1, 512)
                
                # Ensure distinctness and not equal to other elements
                # Simple retry loop
                while new_x1 == x1 or new_x1 == x2 or new_x1 in current_list:
                    new_x1 = random.randint(1, 512)
                while new_x2 == x2 or new_x2 == x1 or new_x2 in current_list or new_x2 == new_x1:
                    new_x2 = random.randint(1, 512)
                
                # Apply move
                new_list = current_list[:]
                new_list[idx1] = new_x1
                new_list[idx2] = new_x2
                
                new_mask = list_to_mask(new_list)
                new_score = get_score(new_mask)
                
                delta = new_score - current_score
                if delta > 0 or random.random() < pow(2.71828, delta / temp):
                    current_mask = new_mask
                    current_score = new_score
                    current_list = new_list
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
            else:
                # 1-swap
                idx1 = random.randint(0, 31)
                x1 = current_list[idx1]
                new_x1 = random.randint(1, 512)
                while new_x1 == x1 or new_x1 in current_list:
                    new_x1 = random.randint(1, 512)
                
                new_list = current_list[:]
                new_list[idx1] = new_x1
                
                new_mask = list_to_mask(new_list)
                new_score = get_score(new_mask)
                
                delta = new_score - current_score
                if delta > 0 or random.random() < pow(2.71828, delta / temp):
                    current_mask = new_mask
                    current_score = new_score
                    current_list = new_list
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
            
            temp *= cooling

    return mask_to_list(best_mask)
