import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    # 115s deadline to be safe
    deadline = 115.0 

    # Handle specific case q=3, n=6, radius=1 efficiently
    # If parameters differ, fallback to a simpler generic greedy (though task implies specific)
    if q != 3 or n != 6 or radius != 1:
         # Fallback: Simple Greedy
         total_words = q ** n
         def word_to_idx(w):
             idx = 0
             for i in range(n):
                 idx = idx * q + w[i]
             return idx
         def idx_to_word(idx):
             w = [0] * n
             for i in range(n-1, -1, -1):
                 w[i] = idx % q
                 idx //= q
             return w
         
         balls = [0] * total_words
         for i in range(total_words):
             w = idx_to_word(i)
             mask = 1 << i
             # Radius 1 neighbors
             for pos in range(n):
                 orig = w[pos]
                 for new_val in range(q):
                     if new_val == orig: continue
                     nw = w.copy()
                     nw[pos] = new_val
                     j = word_to_idx(tuple(nw))
                     mask |= (1 << j)
             balls[i] = mask
             
         uncovered = (1 << total_words) - 1
         code = []
         while uncovered:
             best_idx = -1
             best_cnt = 0
             for i in range(total_words):
                 inter = uncovered & balls[i]
                 if inter:
                     c = inter.bit_count()
                     if c > best_cnt:
                         best_cnt = c
                         best_idx = i
             if best_idx == -1: break
             code.append(best_idx)
             uncovered &= ~balls[best_idx]
             if time.time() > deadline: break
         return [idx_to_word(i) for i in code]

    # Specific optimized path for q=3, n=6, R=1
    total_words = 729
    
    def word_to_idx(w):
        # w is tuple or list of 6 ints
        return w[0]*3**5 + w[1]*3**4 + w[2]*3**3 + w[3]*3**2 + w[4]*3 + w[5]

    def idx_to_word(idx):
        w = [0] * 6
        w[5] = idx % 3; idx //= 3
        w[4] = idx % 3; idx //= 3
        w[3] = idx % 3; idx //= 3
        w[2] = idx % 3; idx //= 3
        w[1] = idx % 3; idx //= 3
        w[0] = idx % 3
        return w

    # Precompute balls
    balls = [0] * total_words
    all_words = [idx_to_word(i) for i in range(total_words)]
    
    for i, w in enumerate(all_words):
        mask = 1 << i
        w0, w1, w2, w3, w4, w5 = w
        # 6 positions * 2 changes = 12 neighbors
        # Pos 0
        for nv in range(3):
            if nv != w0:
                j = nv*3**5 + w1*3**4 + w2*3**3 + w3*3**2 + w4*3 + w5
                mask |= (1 << j)
        # Pos 1
        for nv in range(3):
            if nv != w1:
                j = w0*3**5 + nv*3**4 + w2*3**3 + w3*3**2 + w4*3 + w5
                mask |= (1 << j)
        # Pos 2
        for nv in range(3):
            if nv != w2:
                j = w0*3**5 + w1*3**4 + nv*3**3 + w3*3**2 + w4*3 + w5
                mask |= (1 << j)
        # Pos 3
        for nv in range(3):
            if nv != w3:
                j = w0*3**5 + w1*3**4 + w2*3**3 + nv*3**2 + w4*3 + w5
                mask |= (1 << j)
        # Pos 4
        for nv in range(3):
            if nv != w4:
                j = w0*3**5 + w1*3**4 + w2*3**3 + w3*3**2 + nv*3 + w5
                mask |= (1 << j)
        # Pos 5
        for nv in range(3):
            if nv != w5:
                j = w0*3**5 + w1*3**4 + w2*3**3 + w3*3**2 + w4*3 + nv
                mask |= (1 << j)
        balls[i] = mask

    def reduce_code(code_indices):
        code_indices = list(code_indices)
        changed = True
        while changed:
            changed = False
            for i in range(len(code_indices)):
                # Check if code_indices[i] is redundant
                # Union of all others
                union_others = 0
                for j in range(len(code_indices)):
                    if i != j:
                        union_others |= balls[code_indices[j]]
                        
                if balls[code_indices[i]] & ~union_others == 0:
                    code_indices.pop(i)
                    changed = True
                    break
            if time.time() > deadline:
                break
        return code_indices

    def run_greedy(order=None):
        if order is None:
            order = list(range(total_words))
            
        uncovered = (1 << total_words) - 1
        code_indices = []
        
        while uncovered:
            best_idx = -1
            best_covered = 0
            best_inter = 0
            
            for i in order:
                inter = uncovered & balls[i]
                if inter:
                    if inter == uncovered:
                        best_idx = i
                        best_covered = uncovered.bit_count()
                        best_inter = inter
                        break
                    cnt = inter.bit_count()
                    if cnt > best_covered:
                        best_covered = cnt
                        best_idx = i
                        best_inter = inter
                        
            if best_idx == -1:
                break
                
            code_indices.append(best_idx)
            uncovered &= ~best_inter
            
            if time.time() > deadline:
                break
                
        return code_indices

    # 1. Deterministic Greedy
    best_code = run_greedy()
    best_code = reduce_code(best_code)
    best_size = len(best_code)
    
    # 2. Randomized Greedy Restarts
    # If we already have a very good code (<=73), we might stop.
    # But let's try to improve or confirm.
    
    # Use a fixed seed for reproducibility? No, random is fine.
    # random.seed(42) # Optional
    
    while time.time() < deadline - 2:
        # Shuffle the order to change tie-breaking
        order = list(range(total_words))
        random.shuffle(order)
        
        code = run_greedy(order)
        code = reduce_code(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
        # If we hit the theoretical best known (73) or lower bound (71), we can stop.
        # 73 is the best known upper bound.
        if best_size <= 73:
            break
            
    return [idx_to_word(i) for i in best_code]
