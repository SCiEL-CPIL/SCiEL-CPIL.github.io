import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # 5s buffer
    
    N = q ** n
    
    def int_to_word(val, n, q):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        return word

    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val

    # Precompute masks
    masks = [0] * N
    
    if radius == 1:
        for i in range(N):
            word_i = int_to_word(i, n, q)
            mask = 1 << i
            for pos in range(n):
                orig_val = word_i[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    new_word = word_i[:]
                    new_word[pos] = new_val
                    j = word_to_int(new_word, q)
                    mask |= (1 << j)
            masks[i] = mask
    else:
        # Generic fallback
        from collections import deque
        for i in range(N):
            visited = {i}
            queue = deque([(i, 0)])
            mask = 1 << i
            while queue:
                cur, dist = queue.popleft()
                if dist < radius:
                    cur_word = int_to_word(cur, n, q)
                    for pos in range(n):
                        for new_val in range(q):
                            if new_val == cur_word[pos]:
                                continue
                            new_word = cur_word[:]
                            new_word[pos] = new_val
                            nxt = word_to_int(new_word, q)
                            if nxt not in visited:
                                visited.add(nxt)
                                queue.append((nxt, dist + 1))
            for v in visited:
                mask |= (1 << v)
            masks[i] = mask

    def greedy_init():
        uncovered = (1 << N) - 1
        code = []
        code_set = set()
        
        while uncovered:
            if time.time() > deadline:
                break
            
            best_c = -1
            best_gain = -1
            
            for c in range(N):
                if c in code_set:
                    continue
                gain = (masks[c] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            
            if best_c == -1 or best_gain == 0:
                break
                
            code.append(best_c)
            code_set.add(best_c)
            uncovered &= ~masks[best_c]
            
        return code

    def optimize_1opt(code):
        code = list(code)
        code_set = set(code)
        
        cover_count = [0] * N
        for c in code:
            m = masks[c]
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                cover_count[w] += 1
                m ^= lsb
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline:
                    break
                c = code[idx]
                redundant = True
                m = masks[c]
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    if cover_count[w] == 1:
                        redundant = False
                        break
                    m ^= lsb
                    
                if redundant:
                    m = masks[c]
                    while m:
                        lsb = m & -m
                        w = lsb.bit_length() - 1
                        cover_count[w] -= 1
                        m ^= lsb
                    code.pop(idx)
                    code_set.discard(c)
                    improved = True
                    break
                    
        return code

    code = greedy_init()
    if time.time() < deadline - 2:
        code = optimize_1opt(code)
        
    result = [int_to_word(c, n, q) for c in code]
    
    if not result:
        result = [int_to_word(0, n, q)]
        
    return result
