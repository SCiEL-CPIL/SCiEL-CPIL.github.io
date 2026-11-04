import time
import random

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 110.0
    
    total_words = q ** n
    
    def word_to_int(word):
        val = 0
        for w in word:
            val = val * q + w
        return val

    def int_to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = int_to_word(i)
        neighbors[i].append(i)
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                neighbors[i].append(word_to_int(w))
        # No need to restore w as it is re-decoded for next i

    best_code = None
    best_size = total_words + 1
    
    max_restarts = 2000
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        is_uncovered = [True] * total_words
        candidates = list(range(total_words))
        random.shuffle(candidates)
        
        code = []
        
        while candidates:
            best_gain = -1
            best_candidates = []
            
            for c in candidates:
                if not is_uncovered[c]:
                    continue
                
                gain = 0
                for nb in neighbors[c]:
                    if is_uncovered[nb]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [c]
                elif gain == best_gain:
                    best_candidates.append(c)
                    
            # Pick randomly from best candidates to break symmetry
            best_cand = random.choice(best_candidates)
            code.append(int_to_word(best_cand))
            
            # Mark neighbors as covered
            for nb in neighbors[best_cand]:
                if is_uncovered[nb]:
                    is_uncovered[nb] = False
            
            # Update candidates list
            candidates = [c for c in candidates if is_uncovered[c]]
            
            if not candidates:
                break
                
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
            if best_size <= 73:
                break

    if best_code is None:
        return [int_to_word(i) for i in range(total_words)]
        
    return best_code
