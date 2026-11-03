import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    total_words = q ** n
    if total_words > 100000:
        # Fallback for larger n if needed, but task is n=8
        pass

    # Pre-compute neighbors for all words
    # Word is an integer 0 to total_words - 1
    # Convert to base-q digits, modify, convert back
    neighbors = []
    for w in range(total_words):
        # Decode
        digits = [0] * n
        temp = w
        for i in range(n - 1, -1, -1):
            digits[i] = temp % q
            temp //= q
        
        neigh = set()
        neigh.add(w)
        for i in range(n):
            orig = digits[i]
            for val in range(q):
                if val == orig:
                    continue
                digits[i] = val
                # Encode
                new_w = 0
                pow_q = 1
                for j in range(n - 1, -1, -1):
                    new_w += digits[j] * pow_q
                    pow_q *= q
                neigh.add(new_w)
            digits[i] = orig
        
        neighbors.append(list(neigh))

    covered = np.zeros(total_words, dtype=np.uint8)
    code = []
    
    # Greedy selection
    while True:
        if time.time() > deadline:
            break
            
        best_w = -1
        max_gain = 0
        
        # Find word with max uncovered neighbors
        for w in range(total_words):
            gain = 0
            for v in neighbors[w]:
                if covered[v] == 0:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_w = w
                
        if max_gain == 0:
            break
            
        code.append(best_w)
        
        # Update coverage
        for v in neighbors[best_w]:
            covered[v] = 1

    # Convert integers back to lists of digits
    result = []
    for w in code:
        digits = [0] * n
        temp = w
        for i in range(n - 1, -1, -1):
            digits[i] = temp % q
            temp //= q
        result.append(digits)
        
    return result
