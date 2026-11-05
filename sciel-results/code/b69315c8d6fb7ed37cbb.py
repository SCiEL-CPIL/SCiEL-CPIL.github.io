import itertools
import time
import random

# Precompute neighbors for all words in {0,1,2}^7
# Total words: 3^7 = 2187
# Neighbors: self + 7 positions * 2 changes = 15 neighbors
N = 2187
ALPHABET = 3
LENGTH = 7

# Map integer to list of digits and vice versa
def int_to_word(x):
    w = [0] * LENGTH
    for i in range(LENGTH):
        w[i] = x % ALPHABET
        x //= ALPHABET
    return w

def word_to_int(w):
    x = 0
    for i in range(LENGTH - 1, -1, -1):
        x = x * ALPHABET + w[i]
    return x

# Precompute neighbors
NEIGHBORS = []
for i in range(N):
    w = int_to_word(i)
    nbs = [i]
    for pos in range(LENGTH):
        orig = w[pos]
        for val in range(ALPHABET):
            if val != orig:
                w[pos] = val
                nbs.append(word_to_int(w))
        w[pos] = orig
    NEIGHBORS.append(nbs)

def construct(q, n, radius):
    # The task guarantees q=3, n=7, radius=1 for this specific problem instance.
    # However, we should handle it generically if possible, or just hardcode for speed if the contract implies fixed params.
    # The contract says: "the evaluator passes this task's alphabet size q, length n and covering radius."
    # But the specific task is K_3(7,1).
    # If q, n, radius are different, the precomputation is invalid.
    # Given the "Football pool problem for 7 matches" context, it's safe to assume q=3, n=7, radius=1.
    # But to be safe, we can check. If not 3,7,1, fall back to a simpler method or raise error?
    # The prompt says "candidate interface... evaluator passes this task's...". It implies the specific instance.
    # I will assume q=3, n=7, radius=1. If not, I will return a trivial solution or raise.
    # Actually, let's just run the greedy for the specific case.
    
    if not (q == 3 and n == 7 and radius == 1):
        # Fallback for other cases (not expected for this specific task contract)
        # Simple greedy without precomputed neighbors
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        def hamming(w1, w2):
            return sum(1 for a, b in zip(w1, w2) if a != b)
        
        def get_neighbors(w):
            nbs = [w]
            for i in range(n):
                for v in range(q):
                    if v != w[i]:
                        nw = list(w)
                        nw[i] = v
                        nbs.append(tuple(nw))
            return nbs

        uncovered = set(all_words)
        code = []
        deadline = time.time() + 110 # 10s margin
        
        while uncovered and time.time() < deadline:
            best_word = None
            best_gain = -1
            for w in all_words:
                gain = 0
                for nb in get_neighbors(w):
                    if nb in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            if best_gain <= 0:
                break
            code.append(list(best_word))
            for nb in get_neighbors(best_word):
                if nb in uncovered:
                    uncovered.remove(nb)
        
        return code

    # Specific case: q=3, n=7, radius=1
    # Use precomputed NEIGHBORS
    
    best_code = None
    start_time = time.time()
    deadline = start_time + 110 # Leave 10s for safety
    
    # Run multiple iterations with different random seeds for tie-breaking
    for seed in range(100):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        is_uncovered = [True] * N
        remaining = N
        code = []
        
        while remaining > 0:
            best_idx = -1
            best_gain = 0
            
            for c in range(N):
                # Quick check: if we already covered this candidate's center, 
                # its gain might be low, but it could still cover uncovered neighbors.
                # However, if the center itself is covered, and all neighbors are covered, gain is 0.
                # We just compute gain.
                gain = 0
                for nb in NEIGHBORS[c]:
                    if is_uncovered[nb]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = c
                elif gain == best_gain and gain > 0:
                    # Tie-breaking: random
                    if random.random() < 0.5:
                        best_idx = c
                        # Note: This random tie-break might pick a worse candidate in long run?
                        # Usually, any is fine. Or we can prefer lower index for determinism in some runs.
                        # Let's stick to random to explore different local optima.
            
            if best_idx == -1:
                break
            
            code.append(best_idx)
            
            # Update uncovered
            for nb in NEIGHBORS[best_idx]:
                if is_uncovered[nb]:
                    is_uncovered[nb] = False
                    remaining -= 1
            
            # Safety check inside loop
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
            
        if len(code) < len(best_code) if best_code else True:
            best_code = code
            
    # Convert indices back to words
    if best_code is None:
        # Fallback: return empty or trivial
        return []
        
    result = [int_to_word(idx) for idx in best_code]
    return result
