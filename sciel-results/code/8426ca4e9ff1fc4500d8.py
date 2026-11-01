import numpy as np
import time
import itertools

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    N = q ** n
    words = []
    for i in range(N):
        w = []
        x = i
        for _ in range(n):
            w.append(x % q)
            x //= q
        words.append(w)
        
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    M = np.zeros((N, N), dtype=np.int8)
    
    for i, w in enumerate(words):
        M[i, i] = 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                idx = word_to_idx[tuple(w)]
                M[i, idx] = 1
                w[pos] = orig_val

    def run_greedy():
        covered = np.zeros(N, dtype=bool)
        code_indices = []
        
        while not covered.all():
            uncovered = ~covered
            gains = M.dot(uncovered.astype(np.int32))
            
            noise = np.random.rand(N) * 2.0
            scores = gains + noise
            
            idx = np.argmax(scores)
            
            covered |= M[idx]
            code_indices.append(idx)
            
        return code_indices

    def prune(indices):
        if not indices:
            return []
            
        counts = np.zeros(N, dtype=np.int32)
        for idx in indices:
            counts += M[idx]
            
        new_indices = []
        shuffled_indices = indices[:]
        np.random.shuffle(shuffled_indices)
        
        for idx in shuffled_indices:
            counts -= M[idx]
            if np.min(counts) >= 1:
                pass
            else:
                counts += M[idx]
                new_indices.append(idx)
                
        return new_indices

    best_size = float('inf')
    best_code = []
    
    start_time = time.time()
    deadline = 100.0
    
    while time.time() - start_time < deadline:
        S = run_greedy()
        S = prune(S)
        
        if len(S) < best_size:
            best_size = len(S)
            best_code = [words[i] for i in S]
            
    return best_code
