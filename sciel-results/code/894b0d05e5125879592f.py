import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # 1. Generate standard recursive superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
        
    S = list(word)
    L = len(S)
    
    base = n + 1
    base_powers = [base**i for i in range(n)]
    
    def encode_window(S, start):
        val = 0
        for j in range(n):
            val += S[start + j] * base_powers[j]
        return val

    counts = {}
    for i in range(L - n + 1):
        w = encode_window(S, i)
        counts[w] = counts.get(w, 0) + 1
        
    best_S = S[:]
    best_len = L
    
    def get_affected(S, i, rem_len):
        if rem_len == 1:
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)
            old = [encode_window(S, j) for j in range(start_old, end_old + 1)]
            
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, len(S) - 1 - n)
            new = []
            for k in range(start_new, end_new + 1):
                val = 0
                for j in range(n):
                    idx = k + j
                    if idx >= i: idx += 1
                    val += S[idx] * base_powers[j]
                new.append(val)
            return old, new
        elif rem_len == 2:
            start_old = max(0, i - n + 1)
            end_old = min(i + 1, len(S) - n)
            old = [encode_window(S, j) for j in range(start_old, end_old + 1)]
            
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, len(S) - 2 - n)
            new = []
            for k in range(start_new, end_new + 1):
                val = 0
                for j in range(n):
                    idx = k + j
                    if idx >= i: idx += 2
                    val += S[idx] * base_powers[j]
                new.append(val)
            return old, new
        else:
            return [], []

    def try_delete(S, counts, i, rem_len):
        if i < 0 or i + rem_len > len(S):
            return False, S, counts
        old, new = get_affected(S, i, rem_len)
        new_set = set(new)
        for p in old:
            if counts.get(p, 0) == 1 and p not in new_set:
                return False, S, counts
        for p in old:
            counts[p] -= 1
            if counts[p] == 0: del counts[p]
        for p in new:
            counts[p] = counts.get(p, 0) + 1
        return True, S[:i] + S[i+rem_len:], counts

    # SA Loop
    T = 1.0
    while time.time() < deadline:
        T *= 0.995
        if T < 0.01: T = 0.01
        
        improved = False
        # Try random deletions
        for _ in range(2000):
            if time.time() > deadline: break
            if len(S) <= n: break
            i = random.randint(0, len(S) - 1)
            rem = 1 if random.random() > 0.1 else 2
            valid, new_S, new_counts = try_delete(S, counts, i, rem)
            if valid:
                S = new_S
                counts = new_counts
                improved = True
                if len(S) < best_len:
                    best_len = len(S)
                    best_S = S[:]
        
        if not improved:
            # Perturbation: Insert
            if len(S) < 8000: # Avoid memory blowup
                i = random.randint(0, len(S))
                val = random.randint(1, n)
                new_S = S[:i] + [val] + S[i:]
                # Recalculate counts for safety and simplicity if L is not huge
                # But L is ~6000. Recalc is O(6000*7) = 42k ops. Fast.
                new_counts = {}
                for j in range(len(new_S) - n + 1):
                    w = encode_window(new_S, j)
                    new_counts[w] = new_counts.get(w, 0) + 1
                # Accept if random < T
                if random.random() < T:
                    S = new_S
                    counts = new_counts

    return "".join(map(str, best_S))
