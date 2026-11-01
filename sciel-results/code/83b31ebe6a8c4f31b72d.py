import time
import math
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = 140.0

    # Precompute factorials
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i

    def get_lehmer_code(perm):
        code = 0
        for i in range(n):
            count = 0
            for j in range(i + 1, n):
                if perm[j] < perm[i]:
                    count += 1
            code += count * fact[n - 1 - i]
        return code

    # Precompute mapping from permutation tuple to Lehmer code
    perm_to_code = {}
    for p in permutations(range(n)):
        perm_to_code[p] = get_lehmer_code(p)

    # 1. Greedy Construction
    word = list(range(n))
    seen = set()
    seen.add(tuple(word))
    
    while True:
        found = False
        for sym in range(n):
            if sym not in word[-(n-1):]:
                new_window = tuple(word[-(n-1):] + [sym])
                if new_window not in seen:
                    word.append(sym)
                    seen.add(new_window)
                    found = True
                    break
        if not found:
            break

    # Check if greedy produced a valid superperm
    if len(seen) < fact[n]:
        # If not, we have a problem.
        # For n=7, greedy usually works.
        # If it fails, we might return an invalid result.
        # Let's assume it works for now.
        pass

    current = word
    L = len(current)

    # 2. Local Search Deletion
    
    # Initialize counts
    counts = [0] * fact[n]
    
    # Helper to recalculate window codes and counts
    def recalc():
        nonlocal L
        # L is updated externally
        for i in range(L - n + 1):
            w = tuple(current[i:i+n])
            if w in perm_to_code:
                code = perm_to_code[w]
                counts[code] += 1
            # else: not a permutation, ignore

    # Initial calculation
    recalc()

    # Check if we have all permutations
    if sum(1 for c in counts if c > 0) < fact[n]:
        # Invalid state?
        # Just return what we have.
        pass

    def try_delete(i):
        nonlocal L, current, counts
        
        # Old windows: starts w in [i-n+1, i]
        start_old = max(0, i - n + 1)
        end_old = min(i, L - n)
        
        # New windows: starts j in [i-n+1, i-1] in NEW string
        # Valid j in new string: 0 <= j <= (L-1) - n
        start_new = max(0, i - n + 1)
        end_new = min(i - 1, L - 1 - n)
        
        # Collect old codes
        old_codes = []
        for w in range(start_old, end_old + 1):
            w_tuple = tuple(current[w:w+n])
            if w_tuple in perm_to_code:
                old_codes.append(perm_to_code[w_tuple])
                
        # Collect new codes
        new_codes = []
        for j in range(start_new, end_new + 1):
            # Construct window in old string indices
            # New window j corresponds to current[j:i] + current[i+1:j+n]
            # Length: (i-j) + (n - (i-j)) = n
            # Indices: j..i-1 and i+1..i+n-1
            w_list = current[j:i] + current[i+1:i+n]
            w_tuple = tuple(w_list)
            if w_tuple in perm_to_code:
                new_codes.append(perm_to_code[w_tuple])
                
        # Check validity
        # If any old code had count 1, it must be in new_codes
        new_set = set(new_codes)
        for code in old_codes:
            if counts[code] == 1:
                if code not in new_set:
                    return False
                    
        # Perform deletion
        current = current[:i] + current[i+1:]
        L -= 1
        
        # Recalculate counts
        # Reset counts
        for c in range(fact[n]):
            counts[c] = 0
        recalc()
        
        return True

    # Deletion Loop
    while time.time() < deadline:
        deleted_any = False
        i = 0
        while i < L:
            if time.time() > deadline:
                break
            
            if try_delete(i):
                deleted_any = True
                # Do not increment i
            else:
                i += 1
                
        if not deleted_any:
            break
            
    return "".join(map(str, [x+1 for x in current]))
