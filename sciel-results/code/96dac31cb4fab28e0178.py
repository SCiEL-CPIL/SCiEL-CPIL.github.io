import itertools
import time

def construct(n):
    if n == 1:
        return "1"
    
    # Generate all permutations
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Precompute prefix maps
    # D[k] maps prefix tuple of length k to list of perms starting with it
    D = [None] * (n + 1)
    for k in range(1, n + 1):
        D[k] = {}
        for p in perms:
            prefix = p[:k]
            if prefix not in D[k]:
                D[k][prefix] = []
            D[k][prefix].append(p)
    
    # Greedy construction
    # Start with the first permutation
    current = perms[0]
    word = list(current)
    unused = set(perms[1:])
    
    start_time = time.time()
    timeout = 100.0 # Safety margin
    
    for _ in range(N - 1):
        if time.time() - start_time > timeout:
            break
            
        found = False
        # Try to find max overlap k
        # k can be n-1 down to 1 (k=n implies same perm, which is used)
        for k in range(n - 1, 0, -1):
            suffix = current[n - k:]
            candidates = D[k].get(suffix, [])
            
            # Find an unused candidate
            for p in candidates:
                if p in unused:
                    # Use this permutation
                    unused.remove(p)
                    # Append the non-overlapping part
                    word.extend(p[k:])
                    current = p
                    found = True
                    break
            if found:
                break
        
        if not found:
            # Fallback: pick any unused perm (should not happen often if n>=2)
            # Just pick the first one in unused
            # This breaks the "ends with perm" invariant? No, we append full perm.
            # But we need to update current.
            # Actually, if k=0, we append full perm.
            # Let's just pick one.
            p = next(iter(unused))
            unused.remove(p)
            word.extend(p)
            current = p

    return "".join(map(str, word))
