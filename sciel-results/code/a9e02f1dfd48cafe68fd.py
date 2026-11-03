import itertools
import time

def construct(n):
    if n == 1:
        return "1"
    
    perms = list(itertools.permutations(range(1, n + 1)))
    remaining = set(perms)
    
    # Pre-compute suffix indices for all permutations?
    # No, we build them dynamically or pre-compute for all and update.
    # Pre-computing for all is easier.
    # suffix_indices[k] is a dict: suffix_tuple -> set of perms
    
    suffix_indices = [None] * n
    for k in range(1, n):
        suffix_indices[k] = {}
        for p in perms:
            s = p[:k]
            if s not in suffix_indices[k]:
                suffix_indices[k][s] = set()
            suffix_indices[k][s].add(p)
            
    current = perms[0]
    word = list(current)
    remaining.remove(current)
    
    # Update suffix_indices
    for k in range(1, n):
        s = current[:k]
        if s in suffix_indices[k]:
            suffix_indices[k][s].discard(current)
            if not suffix_indices[k][s]:
                del suffix_indices[k][s]
                
    while remaining:
        found = False
        for k in range(n - 1, 0, -1):
            suffix = tuple(word[-k:])
            candidates = suffix_indices[k].get(suffix, None)
            if candidates:
                # Pick one candidate
                p = next(iter(candidates))
                # Append
                word.extend(p[k:])
                remaining.remove(p)
                
                # Update suffix_indices for p
                for kk in range(1, n):
                    s = p[:kk]
                    if s in suffix_indices[kk]:
                        suffix_indices[kk][s].discard(p)
                        if not suffix_indices[kk][s]:
                            del suffix_indices[kk][s]
                            
                found = True
                break
        if not found:
            break
            
    return "".join(map(str, word))
