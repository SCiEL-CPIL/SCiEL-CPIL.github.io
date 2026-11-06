import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    p = 101
    # Precompute modular inverses
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, p)
    
    # best_subset[r] = smallest-size subset with inverse-sum == r mod 101
    best_subset = [None] * 101
    
    # Iterate by subset size k from 1 to 10
    # First subset found for each residue at size k is optimal (smallest possible)
    for k in range(1, 11):
        if time.time() > deadline:
            break
        for combo in itertools.combinations(range(1, 11), k):
            residue = sum(inv[x] for x in combo) % p
            if best_subset[residue] is None:
                best_subset[residue] = list(combo)
    
    # Build result
    result = []
    for r in range(101):
        if best_subset[r] is not None:
            result.append(best_subset[r])
        else:
            # Should not happen per problem guarantee, but handle gracefully
            # Find any subset that gives residue r
            found = False
            for k2 in range(1, 11):
                for combo in itertools.combinations(range(1, 11), k2):
                    if sum(inv[x] for x in combo) % p == r:
                        result.append(list(combo))
                        found = True
                        break
                if found:
                    break
            if not found:
                # Absolute fallback (should never occur)
                result.append([1])
    
    return result
