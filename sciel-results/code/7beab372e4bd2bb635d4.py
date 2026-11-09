import random

N = 256

def construct():
    best_set = None
    best_size = N + 1
    
    # We run multiple randomized greedy attempts
    # Each attempt uses a random permutation of [1, N]
    # and greedily adds elements if they maintain the Sidon property.
    # This guarantees an inclusion-maximal set.
    
    # Number of attempts: 2000 should be fast enough
    # and likely to find a small set.
    for _ in range(2000):
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        selected = []
        sums = set()
        
        for x in candidates:
            # Check if x can be added
            # New sums are x + s for s in selected, and 2*x
            # We need to ensure all these are distinct and not in existing sums
            new_sums = []
            valid = True
            for s in selected:
                val = x + s
                if val in sums or val in new_sums:
                    valid = False
                    break
                new_sums.append(val)
            
            if valid:
                val = 2 * x
                if val in sums or val in new_sums:
                    valid = False
                else:
                    new_sums.append(val)
            
            if valid:
                selected.append(x)
                sums.update(new_sums)
                
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            # Optional: early termination if we hit a theoretical lower bound?
            # Max Sidon set size is ~sqrt(N). Maximal sets can be smaller.
            # For N=256, sqrt(256)=16.
            # A maximal Sidon set can be as small as ~log(N) or similar?
            # Actually, the minimum size of a maximal Sidon set is not trivially bounded below by sqrt(N).
            # But typically it's around 10-15 for N=256.
            # If we find size 10, it's very good.
            if best_size <= 10:
                break

    return best_set
