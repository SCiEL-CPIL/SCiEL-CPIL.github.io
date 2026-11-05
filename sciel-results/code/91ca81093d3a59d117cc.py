import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # Precompute sums and divisibility?
    # For a given x, and current set S, checking validity is O(|S|^2) worst case.
    # With |S| ~ 140, that's ~20k ops.
    # If we do 1000 swaps, that's 20M ops. Python might be slow.
    # We need to optimize the conflict check.

    def is_valid(S):
        """Verify the set S is valid. Returns True/False."""
        L = sorted(S)
        for i in range(len(L)):
            a = L[i]
            for j in range(i + 1, len(L)):
                b = L[j]
                for k in range(j, len(L)):
                    c = L[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add(x, S):
        """Check if x can be added to S."""
        # x is a: a < b <= c
        # b, c in S, b > x, c >= b
        # (b+c) % x != 0
        for b in S:
            if b <= x:
                continue
            for c in S:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # x is b: a < b <= c
        # a, c in S, a < x, c >= x
        # (x+c) % a != 0
        for a in S:
            if a >= x:
                continue
            for c in S:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # x is c: a < b <= c
        # a, b in S, a < x, b <= x (and b > a)
        # (b+x) % a != 0
        for a in S:
            if a >= x:
                continue
            for b in S:
                if b <= a or b > x:
                    continue
                if (b + x) % a == 0:
                    return False
        
        return True

    def get_blockers(r, S):
        """Find elements in S that block r."""
        blockers = set()
        
        # 1. r is a: pairs (b,c) in S with b>r, c>=b and r | b+c
        for b in S:
            if b <= r:
                continue
            for c in S:
                if c < b:
                    continue
                if (b + c) % r == 0:
                    blockers.add(b)
                    blockers.add(c)
        
        # 2. r is b: pairs (a,c) in S with a<r, c>=r and a | r+c
        for a in S:
            if a >= r:
                continue
            for c in S:
                if c < r:
                    continue
                if (r + c) % a == 0:
                    blockers.add(a)
                    blockers.add(c)
        
        # 3. r is c: pairs (a,b) in S with a<b<=r and a | b+r
        for a in S:
            if a >= r:
                continue
            for b in S:
                if b <= a or b > r:
                    continue
                if (b + r) % a == 0:
                    blockers.add(a)
                    blockers.add(b)
        
        return blockers

    def local_search(S, R):
        """
        S: current set (list or set)
        R: remaining elements (list)
        Returns improved S, R
        """
        S = set(S)
        R = [x for x in R if x not in S]
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Try to add
            for i in range(len(R)):
                if time.time() > deadline - 5:
                    break
                r = R[i]
                if can_add(r, S):
                    S.add(r)
                    R.pop(i)
                    improved = True
                    break
            
            if improved:
                continue

            # 2. Try to swap: remove s, add r
            # Strategy: Pick a random r from R. Find blockers. Remove a blocker.
            if R:
                r = random.choice(R)
                blockers = get_blockers(r, S)
                if blockers:
                    # Try each blocker to see if removing it allows r
                    # To be faster, just try one or a few
                    for s in blockers:
                        if can_add(r, S - {s}):
                            S.remove(s)
                            S.add(r)
                            R.remove(r)
                            R.append(s)
                            improved = True
                            break
                            # If we found one, we break the blocker loop
                            # and continue the main loop (improved=True)
                    
                    # If no blocker removal worked, we might try a 2-swap?
                    # For now, stick to 1-swap to keep it fast.
        
        return S, R

    best_S = set()
    best_size = 0
    
    # Multi-start
    random.seed(42)
    for trial in range(100): # Many trials
        if time.time() > deadline - 10:
            break
        
        # Different initializations
        if trial % 3 == 0:
            # Start with top half
            start_set = set(range(128, 257))
            # Prune invalid elements greedily from start_set?
            # Or just let local search fix it?
            # Actually, if start_set is invalid, can_add might behave weirdly if we assume S is valid.
            # We must ensure S is always valid during local search.
            # So we need to construct a valid initial set.
            
            # Greedy construction of valid set from a candidate pool
            S = set()
            # Add elements in a specific order
            # Order: Descending? Ascending?
            # Descending is usually better for "large summand" constraints
            candidates = list(range(256, 0, -1))
            random.shuffle(candidates) # Randomize order for diversity
            
            for x in candidates:
                if time.time() > deadline - 10:
                    break
                if can_add(x, S):
                    S.add(x)
            
            R = [x for x in range(1, 257) if x not in S]
            
        elif trial % 3 == 1:
            # Start with top 1/3
            S = set()
            candidates = list(range(170, 257)) + list(range(1, 170))
            random.shuffle(candidates)
            for x in candidates:
                if time.time() > deadline - 10:
                    break
                if can_add(x, S):
                    S.add(x)
            R = [x for x in range(1, 257) if x not in S]
            
        else:
            # Random subset of top half, then fill
            S = set()
            top_half = list(range(128, 257))
            random.shuffle(top_half)
            for x in top_half:
                if time.time() > deadline - 10:
                    break
                if can_add(x, S):
                    S.add(x)
            # Then try to add smaller ones
            bottom_half = list(range(1, 128))
            random.shuffle(bottom_half)
            for x in bottom_half:
                if time.time() > deadline - 10:
                    break
                if can_add(x, S):
                    S.add(x)
            R = [x for x in range(1, 257) if x not in S]

        # Local Search
        S, R = local_search(S, R)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

    return sorted(best_S)

if __name__ == "__main__":
    t0 = time.time()
    S = construct()
    t1 = time.time()
    print(f"Size: {len(S)}, Time: {t1-t0:.4f}s")
    # Verify
    valid = True
    L = len(S)
    for i in range(L):
        a = S[i]
        for j in range(i+1, L):
            b = S[j]
            for k in range(j, L):
                c = S[k]
                if (b+c) % a == 0:
                    valid = False
                    print(f"Invalid: {a} divides {b}+{c}")
                    break
            if not valid: break
        if not valid: break
    print("Valid" if valid else "Invalid")
