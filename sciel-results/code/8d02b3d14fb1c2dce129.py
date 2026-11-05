import time

def construct():
    """
    Construct a nonempty subset S of [1, 256] maximizing cardinality
    such that for all a < b <= c in S, a does not divide b+c.
    
    Strategy:
    1. Start with the "safe" large interval [171, 256] which is guaranteed valid.
    2. Greedily add smaller elements from 170 down to 1 if they maintain validity.
    3. Use a sum-frequency array to speed up divisibility checks.
    4. Add a deadline guard to ensure we finish well before 60s.
    """
    n = 256
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin
    
    # Initialize with the guaranteed valid set [171, 256]
    S = list(range(171, n + 1))
    
    # Precompute sum frequency array for current S
    # sums[b+c] counts how many pairs (b,c) with b<=c give that sum
    max_sum = 2 * n  # max possible b+c is 256+256=512
    sum_freq = [0] * (max_sum + 1)
    
    def rebuild_sum_freq(S):
        """Rebuild the sum frequency array from scratch."""
        for i in range(len(sum_freq)):
            sum_freq[i] = 0
        L = len(S)
        for i in range(L):
            b = S[i]
            for j in range(i, L):
                c = S[j]
                sum_freq[b + c] += 1
    
    rebuild_sum_freq(S)
    
    # We will try to add elements from 170 down to 1.
    for x in range(170, 0, -1):
        if time.time() > deadline:
            break
        
        # Check if x divides any b+c for b<=c in S
        # Multiples of x in range [2*min(S), 2*max(S)]
        min_S = min(S)
        max_S = max(S)
        min_sum = 2 * min_S
        max_sum_val = 2 * max_S
        
        # Check each multiple of x in [min_sum, max_sum_val]
        valid = True
        m = (min_sum + x - 1) // x * x  # first multiple >= min_sum
        while m <= max_sum_val:
            if sum_freq[m] > 0:
                valid = False
                break
            m += x
        
        if valid:
            # Add x to S and update sum_freq
            S.append(x)
            # Update sum_freq with new pairs involving x
            # Pairs: (x, s) for s in S where s >= x (but x is smallest, so all s > x)
            # Actually, x < all current elements, so pairs are (x, s) for all s in S_old
            # And (x, x) is not possible since we only add once
            for s in S[:-1]:  # all elements except the newly added x
                sum_freq[x + s] += 1
            # Also need to account for the fact that x is now in S, 
            # but we don't add (x,x) because b<=c and we're checking distinct positions?
            # Wait, the condition allows b=c, so (x,x) would be a valid pair if x were added twice,
            # but we only add each number once. So no (x,x) pair.
            # But we do have pairs (x, s) for all s in old S, which we just added.
    
    # Final verification (optional, but good for safety if time permits)
    if time.time() < deadline - 5:
        S.sort()
        valid_final = True
        L = len(S)
        for i in range(L):
            a = S[i]
            for j in range(i + 1, L):
                b = S[j]
                for k in range(j, L):
                    c = S[k]
                    if (b + c) % a == 0:
                        valid_final = False
                        break
                if not valid_final:
                    break
            if not valid_final:
                break
        
        if not valid_final:
            # Fallback to the guaranteed valid set
            return list(range(171, 257))

    return S

# Test harness (not included in submission, just for local verification)
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
