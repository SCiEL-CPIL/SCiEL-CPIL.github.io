import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Precomputation
    # Mult[d] is a bitset (int) where bit i is set if i is a multiple of d.
    # Range is 2..256. We use 1-based indexing for bits? No, just use value as index.
    # Bit 0 is unused or for 1. Bit 2 for 2, ..., Bit 256 for 256.
    # Python ints have arbitrary precision, so this is fine.
    MAX_VAL = 256
    Mult = [0] * (MAX_VAL + 1)
    for d in range(2, MAX_VAL + 1):
        mask = 0
        for m in range(d, MAX_VAL + 1, d):
            mask |= (1 << m)
        Mult[d] = mask
    
    # Precompute gcds? gcd is fast enough.
    
    def is_valid_addition(S_bits, S_list, x):
        """
        Check if x can be added to S.
        S_bits: bitset of S
        S_list: list of elements in S
        x: candidate
        Returns True if valid.
        """
        # Condition A: For all a in S, a does not divide b*x for any b in S, b != a.
        # Equivalent: For all a in S, let k = a // gcd(a, x).
        # No b in S (b != a) is a multiple of k.
        for a in S_list:
            g = gcd(a, x)
            k = a // g
            # We need to check if any b in S (b != a) is a multiple of k.
            # Mult[k] contains all multiples of k.
            # Intersect with S_bits to see which elements in S are multiples of k.
            # If the result has any bit set other than a, it's a conflict.
            # Note: If k > MAX_VAL, Mult[k] is 0 (no multiples in range), so safe.
            if k <= MAX_VAL:
                conflict_mask = Mult[k] & S_bits
                # Remove a from conflict_mask to see if any other element is there
                conflict_mask &= ~(1 << a)
                if conflict_mask != 0:
                    return False
            else:
                # If k > MAX_VAL, no number in 2..256 is a multiple of k except potentially k itself if k<=256?
                # No, if k > 256, no multiple in range.
                # Wait, if k is large, e.g., 300, no multiple in 2..256.
                # So no conflict.
                pass
        
        # Condition B: For all b, c in S (b != c), x does not divide b*c.
        # Equivalent: For each b in S, let req = x // gcd(b, x).
        # No c in S (c != b) is a multiple of req.
        for b in S_list:
            g = gcd(b, x)
            req = x // g
            if req <= MAX_VAL:
                conflict_mask = Mult[req] & S_bits
                conflict_mask &= ~(1 << b)
                if conflict_mask != 0:
                    return False
            else:
                # If req > 256, no multiple in range.
                pass
        
        return True

    def greedy(order):
        S_bits = 0
        S_list = []
        for x in order:
            if time.time() > deadline:
                break
            if is_valid_addition(S_bits, S_list, x):
                S_bits |= (1 << x)
                S_list.append(x)
        return S_list

    def local_search(S_list, max_iter=100):
        S_bits = 0
        for x in S_list:
            S_bits |= (1 << x)
        
        all_nums = set(range(2, 257))
        current_set = set(S_list)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add
            for x in all_nums - current_set:
                if time.time() > deadline:
                    return S_list
                if is_valid_addition(S_bits, S_list, x):
                    S_bits |= (1 << x)
                    S_list.append(x)
                    current_set.add(x)
                    improved = True
                    break # Restart loop with new set
            
            if not improved:
                # Try to swap: remove a, add y
                # We want to maximize size, so swaps keep size same.
                # But maybe a swap allows adding a third element?
                # Let's try simple 1-swap first to unblock?
                # Actually, if we can't add, maybe removing a small "bad" element allows adding two?
                # That's complex. Let's stick to 1-swap if it leads to a state where we can add?
                # Or just try to find a better set of same size?
                # The objective is cardinality. So we only care about increasing size.
                # If we can't add, we are stuck in a local optimum for size.
                # We can try to remove one element and see if we can add two?
                # That's a 2-move.
                
                # Let's try removing each element and seeing if we can add any new element.
                # If we remove a and add y, size is same.
                # If we remove a and add y, and then add z, size increases.
                
                for a in S_list:
                    if time.time() > deadline:
                        return S_list
                    # Remove a
                    temp_bits = S_bits & ~(1 << a)
                    temp_list = [x for x in S_list if x != a]
                    temp_set = current_set - {a}
                    
                    # Try to add any y not in temp_set
                    added = False
                    for y in all_nums - temp_set:
                        if time.time() > deadline:
                            return S_list
                        if is_valid_addition(temp_bits, temp_list, y):
                            # We found a swap a -> y
                            # Now check if we can add another z?
                            # If we just swap, size is same.
                            # Let's update to this swapped state and continue search from there.
                            # This might open up possibilities.
                            S_bits = temp_bits | (1 << y)
                            S_list = temp_list + [y]
                            current_set = temp_set | {y}
                            improved = True
                            added = True
                            break
                    
                    if added:
                        break
            
        return S_list

    best_set = []
    
    # Strategy 1: Descending order
    order1 = list(range(256, 1, -1))
    S1 = greedy(order1)
    if len(S1) > len(best_set):
        best_set = S1
        
    # Strategy 2: Primes first, then composites descending
    primes = [a for a in range(2, 257) if all(a % d for d in range(2, int(a**0.5) + 1))]
    composites = [a for a in range(2, 257) if a not in primes]
    order2 = primes + sorted(composites, reverse=True)
    S2 = greedy(order2)
    if len(S2) > len(best_set):
        best_set = S2
        
    # Strategy 3: Random shuffles
    random.seed(42)
    all_nums = list(range(2, 257))
    for i in range(100): # More iterations possible due to speedup
        if time.time() > deadline:
            break
        random.shuffle(all_nums)
        Sr = greedy(all_nums)
        if len(Sr) > len(best_set):
            best_set = Sr
            
    # Apply local search to the best set found so far
    if best_set:
        improved_set = local_search(best_set)
        if len(improved_set) > len(best_set):
            best_set = improved_set

    # Final check
    # Convert to list of ints
    return [int(x) for x in best_set]
