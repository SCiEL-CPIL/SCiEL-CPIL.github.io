import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    candidates = list(range(2, 129))
    
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0:
                return False
        return True
        
    primes = [x for x in candidates if is_prime(x)]
    
    def get_all_products(seq):
        """Compute all nonempty consecutive block products for a sequence."""
        prods = set()
        for i in range(len(seq)):
            p = 1
            for j in range(i, len(seq)):
                p *= seq[j]
                prods.add(p)
        return prods
    
    def check_insertion_valid(seq, pos, c, existing_prods):
        """
        Check if inserting c at position pos in seq maintains distinct block products.
        Only checks new products involving c and verifies they don't conflict with existing.
        Returns True if valid.
        """
        n = len(seq)
        # New products involving c:
        # 1. Products starting at pos: c * seq[pos] * ... * seq[j] for j >= pos
        # 2. Products ending at pos: seq[i] * ... * seq[pos-1] * c for i < pos
        # 3. Products spanning c: seq[i] * ... * seq[pos-1] * c * seq[pos] * ... * seq[j]
        
        # First, collect all new products
        new_prods = set()
        
        # Products that start at c (position pos) and extend right
        p = c
        new_prods.add(p)
        for j in range(pos, n):
            p *= seq[j]
            new_prods.add(p)
        
        # Products that end at c (position pos) and extend left
        p = c
        for i in range(pos - 1, -1, -1):
            p *= seq[i]
            new_prods.add(p)
        
        # Products that span c: left part * c * right part
        # Left part: seq[i]*...*seq[pos-1], right part: seq[pos]*...*seq[j]
        # Compute left products from pos-1 going left
        left_prods = [1]
        lp = 1
        for i in range(pos - 1, -1, -1):
            lp *= seq[i]
            left_prods.append(lp)
        
        # Compute right products from pos going right
        right_prods = [1]
        rp = 1
        for j in range(pos, n):
            rp *= seq[j]
            right_prods.append(rp)
        
        # All spanning products: left * c * right
        for l in left_prods:
            for r in right_prods:
                new_prods.add(l * c * r)
        
        # Check for duplicates within new_prods
        if len(new_prods) != len(new_prods):  # This will never be true, but let's check differently
            return False
        
        # Actually, we need to check if any new product already exists in existing_prods
        # or if there are duplicates among the new products themselves
        # Since new_prods is a set, duplicates among new are already handled
        # But we need to check overlap with existing_prods
        for p in new_prods:
            if p in existing_prods:
                return False
        
        return True
    
    def try_insertion_fast(seq, c, existing_prods):
        """Try to insert c into seq at any valid position. Returns (new_seq, new_prods, success)."""
        n = len(seq)
        for pos in range(n + 1):
            # Check strictly increasing
            if pos > 0 and seq[pos-1] >= c:
                continue
            if pos < n and c >= seq[pos]:
                continue
            
            if check_insertion_valid(seq, pos, c, existing_prods):
                new_seq = seq[:pos] + [c] + seq[pos:]
                # Recompute all products for the new sequence
                new_prods = get_all_products(new_seq)
                expected_count = len(new_seq) * (len(new_seq) + 1) // 2
                if len(new_prods) == expected_count:
                    return new_seq, new_prods, True
        return None, None, False
    
    best_seq = primes
    best_len = len(best_seq)
    best_prods = get_all_products(best_seq)
    
    # Multiple restarts with different random orders
    for restart in range(30):
        if time.time() > deadline - 5:
            break
            
        current_seq = primes[:]
        current_prods = get_all_products(current_seq)
        used = set(current_seq)
        remaining = [x for x in candidates if x not in used]
        random.shuffle(remaining)
        
        # Greedy insertion
        progress = True
        while progress:
            progress = False
            for idx in range(len(remaining)):
                if time.time() > deadline - 2:
                    break
                c = remaining[idx]
                new_seq, new_prods, success = try_insertion_fast(current_seq, c, current_prods)
                if success:
                    current_seq = new_seq
                    current_prods = new_prods
                    used.add(c)
                    remaining.pop(idx)
                    if len(current_seq) > best_len:
                        best_seq = current_seq[:]
                        best_len = len(current_seq)
                        best_prods = current_prods
                    progress = True
                    break # Restart the loop with the new sequence
            
        if time.time() > deadline - 5:
            break

    return best_seq
