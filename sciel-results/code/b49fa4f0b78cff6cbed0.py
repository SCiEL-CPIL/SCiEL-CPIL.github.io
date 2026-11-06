import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    # All numbers in [2, 128]
    all_numbers = list(range(2, 129))
    primes = [p for p in all_numbers if is_prime(p)]
    composites = [c for c in all_numbers if not is_prime(c)]
    
    # Start with all primes (known valid, length 30)
    best_seq = sorted(primes)
    best_len = len(best_seq)
    
    def is_valid_fast(seq):
        """Check if all consecutive block products are distinct using efficient method."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True
    
    def num_distinct_prime_factors(n):
        count = 0
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count += 1
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            count += 1
        return count
    
    # Greedy insertion: try adding composites in order of "safety"
    # Prefer composites with fewer distinct prime factors (prime powers are safest)
    composites_sorted = sorted(composites, key=lambda x: num_distinct_prime_factors(x))
    
    current_seq = list(best_seq)
    for c in composites_sorted:
        if time.time() > deadline:
            break
        
        # Binary search for insertion point (strictly increasing)
        lo, hi = 0, len(current_seq)
        while lo < hi:
            mid = (lo + hi) // 2
            if current_seq[mid] < c:
                lo = mid + 1
            else:
                hi = mid
        insert_pos = lo
        
        if insert_pos < len(current_seq) and current_seq[insert_pos] == c:
            continue
        
        new_seq = current_seq[:insert_pos] + [c] + current_seq[insert_pos:]
        
        if is_valid_fast(new_seq):
            current_seq = new_seq
    
    if len(current_seq) > best_len:
        best_len = len(current_seq)
        best_seq = current_seq
    
    # Try multiple greedy passes with different orderings
    for trial in range(8):
        if time.time() > deadline:
            break
        
        if trial % 2 == 0:
            order = sorted(composites, key=lambda x: num_distinct_prime_factors(x))
        else:
            order = sorted(composites, key=lambda x: -num_distinct_prime_factors(x))
        
        current_seq = list(primes)
        for c in order:
            if time.time() > deadline:
                break
            
            lo, hi = 0, len(current_seq)
            while lo < hi:
                mid = (lo + hi) // 2
                if current_seq[mid] < c:
                    lo = mid + 1
                else:
                    hi = mid
            insert_pos = lo
            
            if insert_pos < len(current_seq) and current_seq[insert_pos] == c:
                continue
            
            new_seq = current_seq[:insert_pos] + [c] + current_seq[insert_pos:]
            
            if is_valid_fast(new_seq):
                current_seq = new_seq
        
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = current_seq
    
    # Backtracking search: try to find longer sequences by removing some primes and adding more composites
    # This is expensive, so only do it if we have time
    if time.time() < deadline - 10.0:
        # Try removing one prime at a time and see if we can add 2+ composites
        for i in range(len(primes)):
            if time.time() > deadline:
                break
            
            # Remove prime i
            seq_without = primes[:i] + primes[i+1:]
            removed_prime = primes[i]
            
            # Try to add composites to this reduced sequence
            current_seq = list(seq_without)
            composites_to_try = [c for c in composites if c != removed_prime]
            composites_to_try.sort(key=lambda x: num_distinct_prime_factors(x))
            
            added_count = 0
            for c in composites_to_try:
                if time.time() > deadline:
                    break
                
                lo, hi = 0, len(current_seq)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if current_seq[mid] < c:
                        lo = mid + 1
                    else:
                        hi = mid
                insert_pos = lo
                
                if insert_pos < len(current_seq) and current_seq[insert_pos] == c:
                    continue
                
                new_seq = current_seq[:insert_pos] + [c] + current_seq[insert_pos:]
                
                if is_valid_fast(new_seq):
                    current_seq = new_seq
                    added_count += 1
            
            # If we added more composites than we removed primes, we have a better sequence
            if len(current_seq) > best_len:
                best_len = len(current_seq)
                best_seq = current_seq
    
    return sorted(best_seq)
