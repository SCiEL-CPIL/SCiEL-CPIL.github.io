import time
import random
import sys

def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    best_seq = []
    best_len = 0
    
    # Precompute prime factorization for all numbers in [2, 128]
    # This allows fast product comparison via exponent vectors
    max_val = 128
    primes = []
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, max_val + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, max_val + 1, i):
                if j <= max_val:
                    is_prime[j] = False
    
    # Map each number to its prime exponent vector (tuple of exponents)
    # This makes multiplication and equality checks very fast
    num_to_vec = {}
    for n in range(2, max_val + 1):
        temp = n
        vec = [0] * len(primes)
        for i, p in enumerate(primes):
            while temp % p == 0:
                vec[i] += 1
                temp //= p
        num_to_vec[n] = tuple(vec)
    
    def add_vec(v1, v2):
        return tuple(a + b for a, b in zip(v1, v2))
    
    def check_validity(seq):
        """
        Verify that a sequence is strictly increasing and all consecutive block products are distinct.
        Returns True if valid, False otherwise.
        """
        if not seq:
            return True
        if seq != sorted(seq):
            return False
        if len(seq) != len(set(seq)):
            return False
        if min(seq) < 2 or max(seq) > 128:
            return False
            
        products = set()
        n = len(seq)
        for i in range(n):
            # Use vectors for comparison to avoid huge integers? 
            # Actually, integers up to 128^128 are huge but Python handles big ints.
            # However, vector comparison is O(P) where P is number of primes (~30).
            # Big int comparison is O(digits). For 128^128, digits ~ 240.
            # Vector is likely faster or comparable. Let's use vectors for consistency.
            vec = (0,) * len(primes)
            for j in range(i, n):
                vec = add_vec(vec, num_to_vec[seq[j]])
                if vec in products:
                    return False
                products.add(vec)
        return True

    def get_conflict_score(candidate, current_seq, current_prods):
        """
        Estimate how many future options 'candidate' blocks.
        We count how many numbers x > candidate (or just any x not in current_seq)
        would conflict if added after candidate.
        This is a heuristic for 'Most Constrained Variable' - pick the one that leaves
        the most options, i.e., minimizes conflicts.
        """
        if not current_seq:
            return 0
            
        conflicts = 0
        n = len(current_seq)
        last = current_seq[-1]
        
        # Check conflicts with existing products
        # New products formed by appending candidate:
        # [candidate], [last, candidate], ..., [current_seq, candidate]
        suffix_vec = (0,) * len(primes)
        new_prods = []
        for i in range(n - 1, -1, -1):
            suffix_vec = add_vec(suffix_vec, num_to_vec[current_seq[i]])
            p_vec = add_vec(suffix_vec, num_to_vec[candidate])
            new_prods.append(p_vec)
        new_prods.append(num_to_vec[candidate])
        
        # Check if any new prod is in current_prods
        for p in new_prods:
            if p in current_prods:
                return float('inf') # Invalid immediately
        
        # Check internal conflicts among new_prods
        if len(new_prods) != len(set(new_prods)):
            return float('inf')
        
        # Heuristic: Count how many potential next numbers (x > candidate) would be blocked
        # This is expensive, so we approximate.
        # A number x is blocked if any block ending in candidate * x creates a conflict,
        # or if x itself conflicts.
        # Simplified heuristic: Count how many primes in candidate's factorization are "rare"
        # or just use a simple random probe.
        
        # Better heuristic: Try a few random future numbers and see if they conflict.
        # But for speed, let's just return 0 for valid candidates and rely on the search order.
        # Actually, a good heuristic is: prefer numbers with fewer prime factors (simpler)
        # or numbers that are "coprime" to recent elements to reduce product collisions.
        
        # Let's use a "flexibility" score: 
        # How many numbers in [candidate+1, 128] are coprime to the last few elements?
        # This is a proxy for keeping options open.
        flexibility = 0
        for x in range(candidate + 1, 129):
            if x in current_seq:
                continue
            # Check if x can be appended to (current_seq + [candidate])
            # This is too expensive to do fully for every candidate.
            # Instead, just check if x is coprime to candidate.
            # If gcd(x, candidate) == 1, it's less likely to create product collisions with suffixes.
            if math.gcd(x, candidate) == 1:
                flexibility += 1
        return -flexibility # Minimize this score (maximize flexibility)

    import math

    def backtrack(current_seq, current_prods, available):
        """
        Recursive backtracking with forward checking.
        available: sorted list of numbers > current_seq[-1] that are candidates.
        """
        nonlocal best_seq, best_len
        
        if time.time() > deadline:
            return
            
        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = current_seq[:]
        
        if not available:
            return
            
        # Order available candidates by heuristic (least constraining / most flexible)
        # We want to try candidates that leave the most options.
        # Score: -1 if it conflicts, else some flexibility metric.
        scored_candidates = []
        last = current_seq[-1] if current_seq else 1
        
        for x in available:
            if x <= last:
                continue
            # Quick check: does x conflict with current_prods?
            # We need to check if appending x creates any duplicate product.
            n = len(current_seq)
            suffix_vec = (0,) * len(primes)
            new_prods = []
            valid = True
            for i in range(n - 1, -1, -1):
                suffix_vec = add_vec(suffix_vec, num_to_vec[current_seq[i]])
                p_vec = add_vec(suffix_vec, num_to_vec[x])
                if p_vec in current_prods:
                    valid = False
                    break
                new_prods.append(p_vec)
            if not valid:
                continue
            p_vec = num_to_vec[x]
            if p_vec in current_prods:
                continue
            new_prods.append(p_vec)
            if len(new_prods) != len(set(new_prods)):
                continue
            
            # Valid candidate. Calculate heuristic score.
            # Heuristic: Prefer numbers with smaller values? Or larger?
            # Usually, smaller numbers are more "flexible" because they have fewer prime factors?
            # Or maybe larger numbers are better because they are "further away" from collisions?
            # Let's try: Prefer numbers that are coprime to the last element.
            # This reduces the chance of product collisions with suffixes.
            if current_seq:
                gcd_val = math.gcd(x, current_seq[-1])
            else:
                gcd_val = 1
            
            # Score: lower is better. 
            # If gcd is 1, score is low (good). If gcd > 1, score is higher (bad).
            # Also, prefer smaller numbers? 
            # Let's use a combination: gcd_val * 10 + x
            score = gcd_val * 10 + x
            scored_candidates.append((score, x))
            
        if not scored_candidates:
            return
            
        # Sort by score
        scored_candidates.sort()
        
        # To avoid getting stuck in a single deep path, we can limit the branching factor
        # or use randomization.
        # For now, try the best few candidates.
        for score, x in scored_candidates[:5]: # Branching factor 5
            if time.time() > deadline:
                return
            
            # Recurse
            new_seq = current_seq + [x]
            new_prods = set(current_prods)
            
            # Add new products
            n = len(current_seq)
            suffix_vec = (0,) * len(primes)
            for i in range(n - 1, -1, -1):
                suffix_vec = add_vec(suffix_vec, num_to_vec[current_seq[i]])
                new_prods.add(add_vec(suffix_vec, num_to_vec[x]))
            new_prods.add(num_to_vec[x])
            
            # Update available: remove x and anything <= x? No, just remove x.
            # Since sequence must be increasing, available must be > x.
            new_available = [a for a in available if a > x]
            
            backtrack(new_seq, new_prods, new_available)
            
            if time.time() > deadline:
                return

    # Initial call
    initial_available = list(range(2, 129))
    backtrack([], set(), initial_available)
    
    # If backtracking didn't find anything (unlikely), fallback to simple greedy
    if not best_seq:
        seq = []
        prods = set()
        for x in range(2, 129):
            n = len(seq)
            suffix_vec = (0,) * len(primes)
            new_prods = []
            valid = True
            for i in range(n - 1, -1, -1):
                suffix_vec = add_vec(suffix_vec, num_to_vec[seq[i]])
                p_vec = add_vec(suffix_vec, num_to_vec[x])
                if p_vec in prods:
                    valid = False
                    break
                new_prods.append(p_vec)
            if not valid:
                continue
            p_vec = num_to_vec[x]
            if p_vec in prods:
                continue
            new_prods.append(p_vec)
            if len(new_prods) != len(set(new_prods)):
                continue
            seq.append(x)
            prods.update(new_prods)
        best_seq = seq
        best_len = len(seq)
        
    return best_seq

# Helper for gcd if math is not imported inside function scope properly (it is, but just in case)
import math

if __name__ == "__main__":
    seq = construct()
    print(f"Length: {len(seq)}")
    print(f"Sequence: {seq}")
