import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill
    
    # Precompute primes up to 128
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    all_numbers = list(range(2, 129))
    primes = [x for x in all_numbers if is_prime(x)]
    
    # Precompute prime exponent vectors for all numbers 2..128
    prime_list = primes
    num_primes = len(prime_list)
    prime_to_idx = {p: i for i, p in enumerate(prime_list)}
    
    # Exponent vector for each number
    exp_vecs = {}
    for n in all_numbers:
        vec = [0] * num_primes
        temp = n
        for pi, p in enumerate(prime_list):
            if p * p > temp:
                break
            while temp % p == 0:
                vec[pi] += 1
                temp //= p
        if temp > 1:
            vec[prime_to_idx[temp]] += 1
        exp_vecs[n] = tuple(vec)
        
    def add_vecs(v1, v2):
        return tuple(a + b for a, b in zip(v1, v2))

    def check_sequence_valid(seq):
        """
        Check if all consecutive block products are distinct.
        Uses prefix sums of exponent vectors for O(N^2) checks.
        """
        n = len(seq)
        if n == 0:
            return True
        
        # Prefix sum of exponent vectors
        # prefix[i] = sum of exp_vecs[seq[0]] ... exp_vecs[seq[i-1]]
        prefix = [tuple(0 for _ in range(num_primes))]
        for i in range(n):
            prev = prefix[-1]
            curr_vec = exp_vecs[seq[i]]
            new_prefix = tuple(prev[j] + curr_vec[j] for j in range(num_primes))
            prefix.append(new_prefix)
            
        prods = set()
        for i in range(n):
            for j in range(i + 1, n + 1):
                # Product of seq[i:j] is prefix[j] - prefix[i]
                vec = tuple(prefix[j][k] - prefix[i][k] for k in range(num_primes))
                if vec in prods:
                    return False
                prods.add(vec)
        return True

    def try_insert(seq, c):
        """
        Try to insert c into seq at a valid position.
        Returns new_seq if successful, else None.
        """
        n = len(seq)
        valid_positions = []
        for i in range(n + 1):
            if i > 0 and seq[i-1] >= c:
                continue
            if i < n and c >= seq[i]:
                continue
            valid_positions.append(i)
        
        if not valid_positions:
            return None
        
        # Sort by centrality to find "dense" spots first? Or just random?
        # Let's try random order to avoid bias, or sort by position.
        # Actually, checking validity is O(N^2), so we want to minimize checks.
        # Let's just try them in order.
        random.shuffle(valid_positions)
        
        for i in valid_positions:
            new_seq = seq[:i] + [c] + seq[i:]
            if check_sequence_valid(new_seq):
                return new_seq
        return None

    def greedy_construct():
        seq = []
        for num in all_numbers:
            # Try to add num to the end (since we iterate in increasing order, appending maintains sorted order)
            # But we can also insert? No, greedy usually appends.
            # Since we want strictly increasing, and we iterate 2..128, appending is the only valid move if we haven't skipped.
            # Wait, if we skip a number, we can't go back.
            # So greedy is: for each num, if appending num to current seq is valid, do it.
            new_seq = seq + [num]
            if check_sequence_valid(new_seq):
                seq = new_seq
        return seq

    def local_search(seq, max_iter=500):
        current = seq[:]
        used = set(current)
        remaining = [x for x in all_numbers if x not in used]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            random.shuffle(remaining)
            for c in remaining:
                new_seq = try_insert(current, c)
                if new_seq is not None:
                    current = new_seq
                    used = set(current)
                    remaining = [x for x in all_numbers if x not in used]
                    improved = True
                    break
            if not improved:
                break
        return current

    best_seq = []
    
    # 1. Greedy Initialization
    # Try a few different greedy strategies?
    # Standard greedy: smallest to largest.
    seq1 = greedy_construct()
    
    # Try greedy from largest to smallest?
    # Since sequence must be increasing, we can't just reverse.
    # But we can try to build a set and sort it.
    # Let's stick to one good greedy: Standard.
    
    if len(seq1) > len(best_seq):
        best_seq = seq1
        
    # 2. Local Search Improvement
    if time.time() < deadline - 5:
        improved_seq = local_search(best_seq)
        if len(improved_seq) > len(best_seq):
            best_seq = improved_seq
            
    # 3. Random Restarts if time permits
    # We can randomize the greedy order slightly?
    # Or just run local search on different random subsets?
    # Let's try a few random restarts of greedy with some skipping probability.
    
    for _ in range(10):
        if time.time() > deadline - 5:
            break
            
        # Randomized Greedy: with small prob skip a number even if valid?
        # Or just randomize the local search starting point.
        # Let's try starting with a random subset of the best sequence and expanding.
        
        # Or better: Randomize the order of "remaining" in local search more aggressively.
        # The current local search is pretty good.
        
        # Let's try a different initialization:
        # Start with all primes, then add composites.
        seq2 = []
        for num in all_numbers:
             if is_prime(num):
                 new_seq = seq2 + [num]
                 if check_sequence_valid(new_seq):
                     seq2 = new_seq
        # Now add composites
        for num in all_numbers:
             if not is_prime(num):
                 # Try inserting
                 # This is expensive. Let's just append if valid?
                 # No, must be increasing.
                 # If we just append, it might not be optimal.
                 # Let's use try_insert for composites on the prime-based seq.
                 new_seq = try_insert(seq2, num)
                 if new_seq:
                     seq2 = new_seq
                 
        if len(seq2) > len(best_seq):
            best_seq = seq2
            # Improve this new best
            if time.time() < deadline - 5:
                improved_seq = local_search(best_seq)
                if len(improved_seq) > len(best_seq):
                    best_seq = improved_seq

    return best_seq
