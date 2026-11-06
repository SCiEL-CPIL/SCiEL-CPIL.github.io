import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin

    # 1. Precomputation
    # Primes up to 128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    P = len(primes)
    
    sigs = [None] * 129
    for i in range(2, 129):
        sig = [0] * P
        n = i
        for idx, p in enumerate(primes):
            if p * p > n:
                break
            while n % p == 0:
                sig[idx] += 1
                n //= p
        if n > 1:
            idx = prime_to_idx[n]
            sig[idx] += 1
        sigs[i] = tuple(sig)

    def add_sig(a, b):
        return tuple(x + y for x, y in zip(a, b))

    def is_valid_sequence(vals):
        """
        Check if the sequence of values has distinct products for all non-empty consecutive blocks.
        vals must be strictly increasing.
        """
        if not vals:
            return True
        
        # We need to check all subarrays.
        # Number of subarrays is N*(N+1)/2.
        # We store the signature of the product.
        seen = set()
        n = len(vals)
        for i in range(n):
            current_prod_sig = [0] * P
            for j in range(i, n):
                # Add vals[j] to current_prod_sig
                for k in range(P):
                    current_prod_sig[k] += sigs[vals[j]][k]
                sig_tuple = tuple(current_prod_sig)
                if sig_tuple in seen:
                    return False
                seen.add(sig_tuple)
        return True

    def greedy_construct(skip_prob=0.0, seed=42):
        """
        Greedily build a sequence by iterating 2..128.
        If adding x is valid, add it.
        With probability skip_prob, skip x even if valid (to explore).
        """
        random.seed(seed)
        seq = []
        for x in range(2, 129):
            # Check if adding x is valid
            # Optimization: Only check subarrays ending at x
            # But is_valid_sequence is simple enough for small N.
            # Let's use a faster check for the greedy step.
            
            # Fast check for adding x to the end of seq
            # We need to check if any new subarray product ending at x collides with existing products.
            # Existing products are all subarray products of seq.
            # New products are:
            # [x], [seq[-1], x], [seq[-2], seq[-1], x], ...
            
            # To make this fast, we should maintain the set of existing products.
            # But greedy_construct is called multiple times.
            # Let's just use is_valid_sequence for simplicity first, 
            # but that's O(N^3) per step? No, O(N^2) per step.
            # Total O(N^3) for full greedy. N=127. 127^3 = 2M. Python might be slow.
            
            # Let's implement a faster incremental check.
            
            # Actually, let's stick to is_valid_sequence but optimize the inner loop.
            # Or better, implement a function `can_add(seq, x)` that is O(N^2) worst case but usually fast.
            
            # Let's try a hybrid:
            # Maintain `seq_sigs` and `products_set`?
            # If we do that, we can't easily restart with different skip patterns without re-running.
            
            # Given the time limit, let's try to make `is_valid_sequence` as fast as possible.
            # Using tuples of integers is okay.
            
            if random.random() < skip_prob:
                continue
            
            candidate_seq = seq + [x]
            if is_valid_sequence(candidate_seq):
                seq = candidate_seq
            # If invalid, we simply skip x. We don't backtrack in greedy.
        return seq

    def local_search(seq):
        """
        Try to improve the sequence by inserting excluded elements.
        """
        current_seq = list(seq)
        current_set = set(current_seq)
        excluded = [x for x in range(2, 129) if x not in current_set]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to insert each excluded element
            # To avoid O(N^2) checks, we can just try them one by one.
            for z in excluded:
                if time.time() > deadline:
                    break
                # Find insertion position
                # Since current_seq is sorted, we can use bisect
                import bisect
                pos = bisect.bisect_left(current_seq, z)
                candidate_seq = current_seq[:pos] + [z] + current_seq[pos:]
                
                if is_valid_sequence(candidate_seq):
                    current_seq = candidate_seq
                    current_set.add(z)
                    excluded.remove(z)
                    improved = True
                    # Restart the loop to try other insertions with the new state
                    # Or just continue? Restarting is safer to catch cascading insertions.
                    break 
            # If no single insertion worked, try removal + insertion?
            # For now, let's just do single insertions.
            # We can add a pass for "remove one, try to insert two" later if needed.
            
        return current_seq

    best_seq = []
    
    # Strategy 1: Pure Greedy
    seq = greedy_construct(skip_prob=0.0, seed=1)
    seq = local_search(seq)
    if len(seq) > len(best_seq):
        best_seq = seq

    # Strategy 2: Randomized Greedy + Local Search
    # Try different skip probabilities to escape local optima of the greedy
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        # Vary skip probability
        p = random.uniform(0.0, 0.3)
        seed = random.randint(0, 10000)
        seq = greedy_construct(skip_prob=p, seed=seed)
        seq = local_search(seq)
        if len(seq) > len(best_seq):
            best_seq = seq

    # Strategy 3: Aggressive Local Search on Best
    # Try to remove elements to allow more insertions
    if time.time() < deadline:
        current_best = list(best_seq)
        current_len = len(current_best)
        
        # Try removing one element and then running local_search
        for i in range(len(current_best)):
            if time.time() > deadline:
                break
            orig = current_best[i]
            temp_seq = current_best[:i] + current_best[i+1:]
            # Run local search on this reduced sequence
            improved_seq = local_search(temp_seq)
            if len(improved_seq) > current_len:
                current_best = improved_seq
                current_len = len(current_best)
                # Continue trying to remove more?
                # Reset loop index to 0 to try removing from the new best
                i = 0
                continue
            i += 1
        if len(current_best) > len(best_seq):
            best_seq = current_best

    return best_seq
