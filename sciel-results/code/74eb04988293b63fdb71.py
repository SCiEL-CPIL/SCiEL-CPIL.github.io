import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55

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
    composites = [x for x in all_numbers if not is_prime(x)]
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    # Precompute prime exponent vectors
    exp_vecs = {}
    for n in all_numbers:
        vec = [0] * num_primes
        temp = n
        for pi, p in enumerate(primes):
            if p * p > temp:
                break
            while temp % p == 0:
                vec[pi] += 1
                temp //= p
        if temp > 1:
            vec[prime_to_idx[temp]] += 1
        exp_vecs[n] = tuple(vec)

    def check_sequence_valid(seq):
        """Check if a sequence has all distinct consecutive block products using prefix sums."""
        n = len(seq)
        if n == 0:
            return True
        # Build prefix sum of exponent vectors
        prefix = [None] * (n + 1)
        prefix[0] = (0,) * num_primes
        for i in range(n):
            prev = prefix[i]
            ej = exp_vecs[seq[i]]
            prefix[i + 1] = tuple(prev[k] + ej[k] for k in range(num_primes))
        prods = set()
        for i in range(n):
            for j in range(i + 1, n + 1):
                pv = prefix[j]
                qv = prefix[i]
                tv = tuple(pv[k] - qv[k] for k in range(num_primes))
                if tv in prods:
                    return False
                prods.add(tv)
        return True

    def try_insert(seq, c):
        """Try to insert c into seq at best position. Returns new_seq or None."""
        n = len(seq)
        valid_positions = []
        for i in range(n + 1):
            if i > 0 and seq[i - 1] >= c:
                continue
            if i < n and c >= seq[i]:
                continue
            valid_positions.append(i)
        
        if not valid_positions:
            return None
        
        mid = n / 2.0
        valid_positions.sort(key=lambda i: abs(i - mid))
        
        for i in valid_positions:
            new_seq = seq[:i] + [c] + seq[i:]
            if check_sequence_valid(new_seq):
                return new_seq
        return None

    def try_remove_reinsert(seq, c):
        """Remove one element from seq and try to insert c. Returns new_seq or None."""
        n = len(seq)
        for i in range(n):
            base = seq[:i] + seq[i + 1:]
            result = try_insert(base, c)
            if result is not None:
                return result
        return None

    def local_search(seq, max_iter=500):
        """Improve sequence through local search with insert and swap moves."""
        current = seq[:]
        used = set(current)
        remaining = [x for x in all_numbers if x not in used]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            
            # Try insertions
            rng_local = random.Random(int(time.time() * 1000) % 100000)
            remaining_copy = remaining[:]
            rng_local.shuffle(remaining_copy)
            for c in remaining_copy:
                new_seq = try_insert(current, c)
                if new_seq is not None:
                    current = new_seq
                    used = set(current)
                    remaining = [x for x in all_numbers if x not in used]
                    improved = True
                    break
            
            # Try swaps: remove one, add another (only if same length or longer)
            if not improved:
                for i in range(len(current)):
                    removed = current[i]
                    base = current[:i] + current[i + 1:]
                    for c in remaining:
                        new_seq = try_insert(base, c)
                        if new_seq is not None and len(new_seq) >= len(current):
                            current = new_seq
                            used = set(current)
                            remaining = [x for x in all_numbers if x not in used]
                            improved = True
                            break
                    if improved:
                        break
            
            # Try remove + reinsert different element (escape local optimum)
            if not improved:
                for i in range(len(current)):
                    removed = current[i]
                    base = current[:i] + current[i + 1:]
                    for c in remaining:
                        new_seq = try_insert(base, c)
                        if new_seq is not None:
                            current = new_seq
                            used = set(current)
                            remaining = [x for x in all_numbers if x not in used]
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return current

    # ========== MIS BRANCH-AND-BOUND APPROACH ==========
    # Since the sequence must be strictly increasing, the order is fixed once we choose the subset.
    # So we are looking for the largest subset S of {2,...,128} such that sorted(S) is valid.
    
    # We can model this as a Maximum Independent Set problem on a conflict graph,
    # but the conflict is not pairwise; it depends on the whole sequence.
    # However, we can use a greedy approach with backtracking, or a simple branch-and-bound.
    
    # Let's try a branch-and-bound that decides for each number (in increasing order) whether to include it.
    # To make it efficient, we'll use a heuristic: always try to include the number first.
    
    # Actually, a simpler and often effective approach for this specific problem is to
    # use the fact that primes are "independent" in a strong sense.
    # Let's combine the greedy insertion with a more systematic search.
    
    def mis_search():
        """Branch and bound to find longest valid sequence."""
        # We'll build the sequence left to right, choosing from remaining numbers
        # in increasing order (since sequence must be strictly increasing).
        
        best_seq = []
        
        def backtrack(seq, start_idx):
            nonlocal best_seq
            if time.time() > deadline:
                return
            if len(seq) > len(best_seq):
                best_seq = seq[:]
            
            # Upper bound: number of remaining elements + current length
            remaining_count = len(all_numbers) - start_idx
            if len(seq) + remaining_count <= len(best_seq):
                return
            
            for i in range(start_idx, len(all_numbers)):
                c = all_numbers[i]
                new_seq = seq + [c]
                if check_sequence_valid(new_seq):
                    backtrack(new_seq, i + 1)
                # Even if invalid, we might still find a better solution by skipping c
                # But this is expensive, so we'll rely on the greedy approach for most of the work
        
        backtrack([], 0)
        return best_seq

    # Build initial sequences using different strategies
    def build_initial(seed_val):
        """Build an initial sequence using randomized greedy."""
        rng = random.Random(seed_val)
        
        # Strategy: start with subset of primes, then add composites
        prime_subset_size = rng.randint(10, len(primes))
        prime_subset = rng.sample(primes, prime_subset_size)
        seq = sorted(prime_subset)
        
        if not check_sequence_valid(seq):
            seq = primes[:]
        
        used = set(seq)
        remaining = [x for x in all_numbers if x not in used]
        rng.shuffle(remaining)
        
        progress = True
        while progress and time.time() < deadline:
            progress = False
            for c in remaining[:]:
                new_seq = try_insert(seq, c)
                if new_seq is not None:
                    seq = new_seq
                    remaining.remove(c)
                    progress = True
                    break
        
        return seq

    # First, try the MIS branch-and-bound (with a time limit)
    # But it might be too slow, so we'll run it in the background or with a short timeout
    
    # Let's use a hybrid approach:
    # 1. Start with primes (known to be valid)
    # 2. Add composites using greedy insertion
    # 3. Use local search to improve
    # 4. Try multiple random restarts
    
    # Start with all primes
    seq = primes[:]
    if not check_sequence_valid(seq):
        seq = []
    
    # Add composites using greedy
    used = set(seq)
    remaining = [x for x in composites if x not in used]
    rng = random.Random(42)
    rng.shuffle(remaining)
    
    for c in remaining:
        if time.time() > deadline:
            break
        new_seq = try_insert(seq, c)
        if new_seq is not None:
            seq = new_seq
    
    # Local search
    seq = local_search(seq, max_iter=500)
    
    # Try multiple random restarts
    best_seq = seq
    for restart in range(30):
        if time.time() > deadline - 5:
            break
        s2 = build_initial(42 + restart)
        s2 = local_search(s2, max_iter=400)
        if len(s2) > len(best_seq):
            best_seq = s2
    
    # If we have time, try a more systematic search
    if time.time() < deadline - 5:
        # Try to add more numbers to the best sequence found
        used = set(best_seq)
        remaining = [x for x in all_numbers if x not in used]
        rng = random.Random(123)
        rng.shuffle(remaining)
        
        for c in remaining:
            if time.time() > deadline - 3:
                break
            new_seq = try_insert(best_seq, c)
            if new_seq is not None:
                best_seq = new_seq
                used = set(best_seq)
                remaining = [x for x in all_numbers if x not in used]
        
        best_seq = local_search(best_seq, max_iter=300)
    
    return best_seq
