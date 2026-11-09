import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # Precompute prime factorization vectors for numbers 2..128
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

    primes = [x for x in range(2, 129) if is_prime(x)]
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    # Exponent vector for each number 2..128
    exp_vecs = {}
    for n in range(2, 129):
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

    # Packed integer representation for fast hashing
    # Max exponent for any prime in any product of numbers <= 128:
    # Smallest prime is 2. Max power of 2 in 128 is 7.
    # Product of two numbers <= 128: max power of 2 is 14.
    # Product of k numbers: max power is 7k.
    # Max length of sequence is 127. 7 * 127 = 889.
    # 10 bits is enough for exponents up to 1023.
    BITS = 10
    MASK = (1 << BITS) - 1

    def pack_vec(vec):
        res = 0
        for i, e in enumerate(vec):
            res |= e << (i * BITS)
        return res

    # Precompute packed exponent vectors for each number
    packed_exp = {}
    for n in range(2, 129):
        packed_exp[n] = pack_vec(exp_vecs[n])

    def check_valid(seq):
        """Check if sequence has all distinct consecutive block products using prefix sums."""
        n = len(seq)
        if n == 0:
            return True
        
        # Build prefix packed values
        # prefix[i] represents the packed sum of exponents for seq[0:i]
        # We use a list of integers.
        prefix = [0]
        # We can't just use integer addition for packed vectors if there's carry-over between fields.
        # However, since we are summing exponents, and each field is independent,
        # we can use integer addition IF we ensure no field overflows its BITS.
        # Max exponent sum: 127 * 7 = 889 < 1024. So integer addition is safe!
        
        for i in range(n):
            prefix.append(prefix[-1] + packed_exp[seq[i]])
        
        prods = set()
        for i in range(n):
            pi = prefix[i]
            for j in range(i + 1, n + 1):
                pj = prefix[j]
                # The difference pj - pi gives the packed exponent vector for the block seq[i:j]
                # Because there is no carry-over, subtraction works field-wise.
                packed = pj - pi
                if packed in prods:
                    return False
                prods.add(packed)
        return True

    def try_insert(seq, c):
        """Try to insert c into sorted seq. Return new seq if valid, else None."""
        # Find position
        pos = 0
        while pos < len(seq) and seq[pos] < c:
            pos += 1
        if pos < len(seq) and seq[pos] == c:
            return None # Already present
        
        candidate = seq[:pos] + [c] + seq[pos:]
        if check_valid(candidate):
            return candidate
        return None

    def greedy_forward():
        seq = []
        for n in range(2, 129):
            new_seq = try_insert(seq, n)
            if new_seq is not None:
                seq = new_seq
        return seq

    def greedy_backward():
        seq = []
        # Iterate from largest to smallest
        for n in range(128, 1, -1):
            new_seq = try_insert(seq, n)
            if new_seq is not None:
                seq = new_seq
        return seq

    # Generate initial candidates
    seq_f = greedy_forward()
    seq_b = greedy_backward()
    
    best_seq = seq_f if len(seq_f) >= len(seq_b) else seq_b
    
    # Local Search to improve
    def local_search(start_seq, max_iter=5000):
        current = start_seq[:]
        used = set(current)
        remaining = [x for x in range(2, 129) if x not in used]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            
            # 1. Try insertions
            for c in remaining[:]:
                new_seq = try_insert(current, c)
                if new_seq is not None:
                    current = new_seq
                    used = set(current)
                    remaining = [x for x in range(2, 129) if x not in used]
                    improved = True
                    break
            
            # 2. Try swaps (remove one, add another) if no insertion possible
            if not improved:
                # To speed up, we can limit swaps to "nearby" values or random sampling
                # But N is small, so full search is okay if we break early.
                # We want to find a swap that allows a future insertion.
                # Simple hill climbing: swap if it doesn't decrease length (it doesn't) 
                # but might unlock insertions.
                # However, checking validity for every swap is O(N^2) per swap.
                # Number of swaps: N * (127-N) ~ 50 * 70 = 3500.
                # 3500 * 2500 ops = 8.75M ops per iteration.
                # This might be too slow for 50s if we do many iterations.
                
                # Optimization: Only try swaps if we are stuck.
                # And maybe limit to a subset of swaps per iteration.
                
                # Let's try a few random swaps to avoid getting stuck in local minima
                # that require a specific swap.
                if remaining:
                    # Try up to 20 random swaps
                    for _ in range(20):
                        if time.time() > deadline:
                            break
                        if not current or not remaining:
                            break
                        i = random.randrange(len(current))
                        j = random.randrange(len(remaining))
                        
                        rem_val = current[i]
                        add_val = remaining[j]
                        
                        # Remove rem_val
                        base = current[:i] + current[i+1:]
                        
                        # Try to add add_val
                        new_seq = try_insert(base, add_val)
                        if new_seq is not None:
                            # Check if this state is "better" or at least valid.
                            # Since we just swapped, length is same.
                            # We accept it if it allows further insertions later.
                            # For now, accept any valid swap to change the landscape.
                            current = new_seq
                            used = set(current)
                            remaining = [x for x in range(2, 129) if x not in used]
                            improved = True
                            break

            if not improved:
                # If no simple move works, we might be stuck.
                # Try a more aggressive move: Remove one element, then try ALL insertions.
                # This is effectively a 1-step lookahead.
                found = False
                for i in range(len(current)):
                    if time.time() > deadline:
                        break
                    rem_val = current[i]
                    base = current[:i] + current[i+1:]
                    # Try to insert any from remaining
                    for c in remaining:
                        new_seq = try_insert(base, c)
                        if new_seq is not None:
                            current = new_seq
                            used = set(current)
                            remaining = [x for x in range(2, 129) if x not in used]
                            improved = True
                            found = True
                            break
                    if found:
                        break
                
                if not improved:
                    break
        
        return current

    # Run local search on the best initial sequence
    improved_seq = local_search(best_seq, max_iter=1000)
    if len(improved_seq) > len(best_seq):
        best_seq = improved_seq

    # Try local search on the other initial sequence if it was close
    other_seq = seq_b if best_seq == seq_f else seq_f
    if len(other_seq) >= len(best_seq) - 2:
        improved_other = local_search(other_seq, max_iter=1000)
        if len(improved_other) > len(best_seq):
            best_seq = improved_other

    # Run a few more random restarts with local search
    for _ in range(20):
        if time.time() > deadline:
            break
        
        # Randomized greedy
        candidates = list(range(2, 129))
        random.shuffle(candidates)
        
        s2 = []
        for n in candidates:
            if time.time() > deadline:
                break
            ns = try_insert(s2, n)
            if ns is not None:
                s2 = ns
        
        s2 = local_search(s2, max_iter=500)
        if len(s2) > len(best_seq):
            best_seq = s2

    return best_seq
