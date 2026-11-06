import math
import time

def construct():
    t_start = time.time()
    DEADLINE = 85  # Safety margin

    # Sieve of Eratosthenes for primality up to 256
    MAX_SUM = 256
    is_prime = [True] * (MAX_SUM + 1)
    if MAX_SUM >= 0:
        is_prime[0] = False
    if MAX_SUM >= 1:
        is_prime[1] = False
    for i in range(2, int(MAX_SUM ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_SUM + 1, i):
                is_prime[j] = False

    def check_pairwise_coprime(nums):
        """Check if all distinct numbers in the list are pairwise coprime."""
        distinct = list(set(nums))
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                if math.gcd(distinct[i], distinct[j]) > 1:
                    return False
        return True

    def get_score(A, B):
        """Calculate the score: number of distinct sums if pairwise coprime, else 0."""
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        if len(distinct) < 2:
            return 0
        if check_pairwise_coprime(distinct):
            return len(distinct)
        return 0

    # Initialize with a known valid set (score 5)
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = get_score(best_A, best_B)

    # Search for AP solutions A = {a, a+d1, a+2d1}, B = {b, b+d2, b+2d2}
    # We want 9 distinct sums, ideally all prime.
    # If gcd(d1, d2) == 1, the 9 offsets are distinct.
    
    max_d = 63  # Max step such that 2*d <= 127
    
    # Precompute coprime pairs of steps
    # We iterate d1, d2. To find primes quickly, smaller sums are denser.
    # However, any valid 9-prime solution is optimal.
    
    for d1 in range(1, max_d + 1):
        if time.time() - t_start > DEADLINE:
            break
        a_max = 128 - 2 * d1
        if a_max < 1:
            continue
            
        for d2 in range(1, max_d + 1):
            if time.time() - t_start > DEADLINE:
                break
            if math.gcd(d1, d2) != 1:
                continue
                
            b_max = 128 - 2 * d2
            if b_max < 1:
                continue
                
            # Offsets: i*d1 + j*d2 for i,j in 0,1,2
            offsets = [i * d1 + j * d2 for i in range(3) for j in range(3)]
            
            for a in range(1, a_max + 1):
                if time.time() - t_start > DEADLINE:
                    break
                
                # Check if there exists a b such that all a+b+offset is prime
                found_b = -1
                # We can iterate b or vectorize. Since b_max is small, simple loop is fine.
                # But to be safe and fast, we can pre-filter b based on the first few sums.
                
                # Optimization: The sum a+b must be such that a+b+offset_k is prime for all k.
                # This implies strong constraints on a+b mod small primes.
                
                for b in range(1, b_max + 1):
                    base = a + b
                    # Quick check: are all 9 sums prime?
                    all_prime = True
                    for off in offsets:
                        s = base + off
                        if s > MAX_SUM or not is_prime[s]:
                            all_prime = False
                            break
                    
                    if all_prime:
                        A = [a, a + d1, a + 2 * d1]
                        B = [b, b + d2, b + 2 * d2]
                        score = get_score(A, B)
                        if score == 9:
                            return [A, B]
                        elif score > best_score:
                            best_score = score
                            best_A = A
                            best_B = B
                        # Found a 9-prime solution, which is optimal. Return immediately.
                        # Actually, the problem says maximize cardinality. 9 is max.
                        # So we can return immediately upon finding any valid 9.
                        # The loop above ensures distinctness via gcd(d1,d2)=1.
                        # And primality ensures pairwise coprime.
                        return [A, B]

    # Fallback random search if AP search failed (unlikely)
    if time.time() - t_start < DEADLINE - 5:
        import random
        rng = random.Random(123)
        for _ in range(50000):
            if time.time() - t_start > DEADLINE:
                break
            A = rng.sample(range(1, 129), 3)
            B = rng.sample(range(1, 129), 3)
            s = get_score(A, B)
            if s > best_score:
                best_score = s
                best_A = A
                best_B = B
                if s == 9:
                    return [A, B]

    return [best_A, best_B]
