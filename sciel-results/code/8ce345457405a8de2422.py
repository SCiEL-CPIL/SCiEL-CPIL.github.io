import math
import heapq

def phi(n):
    result = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n = n // p
            result -= result // p
        p += 1
    if n > 1:
        result -= result // n
    return result

def construct():
    # Target totient divisor
    target = 120
    limit = 1000000
    
    # Precompute primes up to a reasonable limit (e.g., 100)
    # Primes > 100 will result in n > 100 * 100 = 10000 quickly, 
    # and phi(p) = p-1. For p > 100, p-1 is not divisible by 120 unless p is specific.
    # But we just need to find the minimal n.
    # We'll use a priority queue to find the minimal n.
    # State: (n, phi_n_mod_target, set_of_primes)
    # To avoid large sets, we can just check divisibility by primes when expanding.
    
    # Start with n=1, phi(1)=1
    # (n, phi_mod, frozenset_of_primes)
    # We use a visited set to avoid re-processing the same (n, phi_mod) pair? 
    # No, different n can have same phi_mod. But we want minimal n.
    # Dijkstra on n.
    
    # We need a list of candidate prime powers.
    # p^k such that p^k <= limit.
    # And we need p to be prime.
    
    # Let's generate prime powers for primes up to 1000.
    primes = []
    for i in range(2, 1000):
        is_prime = True
        for j in range(2, int(math.sqrt(i)) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
            
    # For each prime, generate powers
    prime_powers = []
    for p in primes:
        pk = p
        while pk <= limit:
            # phi(p^k) = p^(k-1) * (p-1)
            # We calculate phi(pk) directly
            phi_pk = phi(pk)
            prime_powers.append((p, pk, phi_pk))
            if pk > limit // p:
                break
            pk *= p
            
    # Dijkstra
    # (n, phi_mod, frozenset(primes_used))
    # We store the set of primes used to ensure we only multiply by NEW primes.
    # Initial state: (1, 1, frozenset())
    
    pq = [(1, 1, frozenset())]
    visited = set() # (n, phi_mod) might not be enough if we need to know primes to expand correctly?
    # Actually, if we reach n with a certain phi_mod, any other path to same n is irrelevant.
    # But different n can have same phi_mod.
    # We just want the minimal n.
    # If we pop n, it is the minimal n for that state.
    # We can just check if we've seen n? No, we might reach n with different phi_mod?
    # No, phi(n) is unique for a given n. So state is just n.
    # But we need to know phi(n) mod 120 to check the condition.
    # And we need to know the prime factors to generate next steps.
    
    # So, state in PQ: (n, phi_mod)
    # But to generate next steps, we need to know which primes are NOT in n.
    # We can compute prime factors of n on the fly or store them.
    # Storing them is better for speed.
    
    pq = [(1, 1, frozenset())]
    # We don't need a visited set for n if we use Dijkstra, because the first time we pop n, it's the minimal n.
    # But we might push the same n multiple times? 
    # If we reach n=6 via 2*3 and 3*2, we push it twice.
    # We can use a visited set for n to avoid re-processing.
    
    visited_n = set()
    
    while pq:
        n, phi_mod, primes_used = heapq.heappop(pq)
        
        if n in visited_n:
            continue
        visited_n.add(n)
        
        if phi_mod % target == 0:
            return n
            
        # Expand: multiply by p^k where p is not in primes_used
        for p, pk, phi_pk in prime_powers:
            if p in primes_used:
                continue
                
            new_n = n * pk
            if new_n > limit:
                continue
                
            new_phi_mod = (phi_mod * phi_pk) % target
            new_primes = primes_used | frozenset([p])
            
            heapq.heappush(pq, (new_n, new_phi_mod, new_primes))
            
    return -1 # Should not reach here
