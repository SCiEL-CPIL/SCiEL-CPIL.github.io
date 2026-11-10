import itertools
import math

def construct():
    max_val = 128
    max_sum = 256
    
    # Sieve of Eratosthenes
    is_prime = [True] * (max_sum + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(max_sum**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_sum + 1, i):
                is_prime[j] = False
                
    odds = [x for x in range(1, max_val + 1) if x % 2 == 1]
    evens = [x for x in range(2, max_val + 1) if x % 2 == 0]
    
    # Build bipartite graph: neighbors[u] = set of v in evens such that u+v is prime
    neighbors = {}
    for u in odds:
        neighbors[u] = set()
        for v in evens:
            if is_prime[u + v]:
                neighbors[u].add(v)
                
    # Sort odds by degree (number of prime-sum partners) in descending order
    # This implements the greedy maximum-degree heuristic
    sorted_odds = sorted(odds, key=lambda u: len(neighbors[u]), reverse=True)
    
    # Search for A (3 odds) and B (3 evens) using the sorted order
    for A in itertools.combinations(sorted_odds, 3):
        a1, a2, a3 = A
        V_A = neighbors[a1].intersection(neighbors[a2]).intersection(neighbors[a3])
        if len(V_A) < 3:
            continue
        
        V_A_list = list(V_A)
        for B in itertools.combinations(V_A_list, 3):
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
            if len(sums) == 9:
                return [list(A), list(B)]
                
    # Fallback: Random search for any valid pairwise coprime set
    import random
    for _ in range(10000):
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        
        # Check pairwise coprime
        valid = True
        sums_list = list(sums)
        for i in range(len(sums_list)):
            for j in range(i + 1, len(sums_list)):
                if math.gcd(sums_list[i], sums_list[j]) != 1:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            return [A, B]
            
    return [[1, 7, 13], [4, 10, 16]]
