import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # Helper: LCM calculation
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Helper: count distinct prime factors
    def num_distinct_primes(x):
        count = 0
        d = 2
        while d * d <= x:
            if x % d == 0:
                count += 1
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            count += 1
        return count

    # Generate primes and prime powers
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(n)
    prime_powers = set()
    for p in primes:
        power = p
        while power <= n:
            prime_powers.add(power)
            power *= p
    
    # Precompute all bad triples
    bad_triples = []
    node_to_triples = {x: [] for x in numbers}
    
    for i in range(n):
        a = numbers[i]
        for j in range(i + 1, n):
            b = numbers[j]
            lab = lcm(a, b)
            for k in range(j + 1, n):
                c = numbers[k]
                if lcm(a, c) == lab and lcm(b, c) == lab:
                    bad_triples.append((a, b, c))
                    node_to_triples[a].append(len(bad_triples) - 1)
                    node_to_triples[b].append(len(bad_triples) - 1)
                    node_to_triples[c].append(len(bad_triples) - 1)
    
    if time.time() > deadline:
        return list(prime_powers | {1})
    
    def is_valid_set(s):
        """Check if set s contains no bad triple."""
        s_set = set(s)
        for t in bad_triples:
            a, b, c = t
            if a in s_set and b in s_set and c in s_set:
                return False
        return True
    
    def greedy_addition(core_set):
        """Start from core, greedily add remaining elements."""
        current = set(core_set)
        remaining = [x for x in range(1, n + 1) if x not in current]
        remaining.sort(key=lambda x: (num_distinct_primes(x), x))
        
        for x in remaining:
            if time.time() > deadline:
                break
            
            valid = True
            groups = {}
            for a in current:
                L = lcm(x, a)
                if L not in groups:
                    groups[L] = []
                groups[L].append(a)
            
            for L, elements in groups.items():
                if len(elements) < 2:
                    continue
                for i in range(len(elements)):
                    for j in range(i + 1, len(elements)):
                        if lcm(elements[i], elements[j]) == L:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                current.add(x)
        
        return current
    
    def greedy_removal(initial_set):
        """Greedy removal from initial set."""
        current = set(initial_set)
        
        # Compute degrees for nodes in current
        degrees = {}
        for x in current:
            deg = 0
            for t_idx in node_to_triples[x]:
                t = bad_triples[t_idx]
                if all(v in current for v in t):
                    deg += 1
            degrees[x] = deg
        
        while True:
            if time.time() > deadline:
                break
            
            # Find active triple with max degree sum
            best_triple = None
            best_sum = -1
            
            for t in bad_triples:
                a, b, c = t
                if a in current and b in current and c in current:
                    s = degrees.get(a, 0) + degrees.get(b, 0) + degrees.get(c, 0)
                    if s > best_sum:
                        best_sum = s
                        best_triple = t
            
            if best_triple is None:
                break
            
            a, b, c = best_triple
            # Pick node with max degree
            candidates = [x for x in (a, b, c) if x in current]
            if not candidates:
                break
            max_deg_node = max(candidates, key=lambda x: degrees.get(x, 0))
            
            # Update degrees before removal
            for t_idx in node_to_triples[max_deg_node]:
                t = bad_triples[t_idx]
                others = [x for x in t if x != max_deg_node]
                u, v = others
                if u in current and v in current:
                    degrees[u] = degrees.get(u, 0) - 1
                    degrees[v] = degrees.get(v, 0) - 1
            
            current.remove(max_deg_node)
            degrees.pop(max_deg_node, None)
        
        return current
    
    # Strategy 1: Greedy addition from prime power core
    best_set = greedy_addition(prime_powers | {1})
    
    # Strategy 2: Greedy removal from full set
    if time.time() < deadline - 10:
        full_set = greedy_removal(set(numbers))
        if len(full_set) > len(best_set):
            best_set = full_set
    
    # Strategy 3: Hybrid - greedy add from a larger core
    if time.time() < deadline - 10:
        # Core: prime powers + 1 + some composites with few prime factors
        extra_core = set()
        for x in range(1, n + 1):
            if x not in (prime_powers | {1}) and num_distinct_primes(x) <= 2:
                extra_core.add(x)
        hybrid_core = prime_powers | {1} | extra_core
        hybrid_set = greedy_addition(hybrid_core)
        if len(hybrid_set) > len(best_set):
            best_set = hybrid_set
    
    # Local search polish: try swapping elements
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            best_set_list = list(best_set)
            excluded = [x for x in range(1, n + 1) if x not in best_set]
            
            for remove_elem in best_set_list:
                if time.time() > deadline - 2:
                    break
                new_set = set(best_set)
                new_set.remove(remove_elem)
                
                # Try to add excluded elements
                added = 0
                for add_elem in excluded:
                    if time.time() > deadline - 2:
                        break
                    test_set = new_set | {add_elem}
                    if is_valid_set(test_set):
                        new_set = test_set
                        added += 1
                        excluded.remove(add_elem)
                
                if len(new_set) > len(best_set):
                    best_set = new_set
                    excluded = [x for x in range(1, n + 1) if x not in best_set]
                    improved = True
                    break
    
    return list(best_set)
