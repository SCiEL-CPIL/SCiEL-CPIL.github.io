import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes_1000 = sieve(1000)
    
    candidate_sets = []
    
    # Hardcoded good sets
    s1 = [3, 5, 7, 11, 13, 17, 19, 23]
    candidate_sets.append(s1)
    s2 = [2, 7, 11, 13, 17, 19, 23]
    candidate_sets.append(s2)
    s3 = [2, 3, 13, 17]
    candidate_sets.append(s3)
    
    # Generate random valid sets
    random.seed(42)
    for _ in range(50):
        s = set()
        total = 0.0
        ps = primes_1000[:]
        random.shuffle(ps)
        for p in ps:
            if total + 1.0/p <= 1.0:
                s.add(p)
                total += 1.0/p
        if len(s) >= 5:
            candidate_sets.append(list(s))
            
    unique_sets = []
    seen = set()
    for s in candidate_sets:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            unique_sets.append(s)
            
    best_L = 0
    best_classes = []
    
    L_MAX = 2000
    
    def get_L_from_count(count):
        L = 0
        for k in range(1, L_MAX + 1):
            if count[k] > 0:
                L = k
            else:
                break
        return L

    def update_counts(count, p, r_old, r_new):
        start = r_old
        if start == 0:
            start = p
        for k in range(start, L_MAX + 1, p):
            count[k] -= 1
        start = r_new
        if start == 0:
            start = p
        for k in range(start, L_MAX + 1, p):
            count[k] += 1

    for primes_set in unique_sets:
        if time.time() > deadline:
            break
            
        residues = {p: random.randint(0, p-1) for p in primes_set}
        
        count = [0] * (L_MAX + 1)
        for p, r in residues.items():
            start = r
            if start == 0:
                start = p
            for k in range(start, L_MAX + 1, p):
                count[k] += 1
                
        current_L = get_L_from_count(count)
        
        if current_L > best_L:
            best_L = current_L
            best_classes = [[p, r] for p, r in residues.items()]
            
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        iterations = 0
        
        while T > T_min and time.time() < deadline - 5:
            iterations += 1
            if iterations % 1000 == 0 and time.time() > deadline - 5:
                break
                
            p = random.choice(primes_set)
            r_old = residues[p]
            r_new = random.randint(0, p-1)
            
            if r_old == r_new:
                continue
                
            update_counts(count, p, r_old, r_new)
            new_L = get_L_from_count(count)
            
            delta = new_L - current_L
            if delta > 0:
                current_L = new_L
                residues[p] = r_new
                if current_L > best_L:
                    best_L = current_L
                    best_classes = [[p, r] for p, r in residues.items()]
            else:
                if random.random() < pow(2.718, delta / T):
                    current_L = new_L
                    residues[p] = r_new
            T *= cooling
            
    return best_classes
# EVOLVE-BLOCK-START
def construct():
    import time
    import random

    start_time = time.time()
    deadline = start_time + 110

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes_1000 = sieve(1000)

    candidate_sets = []
    s1 = [3, 5, 7, 11, 13, 17, 19, 23]
    candidate_sets.append(s1)
    s2 = [2, 7, 11, 13, 17, 19, 23]
    candidate_sets.append(s2)
    s3 = [2, 3, 13, 17]
    candidate_sets.append(s3)

    random.seed(42)
    for _ in range(50):
        s = set()
        total = 0.0
        ps = primes_1000[:]
        random.shuffle(ps)
        for p in ps:
            if total + 1.0/p <= 1.0:
                s.add(p)
                total += 1.0/p
        if len(s) >= 5:
            candidate_sets.append(list(s))

    unique_sets = []
    seen = set()
    for s in candidate_sets:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            unique_sets.append(s)

    best_L = 0
    best_classes = []

    L_MAX = 2000

    def get_L_from_count(count):
        L = 0
        for k in range(1, L_MAX + 1):
            if count[k] > 0:
                L = k
            else:
                break
        return L

    def update_counts(count, p, r_old, r_new):
        start = r_old
        if start == 0:
            start = p
        for k in range(start, L_MAX + 1, p):
            count[k] -= 1
        start = r_new
        if start == 0:
            start = p
        for k in range(start, L_MAX + 1, p):
            count[k] += 1

    for primes_set in unique_sets:
        if time.time() > deadline:
            break

        residues = {p: random.randint(0, p-1) for p in primes_set}

        count = [0] * (L_MAX + 1)
        for p, r in residues.items():
            start = r
            if start == 0:
                start = p
            for k in range(start, L_MAX + 1, p):
                count[k] += 1

        current_L = get_L_from_count(count)

        if current_L > best_L:
            best_L = current_L
            best_classes = [[p, r] for p, r in residues.items()]

        T = 1.0
        T_min = 0.01
        cooling = 0.999
        iterations = 0

        while T > T_min and time.time() < deadline - 5:
            iterations += 1
            if iterations % 1000 == 0 and time.time() > deadline - 5:
                break

            p = random.choice(primes_set)
            r_old = residues[p]
            r_new = random.randint(0, p-1)

            if r_old == r_new:
                continue

            update_counts(count, p, r_old, r_new)
            new_L = get_L_from_count(count)

            delta = new_L - current_L
            if delta > 0:
                current_L = new_L
                residues[p] = r_new
                if current_L > best_L:
                    best_L = current_L
                    best_classes = [[p, r] for p, r in residues.items()]
            else:
                if random.random() < pow(2.718, delta / T):
                    current_L = new_L
                    residues[p] = r_new
            T *= cooling

    return best_classes
# EVOLVE-BLOCK-END
