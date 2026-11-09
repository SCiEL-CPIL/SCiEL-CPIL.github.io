import time

# EVOLVE-BLOCK-START
P = {'n': 256}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin
    
    n = P['n']
    
    # Precompute: for each a, which pairs (b,c) with b<c, b!=a, c!=a have a | b*c
    # We'll use a greedy approach to build the largest valid set
    
    # First, let's try a greedy algorithm with different orderings
    # For each candidate a, count how many pairs (b,c) in [2..n] have a | b*c
    # Prefer numbers with fewer such pairs (they're "safer" to include)
    
    candidates = list(range(2, n + 1))
    
    # Count conflicts: for each a, number of pairs (b,c) with b<c, b!=a, c!=a, a|b*c
    conflict_count = {}
    for a in candidates:
        cnt = 0
        for b in candidates:
            if b == a:
                continue
            for c in candidates:
                if c == a or c == b:
                    continue
                if (b * c) % a == 0:
                    cnt += 1
        conflict_count[a] = cnt
    
    if time.time() > deadline:
        # Fallback to primes
        is_prime = [True] * (n + 1)
        is_prime[0] = False
        if n >= 1:
            is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    # Sort by conflict count (ascending) - prefer less conflicting numbers
    sorted_candidates = sorted(candidates, key=lambda x: conflict_count[x])
    
    # Greedily build set
    selected = []
    selected_set = set()
    
    for a in sorted_candidates:
        if time.time() > deadline - 5:
            break
        # Check if adding a violates constraint with existing selected elements
        valid = True
        for i, b in enumerate(selected):
            for c in selected[i+1:]:
                # Check if a | b*c
                if (b * c) % a == 0:
                    valid = False
                    break
                # Check if b | a*c
                if (a * c) % b == 0:
                    valid = False
                    break
                # Check if c | a*b
                if (a * b) % c == 0:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            selected.append(a)
            selected_set.add(a)
    
    if time.time() > deadline:
        # Fallback to primes
        is_prime = [True] * (n + 1)
        is_prime[0] = False
        if n >= 1:
            is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    # Verify the result
    for i, a in enumerate(selected):
        for j, b in enumerate(selected):
            if i == j:
                continue
            for k, c in enumerate(selected):
                if i == k or j == k:
                    continue
                if (b * c) % a == 0:
                    # Invalid! Fallback to primes
                    is_prime = [True] * (n + 1)
                    is_prime[0] = False
                    if n >= 1:
                        is_prime[1] = False
                    for p in range(2, int(n**0.5) + 1):
                        if is_prime[p]:
                            for q in range(p*p, n + 1, p):
                                is_prime[q] = False
                    return [x for x in range(2, n + 1) if is_prime[x]]
    
    return selected
# EVOLVE-BLOCK-END
