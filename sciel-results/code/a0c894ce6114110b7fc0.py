"""
Combined program: Maximize subset of [1, 128] with no equal-gcd triples.
Uses Bitset representation and Iterated Local Search with Kicks.
"""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    deadline = time.time() + 80  # Safety margin
    
    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute gcd table for fast access
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Precompute prime factors count for ordering
    def count_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    pf_counts = {n: count_prime_factors(n) for n in nums}

    def is_valid_addition(S_mask, x):
        """
        Check if x can be added to the set represented by S_mask.
        S_mask is an integer bitmask where bit i is 1 if (i+1) is in S.
        """
        # Extract elements in S for faster iteration if needed, 
        # but we can just iterate 1..N and check bits.
        # However, we need to check triples {x, a, b} for a, b in S.
        
        # Optimization: Group elements in S by gcd(x, a)
        groups = {}
        for a in range(1, N + 1):
            if (S_mask >> a) & 1:
                g = gcd_table[x][a]
                if g not in groups:
                    groups[g] = []
                groups[g].append(a)
        
        # For each group with gcd g, check if any pair (a, b) has gcd(a, b) == g
        for g, members in groups.items():
            m = len(members)
            for i in range(m):
                a = members[i]
                for j in range(i + 1, m):
                    b = members[j]
                    if gcd_table[a][b] == g:
                        return False
        return True

    def get_elements(mask):
        return [i + 1 for i in range(N) if (mask >> i) & 1]

    def greedy_construct(order):
        S_mask = 0
        for x in order:
            if is_valid_addition(S_mask, x):
                S_mask |= (1 << x)
        return S_mask

    def local_search_add(S_mask):
        """Try to add any element from complement. Returns (new_mask, added_x) or (S_mask, -1)"""
        current_size = bin(S_mask).count("1")
        for x in range(1, N + 1):
            if not ((S_mask >> x) & 1):
                if is_valid_addition(S_mask, x):
                    return S_mask | (1 << x), x
        return S_mask, -1

    def local_search_swap(S_mask, max_swaps=50):
        """
        Try to swap elements to potentially allow more additions.
        We don't just swap for size (size stays same), we swap to change the structure.
        However, a simple swap might not help immediately. 
        A better approach: Randomly remove an element, then try to add many.
        This is handled by the 'kick' in ILS.
        """
        pass

    # Initial Greedy Construction with multiple orderings
    best_mask = 0
    best_size = 0
    
    orderings = []
    # 1. Ascending
    orderings.append(nums)
    # 2. Descending
    orderings.append(nums[::-1])
    # 3. By prime factor count (ascending)
    orderings.append(sorted(nums, key=lambda n: pf_counts[n]))
    # 4. By prime factor count (descending)
    orderings.append(sorted(nums, key=lambda n: -pf_counts[n]))
    # 5. Random shuffles
    rng = random.Random(42)
    for _ in range(10):
        shuffled = nums[:]
        rng.shuffle(shuffled)
        orderings.append(shuffled)

    for order in orderings:
        if time.time() > deadline - 10:
            break
        mask = greedy_construct(order)
        # Try to improve with local search additions
        while True:
            if time.time() > deadline - 5:
                break
            new_mask, _ = local_search_add(mask)
            if new_mask == mask:
                break
            mask = new_mask
        
        size = bin(mask).count("1")
        if size > best_size:
            best_size = size
            best_mask = mask

    # Iterated Local Search with Kicks
    # We try to kick out some elements and re-add them to escape local optima
    current_mask = best_mask
    current_size = best_size
    rng_ils = random.Random(123)
    
    kick_sizes = [1, 2, 3]
    
    while time.time() < deadline - 10:
        # Perform a kick: remove k random elements
        k = rng_ils.choice(kick_sizes)
        elements = get_elements(current_mask)
        if len(elements) < k:
            break
        
        kick_elements = rng_ils.sample(elements, k)
        
        kicked_mask = current_mask
        for x in kick_elements:
            kicked_mask &= ~(1 << x)
        
        # Try to recover/add elements greedily/randomly to the kicked set
        # We want to see if we can get back to current_size and exceed it
        recovery_mask = kicked_mask
        
        # Try adding elements in random order
        candidates = [x for x in nums if not ((recovery_mask >> x) & 1)]
        rng_ils.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 5:
                break
            if is_valid_addition(recovery_mask, x):
                recovery_mask |= (1 << x)
        
        # Try local search additions on the recovered set
        while True:
            if time.time() > deadline - 5:
                break
            new_mask, _ = local_search_add(recovery_mask)
            if new_mask == recovery_mask:
                break
            recovery_mask = new_mask
            
        recovery_size = bin(recovery_mask).count("1")
        
        # Accept if better
        if recovery_size > current_size:
            current_mask = recovery_mask
            current_size = recovery_size
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            # Reset kick size to 1 if we made progress? Or keep random.
        else:
            # Optionally accept worse solutions with some probability (Simulated Annealing)
            # But for now, let's stick to hill climbing with restarts/kicks
            pass

    # Final check
    if best_size == 0:
        return [1, 2]
        
    return get_elements(best_mask)

# EVOLVE-BLOCK-END
