"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    N = 512
    B = 512

    # Precompute compatibility: compat[i] is a set of j such that lcm(i, j) <= B
    # Using sets for O(1) lookup.
    compat = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            if (i * j) // g <= B:
                compat[i].add(j)
                if i != j:
                    compat[j].add(i)
                else:
                    compat[i].add(i) # Ensure self-compatibility for logic checks if needed, though usually we check distinct pairs

    def is_compatible(x, S_set):
        """Check if x is compatible with all elements in S_set."""
        # Optimization: Check against the "bottleneck" or just iterate.
        # Since S is small (<30), iterating is fast.
        for y in S_set:
            if x not in compat[y]:
                return False
        return True

    def try_add(S, S_set):
        """Try to add any single element to S."""
        for x in range(1, N + 1):
            if x in S_set:
                continue
            if is_compatible(x, S_set):
                S.append(x)
                S_set.add(x)
                return True
        return False

    def try_swap(S, S_set):
        """
        Try to remove one element a and add two elements b, c.
        This increases size by 1.
        """
        # Iterate over potential elements to remove
        for a in list(S):
            # Create a temporary set without a
            S_prime_set = S_set - {a}
            S_prime_list = [x for x in S if x != a]
            
            # Find all candidates b that can be added to S_prime
            valid_b = []
            for b in range(1, N + 1):
                if b in S_set:
                    continue
                if is_compatible(b, S_prime_set):
                    valid_b.append(b)
            
            # We need to find two distinct b, c in valid_b such that they are compatible with each other.
            # To be efficient, we can just check pairs.
            # valid_b size is typically small enough (maybe < 50) that O(k^2) is fine.
            found = False
            k = len(valid_b)
            for i in range(k):
                b = valid_b[i]
                for j in range(i + 1, k):
                    c = valid_b[j]
                    if b in compat[c]:
                        # Valid swap found: remove a, add b, c
                        S.remove(a)
                        S_set.remove(a)
                        S.append(b)
                        S_set.add(b)
                        S.append(c)
                        S_set.add(c)
                        found = True
                        break
                if found:
                    break
            if found:
                return True
        
        return False

    # --- Strategy: Multiple diverse starts + Local Search ---
    
    best_S = []
    best_size = 0

    # 1. Seed with divisors of highly composite numbers
    # 360 = 2^3 * 3^2 * 5 (24 divisors)
    # 480 = 2^5 * 3 * 5 (24 divisors)
    # 420 = 2^2 * 3 * 5 * 7 (24 divisors)
    # 720 is > 512, but its divisors <= 512 might form a larger set?
    # Let's try seeds based on divisors of numbers close to 512 with many factors.
    
    seeds = [
        list(range(1, N + 1)), # Random-ish start (actually sorted)
        list(range(N, 0, -1)), # Reverse
    ]
    
    # Add divisor sets of specific numbers
    for m in [360, 420, 480, 432, 336, 288, 240, 192]:
        divs = [i for i in range(1, N + 1) if m % i == 0]
        seeds.append(divs)
        
    # Also try a greedy construction from scratch with different orderings
    # Ordering by degree (number of compatible neighbors) descending
    deg_order = sorted(range(1, N + 1), key=lambda x: len(compat[x]), reverse=True)
    seeds.append(deg_order)
    seeds.append(sorted(deg_order, reverse=True))
    
    # Random shuffles
    import random
    random.seed(42) # Fixed seed for reproducibility, but we can vary it in loop
    for _ in range(5):
        shuffled = list(range(1, N + 1))
        random.shuffle(shuffled)
        seeds.append(shuffled)

    def greedy_fill(S, S_set):
        """Add compatible vertices greedily in the order of S."""
        # Note: S here is an ordering list, not the current set.
        # We assume S_set is the current set.
        # We iterate through the ordering and add if possible.
        for x in S:
            if x in S_set:
                continue
            if is_compatible(x, S_set):
                S_set.add(x)
        return S_set

    for seed_order in seeds:
        if time.time() > deadline:
            break
            
        # Initialize set with the first compatible element from seed_order?
        # Or just start empty and greedily fill?
        # Starting empty and greedily filling with seed_order is a valid heuristic.
        
        current_S = []
        current_S_set = set()
        
        # Greedy fill
        greedy_fill(seed_order, current_S_set)
        current_S = list(current_S_set)
        
        # Local search loop
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add
            if try_add(current_S, current_S_set):
                improved = True
                if len(current_S) > best_size:
                    best_size = len(current_S)
                    best_S = list(current_S)
                continue # Restart to allow more additions
            
            # Try to swap
            if try_swap(current_S, current_S_set):
                improved = True
                if len(current_S) > best_size:
                    best_size = len(current_S)
                    best_S = list(current_S)
                continue
            
            # If no improvement, break
            if not improved:
                break
        
        # Update best
        if len(current_S) > best_size:
            best_size = len(current_S)
            best_S = list(current_S)

    # Final polish: Try one more pass on the best set found
    # Sometimes a different local search trajectory can squeeze out 1 more
    if best_S and time.time() < deadline - 5:
        current_S = list(best_S)
        current_S_set = set(best_S)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            if try_add(current_S, current_S_set):
                improved = True
                if len(current_S) > best_size:
                    best_size = len(current_S)
                    best_S = list(current_S)
                continue
            if try_swap(current_S, current_S_set):
                improved = True
                if len(current_S) > best_size:
                    best_size = len(current_S)
                    best_S = list(current_S)
                continue
            if not improved:
                break

    return list(best_S)

# EVOLVE-BLOCK-END
