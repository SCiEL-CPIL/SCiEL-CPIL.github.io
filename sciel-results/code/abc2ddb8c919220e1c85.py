import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    n = 512

    # Precompute lcm matrix and compatibility
    compat_bits = [0] * (n + 1)
    for i in range(1, n + 1):
        bits = 0
        for j in range(1, n + 1):
            if i == j:
                continue
            g = gcd(i, j)
            l = (i // g) * j
            if l <= n:
                bits |= (1 << (j - 1))
        compat_bits[i] = bits

    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    # Compute degree of each node
    degrees = [0] * (n + 1)
    for i in range(1, n + 1):
        degrees[i] = popcount(compat_bits[i])

    # Strategy: Use divisor structure for strong initial solution
    # All powers of 2 up to 512 are mutually compatible (lcm is the larger one)
    # Also, any number dividing 512 is compatible with all powers of 2
    # Build initial set from divisors of 512, then try to add more

    # Get all divisors of 512
    divisors_512 = []
    for i in range(1, n + 1):
        if 512 % i == 0:
            divisors_512.append(i)
    
    # Sort by degree descending for greedy
    sorted_nodes = sorted(range(1, n + 1), key=lambda x: -degrees[x])
    
    # Greedy initialization
    greedy_clique = []
    greedy_set = set()
    for v in sorted_nodes:
        if all((1 << (u - 1)) & compat_bits[v] for u in greedy_set):
            greedy_clique.append(v)
            greedy_set.add(v)
    
    best_clique = list(greedy_clique)
    best_size = len(greedy_clique)

    # Local search: try to improve by swapping elements
    # For each element in the set, try removing it and adding a better one
    S = set(best_clique)
    
    def can_add(v, current_set):
        """Check if v can be added to current_set"""
        for u in current_set:
            if not (compat_bits[v] & (1 << (u - 1))):
                return False
        return True

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for remove_v in list(S):
            S_without = S - {remove_v}
            # Try adding any node not in S that is compatible with S_without
            for add_v in range(1, n + 1):
                if add_v not in S_without:
                    if can_add(add_v, S_without):
                        # Check if this swap improves (it should if add_v has higher degree or we just want to explore)
                        # For now, just take the first improvement
                        S = S_without | {add_v}
                        improved = True
                        break
            if improved:
                break
            if time.time() > deadline:
                break

    if len(S) > best_size:
        best_clique = list(S)
        best_size = len(S)

    # Final polish: try to add any remaining compatible nodes
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(1, n + 1):
            if i not in S:
                if can_add(i, S):
                    S.add(i)
                    changed = True

    if len(S) > best_size:
        best_clique = list(S)
        best_size = len(S)

    return best_clique
