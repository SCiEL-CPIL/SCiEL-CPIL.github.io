"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin within 90s limit
    
    N = 512
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    def get_divisors(n):
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return divs
    
    # Strategy: Find maximum clique in the graph where edge (a,b) exists if lcm(a,b) <= 512
    # Key insight: If all elements divide some M <= 512, then lcm(a,b) divides M, so lcm(a,b) <= M <= 512.
    # So the set of all divisors of any M <= 512 is a valid clique.
    # But there might be larger cliques that aren't just divisors of a single number.
    
    # First, try all divisor sets of numbers 1..512
    best_set = set()
    best_size = 0
    
    for M in range(1, N + 1):
        divs = set(get_divisors(M))
        # Verify it's valid (it should be by construction, but let's be safe)
        if len(divs) > best_size:
            # Verify pairwise lcm
            valid = True
            divs_list = sorted(divs)
            for i in range(len(divs_list)):
                for j in range(i + 1, len(divs_list)):
                    if lcm(divs_list[i], divs_list[j]) > N:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                best_set = divs
                best_size = len(divs)
    
    # Now try to expand: start from best divisor set and greedily add elements
    # Also try starting from other highly composite numbers and expanding
    
    # Build adjacency: which pairs have lcm <= 512?
    # For efficiency, precompute for each element which others it's compatible with
    # compatible[i] = set of j > i such that lcm(i, j) <= 512
    
    # Actually, let's use a different approach: 
    # A set S is valid iff for every pair a,b in S, lcm(a,b) <= 512.
    # This is equivalent to: the lcm of ALL elements in S is <= 512? No, that's not right.
    # lcm(a,b) <= 512 for all pairs does NOT imply lcm of all elements <= 512.
    # Example: {2, 3, 5}: lcm(2,3)=6<=512, lcm(2,5)=10<=512, lcm(3,5)=15<=512, but lcm(2,3,5)=30.
    # Actually lcm of all is 30 which is still <= 512. Let me think again...
    # If lcm(a,b) <= 512 for all pairs, then the overall lcm L divides... hmm.
    # Actually, if lcm(a,b) <= 512 for all pairs a,b in S, then for any prime power p^k in the overall lcm,
    # there exist a,b in S such that p^k divides lcm(a,b). Since lcm(a,b) <= 512, p^k <= 512.
    # So the overall lcm is a product of prime powers each <= 512, but their product could exceed 512.
    # Wait no. The overall lcm L = lcm of all elements. For each prime p, let p^k be the max power in any element.
    # Then p^k divides some element a in S. And for any other element b in S, lcm(a,b) is a multiple of p^k.
    # So p^k <= lcm(a,b) <= 512. Thus each prime power in L is <= 512, but L itself could be larger.
    # Example: S = {8, 9, 25}. lcm(8,9)=72, lcm(8,25)=200, lcm(9,25)=225, all <= 512. But lcm(8,9,25)=1800 > 512.
    # So the overall lcm can exceed 512 even if all pairwise lcms are <= 512.
    
    # So the divisor-of-M approach gives valid sets but may not be optimal.
    # Let's try a greedy approach to find larger sets.
    
    # Greedy: start with empty set, add elements that are compatible with all current elements.
    # Use multiple starting points.
    
    def is_compatible(x, S):
        for y in S:
            if lcm(x, y) > N:
                return False
        return True
    
    # Try greedy from various starting points
    for start_elem in range(1, N + 1):
        if time.time() > deadline:
            break
        S = {start_elem}
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if x not in S and is_compatible(x, S):
                    S.add(x)
                    changed = True
                    if len(S) > best_size:
                        best_set = S
                        best_size = len(S)
    
    # Also try: start from divisor sets of highly composite numbers and expand
    highly_composite = [1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 480, 420, 336, 288, 168, 84, 42, 21, 14, 7, 3, 5, 10, 20, 40, 80, 160, 320, 64, 128, 256, 192, 96, 48, 72, 144, 288, 384]
    for M in highly_composite:
        if M > N:
            continue
        if time.time() > deadline:
            break
        S = set(get_divisors(M))
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if x not in S and is_compatible(x, S):
                    S.add(x)
                    changed = True
                    if len(S) > best_size:
                        best_set = S
                        best_size = len(S)
    
    # Final verification
    best_list = sorted(best_set)
    for i in range(len(best_list)):
        for j in range(i + 1, len(best_list)):
            if lcm(best_list[i], best_list[j]) > N:
                # This shouldn't happen, but if it does, return a safe fallback
                return list(range(1, 23))
    
    return best_list
# EVOLVE-BLOCK-END
