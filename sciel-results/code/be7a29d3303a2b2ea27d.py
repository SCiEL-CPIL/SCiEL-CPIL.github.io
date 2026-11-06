"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    deadline = time.time() + 50  # Strict safety margin
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute LCM for all pairs
    lcm = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            lcm[i][j] = i * j // gcd(i, j)
            
    # Find all bad triples
    # Use sorted tuples for faster hashing and comparison than frozensets
    bad_triples = set()
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            lab = lcm[a][b]
            for c in range(b+1, N+1):
                if lcm[a][c] == lab and lcm[b][c] == lab:
                    bad_triples.add((a, b, c))
    
    # Precompute degrees for heuristic ordering
    degrees = [0] * (N + 1)
    for a, b, c in bad_triples:
        degrees[a] += 1
        degrees[b] += 1
        degrees[c] += 1

    def can_add(x, S):
        """Check if x can be added to set S without creating a bad triple."""
        # O(|S|^2) check, which is fast for |S| < 128
        # We check all pairs in S to see if (x, y, z) is a bad triple
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                y = S_list[i]
                z = S_list[j]
                # Check all 3 permutations of (x, y, z) in bad_triples
                # Since bad_triples are sorted tuples, we just sort the triple
                t = (x, y, z)
                t_sorted = tuple(sorted(t))
                if t_sorted in bad_triples:
                    return False
        return True

    def is_valid(S):
        """Verify that S contains no bad triple."""
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                for k in range(j+1, len(S_list)):
                    t_sorted = tuple(sorted([S_list[i], S_list[j], S_list[k]]))
                    if t_sorted in bad_triples:
                        return False
        return True

    best_set = set()
    best_size = 0
    
    # Strategy 1: Greedy by Degree (Lowest degree first often allows more elements)
    def greedy_fill(S, order):
        for x in order:
            if x not in S and can_add(x, S):
                S.add(x)
        return S

    # Local Search: Try to swap out one element to allow adding two others?
    # Or simply try to add any remaining element (polish)
    def polish(S):
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for x in elements:
                if x not in S and can_add(x, S):
                    S.add(x)
                    changed = True
        return S

    # Run multiple strategies
    strategies = []
    
    # 1. Natural order
    strategies.append(elements[:])
    
    # 2. Reverse natural order
    strategies.append(elements[::-1])
    
    # 3. Sorted by degree (ascending)
    strategies.append(sorted(elements, key=lambda x: degrees[x]))
    
    # 4. Sorted by degree (descending)
    strategies.append(sorted(elements, key=lambda x: -degrees[x]))
    
    # 5-10. Random shuffles
    for _ in range(5):
        random.shuffle(elements)
        strategies.append(elements[:])
    
    for order in strategies:
        if time.time() > deadline - 5:
            break
        S = set()
        greedy_fill(S, order)
        S = polish(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
    # Final verification
    if not is_valid(best_set):
        # Fallback
        best_set = {1}
        
    return list(best_set)
# EVOLVE-BLOCK-END
