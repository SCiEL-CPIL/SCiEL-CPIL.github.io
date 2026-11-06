import time
import random
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute gcd matrix
    gcd_mat = [[1] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Precompute bad triples
    # bad_triples[i] is a list of (j, k) such that (i, j, k) is a bad triple
    # i < j < k
    bad_triples = {i: [] for i in range(1, N + 1)}
    
    # We need to check all triples (a, b, c) with a < b < c
    # and mark them in bad_triples for a, b, and c?
    # For efficiency in checking if adding x to S is valid,
    # we need to know if there exist a, b in S such that (x, a, b) is bad.
    # So for each x, we store a set of pairs (a, b) that conflict with x.
    
    conflict_pairs = {i: set() for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_mat[a][b]
            for c in range(b + 1, N + 1):
                if gcd_mat[a][c] == gab and gcd_mat[b][c] == gab:
                    # (a, b, c) is a bad triple
                    conflict_pairs[a].add((b, c))
                    conflict_pairs[b].add((a, c))
                    conflict_pairs[c].add((a, b))

    # Greedy Initialization
    def greedy_construct():
        S = set()
        # Try adding elements in random order? Or sorted?
        # Sorted is deterministic. Let's try a few random permutations.
        candidates = numbers[:]
        random.shuffle(candidates)
        for x in candidates:
            # Check if x can be added to S
            valid = True
            for (a, b) in conflict_pairs[x]:
                if a in S and b in S:
                    valid = False
                    break
            if valid:
                S.add(x)
        return S

    # Local Search
    def local_search(S, max_iters=10000):
        S = set(S)
        current_size = len(S)
        stuck_count = 0
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try to add an element
            added = False
            # Sort candidates by number of conflicts with S?
            # Just iterate over not-in-S
            candidates_to_add = [x for x in numbers if x not in S]
            random.shuffle(candidates_to_add)
            
            for x in candidates_to_add:
                valid = True
                for (a, b) in conflict_pairs[x]:
                    if a in S and b in S:
                        valid = False
                        break
                if valid:
                    S.add(x)
                    current_size += 1
                    added = True
                    stuck_count = 0
                    break
            
            if not added:
                stuck_count += 1
                # If stuck, try a swap
                # Pick random u in S, random v not in S
                if S:
                    u = random.choice(list(S))
                    v = random.choice([x for x in numbers if x not in S])
                    if u != v:
                        # Check if S \ {u} U {v} is valid
                        S_new = S.copy()
                        S_new.remove(u)
                        S_new.add(v)
                        
                        # Check validity of v against S_new
                        # Note: removing u cannot create new conflicts, only v might create them.
                        # But v might conflict with pairs in S_new.
                        valid = True
                        for (a, b) in conflict_pairs[v]:
                            if a in S_new and b in S_new:
                                valid = False
                                break
                        
                        if valid:
                            S = S_new
                            # Size remains same
                            stuck_count = 0 # Reset? Or keep?
                            # Let's reset to allow more exploration
                        else:
                            # Try a few more random swaps?
                            pass
                else:
                    break
                    
        return S

    best_S = set()
    
    # Run multiple restarts
    while time.time() < deadline:
        # Greedy
        S = greedy_construct()
        # Local Search
        S = local_search(S)
        
        if len(S) > len(best_S):
            best_S = S
            # If we reached a high size, we can tighten the deadline or just continue
            # Let's continue to see if we can do better
            
    return list(best_S)

# EVOLVE-BLOCK-END
