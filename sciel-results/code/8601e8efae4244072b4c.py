import time
import random
from math import gcd
from itertools import combinations

N = 128
ALL = list(range(1, N + 1))

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    # Precompute GCD
    g = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            g[i][j] = gcd(i, j)
            g[j][i] = g[i][j]
            
    # Precompute all bad triples
    bad_triples = []
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            d = g[a][b]
            for c in range(b+1, N+1):
                if g[a][c] == d and g[b][c] == d:
                    bad_triples.append((a,b,c))
                    
    def is_valid(S):
        S_set = set(S)
        for t in bad_triples:
            if all(x in S_set for x in t):
                return False
        return True
        
    def can_add(x, S_list):
        # Check if adding x creates a bad triple
        # x must not be in S_list
        S_set = set(S_list)
        for a in S_list:
            d_xa = g[x][a]
            for b in S_list:
                if a == b: continue
                if g[x][b] == d_xa and g[a][b] == d_xa:
                    return False
        return True

    best_set = []
    
    # Greedy Removal
    S = set(ALL)
    # We need to efficiently find bad triples in S.
    # Since N is small, we can just iterate bad_triples.
    
    # To pick which element to remove, we count frequency in S.
    # We will do this in a loop.
    
    # Optimization:
    # Instead of removing one by one, we can identify a large set of conflicts.
    
    # Let's just run multiple random greedy removals.
    
    for restart in range(10):
        if time.time() > deadline:
            break
            
        S = set(ALL)
        
        # Shuffle bad triples to break symmetry
        triples = list(bad_triples)
        random.shuffle(triples)
        
        while True:
            if time.time() > deadline:
                break
                
            found = False
            for t in triples:
                if all(x in S for x in t):
                    # Found a bad triple
                    found = True
                    a,b,c = t
                    # Decide which to remove
                    # Count how many bad triples each is in (within S)
                    # This is expensive. Use a heuristic: remove the one with smallest "value" or random.
                    # Or remove the one that appears most often in the *remaining* triples?
                    # For now, random.
                    victim = random.choice([a,b,c])
                    S.discard(victim)
                    break
            
            if not found:
                break
                
        # Local Search
        S_list = list(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try add
            for x in ALL:
                if x in S: continue
                if can_add(x, S_list):
                    S.add(x)
                    S_list.append(x)
                    improved = True
                    break
            if not improved:
                # Try swap
                for x in S_list:
                    for y in ALL:
                        if y in S: continue
                        S_trial = S - {x}
                        if can_add(y, list(S_trial)):
                            S = S_trial | {y}
                            S_list = list(S)
                            improved = True
                            break
                    if improved: break
                    
        if len(S) > len(best_set):
            best_set = list(S)
            
    return best_set
