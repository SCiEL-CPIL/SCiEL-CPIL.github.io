import math
import random
import time
from itertools import combinations

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0  # 55 seconds to be safe
    
    n = 128
    elements = list(range(1, n + 1))
    
    # 1. Precompute LCMs and Bad Triples
    # lcm[i][j] for 1 <= i < j <= 128
    # We only need to check triples (a,b,c)
    # Optimization: 
    # A triple (a,b,c) is bad if lcm(a,b)==lcm(a,c)==lcm(b,c).
    # Let's precompute lcm for all pairs.
    lcm_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            lcm_table[i][j] = i * j // math.gcd(i, j)
            lcm_table[j][i] = lcm_table[i][j]
            
    bad_triples = []
    # To speed up, we can group by LCM?
    # If lcm(a,b) = L, then a and b are divisors of L.
    # Let's just iterate all triples. 128C3 is ~350k. 
    # 350k * 3 LCM lookups is very fast in Python if optimized, 
    # but even with math.lcm it's fine.
    
    # We need to store for each x, the pairs (y,z) that form a bad triple with x.
    # triples_by_vertex[x] = list of (y,z)
    triples_by_vertex = {i: [] for i in range(1, n+1)}
    
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            lab = lcm_table[a][b]
            for c in range(b+1, n+1):
                # Check if lcm(a,c) == lab and lcm(b,c) == lab
                # We can precompute lcm(a,c) and lcm(b,c) or just use table
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
                    triples_by_vertex[a].append((b, c))
                    triples_by_vertex[b].append((a, c))
                    triples_by_vertex[c].append((a, b))
                    
    # 2. Greedy Initialization
    S = set()
    # Sort by degree in hypergraph? Or just natural order?
    # Elements with fewer bad triples are "safer" to add.
    # Let's sort by len(triples_by_vertex[i])
    sorted_elements = sorted(elements, key=lambda x: len(triples_by_vertex[x]))
    
    for x in sorted_elements:
        # Check if x can be added without creating a bad triple
        # A bad triple is created if there exist y,z in S such that (y,z) is in triples_by_vertex[x]
        can_add = True
        for y, z in triples_by_vertex[x]:
            if y in S and z in S:
                can_add = False
                break
        if can_add:
            S.add(x)
            
    best_S = set(S)
    best_size = len(S)
    
    # 3. Local Search / Simulated Annealing
    # We will try to improve the size.
    # Since we want to maximize size, and the constraint is "no bad triples",
    # we can try to add elements and remove conflicting ones.
    
    # Current violations = 0 since S is valid.
    # We will use a move that adds one element and removes necessary others.
    # But simpler: Random Restart + Greedy?
    # Or: Try to add a random x not in S. If it conflicts, remove one of the conflicting elements.
    # If we add x and remove y, size stays same. If we can remove y and x doesn't conflict with others, we might be able to add another later.
    
    # Let's use a simple hill climbing with restarts.
    # Move: Pick x not in S. 
    # If x is valid with S, add it.
    # Else, pick a conflict (y,z). Remove y. Add x? 
    # If we remove y, the triple (x,y,z) is broken.
    # But x might conflict with other pairs.
    # So we must remove ALL y in S that form a bad triple with x and some z in S.
    # Let Conflicts(x, S) = { y in S | exists z in S \ {y} such that {x,y,z} is bad }
    # If we add x, we must remove Conflicts(x, S).
    # New Size = |S| + 1 - |Conflicts(x, S)|.
    # We accept if New Size > |S|. i.e., |Conflicts(x, S)| == 0.
    # This just finds a valid addition.
    
    # To escape local optima, we might accept moves that keep size same (swaps) or even decrease slightly (annealing).
    # But given the time, let's try multiple random orderings for Greedy.
    
    random.seed(42)
    
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        # Randomized Greedy
        S_curr = set()
        order = elements.copy()
        random.shuffle(order)
        # Bias towards lower degree?
        # Let's just shuffle.
        
        for x in order:
            # Check validity
            valid = True
            for y, z in triples_by_vertex[x]:
                if y in S_curr and z in S_curr:
                    valid = False
                    break
            if valid:
                S_curr.add(x)
                
        if len(S_curr) > best_size:
            best_size = len(S_curr)
            best_S = set(S_curr)
            
    # 4. Local Search on best_S
    # Try to add any element that is currently not in best_S but is valid.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in elements:
            if x in best_S:
                continue
            # Check if x can be added
            valid = True
            for y, z in triples_by_vertex[x]:
                if y in best_S and z in best_S:
                    valid = False
                    break
            if valid:
                best_S.add(x)
                best_size += 1
                improved = True
                
    # 5. Try to swap elements to increase size?
    # If we remove one element, we might be able to add two?
    # This is a 1-for-2 swap.
    # Try removing each y in best_S.
    # Check if we can add two elements x1, x2 from complement.
    # This is expensive.
    
    # Given the sparsity, the greedy should be very close to optimal.
    # The number of bad triples is likely small.
    
    return list(best_S)

# Test locally? No, just output.
