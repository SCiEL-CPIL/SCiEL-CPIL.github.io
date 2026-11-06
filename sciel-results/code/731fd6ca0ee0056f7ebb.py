import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N = 128
    elements = list(range(1, N + 1))
    
    # 1. Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
            
    # 2. Precompute BadPairs[x]
    # BadPairs[x] contains pairs (a, b) with a < b, a != x, b != x
    # such that gcd(a,b) == gcd(a,x) == gcd(b,x)
    BadPairs = [set() for _ in range(N + 1)]
    
    # Iterate through all triples to populate BadPairs
    # This is O(N^3) = 128^3 ~ 2M iterations. Fast enough.
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd_table[a][b]
            for c in range(b + 1, N + 1):
                if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                    # (a,b,c) is a bad triangle
                    # For x=c, the conflicting pair is (a,b)
                    # For x=b, the conflicting pair is (a,c)
                    # For x=a, the conflicting pair is (b,c)
                    BadPairs[c].add((a, b))
                    BadPairs[b].add((a, c))
                    BadPairs[a].add((b, c))
    
    # Convert sets to lists for faster iteration
    for i in range(N + 1):
        BadPairs[i] = list(BadPairs[i])
        
    # 3. Greedy Construction with Degree Ordering
    # Sort elements by number of bad pairs (ascending)
    # Elements with fewer conflicts are "safer" to pick
    degree = [len(BadPairs[i]) for i in range(N + 1)]
    
    def greedy_construct(order):
        selected = set()
        # We iterate through the given order
        for x in order:
            # Check if x can be added
            can_add = True
            for (a, b) in BadPairs[x]:
                if a in selected and b in selected:
                    can_add = False
                    break
            if can_add:
                selected.add(x)
        return selected

    def local_search(initial_set, max_steps=1000):
        """
        Perform 1-1 swap local search.
        Try to swap an element in S with one not in S to increase size or maintain size while reducing conflicts?
        Actually, we only care about VALID sets (size is the objective).
        So we look for swaps that keep the set valid and increase size?
        No, swap keeps size constant.
        We want to ADD elements.
        So local search should try to find a "hole" to fill.
        
        Better: 
        1. Start with valid set S.
        2. Try to add any x not in S. If valid, add it.
        3. If no single add works, try a swap: remove y from S, add x.
           If S\{y} U {x} is valid, and |S\{y} U {x}| == |S|, it doesn't help directly.
           But it might allow a subsequent add.
           
        Simpler: Just do many greedy restarts.
        """
        S = set(initial_set)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add any element
            for x in range(1, N + 1):
                if x not in S:
                    can_add = True
                    for (a, b) in BadPairs[x]:
                        if a in S and b in S:
                            can_add = False
                            break
                    if can_add:
                        S.add(x)
                        improved = True
                        break # Restart the loop to try adding more
        return S

    best_set = set()
    best_size = 0
    
    # Generate base order: sorted by degree
    base_order = sorted(elements, key=lambda x: degree[x])
    
    # Run multiple restarts
    # 1. Deterministic degree-based greedy
    current_set = greedy_construct(base_order)
    current_set = local_search(current_set)
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = set(current_set)
        
    # 2. Randomized restarts
    # We have ~80s. Each greedy run is fast (~0.01s).
    # Local search is also fast.
    # Let's do 500 restarts.
    
    for i in range(500):
        if time.time() > deadline - 5:
            break
            
        # Perturb the base order
        order = base_order[:]
        # Randomly swap a few elements to break symmetry
        for _ in range(5):
            idx1 = random.randint(0, N - 1)
            idx2 = random.randint(0, N - 1)
            order[idx1], order[idx2] = order[idx2], order[idx1]
            
        current_set = greedy_construct(order)
        current_set = local_search(current_set)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
    # 3. Final Polish: Try to add any remaining elements to best_set
    for x in range(1, N + 1):
        if x not in best_set:
            can_add = True
            for (a, b) in BadPairs[x]:
                if a in best_set and b in best_set:
                    can_add = False
                    break
            if can_add:
                best_set.add(x)
                
    return list(best_set)
