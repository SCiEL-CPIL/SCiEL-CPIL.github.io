"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # 1. Precompute LCM table for speed
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l

    # 2. Identify all bad triples
    # A triple (a, b, c) with a < b < c is bad if:
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_triples = set()
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                lac = lcm_table[a][c]
                if lac != lab:
                    continue
                lbc = lcm_table[b][c]
                if lbc == lab:
                    bad_triples.add((a, b, c))
                    
    if time.time() > deadline - 10:
        # Fallback to simple greedy if precomputation is too slow (unlikely for 128)
        selected = []
        for x in range(1, 129):
            if all(not (lcm_table[a][b] == lcm_table[a][x] == lcm_table[b][x]) for a, b in combinations(selected, 2)):
                selected.append(x)
        return selected

    # 3. Build conflict structure for Branch and Bound
    # We want to find the maximum independent set in the 3-uniform hypergraph of bad triples.
    # This is equivalent to finding the largest subset with no bad triples.
    
    # Let's use a recursive backtracking with pruning (Branch and Bound).
    # State: current set S, next candidate index i.
    # To prune, we can use an upper bound on the number of additional elements we can add.
    # A simple upper bound is the number of remaining candidates.
    # A tighter bound: For each remaining candidate x, if it conflicts with some pair in S, it can't be added.
    # Actually, the standard MIP approach or a smart backtracking is needed.
    
    # Given N=128, a naive backtracking is too slow. 
    # However, the structure of LCM conflicts is specific.
    # Let's try a hybrid: 
    # 1. Greedy to get a good initial lower bound (LB).
    # 2. Branch and Bound to try to beat LB.
    
    # Greedy initialization
    def greedy_solve(order):
        S = set()
        for x in order:
            can_add = True
            # Check against all pairs in S? No, check if any bad triple exists involving x and two in S.
            # We can precompute for each x, the list of pairs (u,v) that form a bad triple with x.
            # conflict[x] = list of (u, v)
            # If u in S and v in S, then x cannot be added.
            for u, v in conflict[x]:
                if u in S and v in S:
                    can_add = False
                    break
            if can_add:
                S.add(x)
        return S

    # Precompute conflict lists for fast checking
    conflict = {i: [] for i in range(1, n + 1)}
    for a, b, c in bad_triples:
        conflict[c].append((a, b))
        conflict[b].append((a, c))
        conflict[a].append((b, c))

    # Initial Greedy to set LB
    # Try a few orderings to get a strong LB
    best_lb = 0
    best_set = set()
    
    # Ordering 1: Ascending conflict count
    conflict_count = {i: len(conflict[i]) for i in range(1, n + 1)}
    order1 = sorted(numbers, key=lambda x: conflict_count[x])
    S1 = greedy_solve(order1)
    if len(S1) > best_lb:
        best_lb = len(S1)
        best_set = S1

    # Ordering 2: Descending value
    order2 = sorted(numbers, reverse=True)
    S2 = greedy_solve(order2)
    if len(S2) > best_lb:
        best_lb = len(S2)
        best_set = S2
        
    # Ordering 3: Primes first? Primes have low LCMs with others? 
    # Actually, primes p and q have lcm pq. 
    # Let's just try a few randoms if time permits.
    import random
    random.seed(42)
    for _ in range(10):
        order_r = numbers[:]
        random.shuffle(order_r)
        Sr = greedy_solve(order_r)
        if len(Sr) > best_lb:
            best_lb = len(Sr)
            best_set = Sr

    # Now, Branch and Bound to try to improve.
    # We will order the candidates to maximize the chance of finding a large set quickly.
    # We'll use the conflict_count order.
    
    candidates = sorted(numbers, key=lambda x: conflict_count[x])
    # Map number to index in candidates list for efficiency? 
    # Or just iterate through candidates list.
    
    current_set = set()
    best_global = best_lb
    best_global_set = best_set
    
    # Recursive B&B
    # index: index in 'candidates' list to consider next
    # current_size: size of current_set
    
    # To make it faster, we can use bitmasks if N was smaller, but N=128 is too big for single int.
    # We will rely on the sparsity of conflicts.
    
    def can_add(x):
        for u, v in conflict[x]:
            if u in current_set and v in current_set:
                return False
        return True

    def solve(idx, current_size):
        nonlocal best_global, best_global_set
        
        # Pruning: if current_size + (number of remaining candidates) <= best_global, prune
        remaining = len(candidates) - idx
        if current_size + remaining <= best_global:
            return
        
        if time.time() > deadline - 5:
            return
            
        if idx == len(candidates):
            if current_size > best_global:
                best_global = current_size
                best_global_set = set(current_set)
            return
        
        x = candidates[idx]
        
        # Option 1: Exclude x
        solve(idx + 1, current_size)
        
        # Option 2: Include x (if possible)
        if can_add(x):
            current_set.add(x)
            solve(idx + 1, current_size + 1)
            current_set.remove(x)

    # The simple recursive B&B might be too slow if the bound is loose.
    # Let's add a heuristic ordering in the B&B: 
    # The candidates are already sorted by conflict count (low conflict first).
    # This is generally good for B&B.
    
    # Optimization: If the gap between best_global and the theoretical max (128) is small, 
    # and the remaining candidates are few, it might finish.
    # But 128 is large. 
    # Let's limit the depth or use a tighter bound.
    # A tighter bound: 
    # For each remaining candidate y, if it is "blocked" by current_set (i.e., there exists a pair (u,v) in conflict[y] such that u,v in current_set), it cannot be added.
    # This check is O(conflicts) per candidate.
    # We can compute an upper bound on addable candidates from the remaining list.
    
    def get_upper_bound(idx):
        # Count how many candidates from idx onwards are NOT blocked by current_set
        count = 0
        for i in range(idx, len(candidates)):
            x = candidates[i]
            # Check if x is blocked
            blocked = False
            for u, v in conflict[x]:
                if u in current_set and v in current_set:
                    blocked = True
                    break
            if not blocked:
                count += 1
        return count

    def solve_optimized(idx, current_size):
        nonlocal best_global, best_global_set
        
        # Pruning with tighter bound
        if current_size + get_upper_bound(idx) <= best_global:
            return
            
        if time.time() > deadline - 5:
            return
            
        if idx == len(candidates):
            if current_size > best_global:
                best_global = current_size
                best_global_set = set(current_set)
            return
        
        x = candidates[idx]
        
        # Option 1: Exclude x
        solve_optimized(idx + 1, current_size)
        
        # Option 2: Include x (if possible)
        if can_add(x):
            current_set.add(x)
            solve_optimized(idx + 1, current_size + 1)
            current_set.remove(x)

    # Run the optimized B&B
    # Note: get_upper_bound is expensive O(N * C). 
    # If this is too slow, we fall back to the greedy result.
    # Given the 80s deadline, we hope it finds a better solution quickly or prunes effectively.
    
    # To be safe, we only run B&B if we have time.
    if time.time() < deadline - 15:
        solve_optimized(0, 0)

    # Final check: verify the best set found
    if best_global_set:
        # Verify no bad triples
        valid = True
        s_list = sorted(list(best_global_set))
        for a, b, c in combinations(s_list, 3):
            if lcm_table[a][b] == lcm_table[a][c] == lcm_table[b][c]:
                valid = False
                break
        if valid:
            return s_list
        else:
            # Fallback to greedy if B&B returned invalid (shouldn't happen)
            return sorted(list(best_set))
    else:
        return sorted(list(best_set))

# EVOLVE-BLOCK-END
