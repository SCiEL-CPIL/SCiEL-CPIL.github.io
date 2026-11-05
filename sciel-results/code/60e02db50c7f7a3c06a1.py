import time
import math
import random

N = 128

# Precompute exclusion masks
# exclusion_masks[u][v] is a bitmask of w such that {u+1, v+1, w+1} is a forbidden triple
exclusion_masks = [[0] * N for _ in range(N)]

# Precompute combined exclusion for each element: conf[u] = bitmask of all v that conflict with u
# Actually we need pairwise, so keep exclusion_masks but also precompute degree info
conflict_degree = [0] * N

def precompute_masks():
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            L_ab = math.lcm(a, b)
            for c in range(b + 1, N + 1):
                if L_ab % c != 0:
                    continue
                if math.lcm(a, c) == L_ab and math.lcm(b, c) == L_ab:
                    ai, bi, ci = a - 1, b - 1, c - 1
                    exclusion_masks[ai][bi] |= (1 << ci)
                    exclusion_masks[bi][ai] |= (1 << ci)
                    exclusion_masks[ai][ci] |= (1 << bi)
                    exclusion_masks[ci][ai] |= (1 << bi)
                    exclusion_masks[bi][ci] |= (1 << ai)
                    exclusion_masks[ci][bi] |= (1 << ai)
    # Compute conflict degrees
    for u in range(N):
        mask = 0
        for v in range(N):
            mask |= exclusion_masks[u][v]
        conflict_degree[u] = bin(mask).count('1')

precompute_masks()

def popcount(x):
    return x.bit_count()

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_size = 0
    best_set = 0  # Bitmask
    
    # Greedy to get a good initial solution
    def greedy(order):
        S = 0
        excl = 0
        S_list = []
        for x in order:
            if (S >> x) & 1:
                continue
            if (excl >> x) & 1:
                continue
            S |= (1 << x)
            S_list.append(x)
            for u in S_list[:-1]:
                excl |= exclusion_masks[x][u]
        return S, excl

    best_greedy = 0
    # Natural order
    S, _ = greedy(range(N))
    if popcount(S) > popcount(best_greedy):
        best_greedy = S
    
    # Reverse order
    S, _ = greedy(range(N-1, -1, -1))
    if popcount(S) > popcount(best_greedy):
        best_greedy = S
        
    # Random orders
    for _ in range(200):
        if time.time() > deadline - 15:
            break
        order = list(range(N))
        random.shuffle(order)
        S, _ = greedy(order)
        if popcount(S) > popcount(best_greedy):
            best_greedy = S
            
    best_size = popcount(best_greedy)
    best_set = best_greedy

    # Local search: greedily add elements, then try 1-swap improvements
    def local_search(S):
        nonlocal best_size, best_set
        if popcount(S) > best_size:
            best_size = popcount(S)
            best_set = S
        
        S_list = []
        for i in range(N):
            if (S >> i) & 1:
                S_list.append(i)
        
        # Compute excl from current S
        excl = 0
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                excl |= exclusion_masks[S_list[i]][S_list[j]]
        
        # Phase 1: Try to add elements (greedy fill)
        changed = True
        while changed:
            changed = False
            for x in range(N):
                if (S >> x) & 1:
                    continue
                if (excl >> x) & 1:
                    continue
                # x is safe to add
                S |= (1 << x)
                S_list.append(x)
                for u in S_list[:-1]:
                    excl |= exclusion_masks[x][u]
                changed = True
                if popcount(S) > best_size:
                    best_size = popcount(S)
                    best_set = S
                break
        
        # Phase 2: 1-swap local search (remove one, try to add a different one)
        improved = True
        while improved:
            improved = False
            for i in range(len(S_list)):
                if time.time() > deadline - 5:
                    break
                removed = S_list[i]
                # Compute excl without S_list[i]
                excl_no_i = 0
                for a in range(len(S_list)):
                    if a == i:
                        continue
                    for b in range(a+1, len(S_list)):
                        if b == i:
                            continue
                        excl_no_i |= exclusion_masks[S_list[a]][S_list[b]]
                # Try to find an element not in S_list that can be added
                for x in range(N):
                    if (S >> x) & 1:
                        continue
                    if (excl_no_i >> x) & 1:
                        continue
                    # Check x doesn't conflict with any remaining element
                    # excl_no_i already covers conflicts among remaining
                    # Need to check x vs each remaining
                    ok = True
                    for u in S_list:
                        if u == removed:
                            continue
                        if (exclusion_masks[x][u] >> removed) & 1:
                            # This means {x, u, removed} would form a bad triple
                            # But removed is gone, so we need to check if x and u
                            # together with any other element in S_list\{removed} form bad triple
                            pass
                        # Actually excl_no_i already has all conflicts from remaining pairs
                        # We just need x to not be in excl_no_i AND x not to create
                        # new conflicts with remaining elements
                        # excl_no_i contains elements that conflict with some pair in S\{removed}
                        # Adding x creates new pairs (x, u) for u in S\{removed}
                        # Those pairs might forbid additional elements, but that's fine
                        # as long as x itself is not forbidden
                        pass
                    # The key check: x is not in excl_no_i means no pair in S\{removed}
                    # forbids x. But we also need that adding x doesn't create a triple
                    # with two elements already in S\{removed}.
                    # That's exactly what excl_no_i checks: for each pair (a,b) in S\{removed},
                    # exclusion_masks[a][b] tells which c would make {a,b,c} bad.
                    # So if x not in excl_no_i, then no pair in S\{removed} + x forms bad triple.
                    # Additionally, we need that (x, a) pair doesn't forbid any b in S\{removed}.
                    # That's already covered by excl_no_i since it includes exclusion_masks[a][b]
                    # for all pairs a,b in S\{removed}.
                    # Wait, no. excl_no_i = OR of exclusion_masks[a][b] for all pairs a,b in S\{removed}.
                    # This tells us which elements c would form a bad triple with some pair in S\{removed}.
                    # If x is not in excl_no_i, then x doesn't form a bad triple with any pair in S\{removed}.
                    # That's sufficient! We can swap.
                    # Perform the swap
                    S &= ~(1 << removed)
                    S |= (1 << x)
                    S_list[i] = x
                    # Recompute excl
                    excl = 0
                    for a in range(len(S_list)):
                        for b in range(a+1, len(S_list)):
                            excl |= exclusion_masks[S_list[a]][S_list[b]]
                    if popcount(S) > best_size:
                        best_size = popcount(S)
                        best_set = S
                    improved = True
                    break
                if improved:
                    break
            if time.time() > deadline - 5:
                break
        
        return S

    best_set = local_search(best_greedy)

    # B&B with better variable selection
    def bnb(S, R, excl, S_count):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
            
        # Propagate
        R &= ~excl
        
        if R == 0:
            if S_count > best_size:
                best_size = S_count
                best_set = S
            return
            
        # Bound
        if S_count + popcount(R) <= best_size:
            return
            
        # Pick variable: choose from R the one with highest conflict_degree
        # to maximize pruning
        candidates = R
        best_x = -1
        best_deg = -1
        tmp = candidates
        while tmp:
            x = (tmp & -tmp).bit_length() - 1
            if conflict_degree[x] > best_deg:
                best_deg = conflict_degree[x]
                best_x = x
            tmp &= tmp - 1
        
        if best_x < 0:
            return
        
        # Branch 1: Include x
        new_S = S | (1 << best_x)
        new_excl = excl
        tmp = S
        while tmp:
            u = (tmp & -tmp).bit_length() - 1
            new_excl |= exclusion_masks[best_x][u]
            tmp &= tmp - 1
        new_R = R & ~(1 << best_x)
        bnb(new_S, new_R, new_excl, S_count + 1)
        
        # Branch 2: Exclude x
        new_R = R & ~(1 << best_x)
        bnb(S, new_R, excl, S_count)

    # Initialize B&B with the best greedy solution as lower bound
    # Start with all elements available
    if time.time() < deadline - 10:
        initial_R = (1 << N) - 1
        bnb(0, initial_R, 0, 0)
    
    # Convert best_set bitmask to list
    result = []
    for i in range(N):
        if (best_set >> i) & 1:
            result.append(i + 1)
            
    if not result:
        result = [1]
        
    return result
