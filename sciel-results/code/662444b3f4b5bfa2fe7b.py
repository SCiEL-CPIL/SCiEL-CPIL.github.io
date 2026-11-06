import sys
from itertools import combinations
from math import lcm

# Increase recursion limit for deep DFS
sys.setrecursionlimit(10000)

def construct():
    N = 128
    # 1. Precompute LCMs and Bad Triples
    # lcm[i][j] for 1 <= i < j <= N
    # We only need to check triples (a, b, c) with a < b < c
    # Bad if lcm(a,b) == lcm(a,c) == lcm(b,c)
    
    # To speed up, we can just iterate all triples
    # There are ~350k triples. This is fast.
    
    bad_triples = []
    # Store conflicts for each vertex: v -> list of (u, w)
    # u, w are the other two vertices in the bad triple
    conflicts = {i: [] for i in range(1, N+1)}
    
    # Optimization: Precompute lcm for all pairs?
    # lcm(a, b) is cheap.
    
    # Iterate a < b < c
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            lab = lcm(a, b)
            for c in range(b+1, N+1):
                # Check if lcm(a,c) == lab and lcm(b,c) == lab
                # If lcm(a,c) != lab, no need to check lcm(b,c)
                lac = lcm(a, c)
                if lac == lab:
                    lbc = lcm(b, c)
                    if lbc == lab:
                        bad_triples.append((a, b, c))
                        conflicts[a].append((b, c))
                        conflicts[b].append((a, c))
                        conflicts[c].append((a, b))
                        
    # 2. Greedy Initial Solution
    # Sort by some heuristic? Or just 1..N
    # The seed used 1..N. Let's try to get a better initial bound.
    # A simple greedy: pick vertex that conflicts with least others?
    # Or just use the seed's result as initial best.
    
    def get_greedy_solution():
        selected = []
        for x in range(1, N+1):
            valid = True
            for a, b in combinations(selected, 2):
                if lcm(a, b) == lcm(a, x) == lcm(b, x):
                    valid = False
                    break
            if valid:
                selected.append(x)
        return len(selected), selected

    init_size, _ = get_greedy_solution()
    best_size = init_size
    
    # 3. Branch and Bound DFS
    # State: S_mask (selected), A_mask (available)
    # We use 0-indexed bits for 1..128, so bit i represents number i+1?
    # Or bit v-1 represents number v.
    
    # Precompute masks for conflicts?
    # For each v, we have list of pairs (u, w).
    # To check if v is valid given S_mask:
    # For each (u, w) in conflicts[v]:
    #   if (S_mask & (1 << (u-1))) and (S_mask & (1 << (w-1))):
    #       return False
    # return True
    
    # To speed up, we can precompute for each v, a list of masks for the pairs?
    # No, just the indices.
    
    # Helper: popcount
    def popcount(x):
        return bin(x).count('1')
        
    # Recursive search
    # S_mask: bitmask of selected numbers
    # A_mask: bitmask of available numbers
    # current_size: number of selected numbers
    
    # We will iterate through vertices in A_mask.
    # To make it efficient, we can process vertices in a specific order.
    
    # Let's store the list of vertices for each mask? No, just iterate bits.
    
    def dfs(S_mask, A_mask, current_size):
        nonlocal best_size
        
        # Upper bound: current_size + number of available vertices
        # But some available vertices might be invalid due to S_mask
        # Let's filter A_mask first to get valid_A_mask
        
        valid_A_mask = 0
        # Iterate through bits in A_mask
        temp_A = A_mask
        while temp_A:
            # Get lowest set bit
            lsb = temp_A & -temp_A
            v_idx = lsb.bit_length() - 1 # 0-indexed, so number is v_idx + 1
            v = v_idx + 1
            
            # Check if v is valid given S_mask
            is_valid = True
            for u, w in conflicts[v]:
                # Check if u and w are in S_mask
                # u-1 and w-1 are bit indices
                if (S_mask & (1 << (u-1))) and (S_mask & (1 << (w-1))):
                    is_valid = False
                    break
            
            if is_valid:
                valid_A_mask |= lsb
            
            temp_A ^= lsb
            
        # Prune 1: Even if all valid_A_mask are added, can we beat best?
        if current_size + popcount(valid_A_mask) <= best_size:
            return
            
        if valid_A_mask == 0:
            if current_size > best_size:
                best_size = current_size
            return
            
        # Prune 2: Stronger bound?
        # We can't easily compute MIS of valid_A_mask hypergraph quickly.
        # But we can try to pick a vertex.
        
        # Heuristic: Pick a vertex from valid_A_mask.
        # To maximize pruning, pick a vertex that "blocks" many others?
        # Or just pick the first one.
        # Let's pick the vertex with the smallest index for determinism, 
        # but maybe sorting by "constraint" is better.
        # For now, let's just pick the first set bit.
        
        lsb = valid_A_mask & -valid_A_mask
        v_idx = lsb.bit_length() - 1
        v = v_idx + 1
        
        new_A_mask = valid_A_mask & ~lsb
        
        # Branch 1: Include v
        new_S_mask = S_mask | lsb
        dfs(new_S_mask, new_A_mask, current_size + 1)
        
        # Branch 2: Exclude v
        dfs(S_mask, new_A_mask, current_size)
        
    # Initial masks
    # A_mask starts with all 1..128
    # Bit i-1 corresponds to number i
    initial_A_mask = (1 << N) - 1
    initial_S_mask = 0
    
    dfs(initial_S_mask, initial_A_mask, 0)
    
    # Reconstruct the set? 
    # The task asks to "Return nonempty subset". 
    # It doesn't strictly require the *optimal* subset to be returned, 
    # but "Maximize cardinality" implies the value returned (cardinality) is what's measured.
    # "The evaluator measures the raw value (higher is better)."
    # The function returns the subset. The evaluator likely computes len(subset).
    # So we need to return the actual set.
    
    # The DFS above only tracks size. We need to track the set or reconstruct it.
    # Modifying DFS to track the best set is slightly more expensive but necessary.
    
    best_set_mask = 0
    
    def dfs_track(S_mask, A_mask, current_size):
        nonlocal best_size, best_set_mask
        
        if current_size + popcount(A_mask) <= best_size:
            return
            
        # Filter valid_A_mask
        valid_A_mask = 0
        temp_A = A_mask
        while temp_A:
            lsb = temp_A & -temp_A
            v_idx = lsb.bit_length() - 1
            v = v_idx + 1
            
            is_valid = True
            for u, w in conflicts[v]:
                if (S_mask & (1 << (u-1))) and (S_mask & (1 << (w-1))):
                    is_valid = False
                    break
            
            if is_valid:
                valid_A_mask |= lsb
            
            temp_A ^= lsb
            
        if current_size + popcount(valid_A_mask) <= best_size:
            return
            
        if valid_A_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = S_mask
            return
            
        lsb = valid_A_mask & -valid_A_mask
        v_idx = lsb.bit_length() - 1
        v = v_idx + 1
        
        new_A_mask = valid_A_mask & ~lsb
        
        # Branch 1: Include v
        new_S_mask = S_mask | lsb
        dfs_track(new_S_mask, new_A_mask, current_size + 1)
        
        # Branch 2: Exclude v
        dfs_track(S_mask, new_A_mask, current_size)
        
    # Reset best values
    best_size = init_size
    # We need the mask for the greedy solution to initialize best_set_mask
    # Let's just run the greedy to get the mask too
    def get_greedy_mask():
        selected = []
        mask = 0
        for x in range(1, N+1):
            valid = True
            for a, b in combinations(selected, 2):
                if lcm(a, b) == lcm(a, x) == lcm(b, x):
                    valid = False
                    break
            if valid:
                selected.append(x)
                mask |= (1 << (x-1))
        return len(selected), mask
        
    best_size, best_set_mask = get_greedy_mask()
    
    dfs_track(0, initial_A_mask, 0)
    
    # Convert best_set_mask back to list
    result = []
    for i in range(N):
        if best_set_mask & (1 << i):
            result.append(i + 1)
            
    return result

# EVOLVE-BLOCK-END
