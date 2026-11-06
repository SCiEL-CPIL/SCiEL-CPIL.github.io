import time
import random
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N = 128
    elements = list(range(1, N + 1))
    # Map 1..128 to 0..127 for bitset operations
    # index i corresponds to number i+1
    
    # 1. Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
            
    # 2. Identify all "bad" triples (a, b, c) such that 
    # gcd(a,b) == gcd(b,c) == gcd(c,a)
    # We need to find the largest subset of [1, 128] containing no such triple.
    # This is a Maximum Independent Set problem in a 3-uniform hypergraph.
    
    # To use bitsets efficiently, we represent the set of available vertices as a Python integer (bitmask).
    # Bit i is set if number (i+1) is available.
    
    # Precompute for each pair (u, v) the bitmask of all w such that (u, v, w) is a bad triple.
    # u, v, w are 0-indexed.
    # If we select u and v, we cannot select any w in Conflict[u][v].
    # Also, if we select u and v, we cannot select any w such that (u, w, v) is bad, etc.
    # Actually, the condition is symmetric. 
    # Let's precompute:
    # ForbiddenPairs[u][v] = bitmask of w != u, w != v such that gcd(u+1, v+1) == gcd(u+1, w+1) == gcd(v+1, w+1)
    
    # Wait, the condition is "three pairwise greatest common divisors are equal".
    # So gcd(a,b) == gcd(b,c) == gcd(c,a).
    
    # Let's build the list of bad triples first to be safe, then derive constraints.
    # However, for branch and bound, we need fast checks.
    
    # Strategy for Branch and Bound:
    # State: (available_mask, current_set_size)
    # We try to add vertices. 
    # A better formulation for MIS in hypergraph:
    # We select vertices one by one.
    # If we select vertex v, we must remove from 'available' all vertices that form a bad triple with any two previously selected vertices? No.
    # The constraint is: No three selected vertices form a bad triple.
    
    # Let S be the set of selected vertices.
    # When adding a new vertex x to S, we must ensure that for all pairs (a, b) in S, (a, b, x) is NOT a bad triple.
    # This means x must not be in ForbiddenByPair[a][b] for any (a,b) in S.
    
    # This check is O(|S|^2) per addition, which might be slow if done naively in deep branches.
    # However, N=128 is small.
    
    # Alternative: Backtracking with "conflict set" update.
    # Maintain a bitmask `forbidden` which represents vertices that CANNOT be added to the current set S 
    # because they would form a bad triple with some pair in S.
    # Initially forbidden = 0.
    # When we add x to S:
    #   New forbidden vertices are those y such that there exists z in S (z != x) where (x, z, y) is a bad triple.
    #   Also, x itself is removed from available.
    #   We also need to ensure that x itself was not already forbidden.
    
    # Let's precompute `BadWith[x][z]`: bitmask of all y such that (x, z, y) is a bad triple.
    # Note: (x, z, y) is bad iff gcd(x+1, z+1) == gcd(x+1, y+1) == gcd(z+1, y+1).
    
    BadWith = [[0] * N for _ in range(N)]
    
    # Precompute BadWith
    # For efficiency, iterate through all triples? 128^3 / 6 is ~350k. Fast.
    for a in range(N):
        for b in range(a + 1, N):
            g_ab = gcd_table[a + 1][b + 1]
            for c in range(b + 1, N):
                if gcd_table[a + 1][c + 1] == g_ab and gcd_table[b + 1][c + 1] == g_ab:
                    # Triple (a, b, c) is bad
                    # For pair (a, b), c is forbidden
                    # For pair (a, c), b is forbidden
                    # For pair (b, c), a is forbidden
                    BadWith[a][b] |= (1 << c)
                    BadWith[b][a] |= (1 << c)
                    
                    BadWith[a][c] |= (1 << b)
                    BadWith[c][a] |= (1 << b)
                    
                    BadWith[b][c] |= (1 << a)
                    BadWith[c][b] |= (1 << a)
                    
    # Now we have BadWith.
    # If S contains {x, z}, then we cannot add any y in BadWith[x][z].
    
    # Branch and Bound Algorithm:
    # We want to maximize the number of selected vertices.
    # We use a recursive function `solve(available_mask, current_size, forbidden_mask)`.
    # `available_mask`: bits for vertices that can still be considered.
    # `forbidden_mask`: bits for vertices that are currently forbidden due to pairs in the current selection.
    # Note: A vertex is selectable if it is in `available_mask` AND NOT in `forbidden_mask`.
    
    # To make it efficient, we should order vertices.
    # We can use a heuristic: pick the vertex with the highest "degree" of conflicts? 
    # Or just iterate through bits.
    
    # Let's implement a standard branch and bound.
    # Upper bound: current_size + popcount(available_mask & ~forbidden_mask)
    # If upper_bound <= best_found, prune.
    
    # To improve pruning, we can use a greedy upper bound or just the simple count.
    # Given N=128, simple count might be enough if the tree is pruned well.
    # However, the "forbidden" mask grows as we select more vertices, reducing the available candidates.
    
    # Let's add a greedy lower bound to initialize `best_found`.
    def greedy_lower_bound():
        avail = (1 << N) - 1
        forbidden = 0
        count = 0
        selected = []
        
        # Simple greedy: pick available, non-forbidden vertex, update forbidden.
        # To be deterministic and fast, we can just pick the lowest index available.
        # Or sort by some heuristic. Let's try lowest index first for speed in initialization.
        
        # Actually, a better greedy: 
        # At each step, pick the vertex that minimizes the number of new forbidden vertices?
        # That's expensive. Let's stick to a fast greedy for the initial bound.
        
        # We'll do a few random greedy restarts to get a good lower bound.
        best_lb = 0
        for _ in range(10):
            avail = (1 << N) - 1
            forbidden = 0
            count = 0
            while avail & ~forbidden:
                # Pick a random available non-forbidden vertex
                candidates = avail & ~forbidden
                # Get index of a random set bit
                # candidates.bit_length() - 1 is highest. 
                # Let's just pick the lowest set bit for simplicity and speed in this loop?
                # Or random.
                # Let's pick the one with index i.
                # To pick random:
                # We can convert to list? No, too slow.
                # Let's just pick the lowest set bit.
                lsb = candidates & -candidates
                idx = lsb.bit_length() - 1
                
                # Select idx
                count += 1
                selected.append(idx)
                
                # Update forbidden
                # For each z in selected (excluding idx), add BadWith[idx][z] to forbidden
                # We need to iterate through selected.
                for z in selected[:-1]:
                    forbidden |= BadWith[idx][z]
                    
                # Remove idx from available
                avail &= ~(1 << idx)
                
            if count > best_lb:
                best_lb = count
                
        return best_lb

    initial_bound = greedy_lower_bound()
    best_found = initial_bound
    
    # Recursive B&B
    # To avoid recursion depth issues, we could use an explicit stack, but N=128 is small enough 
    # that depth won't exceed 128. Python recursion limit is usually 1000.
    
    # We need to be careful with the time limit.
    
    # Heuristic for ordering: 
    # We can sort vertices by their "potential" or just iterate 0..127.
    # A good heuristic is to process vertices that appear in many bad triples first? 
    # Or vertices that are "conflicting" with many others.
    
    # Let's try a static ordering based on the number of bad triples each vertex is part of.
    # Vertices with more conflicts should be decided earlier (branch on them).
    
    conflict_count = [0] * N
    for a in range(N):
        for b in range(N):
            conflict_count[a] += BadWith[a][b].bit_count()
            
    # Sort indices by conflict_count descending
    order = sorted(range(N), key=lambda x: conflict_count[x], reverse=True)
    
    # We will iterate through `order` in the branching.
    # However, the state `available` and `forbidden` change.
    # It's easier to branch on the "next" vertex in the fixed order that is still available and not forbidden.
    
    def solve(avail, forbidden, current_size, idx_in_order):
        nonlocal best_found
        
        # Pruning
        # Upper bound: current_size + number of available non-forbidden vertices from idx_in_order onwards
        # Note: `avail` contains all bits. `forbidden` contains bits forbidden by current selection.
        # We can only pick from `avail & ~forbidden`.
        
        # Simple bound:
        remaining = avail & ~forbidden
        # We can at most add popcount(remaining)
        if current_size + remaining.bit_count() <= best_found:
            return
        
        # Base case: no more vertices to consider in order?
        # Actually, we should check if there are any candidates left.
        # If `remaining` is 0, we can't add more.
        if remaining == 0:
            if current_size > best_found:
                best_found = current_size
            return
        
        # Find the next vertex to branch on.
        # We iterate through `order` starting from `idx_in_order`.
        # We need the first vertex v in `order[idx_in_order:]` such that v is in `remaining`.
        
        v = -1
        for i in range(idx_in_order, N):
            candidate = order[i]
            if (remaining >> candidate) & 1:
                v = candidate
                break
        
        if v == -1:
            # No more candidates
            if current_size > best_found:
                best_found = current_size
            return
            
        # Branch 1: Exclude v
        # We just move to the next index in order.
        # avail and forbidden remain the same? 
        # Yes, excluding v doesn't change the constraints for other vertices.
        # But we must ensure we don't pick v again. 
        # Since we iterate `order` sequentially, and we move `idx_in_order` to i+1, 
        # we effectively skip v.
        
        # However, `avail` still has v set. 
        # If we just increment idx, we skip v in the search path.
        # But what if v becomes available later? No, availability is monotonic decreasing (we only remove).
        # Forbidden is monotonic increasing.
        # So if v is not picked now, it will never be picked in this branch.
        
        solve(avail, forbidden, current_size, i + 1)
        
        # Branch 2: Include v
        # Check if v is still valid (it is, by definition of `remaining`).
        
        new_forbidden = forbidden
        # For every z in current selection, add BadWith[v][z] to new_forbidden.
        # Wait, we don't store the current selection `S` explicitly in the state to save memory/time?
        # We need to know which vertices are in S to update `forbidden`.
        
        # This is a problem. The state `forbidden` is derived from S.
        # If we don't pass S, we can't update `forbidden` correctly when adding v.
        # We need to know all z in S such that (v, z, y) is bad.
        
        # Solution: Pass the set of selected vertices? 
        # That makes the state large. 
        # Alternative: 
        # The `forbidden` mask at any point is the union of BadWith[x][y] for all pairs (x,y) in S.
        # When we add v, the new forbidden bits are:
        #   old_forbidden | (union of BadWith[v][z] for all z in S)
        
        # We need S. 
        # Since N=128, we can represent S as a bitmask too.
        # State: (avail, forbidden, S_mask, current_size, idx_in_order)
        
        # Let's refactor solve to take S_mask.
        
        # Re-defining solve inside construct to access nonlocal best_found easily?
        # Yes.
        
        # Let's restart the logic with S_mask.
        
        # Actually, I can just implement the recursive function properly now.
        
        pass # Placeholder, will implement below

    # Let's implement the full recursive function with S_mask.
    
    def bb_solve(avail, forbidden, S_mask, current_size, idx_in_order):
        nonlocal best_found
        
        # Pruning
        remaining = avail & ~forbidden
        if current_size + remaining.bit_count() <= best_found:
            return
            
        if remaining == 0:
            if current_size > best_found:
                best_found = current_size
            return
            
        # Find next candidate v from order[idx_in_order:] that is in remaining
        v = -1
        next_idx = idx_in_order
        for i in range(idx_in_order, N):
            candidate = order[i]
            if (remaining >> candidate) & 1:
                v = candidate
                next_idx = i + 1
                break
                
        if v == -1:
            if current_size > best_found:
                best_found = current_size
            return
            
        # Branch 1: Exclude v
        bb_solve(avail, forbidden, S_mask, current_size, next_idx)
        
        # Branch 2: Include v
        # Update forbidden
        # new_forbidden = forbidden | (union of BadWith[v][z] for z in S)
        # We can compute this by iterating through set bits in S_mask.
        
        new_forbidden = forbidden
        s_temp = S_mask
        while s_temp:
            lsb = s_temp & -s_temp
            z_idx = lsb.bit_length() - 1
            new_forbidden |= BadWith[v][z_idx]
            s_temp ^= lsb
            
        # Check if v is still allowed? 
        # v was in `remaining`, so v is not in `forbidden`.
        # But could v be in `new_forbidden`?
        # `new_forbidden` contains vertices y such that (v, z, y) is bad for some z in S.
        # Does it contain v itself? 
        # BadWith[v][z] contains y. Can y == v?
        # BadWith[v][z] is defined for y != v, y != z.
        # So v is not in BadWith[v][z].
        # So v is not in new_forbidden.
        
        # Also, we must remove v from avail.
        new_avail = avail & ~(1 << v)
        
        # New S_mask
        new_S_mask = S_mask | (1 << v)
        
        # Time check
        if time.time() > deadline:
            return
            
        bb_solve(new_avail, new_forbidden, new_S_mask, current_size + 1, next_idx)
        
    # Start B&B
    # Initial state:
    # avail = all bits set
    # forbidden = 0
    # S_mask = 0
    # current_size = 0
    # idx_in_order = 0
    
    initial_avail = (1 << N) - 1
    bb_solve(initial_avail, 0, 0, 0, 0)
    
    # The B&B finds the max size, but not the set.
    # We need to return the set.
    # We can modify the B&B to store the best set found.
    
    # Let's refactor to store the best set.
    
    best_set_mask = 0
    
    def bb_solve_with_set(avail, forbidden, S_mask, current_size, idx_in_order):
        nonlocal best_found, best_set_mask
        
        # Pruning
        remaining = avail & ~forbidden
        if current_size + remaining.bit_count() <= best_found:
            return
            
        if remaining == 0:
            if current_size > best_found:
                best_found = current_size
                best_set_mask = S_mask
            return
            
        # Find next candidate v
        v = -1
        next_idx = idx_in_order
        for i in range(idx_in_order, N):
            candidate = order[i]
            if (remaining >> candidate) & 1:
                v = candidate
                next_idx = i + 1
                break
                
        if v == -1:
            if current_size > best_found:
                best_found = current_size
                best_set_mask = S_mask
            return
            
        # Branch 1: Exclude v
        bb_solve_with_set(avail, forbidden, S_mask, current_size, next_idx)
        
        # Branch 2: Include v
        new_forbidden = forbidden
        s_temp = S_mask
        while s_temp:
            lsb = s_temp & -s_temp
            z_idx = lsb.bit_length() - 1
            new_forbidden |= BadWith[v][z_idx]
            s_temp ^= lsb
            
        new_avail = avail & ~(1 << v)
        new_S_mask = S_mask | (1 << v)
        
        if time.time() > deadline:
            return
            
        bb_solve_with_set(new_avail, new_forbidden, new_S_mask, current_size + 1, next_idx)
        
    # Run B&B
    # To get a good initial bound, we use the greedy one.
    # We also need the greedy set to initialize best_set_mask.
    
    def get_greedy_set():
        avail = (1 << N) - 1
        forbidden = 0
        S_mask = 0
        count = 0
        
        # Use the same order for greedy? 
        # Let's just use random restarts with random order for diversity in LB.
        best_greedy_mask = 0
        best_greedy_count = 0
        
        for _ in range(20):
            # Random shuffle of order
            ord_copy = order[:]
            random.shuffle(ord_copy)
            
            avail = (1 << N) - 1
            forbidden = 0
            S_mask = 0
            count = 0
            
            for v in ord_copy:
                if (avail >> v) & 1 and not (forbidden >> v) & 1:
                    # Can add v
                    S_mask |= (1 << v)
                    count += 1
                    avail &= ~(1 << v)
                    
                    # Update forbidden
                    s_temp = S_mask & ~(1 << v) # Exclude v itself
                    while s_temp:
                        lsb = s_temp & -s_temp
                        z_idx = lsb.bit_length() - 1
                        forbidden |= BadWith[v][z_idx]
                        s_temp ^= lsb
                        
            if count > best_greedy_count:
                best_greedy_count = count
                best_greedy_mask = S_mask
                
        return best_greedy_count, best_greedy_mask

    initial_count, initial_mask = get_greedy_set()
    best_found = initial_count
    best_set_mask = initial_mask
    
    # Run B&B
    bb_solve_with_set(initial_avail, 0, 0, 0, 0)
    
    # Convert best_set_mask to list
    result = []
    for i in range(N):
        if (best_set_mask >> i) & 1:
            result.append(i + 1)
            
    # Final sanity check: is the set valid?
    # It should be, by construction.
    
    return result

# To avoid recursion limit issues, increase it
import sys
sys.setrecursionlimit(2000)

# Note: The above code defines construct, but the recursive function is nested.
# I need to make sure the final construct function returns the list.

# Let's wrap it properly.

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 128
    elements = list(range(1, N + 1))
    
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
            
    BadWith = [[0] * N for _ in range(N)]
    
    for a in range(N):
        for b in range(a + 1, N):
            g_ab = gcd_table[a + 1][b + 1]
            for c in range(b + 1, N):
                if gcd_table[a + 1][c + 1] == g_ab and gcd_table[b + 1][c + 1] == g_ab:
                    BadWith[a][b] |= (1 << c)
                    BadWith[b][a] |= (1 << c)
                    
                    BadWith[a][c] |= (1 << b)
                    BadWith[c][a] |= (1 << b)
                    
                    BadWith[b][c] |= (1 << a)
                    BadWith[c][b] |= (1 << a)
                    
    conflict_count = [0] * N
    for a in range(N):
        for b in range(N):
            conflict_count[a] += BadWith[a][b].bit_count()
            
    order = sorted(range(N), key=lambda x: conflict_count[x], reverse=True)
    
    best_found = 0
    best_set_mask = 0
    
    def get_greedy_set():
        best_greedy_mask = 0
        best_greedy_count = 0
        
        for _ in range(20):
            ord_copy = order[:]
            random.shuffle(ord_copy)
            
            avail = (1 << N) - 1
            forbidden = 0
            S_mask = 0
            count = 0
            
            for v in ord_copy:
                if (avail >> v) & 1 and not (forbidden >> v) & 1:
                    S_mask |= (1 << v)
                    count += 1
                    avail &= ~(1 << v)
                    
                    s_temp = S_mask & ~(1 << v)
                    while s_temp:
                        lsb = s_temp & -s_temp
                        z_idx = lsb.bit_length() - 1
                        forbidden |= BadWith[v][z_idx]
                        s_temp ^= lsb
                        
            if count > best_greedy_count:
                best_greedy_count = count
                best_greedy_mask = S_mask
                
        return best_greedy_count, best_greedy_mask

    initial_count, initial_mask = get_greedy_set()
    best_found = initial_count
    best_set_mask = initial_mask
    
    def bb_solve_with_set(avail, forbidden, S_mask, current_size, idx_in_order):
        nonlocal best_found, best_set_mask
        
        remaining = avail & ~forbidden
        if current_size + remaining.bit_count() <= best_found:
            return
            
        if remaining == 0:
            if current_size > best_found:
                best_found = current_size
                best_set_mask = S_mask
            return
            
        v = -1
        next_idx = idx_in_order
        for i in range(idx_in_order, N):
            candidate = order[i]
            if (remaining >> candidate) & 1:
                v = candidate
                next_idx = i + 1
                break
                
        if v == -1:
            if current_size > best_found:
                best_found = current_size
                best_set_mask = S_mask
            return
            
        bb_solve_with_set(avail, forbidden, S_mask, current_size, next_idx)
        
        new_forbidden = forbidden
        s_temp = S_mask
        while s_temp:
            lsb = s_temp & -s_temp
            z_idx = lsb.bit_length() - 1
            new_forbidden |= BadWith[v][z_idx]
            s_temp ^= lsb
            
        new_avail = avail & ~(1 << v)
        new_S_mask = S_mask | (1 << v)
        
        if time.time() > deadline:
            return
            
        bb_solve_with_set(new_avail, new_forbidden, new_S_mask, current_size + 1, next_idx)
        
    initial_avail = (1 << N) - 1
    bb_solve_with_set(initial_avail, 0, 0, 0, 0)
    
    result = []
    for i in range(N):
        if (best_set_mask >> i) & 1:
            result.append(i + 1)
            
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
