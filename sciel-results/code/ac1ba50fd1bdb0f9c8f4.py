import time
import sys
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin for 60s limit
    
    N = 256
    
    def lcm(a, b):
        return a // gcd(a, b) * b

    # Precompute conflict graph for all pairs in [2, 256]
    # adj[i] is a set of j such that lcm(i, j) <= 256
    # We use 1-based indexing for values, so index 2 corresponds to value 2.
    # Actually, let's just use the value itself as the index for simplicity in the global graph,
    # but for the subgraph solver, we map to local indices.
    
    # Global adjacency: adj_val[v] = set of u such that lcm(u, v) <= 256
    adj_val = [set() for _ in range(N + 1)]
    for a in range(2, N + 1):
        for b in range(a + 1, N + 1):
            if lcm(a, b) <= N:
                adj_val[a].add(b)
                adj_val[b].add(a)
                
    # Helper to solve MWIS on a set of values V
    def solve_mwis(values):
        if not values:
            return 0.0, []
        
        # Sort values to have a consistent order
        verts = sorted(values)
        n = len(verts)
        
        # Map value -> local index
        val_to_idx = {v: i for i, v in enumerate(verts)}
        
        # Precompute local adjacency
        # adj[i] = set of local indices j that conflict with i
        adj = [set() for _ in range(n)]
        for i in range(n):
            vi = verts[i]
            for j in range(i + 1, n):
                vj = verts[j]
                if vj in adj_val[vi]:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Weights
        weights = [1.0 / v for v in verts]
        
        # Greedy initialization
        # Sort by weight descending (which is value ascending)
        # But we want to pick high weight (small value) first?
        # Greedy: pick smallest value that doesn't conflict with already picked.
        # This is equivalent to sorting by weight descending.
        
        chosen_indices = []
        available = set(range(n))
        
        # Order by weight descending (value ascending)
        order = sorted(range(n), key=lambda i: verts[i])
        
        for i in order:
            if i in available:
                chosen_indices.append(i)
                available.remove(i)
                for j in adj[i]:
                    if j in available:
                        available.remove(j)
                        
        best_sum = sum(weights[i] for i in chosen_indices)
        best_set = set(chosen_indices)
        
        # Local Search: 1-swap and 1-add
        # We keep the set of chosen indices.
        # current_set is a set of local indices.
        current_set = set(chosen_indices)
        current_sum = best_sum
        
        # Precompute conflict check for speed?
        # adj[i] is a set.
        
        # We will iterate until no improvement or time limit.
        # To avoid getting stuck in a local minimum, we might need more complex moves,
        # but 1-swap is a good start.
        
        # Optimization: Precompute for each node i, the set of nodes it conflicts with.
        # When checking if j can be added, we check if any neighbor of j is in current_set.
        
        # To speed up "can add j", we can maintain a count of how many neighbors of j are in current_set.
        # Let's try a simpler approach first: just check adj[j] & current_set.
        
        while time.time() < deadline - 5.0:
            improved = False
            
            # 1. Try adding nodes
            # Iterate through all nodes not in current_set
            # If no conflict, add it.
            # Since weights are positive, adding is always beneficial if possible.
            # We should do this in a way that doesn't miss opportunities?
            # Actually, if we can add a node, we should.
            # But adding one might block another.
            # Greedy addition order matters.
            # Let's try adding in order of weight descending (value ascending).
            
            for i in sorted(available, key=lambda x: verts[x]):
                if i in current_set:
                    continue
                # Check conflict
                if not (adj[i] & current_set):
                    current_set.add(i)
                    current_sum += weights[i]
                    improved = True
            
            if not improved:
                # 2. Try 1-swap
                # For each i in current_set, for each j not in current_set:
                # If removing i allows adding j (i.e., j conflicts with current_set \ {i} only if it conflicts with i? No.)
                # j can be added if (adj[j] & (current_set - {i})) is empty.
                # This is equivalent to: all neighbors of j in current_set must be i.
                # i.e., adj[j] & current_set == {i} or empty.
                # And we require weights[j] > weights[i].
                
                # Optimization:
                # For each j not in current_set, find the set of neighbors in current_set: C_j = adj[j] & current_set.
                # If C_j is empty, we would have added it in step 1.
                # So C_j is non-empty.
                # We can swap if there exists i in C_j such that weights[j] > weights[i].
                # And we want the best improvement: max(weights[j] - weights[i]) for i in C_j.
                # If max improvement > 0, we do the swap.
                
                # We can precompute C_j for all j not in current_set?
                # That's O(N^2) per iteration. N=240, N^2=57600. Fast enough.
                
                best_swap_gain = 0
                best_swap_i = -1
                best_swap_j = -1
                
                for j in available:
                    if j in current_set:
                        continue
                    C_j = adj[j] & current_set
                    if not C_j:
                        # Should have been added
                        continue
                    
                    # Find i in C_j with min weight
                    # Since we want to maximize weights[j] - weights[i], we want min weights[i] in C_j
                    min_weight_i = min(weights[i] for i in C_j)
                    # Which i had that weight?
                    # There might be multiple. Pick any.
                    # Actually, we just need the gain.
                    gain = weights[j] - min_weight_i
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_j = j
                        # Find the i
                        for i in C_j:
                            if weights[i] == min_weight_i:
                                best_swap_i = i
                                break
                
                if best_swap_gain > 0:
                    current_set.remove(best_swap_i)
                    current_set.add(best_swap_j)
                    current_sum += best_swap_gain
                    improved = True
            
            if not improved:
                break
                
        return current_sum, [verts[i] for i in current_set]

    # Strategy:
    # The problem asks for a subset of [2, 256].
    # We can decompose by "small" numbers.
    # If we include a small number x, it constrains others.
    # If we don't include any small number, we are free in [17, 256] (roughly).
    
    # Let's define "small" as 2..16.
    # Case 0: No numbers from 2..16.
    # Vertices: 17..256.
    V0 = set(range(17, 257))
    sum0, set0 = solve_mwis(V0)
    best_score = sum0
    best_candidate = set0
    
    # Case x: Include exactly one small number x?
    # Or include a set of small numbers?
    # If we include x and y (both small), they might conflict.
    # lcm(x, y) <= 256 is likely for small x, y.
    # e.g., lcm(2, 3) = 6 <= 256. Conflict.
    # lcm(2, 5) = 10 <= 256. Conflict.
    # Most small numbers conflict with each other.
    # So we likely can only pick at most 1 or 2 small numbers.
    
    # Let's try picking exactly one small number x from 2..16.
    # The rest must be from Vx = {y in [2, 256] \ {x} | lcm(x, y) > 256}.
    
    for x in range(2, 17):
        Vx = []
        for y in range(2, 257):
            if y == x:
                continue
            if lcm(x, y) > 256:
                Vx.append(y)
        
        if not Vx:
            score = 1.0 / x
            set_x = [x]
        else:
            sum_x, set_x = solve_mwis(Vx)
            score = 1.0 / x + sum_x
            set_x.append(x)
            
        if score > best_score:
            best_score = score
            best_candidate = set_x
            
        if time.time() > deadline - 5.0:
            break
            
    # What if we pick 2 small numbers?
    # e.g., x, y.
    # They must not conflict: lcm(x, y) > 256.
    # For x, y in 2..16, lcm(x, y) is small.
    # Max lcm(15, 16) = 240 <= 256.
    # So NO two numbers from 2..16 can be in the set together!
    # Because lcm(x, y) <= 256 for all x, y in 2..16.
    # Proof: Max product 15*16=240. LCM <= Product.
    # So at most one number from 2..16 can be selected.
    
    # Thus, the cases "No small" and "Exactly one small x" cover all possibilities.
    
    return best_candidate
