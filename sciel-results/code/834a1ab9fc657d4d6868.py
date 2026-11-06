import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin from 90s limit
    
    # 1. Preprocessing: Identify valid vertices (2a is not a perfect square)
    def is_square(n):
        r = int(math.isqrt(n))
        return r * r == n

    V = list(range(1, 257))
    V_valid = [a for a in V if not is_square(2 * a)]
    
    # Map values to indices for faster adjacency list handling if needed, 
    # but with N=256, direct value-based checks are fine.
    
    # 2. Build Conflict Graph
    # adj[u] = set of v in V_valid such that u+v is a square
    adj = {v: set() for v in V_valid}
    squares = [i*i for i in range(2, 23)] # 4 to 512
    
    for u in V_valid:
        for s in squares:
            v = s - u
            if v in adj and v != u: # v must be in V_valid and distinct
                adj[u].add(v)
                # adj is symmetric, but we only need to build it once.
                # Since we iterate all u, we will add both directions naturally if we were careful, 
                # but here we just check membership in adj (which is V_valid).
                # Note: if v is not in V_valid, it's not in adj, so it won't be added.
                # We should ensure we don't add self-loops, but 2u is square is already filtered out.
                # So u != v is guaranteed if 2u is not square? 
                # If u+v is square and u=v, then 2u is square. We filtered those out.
                # So u != v is always true for valid pairs.
    
    # 3. Greedy Initialization
    # Sort by degree (number of conflicts)
    degrees = {v: len(adj[v]) for v in V_valid}
    sorted_V = sorted(V_valid, key=lambda x: degrees[x])
    
    S = set()
    R = set(V_valid)
    # conflict_count[v] = number of neighbors of v in S
    conflict_count = {v: 0 for v in V_valid}
    
    for v in sorted_V:
        if v in R:
            if conflict_count[v] == 0:
                S.add(v)
                R.remove(v)
                # Update conflict counts for neighbors
                for u in adj[v]:
                    if u in R:
                        conflict_count[u] += 1
    
    best_size = len(S)
    best_S = set(S)
    
    # 4. Local Search
    # We maintain S, R, conflict_count
    # We will use a loop that tries to improve.
    
    def add_vertex(v):
        S.add(v)
        R.remove(v)
        for u in adj[v]:
            if u in R:
                conflict_count[u] += 1

    def remove_vertex(v):
        S.remove(v)
        R.add(v)
        for u in adj[v]:
            if u in R:
                conflict_count[u] -= 1

    def try_add():
        # Find all v in R with conflict_count 0
        candidates = [v for v in R if conflict_count[v] == 0]
        if candidates:
            # Pick one with min degree in R? Or just first?
            # Min degree in current R might be better for future
            # degree in R = number of neighbors in R
            # We don't maintain degree in R dynamically easily, but we can approximate
            # or just pick random to avoid bias
            v = random.choice(candidates)
            add_vertex(v)
            return True
        return False

    def try_swap():
        # Find v in R with conflict_count 1
        # Let u be the unique neighbor in S
        # Then we can swap u for v
        candidates = []
        for v in R:
            if conflict_count[v] == 1:
                # Find the neighbor in S
                for u in adj[v]:
                    if u in S:
                        candidates.append((u, v))
                        break
        if candidates:
            # Pick a swap
            # Heuristic: prefer v with lower degree?
            u, v = random.choice(candidates)
            remove_vertex(u)
            add_vertex(v)
            return True
        return False

    while time.time() < deadline:
        # Phase 1: Greedy Add
        added = True
        while added:
            if time.time() > deadline:
                break
            added = try_add()
        
        if time.time() > deadline:
            break
            
        # Update best
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
        # Phase 2: Swap
        swapped = True
        while swapped:
            if time.time() > deadline:
                break
            swapped = try_swap()
            
        if time.time() > deadline:
            break
            
        # Update best
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
        # Phase 3: Perturb if stuck
        # If we couldn't add or swap, we are in a local optimum.
        # Remove a few random vertices to escape.
        # How many? Maybe 5-10.
        if len(S) > 0:
            # Pick 5 random vertices to remove
            num_removals = min(5, len(S))
            to_remove = random.sample(list(S), num_removals)
            for v in to_remove:
                remove_vertex(v)
            # Reset conflict counts for R? No, remove_vertex updates them.
            # But we need to ensure conflict_count for R is correct.
            # remove_vertex decrements counts for neighbors in R.
            # The removed vertices are now in R. Their counts were 0 in S, but now they are in R.
            # We don't track conflict_count for S vertices.
            # When a vertex is in R, its conflict_count is the number of its neighbors in S.
            # When we remove v from S, v goes to R.
            # Its conflict_count should be the number of its neighbors in the NEW S.
            # Our remove_vertex function decrements neighbors in R.
            # It does NOT update the conflict_count of v itself (since v was in S, its count wasn't tracked/used).
            # So we need to calculate conflict_count[v] for the newly added v (which is in R now).
            # Actually, simpler: just recalculate conflict_count for all R?
            # N is small (256). Recalculating is O(N * deg).
            # Let's just recalculate to be safe after perturbation.
            for v in R:
                count = 0
                for u in adj[v]:
                    if u in S:
                        count += 1
                conflict_count[v] = count
            
            # Update best
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)

    return list(best_S)
