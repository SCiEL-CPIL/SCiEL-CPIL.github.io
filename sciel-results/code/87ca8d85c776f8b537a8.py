import time
import random
from itertools import combinations

START_TIME = time.time()
DEADLINE = START_TIME + 145.0

def construct():
    import time
    import random
    from itertools import combinations

    start_time = time.time()
    deadline = start_time + 145.0

    # We will try n=21, then n=20, etc.
    # For each n, we run a local search to minimize violations.
    
    for n in [21, 20, 19, 18]:
        if time.time() > deadline - 20:
            break
            
        # Precompute subsets of size >= 6
        # This is done once per n
        subsets = []
        for size in range(6, n + 1):
            for subset in combinations(range(n), size):
                subsets.append(subset)
        
        num_subsets = len(subsets)
        
        # Initialize random graph
        # adj is a list of integers (bitsets)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Precompute masks for each subset?
        # mask[i] is the bitset of vertices in subset i
        masks = [0] * num_subsets
        for i, subset in enumerate(subsets):
            m = 0
            for v in subset:
                m |= (1 << v)
            masks[i] = m
            
        # Function to count violations
        def count_violations(adj):
            count = 0
            for i, subset in enumerate(subsets):
                mask = masks[i]
                # Check if regular
                # Get degrees
                # Optimization: If we just need to check equality, we can compare
                # degrees.
                k = len(subset)
                if k == 0: continue
                first_deg = (adj[subset[0]] & mask).bit_count()
                is_reg = True
                for u in subset[1:]:
                    deg = (adj[u] & mask).bit_count()
                    if deg != first_deg:
                        is_reg = False
                        break
                if is_reg:
                    count += 1
            return count

        # Initial count
        current_violations = count_violations(adj)
        
        # Local Search
        # We want to reach 0 violations.
        # If we reach 0, we can return this graph.
        
        # To speed up, we can track which subsets are currently violating.
        # But updating that set is complex.
        # Let's just re-evaluate affected subsets.
        
        # Affected subsets for edge (u, v):
        # Any subset containing u or v.
        # We can precompute for each vertex u, the list of subset indices containing u.
        # subsets_containing = [set() for _ in range(n)]
        # for i, subset in enumerate(subsets):
        #     for u in subset:
        #         subsets_containing[u].add(i)
        
        # This uses memory but speeds up lookup.
        # For n=20, num_subsets ~ 100k.
        # Each subset has size ~ 10.
        # Total entries ~ 1M.
        # Sets of ints. Fine.
        
        subsets_containing = [set() for _ in range(n)]
        for i, subset in enumerate(subsets):
            for u in subset:
                subsets_containing[u].add(i)

        best_violations = current_violations
        best_adj = adj[:]
        
        temperature = 1.0
        cooling_rate = 0.99
        
        while time.time() < deadline - 5:
            if current_violations == 0:
                break
                
            # Pick random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Determine affected subsets
            affected_indices = subsets_containing[u].union(subsets_containing[v])
            
            # Calculate new violations for affected subsets
            # We need to know the OLD status of these subsets to calculate delta.
            # This requires storing status for all subsets.
            # status[i] = True if regular.
            
            # Let's maintain a status array.
            # This is a structural change in the loop, but necessary for efficiency.
            
            # Initialize status
            if not hasattr(construct, '_status_initialized'):
                # We are in a fresh run for this n
                pass
            
            # Actually, let's just re-calculate the delta by checking old vs new.
            # Old status: We need to know if it was regular BEFORE the flip.
            # New status: Check AFTER flip.
            
            # To avoid storing status, we can just count violations in affected sets
            # before and after?
            # No, we need global count.
            # Let's maintain a global `violation_status` array of booleans.
            
            # Re-initializing status for each restart is expensive.
            # Let's do it once per n.
            
            # This logic is getting complex for a single function.
            # Let's simplify:
            # Just use a simple hill climbing that re-evaluates the WHOLE graph
            # every K steps?
            # No, that's too slow.
            
            # Alternative:
            # Use the `count_violations` function but only on affected subsets?
            # We can't just sum affected because non-affected don't change.
            # So:
            # old_affected_violations = sum(1 for i in affected if status[i])
            # Flip edge
            # new_affected_violations = sum(1 for i in affected if is_regular(i, new_adj))
            # delta = new - old
            # current_violations += delta
            # Update status[i] for i in affected
            
            # We need `status` array.
            # Let's create it.
            
            # Since we are in a loop for n, we can define this inside.
            
            # But wait, I'm editing the existing function.
            # I should replace the whole construct function.
            
            pass

    # Fallback
    return {'n': 5, 'edges': []}
