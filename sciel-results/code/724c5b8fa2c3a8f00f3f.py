"""Rewritten program with a robust backtracking solver."""
import time
import sys
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_instance(n, edges, lists):
        # Precompute adjacency as bitmasks and lists
        adj_mask = [0] * n
        adj_list = [[] for _ in range(n)]
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            adj_list[u].append(v)
            adj_list[v].append(u)
        
        # Precompute color masks for each vertex
        # Color c is bit c in the mask
        color_masks = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            color_masks[v] = mask

        # Initial domains
        domains = color_masks[:]
        
        # assigned[v] = color index or -1
        assigned = [-1] * n
        
        # Track number of unassigned vertices
        unassigned = set(range(n))
        
        def backtrack():
            if not unassigned:
                return True
            
            # MRV: Choose unassigned vertex with smallest domain
            v = None
            min_size = 5 # Max colors is 5
            for u in unassigned:
                size = bin(domains[u]).count('1')
                if size < min_size:
                    min_size = size
                    v = u
                    if size == 0:
                        return False
                    if size == 1:
                        break
            
            # Get available colors for v
            avail_mask = domains[v]
            
            # Try each color in avail_mask
            # We can iterate through bits
            m = avail_mask
            while m:
                # Get lowest set bit
                bit = m & (-m)
                c = bit.bit_length() - 1
                m ^= bit
                
                # Assign v -> c
                assigned[v] = c
                unassigned.remove(v)
                
                # Check consistency with neighbors
                valid = True
                # We need to update domains of neighbors
                # Since we don't have a full DSU implementation here for speed in Python,
                # we do a forward checking step.
                # For each neighbor u of v that is unassigned:
                #   if domains[u] has bit c set, remove it.
                #   if domains[u] becomes 0, invalid.
                
                # To avoid modifying state permanently and backtracking manually for domains,
                # we can just check if any neighbor is ALREADY assigned to c.
                # If so, invalid.
                # If we don't do forward checking, we might explore deep branches that fail later.
                # Given the small size, simple checking might be enough, but let's add basic FC.
                
                # Actually, for this specific problem size, a simple check against assigned neighbors is often sufficient 
                # combined with MRV, but let's try a slightly stronger approach:
                # We will just check assigned neighbors for now. If we hit a dead end, we backtrack.
                
                neighbor_conflict = False
                for u in adj_list[v]:
                    if assigned[u] == c:
                        neighbor_conflict = True
                        break
                
                if neighbor_conflict:
                    # Undo
                    assigned[v] = -1
                    unassigned.add(v)
                    continue
                
                # Recurse
                if backtrack():
                    return True
                
                # Undo
                assigned[v] = -1
                unassigned.add(v)
                
            return False

        # The simple backtracker above might be too slow without FC or DSU.
        # Let's implement a slightly more optimized backtracker with Forward Checking.
        # We need to copy domains or use a stack.
        # Given N <= 408, copying domains (list of ints) is fast.
        
        def backtrack_fc():
            if not unassigned:
                return True
            
            # MRV
            v = None
            min_size = 5
            for u in unassigned:
                size = bin(domains[u]).count('1')
                if size < min_size:
                    min_size = size
                    v = u
                    if size == 0:
                        return False
                    if size == 1:
                        break
            
            avail_mask = domains[v]
            # Order values by least constraining? 
            # For now, just iterate.
            
            # Save state for backtracking
            # We need to restore domains of neighbors.
            # Let's store the old domains of affected neighbors.
            affected = []
            for u in adj_list[v]:
                if u in unassigned:
                    affected.append((u, domains[u]))
            
            m = avail_mask
            while m:
                bit = m & (-m)
                c = bit.bit_length() - 1
                m ^= bit
                
                # Assign
                assigned[v] = c
                unassigned.remove(v)
                
                # Forward Check
                valid = True
                for u, old_dom in affected:
                    if domains[u] & bit:
                        domains[u] ^= bit # Remove color c
                        if domains[u] == 0:
                            valid = False
                            break
                
                if valid:
                    if backtrack_fc():
                        return True
                
                # Undo
                assigned[v] = -1
                unassigned.add(v)
                for u, old_dom in affected:
                    domains[u] = old_dom
                    
            return False

        # The recursive FC might hit recursion limit or be slow in Python due to overhead.
        # Let's try an iterative version or just rely on the fact that these graphs are easy.
        # However, Python recursion limit is usually 1000. Depth can be 408. This is safe.
        
        # Let's optimize the MRV selection slightly by precomputing sizes? No, domains change.
        
        # One optimization: If a variable has 1 value, assign it immediately (Unit Propagation).
        # This can be done in a loop before MRV.
        
        def propagate():
            # Loop until no change
            changed = True
            while changed:
                changed = False
                # Find any unassigned var with domain size 1
                # This is O(N^2) worst case, but N is small.
                # We can just iterate.
                for u in list(unassigned):
                    if bin(domains[u]).count('1') == 1:
                        c = domains[u].bit_length() - 1
                        # Assign u -> c
                        assigned[u] = c
                        unassigned.remove(u)
                        changed = True
                        # Update neighbors
                        for w in adj_list[u]:
                            if w in unassigned:
                                if domains[w] & (1 << c):
                                    domains[w] ^= (1 << c)
                                    if domains[w] == 0:
                                        return False
            return True

        # Integrate propagation into the search
        def backtrack_with_prop():
            if not propagate():
                return False
            if not unassigned:
                return True
            
            # MRV
            v = None
            min_size = 5
            for u in unassigned:
                size = bin(domains[u]).count('1')
                if size < min_size:
                    min_size = size
                    v = u
                    if size == 0:
                        return False
                    if size == 1:
                        # Should have been handled by propagate, but just in case
                        break
            
            avail_mask = domains[v]
            
            affected = []
            for u in adj_list[v]:
                if u in unassigned:
                    affected.append((u, domains[u]))
            
            m = avail_mask
            while m:
                bit = m & (-m)
                c = bit.bit_length() - 1
                m ^= bit
                
                assigned[v] = c
                unassigned.remove(v)
                
                # Apply assignment to neighbors (Forward Check)
                valid = True
                for u, old_dom in affected:
                    if domains[u] & bit:
                        domains[u] ^= bit
                        if domains[u] == 0:
                            valid = False
                            break
                
                if valid:
                    if backtrack_with_prop():
                        return True
                
                # Undo
                assigned[v] = -1
                unassigned.add(v)
                for u, old_dom in affected:
                    domains[u] = old_dom
                    
            return False

        # Increase recursion limit just in case
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(10000)
        
        success = backtrack_with_prop()
        
        sys.setrecursionlimit(old_limit)
        
        if success:
            return assigned
        else:
            # Should not happen for these instances
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # If we run out of time, we must return something valid.
            # Since we process sequentially, if we fail to solve one, we can't return a valid list for it.
            # But the problem says "An instance whose colouring breaks a list or an edge scores 0 for that instance".
            # It also says "A wrong length or type makes the whole result invalid".
            # So we MUST return a list of length n for each instance.
            # If we haven't solved it, we can return a dummy list? No, that scores 0 for that instance.
            # But if we run out of time, we might not have solved previous ones?
            # The loop processes them one by one. If we finish 3 and run out of time on the 4th,
            # we should return the 3 solved ones and a dummy for the 4th?
            # Wait, the return type is "a list of four lists".
            # If I return a list of length 3, it's invalid.
            # So I must ensure I return 4 lists.
            
            # Strategy: If time is running out, fill remaining with [0]*n?
            # This will score 0 for those instances, but keep the result valid.
            
            # However, the solver is fast. This is just a safety net.
            pass
        
        n, edges, lists = instance(splits, seed)
        coloring = solve_instance(n, edges, lists)
        out.append(coloring)
        
        # Check time
        if time.time() > deadline:
             # Fill rest with dummies to maintain structure
             for remaining_splits, remaining_seed in ((20, 701), (40, 702), (60, 703), (100, 704))[len(out):]:
                 n_rem, _, _ = instance(remaining_splits, remaining_seed)
                 out.append([0] * n_rem)
             break

    # Final check: ensure all are Python lists of Python ints
    final_out = []
    for coloring in out:
        final_out.append([int(c) for c in coloring])
        
    return final_out

# EVOLVE-BLOCK-END
