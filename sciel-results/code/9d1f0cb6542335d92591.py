import time
import random

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    def is_valid(edge_set, n):
        """Check if no simple three-edge path has endpoints of equal degree."""
        if not edge_set:
            return True
            
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        # Optimization: Group vertices by degree
        # For any path s-u-v-t, if deg[s] == deg[t], it's invalid.
        # This means: For any edge (u, v), the set of neighbors of u (excluding v)
        # and the set of neighbors of v (excluding u) must not share any degree class
        # in a way that allows s in N(u)\{v} and t in N(v)\{u} with deg[s]=deg[t].
        # Actually, simpler:
        # For each edge (u, v):
        #   Let D_u = {deg[s] for s in adj[u] if s != v}
        #   Let D_v = {deg[t] for t in adj[v] if t != u}
        #   If D_u intersects D_v, there exist s, t with deg[s]=deg[t].
        #   Wait, we also need s != t.
        #   If there is a degree value d in the intersection, we need to ensure there aren't
        #   s in N(u)\{v} with deg[s]=d and t in N(v)\{u} with deg[t]=d such that s != t.
        #   If s == t, then s is a common neighbor of u and v.
        #   Path s-u-v-s is not a simple path (repeats s). The problem says "simple three-edge path".
        #   A simple path cannot repeat vertices. So s-u-v-s is NOT a simple path.
        #   So we only care about s != t.
        
        # Let's check properly.
        for (u, v) in edge_set:
            # Get neighbors of u excluding v
            nu = [s for s in adj[u] if s != v]
            # Get neighbors of v excluding u
            nv = [t for t in adj[v] if t != u]
            
            # We need to find if there exists s in nu, t in nv such that s != t and deg[s] == deg[t]
            # We can use a set of (deg, vertex) or just check.
            # Since n is small, O(deg(u)*deg(v)) is fine.
            # But we can optimize slightly.
            deg_map = {}
            for t in nv:
                d = deg[t]
                if d not in deg_map:
                    deg_map[d] = []
                deg_map[d].append(t)
            
            found_invalid = False
            for s in nu:
                ds = deg[s]
                if ds in deg_map:
                    # There is some t in nv with deg[t] == deg[s]
                    # We need to check if there is a t != s
                    # If len(deg_map[ds]) > 1, definitely exists t != s (since s is not in nv, unless s==t which is checked by s!=t condition? No, s is in nu, t is in nv. s and t are distinct sets? No, they can overlap).
                    # If s is in nv, then s is a common neighbor.
                    # If s is in nv, and deg_map[ds] contains s, we need to see if there is ANOTHER t in deg_map[ds].
                    # If s is NOT in nv, then any t in deg_map[ds] is valid (t != s is automatic).
                    
                    if len(deg_map[ds]) > 1:
                        found_invalid = True
                        break
                    elif len(deg_map[ds]) == 1:
                        t = deg_map[ds][0]
                        if t != s:
                            found_invalid = True
                            break
                        # else t == s, invalid
            if found_invalid:
                return False
        
        return True

    # Known valid solution: K_{7,8}
    # Vertices 0-6 in A, 7-14 in B
    A = list(range(7))
    B = list(range(7, 15))
    base_edges = [(a, b) for a in A for b in B]
    base_set = set(tuple(sorted(e)) for e in base_edges)
    best_set = base_set
    best_count = len(best_set)
    
    if not is_valid(best_set, n):
        # This should not happen, but fallback to empty or something? 
        # K7,8 is valid.
        pass

    # Candidate edges to add: edges within A and within B
    candidates = []
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            candidates.append((A[i], A[j]))
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            candidates.append((B[i], B[j]))
    
    # Shuffle for randomization in local search
    random.shuffle(candidates)
    
    # Greedy construction
    current_set = set(base_set)
    current_edges_list = list(base_set) # Not needed, we use set
    
    for e in candidates:
        if time.time() > deadline:
            break
        if len(current_set) + 1 <= best_count:
            # Even if we add this, we can't beat best yet? 
            # No, we might add more later. But if current + 1 <= best, we might still want to add it if it leads to more.
            # However, if we are strictly increasing, we should check if it's valid.
            pass
        
        test_set = set(current_set)
        test_set.add(tuple(sorted(e)))
        
        if is_valid(test_set, n):
            current_set = test_set
    
    if len(current_set) > best_count:
        best_set = current_set
        best_count = len(current_set)
    
    # Local Search: Hill Climbing
    # Try to add edges that are not in current_set
    not_in_current = [e for e in candidates if tuple(sorted(e)) not in current_set]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(not_in_current)
        for e in not_in_current:
            if time.time() > deadline:
                break
            if len(current_set) + 1 <= best_count:
                continue # Can't improve best, but might allow further improvements?
                # Usually greedy hill climbing just looks for immediate improvement.
            
            test_set = set(current_set)
            test_set.add(tuple(sorted(e)))
            
            if is_valid(test_set, n):
                current_set = test_set
                not_in_current.remove(e)
                improved = True
                if len(current_set) > best_count:
                    best_set = current_set
                    best_count = len(current_set)
                break # Restart scan after improvement
        
        # If no single edge addition works, try removals?
        # The problem asks to maximize edge count.
        # If we are stuck, maybe we should try a different random restart?
        # Given the time limit, let's try a few restarts with different shuffles.
        
    # Let's do a few full restarts of the greedy process with different shuffles
    # to escape local optima.
    for _ in range(10):
        if time.time() > deadline:
            break
        
        random.shuffle(candidates)
        current_set = set(base_set)
        
        for e in candidates:
            if time.time() > deadline:
                break
            test_set = set(current_set)
            test_set.add(tuple(sorted(e)))
            if is_valid(test_set, n):
                current_set = test_set
        
        if len(current_set) > best_count:
            best_set = current_set
            best_count = len(current_set)
            
            # Local search on this new best
            not_in_current = [e for e in candidates if tuple(sorted(e)) not in current_set]
            improved = True
            while improved and time.time() < deadline:
                improved = False
                random.shuffle(not_in_current)
                for e in not_in_current:
                    if time.time() > deadline:
                        break
                    test_set = set(current_set)
                    test_set.add(tuple(sorted(e)))
                    if is_valid(test_set, n):
                        current_set = test_set
                        not_in_current.remove(e)
                        improved = True
                        best_set = current_set
                        best_count = len(current_set)
                        break

    # Final validation and formatting
    if not is_valid(best_set, n):
        # Fallback to base if somehow best became invalid (shouldn't happen)
        best_set = base_set
        best_count = len(base_set)

    result = [list(e) for e in best_set]
    result.sort()
    for i in range(len(result)):
        result[i][0] = int(result[i][0])
        result[i][1] = int(result[i][1])
    
    return result
