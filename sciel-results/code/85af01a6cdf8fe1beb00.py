import random
import time

def construct():
    d = 7
    n_vertices = 1 << d
    all_edges = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))

    edge_index = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)  # 448

    # Precompute for each edge, the list of C4s it belongs to.
    # Each C4 is represented by the set of 4 edge indices.
    # For fast conflict checking, we need to know for a given edge e,
    # which other edges in the current set S form a C4 with e.
    
    # edge_c4s[ei] = list of tuples (a, b, c) where {ei, a, b, c} is a C4
    edge_c4s = [[] for _ in range(num_edges)]
    for (u, v) in all_edges:
        diff = u ^ v
        i = diff.bit_length() - 1
        ei = edge_index[(u, v)]
        for j in range(d):
            if j == i:
                continue
            u_prime = u ^ (1 << j)
            v_prime = v ^ (1 << j)
            e_opp = (min(u_prime, v_prime), max(u_prime, v_prime))
            e_side1 = (min(u, u_prime), max(u, u_prime))
            e_side2 = (min(v, v_prime), max(v, v_prime))
            a = edge_index[e_opp]
            b = edge_index[e_side1]
            c = edge_index[e_side2]
            edge_c4s[ei].append((a, b, c))

    # Helper to check if adding edge ei to set S is valid
    # It is invalid if there exists a C4 (a,b,c) in edge_c4s[ei] such that a,b,c are all in S
    def is_valid_add(ei, S):
        for (a, b, c) in edge_c4s[ei]:
            if a in S and b in S and c in S:
                return False
        return True

    # Greedy construction with random order
    def greedy_construct(order):
        S = set()
        for ei in order:
            if is_valid_add(ei, S):
                S.add(ei)
        return S

    # Local Search: Try to improve S
    # Strategy: 
    # 1. Identify all edges not in S.
    # 2. For each, check if it can be added. If yes, add it.
    # 3. If no, it's blocked by some C4s. 
    #    Try to remove one edge from one of the blocking C4s.
    #    Then try to add the original edge and any others that become valid.
    #    If the net gain is positive, accept.

    def local_search(S, max_iterations=1000):
        S = set(S)
        current_size = len(S)
        no_improve_count = 0
        for _ in range(max_iterations):
            # Step 1: Try to add any valid edge
            improved = False
            for ei in range(num_edges):
                if ei not in S and is_valid_add(ei, S):
                    S.add(ei)
                    improved = True
            if improved:
                no_improve_count = 0
                continue
            else:
                no_improve_count += 1
                if no_improve_count > 5:
                    break
            
            # Step 2: Try a swap
            # Pick a random edge not in S
            candidates = [ei for ei in range(num_edges) if ei not in S]
            if not candidates:
                break
            target_ei = random.choice(candidates)
            
            # Find a blocking C4
            # target_ei is blocked if there is a C4 (a,b,c) in edge_c4s[target_ei] 
            # such that a,b,c are in S.
            blocking_c4 = None
            for (a, b, c) in edge_c4s[target_ei]:
                if a in S and b in S and c in S:
                    blocking_c4 = (a, b, c)
                    break
            
            if blocking_c4 is None:
                # Should not happen if we just checked validity, but maybe S changed?
                # Actually, if is_valid_add is false, there must be a blocking C4.
                continue

            # Try removing one edge from blocking_c4
            # We want to see if removing one allows us to add target_ei and potentially more.
            # Let's try removing each of a, b, c and see the outcome.
            best_new_S = None
            best_gain = 0
            
            for rem_ei in blocking_c4:
                new_S = set(S)
                new_S.remove(rem_ei)
                # Try to add target_ei
                if is_valid_add(target_ei, new_S):
                    new_S.add(target_ei)
                    # Try to add any other edges that are now valid
                    # We can do a quick greedy pass on the remaining unselected edges
                    # But to keep it fast, let's just check if we can add a few more.
                    # Actually, a full greedy pass on unselected edges is O(N * Cost).
                    # Let's just add target_ei and see if size increased (it didn't, it's a swap).
                    # Wait, if we remove 1 and add 1, size is same.
                    # We need to check if this swap opens up space for MORE edges.
                    
                    # Let's do a limited greedy fill
                    added_more = 0
                    # Check a subset of unselected edges for quick wins
                    # To be efficient, we only check edges that were blocked by rem_ei?
                    # That's complex. Let's just try to add any edge that is now valid.
                    # We'll iterate through a random sample or all if small enough.
                    # 448 edges is small. Let's do a full check.
                    for check_ei in range(num_edges):
                        if check_ei not in new_S and is_valid_add(check_ei, new_S):
                            new_S.add(check_ei)
                            added_more += 1
                    
                    gain = added_more # We removed 1, added 1 + added_more. Net gain = added_more.
                    if gain > best_gain:
                        best_gain = gain
                        best_new_S = new_S

            if best_gain > 0:
                S = best_new_S
                no_improve_count = 0
            else:
                # No improvement from this swap. 
                # To avoid getting stuck, we might accept a neutral move (gain 0) with some probability?
                # Or just break if too many neutral moves.
                if random.random() < 0.1:
                    # Accept a neutral move to escape local optimum
                    # We need to pick one that is valid (which they all are, gain 0 means we added 1, removed 1)
                    # Let's just pick the first one we found
                    # Re-evaluate: if best_gain == 0, we didn't update S.
                    # We need to explicitly construct a neutral move if we want to wander.
                    # Let's just break to avoid wasting time if no gain is found in a few tries.
                    pass
        return S

    best_S = set()
    best_size = 0
    
    num_trials = 200
    time_limit = 110.0
    start_time = time.time()
    
    for trial in range(num_trials):
        if time.time() - start_time > time_limit:
            break
        
        # Random order
        order = list(range(num_edges))
        random.shuffle(order)
        
        # Greedy
        S = greedy_construct(order)
        
        # Local Search
        S = local_search(S, max_iterations=500)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
            # print(f"Trial {trial}: Size {best_size}")
            if best_size >= 308:
                break

    return [[all_edges[i][0], all_edges[i][1]] for i in best_S]
