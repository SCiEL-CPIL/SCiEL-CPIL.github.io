"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 135  # Safety margin before 150s limit
    d = 7
    n = 2 ** d
    
    # Precompute all edges
    all_edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    m = len(all_edges)  # 448 edges
    
    # Precompute for each edge, the other 3 edges in each 4-face it belongs to
    edge_faces = [[] for _ in range(m)]
    
    for idx, (x, y) in enumerate(all_edges):
        diff = x ^ y
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            c = x ^ (1 << j)
            e = y ^ (1 << j)
            e1 = (min(x, c), max(x, c))
            e2 = (min(y, e), max(y, e))
            e3 = (min(c, e), max(c, e))
            idx1 = edge_to_idx.get(e1)
            idx2 = edge_to_idx.get(e2)
            idx3 = edge_to_idx.get(e3)
            if idx1 is not None and idx2 is not None and idx3 is not None:
                edge_faces[idx].append((idx1, idx2, idx3))
    
    # For each edge, precompute the set of edges that would be "blocked" if this edge were added
    # An edge idx is blocked by face (a,b,c) if a, b, c are all chosen
    # We need to track: for each edge, which faces could block it
    
    def greedy_init(order=None):
        """Greedy initialization with optional custom edge order."""
        chosen = [False] * m
        count = 0
        edges_to_try = order if order is not None else range(m)
        for idx in edges_to_try:
            ok = True
            for (idx1, idx2, idx3) in edge_faces[idx]:
                if chosen[idx1] and chosen[idx2] and chosen[idx3]:
                    ok = False
                    break
            if ok:
                chosen[idx] = True
                count += 1
        return chosen, count
    
    def local_search(chosen, count, time_limit=10.0):
        """Local search with swap moves."""
        t_start = time.time()
        best_count = count
        best_chosen = chosen.copy()
        
        # Precompute: for each edge, list of (face_idx, conflicting_edges) where face blocks this edge
        # When edge idx is not chosen, it's blocked if any face has all 3 other edges chosen
        
        while time.time() - t_start < time_limit:
            # Try to add a random unchosen edge
            # First, find all unchosen edges
            unchosen = [i for i in range(m) if not chosen[i]]
            if not unchosen:
                break
            
            # Pick a random unchosen edge
            idx = random.choice(unchosen)
            
            # Check if it can be added
            blocking_faces = []
            for fi, (idx1, idx2, idx3) in enumerate(edge_faces[idx]):
                if chosen[idx1] and chosen[idx2] and chosen[idx3]:
                    blocking_faces.append((fi, (idx1, idx2, idx3)))
            
            if not blocking_faces:
                # Can add directly
                chosen[idx] = True
                count += 1
                if count > best_count:
                    best_count = count
                    best_chosen = chosen.copy()
            else:
                # Try swap: remove one conflicting edge from a blocking face, add idx
                # Pick a random blocking face and a random edge from it to remove
                fi, (idx1, idx2, idx3) = random.choice(blocking_faces)
                # Choose which of the 3 to remove - prefer one that's in fewer other blocking situations
                candidates = [idx1, idx2, idx3]
                rand_idx = random.choice(candidates)
                
                # Remove rand_idx and add idx
                chosen[rand_idx] = False
                chosen[idx] = True
                # Count stays the same (swap)
                
                # Now try to add more edges that were previously blocked
                # Do a quick pass to add any edges that can now be added
                for trial in range(50):
                    if time.time() - t_start > time_limit:
                        break
                    trial_unchosen = [i for i in range(m) if not chosen[i]]
                    if not trial_unchosen:
                        break
                    trial_idx = random.choice(trial_unchosen)
                    trial_ok = True
                    for (a, b, c) in edge_faces[trial_idx]:
                        if chosen[a] and chosen[b] and chosen[c]:
                            trial_ok = False
                            break
                    if trial_ok:
                        chosen[trial_idx] = True
                        count += 1
                        if count > best_count:
                            best_count = count
                            best_chosen = chosen.copy()
                        break
                else:
                    # No improvement, consider reverting if count decreased
                    pass
        
        return best_chosen, best_count
    
    # Multi-start with different initializations
    best_overall = 0
    best_overall_chosen = None
    
    # Start 1: Natural order greedy
    chosen1, count1 = greedy_init()
    best_overall = max(best_overall, count1)
    if count1 > best_overall or best_overall_chosen is None:
        best_overall = count1
        best_overall_chosen = chosen1.copy()
    
    # Run local search from start 1
    if time.time() < deadline - 30:
        chosen_ls, count_ls = local_search(chosen1.copy(), count1, time_limit=min(20.0, deadline - time.time() - 10))
        if count_ls > best_overall:
            best_overall = count_ls
            best_overall_chosen = chosen_ls.copy()
    
    # Start 2: Reverse order greedy
    order2 = list(range(m))
    random.shuffle(order2)
    chosen2, count2 = greedy_init(order2)
    if count2 > best_overall:
        best_overall = count2
        best_overall_chosen = chosen2.copy()
    
    # Run local search from start 2
    if time.time() < deadline - 20:
        remaining = min(20.0, deadline - time.time() - 10)
        if remaining > 5:
            chosen_ls2, count_ls2 = local_search(chosen2.copy(), count2, time_limit=remaining)
            if count_ls2 > best_overall:
                best_overall = count_ls2
                best_overall_chosen = chosen_ls2.copy()
    
    # Start 3: Direction-ordered greedy (group by direction)
    order3 = []
    for i in range(d):
        for idx in range(m):
            x, y = all_edges[idx]
            if (x ^ y).bit_length() - 1 == i:
                order3.append(idx)
    chosen3, count3 = greedy_init(order3)
    if count3 > best_overall:
        best_overall = count3
        best_overall_chosen = chosen3.copy()
    
    # Run local search from start 3
    if time.time() < deadline - 10:
        remaining = min(15.0, deadline - time.time() - 5)
        if remaining > 5:
            chosen_ls3, count_ls3 = local_search(chosen3.copy(), count3, time_limit=remaining)
            if count_ls3 > best_overall:
                best_overall = count_ls3
                best_overall_chosen = chosen_ls3.copy()
    
    # Final polish: try to add any remaining edges
    if best_overall_chosen is not None and time.time() < deadline - 5:
        final_chosen = best_overall_chosen.copy()
        final_count = best_overall
        for idx in range(m):
            if not final_chosen[idx]:
                ok = True
                for (a, b, c) in edge_faces[idx]:
                    if final_chosen[a] and final_chosen[b] and final_chosen[c]:
                        ok = False
                        break
                if ok:
                    final_chosen[idx] = True
                    final_count += 1
        if final_count > best_overall:
            best_overall = final_count
            best_overall_chosen = final_chosen
    
    # Convert to output format
    out = []
    for idx in range(m):
        if best_overall_chosen[idx]:
            x, y = all_edges[idx]
            out.append([x, y])
    
    return out
# EVOLVE-BLOCK-END
