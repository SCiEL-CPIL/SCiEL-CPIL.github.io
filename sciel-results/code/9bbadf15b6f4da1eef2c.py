import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    d = 7
    n = 2 ** d  # 128 vertices
    
    # Precompute all edges and their "opposite" edges for each face
    # Edge is (u, v) with u < v
    all_edges = []
    edge_index = {}
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(all_edges)
                all_edges.append((u, v))
                edge_index[(u, v)] = idx
    
    m = len(all_edges)  # 448
    
    # For each edge, precompute the set of "conflicting" edges that would complete a C4
    # An edge (u,v) along dimension i conflicts with edges that, together with two others, form a C4.
    # Actually, for adding edge e=(u,v), it conflicts if there exists a face where the other 3 edges are present.
    # Let's precompute for each edge, the list of "partner" edges that share a face.
    # Edge (u,v) is along dim i. The face is {u, v, u^bit_j, v^bit_j} for j != i.
    # The other 3 edges in that face are: (u, u^bit_j), (v, v^bit_j), (u^bit_j, v^bit_j)
    # So edge e conflicts with the "triplet" of these 3 edges.
    
    # For efficient checking: for each edge e, store the list of (j, e1, e2, e3) where e1,e2,e3 are the other edges in face j
    # Actually, we need: if we add e, and e1,e2,e3 are all present, then we have a C4.
    # So e is "blocked" if for any face containing e, the other 3 edges are all in our set.
    
    # Precompute: for each edge index, list of (e1_idx, e2_idx, e3_idx) for each face
    edge_faces = [[] for _ in range(m)]
    for idx, (u, v) in enumerate(all_edges):
        # Determine dimension
        diff = u ^ v
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            bit_j = 1 << j
            u2 = u ^ bit_j
            v2 = v ^ bit_j
            # The other 3 edges: (u, u2), (v, v2), (u2, v2)
            e1 = (min(u, u2), max(u, u2))
            e2 = (min(v, v2), max(v, v2))
            e3 = (min(u2, v2), max(u2, v2))
            e1_idx = edge_index[e1]
            e2_idx = edge_index[e2]
            e3_idx = edge_index[e3]
            edge_faces[idx].append((e1_idx, e2_idx, e3_idx))
    
    def is_blocked(chosen_set, edge_idx):
        """Check if adding edge_idx would create a C4 given current chosen_set."""
        for (e1, e2, e3) in edge_faces[edge_idx]:
            if e1 in chosen_set and e2 in chosen_set and e3 in chosen_set:
                return True
        return False
    
    def greedy_randomized():
        """Greedy with random edge ordering."""
        chosen = set()
        order = list(range(m))
        random.shuffle(order)
        for idx in order:
            if not is_blocked(chosen, idx):
                chosen.add(idx)
        return chosen
    
    def local_search(chosen):
        """Try to add edges; if blocked, try removing one conflicting edge and re-adding."""
        improved = True
        while improved:
            improved = False
            # Find all missing edges
            missing = [i for i in range(m) if i not in chosen]
            random.shuffle(missing)
            for idx in missing:
                if time.time() > deadline - 2.0:
                    return chosen
                if is_blocked(chosen, idx):
                    # Find a face where the other 3 are present
                    for (e1, e2, e3) in edge_faces[idx]:
                        if e1 in chosen and e2 in chosen and e3 in chosen:
                            # Try removing one of e1, e2, e3 and adding idx
                            # This is a 1-for-1 swap, net change 0, so we need to check if it enables further additions
                            # Actually, let's try: remove e1, add idx. Then see if we can add e1 back or others.
                            # Simpler: just try all 3 removals, pick one that doesn't immediately block
                            for rem in [e1, e2, e3]:
                                new_chosen = chosen - {rem} | {idx}
                                # Check if this is valid (it should be, since we removed one of the blockers)
                                # But we need to ensure no OTHER C4 is formed. Since we only removed one edge and added one,
                                # the only new C4 possible is one involving idx. But we already checked idx's faces:
                                # the face that blocked idx had e1,e2,e3. We removed one of them, so that face is broken.
                                # Could idx form a C4 in another face? Only if that face's other 3 edges are all in new_chosen.
                                # Let's just verify.
                                valid = True
                                for (f1, f2, f3) in edge_faces[idx]:
                                    if f1 in new_chosen and f2 in new_chosen and f3 in new_chosen:
                                        valid = False
                                        break
                                if valid:
                                    # Also need to check that removing rem didn't break validity (it can't, removal only helps)
                                    # But we should check that rem's removal doesn't leave a C4? No, removing edges can't create C4s.
                                    chosen = new_chosen
                                    improved = True
                                    break
                            if improved:
                                break
            # After one pass of trying to add, do another pass? Or just loop until no improvement.
            # The above loop already continues while improved.
        return chosen
    
    best_chosen = set()
    best_size = 0
    
    # First, try a few greedy randomizations
    num_greedy = 20
    for _ in range(num_greedy):
        if time.time() > deadline - 5.0:
            break
        chosen = greedy_randomized()
        if len(chosen) > best_size:
            best_chosen = set(chosen)
            best_size = len(chosen)
        # Local search from this start
        chosen = local_search(chosen)
        if len(chosen) > best_size:
            best_chosen = set(chosen)
            best_size = len(chosen)
    
    # Now, use remaining time for more local search iterations from the best solution
    # or random restarts with local search
    while time.time() < deadline - 2.0:
        # Random restart: take best, perturb, local search
        chosen = set(best_chosen)
        # Perturb: randomly remove k edges
        k = random.randint(5, 20)
        to_remove = random.sample(list(chosen), min(k, len(chosen)))
        for r in to_remove:
            chosen.discard(r)
        # Local search
        chosen = local_search(chosen)
        if len(chosen) > best_size:
            best_chosen = set(chosen)
            best_size = len(chosen)
    
    # Convert back to edge list
    out = []
    for idx in sorted(best_chosen):
        u, v = all_edges[idx]
        out.append([u, v])
    return out
