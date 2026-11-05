import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin under the 150s kill

    N = 16
    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    # ---------- Exact 3-colourability check (bitmask, DSATUR ordering) ----------
    def is_3_colorable(adj_mask):
        # Build static ordering: highest degree first
        order = list(range(N))
        order.sort(key=lambda x: bin(adj_mask[x]).count('1'), reverse=True)
        pos = {v: i for i, v in enumerate(order)}
        later_neighbors = [[] for _ in range(N)]
        for idx in range(N):
            v = order[idx]
            m = adj_mask[v]
            for j in range(idx + 1, N):
                u = order[j]
                if m & (1 << u):
                    later_neighbors[idx].append(j)
        colors = [-1] * N
        forb = [0] * N

        def backtrack(idx):
            if idx == N:
                return True
            avail = 7 & ~forb[idx]
            if avail == 0:
                return False
            c = avail
            while c:
                bit = c & (-c)
                c -= bit
                col = bit.bit_length() - 1
                colors[idx] = col
                touched = []
                for j in later_neighbors[idx]:
                    old = forb[j]
                    new = old | bit
                    forb[j] = new
                    touched.append((j, old))
                if backtrack(idx + 1):
                    return True
                for j, old in touched:
                    forb[j] = old
                colors[idx] = -1
            return False

        return backtrack(0)

    # ---------- Clebsch graph seed ----------
    vectors = [i for i in range(32) if bin(i).count('1') % 2 == 0]
    clebsch_adj = [0] * N
    clebsch_edges = []
    for u_idx in range(N):
        for v_idx in range(u_idx + 1, N):
            if bin(vectors[u_idx] ^ vectors[v_idx]).count('1') == 4:
                clebsch_edges.append([u_idx, v_idx])
                clebsch_adj[u_idx] |= (1 << v_idx)
                clebsch_adj[v_idx] |= (1 << u_idx)

    best_edges = clebsch_edges
    best_count = len(clebsch_edges)  # 40

    def update_best(adj_mask, edges, count):
        nonlocal best_edges, best_count
        if count > best_count:
            best_count = count
            best_edges = [e[:] for e in edges]

    # ---------- Triangle-free edge-add test ----------
    def can_add(adj_mask, u, v):
        return (adj_mask[u] & adj_mask[v]) == 0

    # ---------- Local search: 2-opt edge swaps (structural escape) ----------
    def local_swap(adj_mask, edges, count):
        nonlocal best_edges, best_count
        # Convert edges to set for fast removal
        edge_set = {tuple(e) for e in edges}
        
        while time.time() < deadline:
            # Candidate edges to remove
            removable = list(edge_set)
            random.shuffle(removable)
            progress = False
            
            for ru, rv in removable:
                if time.time() > deadline:
                    break
                # Remove (ru,rv)
                adj_mask[ru] &= ~(1 << rv)
                adj_mask[rv] &= ~(1 << ru)
                edge_set.remove((ru, rv))
                
                # Check if still 4-chromatic
                if is_3_colorable(adj_mask):
                    # Revert
                    adj_mask[ru] |= (1 << rv)
                    adj_mask[rv] |= (1 << ru)
                    edge_set.add((ru, rv))
                    continue
                
                # Try to add 2 edges to increase count by 1
                # Candidate edges to add
                addable = []
                for u in range(N):
                    for v in range(u + 1, N):
                        if not (adj_mask[u] & (1 << v)) and can_add(adj_mask, u, v):
                            addable.append((u, v))
                
                if len(addable) < 2:
                    adj_mask[ru] |= (1 << rv)
                    adj_mask[rv] |= (1 << ru)
                    edge_set.add((ru, rv))
                    continue
                    
                # Pick 2 random addable edges
                random.shuffle(addable)
                for i in range(min(10, len(addable))):
                    for j in range(i+1, min(10, len(addable))):
                        if time.time() > deadline:
                            break
                        au, av = addable[i]
                        bw, bv = addable[j]
                        
                        # Check if adding both is triangle-free
                        # Note: adding au,av and bw,bv simultaneously
                        # Need to check triangles involving both new edges
                        # Simpler: add one, check, add second, check
                        if can_add(adj_mask, au, av):
                            adj_mask[au] |= (1 << av)
                            adj_mask[av] |= (1 << au)
                            if can_add(adj_mask, bw, bv):
                                adj_mask[bw] |= (1 << bv)
                                adj_mask[bv] |= (1 << bw)
                                
                                if not is_3_colorable(adj_mask):
                                    # Accept
                                    edge_set.add((au, av))
                                    edge_set.add((bw, bv))
                                    edges.append([au, av])
                                    edges.append([bw, bv])
                                    count += 1
                                    update_best(adj_mask, edges, count)
                                    progress = True
                                    break
                                else:
                                    # Revert second
                                    adj_mask[bw] &= ~(1 << bv)
                                    adj_mask[bv] &= ~(1 << bw)
                            # Revert first
                            adj_mask[au] &= ~(1 << av)
                            adj_mask[av] &= ~(1 << au)
                        
                        if progress:
                            break
                    if progress:
                        break
                
                # Revert remove if no progress
                if not progress:
                    adj_mask[ru] |= (1 << rv)
                    adj_mask[rv] |= (1 << ru)
                    edge_set.add((ru, rv))
                
                if progress:
                    break
            
            if not progress:
                break
        return adj_mask, edges, count

    # ---------- Main search loop ----------
    # Phase 1: Local search from Clebsch
    adj_mask = clebsch_adj[:]
    edges = [e[:] for e in clebsch_edges]
    count = best_count
    adj_mask, edges, count = local_swap(adj_mask, edges, count)

    # Phase 2: Random restarts
    while time.time() < deadline:
        # Generate random maximal triangle-free graph
        adj_mask = [0] * N
        edges = []
        count = 0
        perm = all_pairs[:]
        random.shuffle(perm)
        for u, v in perm:
            if time.time() > deadline:
                break
            if can_add(adj_mask, u, v):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edges.append([u, v])
                count += 1
        
        # If this random graph is 4-chromatic, try to improve it
        if not is_3_colorable(adj_mask):
            if count > best_count:
                update_best(adj_mask, edges, count)
            
            # Try local swap on this promising graph
            # Only if it's close to best to save time
            if count >= best_count - 5:
                adj_mask, edges, count = local_swap(adj_mask, edges, count)

    return best_edges
