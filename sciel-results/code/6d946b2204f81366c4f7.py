"""Rewritten program: multi-start local search with diverse seeds."""
# EVOLVE-BLOCK-START
import time
import random

N = 15
TOTAL = 1 << N

def _is_valid_fast(adj, deg):
    """Fast validity check using bitmask per edge."""
    for u in range(N):
        for v in range(u + 1, N):
            if not adj[u][v]:
                continue
            # Collect neighbors of u (excluding v) and neighbors of v (excluding u)
            # Group by degree
            deg_groups_u = {}  # degree -> bitmask of neighbors of u
            deg_groups_v = {}  # degree -> bitmask of neighbors of v
            for s in range(N):
                if s == u or s == v:
                    continue
                d = deg[s]
                if adj[s][u]:
                    deg_groups_u[d] = deg_groups_u.get(d, 0) | (1 << s)
                if adj[s][v]:
                    deg_groups_v[d] = deg_groups_v.get(d, 0) | (1 << s)
            # Check: for each degree d, if there exist s in N(u)\{v} and t in N(v)\{u}
            # with deg[s]=deg[t]=d and s!=t, then violation
            for d in deg_groups_u:
                if d in deg_groups_v:
                    mask_u = deg_groups_u[d]
                    mask_v = deg_groups_v[d]
                    # Need to check if there exist s in mask_u, t in mask_v with s != t
                    # This fails only if mask_u and mask_v are both singletons of the same vertex
                    if mask_u & mask_v:
                        # There's overlap - but we need s != t
                        # If mask_u has only one bit and mask_v has only one bit and they're the same,
                        # then s==t is the only possibility, no violation
                        # Otherwise there's a violation
                        u_popcount = bin(mask_u).count('1')
                        v_popcount = bin(mask_v).count('1')
                        overlap = mask_u & mask_v
                        if u_popcount == 1 and v_popcount == 1 and mask_u == mask_v:
                            # Only one vertex in both, s==t, no violation
                            continue
                        # Otherwise: either there are multiple vertices, or the overlap
                        # means we can pick s from mask_u not in overlap or t from mask_v not in overlap
                        # Actually simpler: violation exists iff NOT (mask_u == mask_v and popcount(mask_u)==1)
                        # i.e., we need s in mask_u and t in mask_v with s != t
                        # This is impossible only if mask_u == mask_v and both have exactly 1 bit
                        return False
    return True

def _try_add_edge(adj, deg, u, v):
    """Try adding edge (u,v); return (new_adj, new_deg) if valid, else None."""
    if u == v or adj[u][v]:
        return None
    new_adj = [row[:] for row in adj]
    new_adj[u][v] = new_adj[v][u] = 1
    new_deg = deg[:]
    new_deg[u] += 1
    new_deg[v] += 1
    if _is_valid_fast(new_adj, new_deg):
        return (new_adj, new_deg)
    return None

def _try_swap(adj, deg, remove, add):
    """Remove edge `remove`, add edge `add`. Return (new_adj, new_deg) if valid."""
    (ru, rv) = remove
    (au, av) = add
    if au == av or adj[au][av]:
        return None
    if not adj[ru][rv]:
        return None
    new_adj = [row[:] for row in adj]
    new_adj[ru][rv] = new_adj[rv][ru] = 0
    new_adj[au][av] = new_adj[av][au] = 1
    new_deg = deg[:]
    new_deg[ru] -= 1
    new_deg[rv] -= 1
    new_deg[au] += 1
    new_deg[av] += 1
    if _is_valid_fast(new_adj, new_deg):
        return (new_adj, new_deg)
    return None

def _edge_count(adj):
    return sum(sum(row) for row in adj) // 2

def _greedy_add(adj, deg, deadline):
    """Repeatedly try to add edges until no more can be added or time runs out."""
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        for u in range(N):
            for v in range(u+1, N):
                if adj[u][v]:
                    continue
                res = _try_add_edge(adj, deg, u, v)
                if res is not None:
                    adj, deg = res
                    improved = True
                    break
            if improved:
                break
    return adj, deg

def _local_search(adj, deg, deadline, rng, max_iter=200000):
    """Random 2-swap local search with maintained edge lists. Returns (adj, deg, count)."""
    all_edges = [(u, v) for u in range(N) for v in range(u+1, N)]
    edges_present = [(u, v) for (u, v) in all_edges if adj[u][v]]
    non_edges = [(u, v) for (u, v) in all_edges if not adj[u][v]]
    count = len(edges_present)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        if not non_edges or not edges_present:
            break
        
        # Try a few random swaps before giving up on this iteration
        found = False
        for _ in range(5):
            if time.time() > deadline:
                break
            add = rng.choice(non_edges)
            remove = rng.choice(edges_present)
            res = _try_swap(adj, deg, remove, add)
            if res is not None:
                adj, deg = res
                new_count = _edge_count(adj)
                if new_count > count:
                    count = new_count
                    # Update edge lists
                    remove_set = (remove[0], remove[1])
                    add_set = (add[0], add[1])
                    edges_present = [e for e in edges_present if e != remove_set]
                    edges_present.append(add_set)
                    non_edges = [e for e in non_edges if e != add_set]
                    non_edges.append(remove_set)
                    found = True
                    break
        if found:
            continue
        # If no swap worked, occasionally try to find a valid add
        if rng.random() < 0.1 and non_edges:
            for _ in range(3):
                if time.time() > deadline:
                    break
                idx = rng.randrange(len(non_edges))
                u, v = non_edges[idx]
                res = _try_add_edge(adj, deg, u, v)
                if res is not None:
                    adj, deg = res
                    count += 1
                    non_edges.pop(idx)
                    edges_present.append((u, v))
                    break
    
    return adj, deg, count

def _polish(adj, deg, deadline):
    """Final polish: systematically try all possible adds and swaps."""
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        # Try all adds
        for u in range(N):
            for v in range(u+1, N):
                if adj[u][v]:
                    continue
                res = _try_add_edge(adj, deg, u, v)
                if res is not None:
                    adj, deg = res
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        # Try all swaps
        edges_present = [(u, v) for u in range(N) for v in range(u+1, N) if adj[u][v]]
        non_edges = [(u, v) for u in range(N) for v in range(u+1, N) if not adj[u][v]]
        for remove in edges_present:
            if time.time() > deadline:
                break
            for add in non_edges:
                res = _try_swap(adj, deg, remove, add)
                if res is not None:
                    adj, deg = res
                    improved = True
                    break
            if improved:
                break
    return adj, deg

def _make_seed_k78():
    """Create K7,8 bipartite seed."""
    adj = [[0]*N for _ in range(N)]
    deg = [0]*N
    for a in range(7):
        for b in range(7, 15):
            adj[a][b] = adj[b][a] = 1
    for a in range(7):
        deg[a] = 8
    for b in range(7, 15):
        deg[b] = 7
    return adj, deg

def _make_seed_random(rng, density=0.4):
    """Create a random valid seed by greedily adding edges."""
    adj = [[0]*N for _ in range(N)]
    deg = [0]*N
    # Start with a few random edges
    for _ in range(5):
        u, v = rng.sample(range(N), 2)
        if not adj[u][v]:
            adj[u][v] = adj[v][u] = 1
            deg[u] += 1
            deg[v] += 1
    # Greedily add more edges
    for u in range(N):
        for v in range(u+1, N):
            if adj[u][v]:
                continue
            res = _try_add_edge(adj, deg, u, v)
            if res is not None:
                adj, deg = res
    return adj, deg

def construct():
    start = time.time()
    deadline = start + 80  # 10s safety margin before 90s kill
    
    best_adj = None
    best_deg = None
    best_count = 0
    
    # Multi-start with different seeds
    seeds = []
    
    # Seed 1: K7,8 bipartite
    seeds.append(("k78", _make_seed_k78()))
    
    # Seed 2-4: Random seeds with different densities
    rng_init = random.Random(42)
    for i in range(3):
        adj, deg = _make_seed_random(rng_init, density=0.3 + 0.1*i)
        seeds.append((f"random_{i}", (adj, deg)))
    
    # Process each seed
    num_seeds = len(seeds)
    time_per_seed = (deadline - start - 10) / num_seeds  # reserve 10s for final polish
    
    for idx, (name, (adj, deg)) in enumerate(seeds):
        seed_deadline = start + (idx + 1) * time_per_seed
        if time.time() > seed_deadline - 2:
            break
        
        # Greedy add phase
        adj, deg = _greedy_add(adj, deg, seed_deadline - 3)
        count = _edge_count(adj)
        
        # Local search phase
        rng = random.Random(42 + idx * 1000)
        adj, deg, count = _local_search(adj, deg, seed_deadline - 1, rng, max_iter=100000)
        
        # Final greedy add
        adj, deg = _greedy_add(adj, deg, seed_deadline - 0.5)
        count = _edge_count(adj)
        
        if count > best_count:
            best_count = count
            best_adj = [row[:] for row in adj]
            best_deg = deg[:]
    
    # If we have time left, do more local search and polish on the best solution
    if best_adj is not None and time.time() < deadline - 5:
        remaining = deadline - 5 - time.time()
        if remaining > 2:
            rng = random.Random(999)
            adj, deg, count = _local_search(best_adj, best_deg, deadline - 3, rng, max_iter=300000)
            adj, deg = _greedy_add(adj, deg, deadline - 1)
            count = _edge_count(adj)
            if count > best_count:
                best_count = count
                best_adj = [row[:] for row in adj]
                best_deg = deg[:]
    
    # Final polish pass
    if best_adj is not None and time.time() < deadline - 1:
        adj, deg = _polish(best_adj, best_deg, deadline - 0.5)
        count = _edge_count(adj)
        if count > best_count:
            best_count = count
            best_adj = [row[:] for row in adj]
            best_deg = deg[:]
    
    # Convert to edge list
    if best_adj is None:
        # Fallback: return K7,8
        adj, deg = _make_seed_k78()
        best_adj = adj
        best_deg = deg
    
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if best_adj[u][v]:
                edges.append([int(u), int(v)])
    return edges
# EVOLVE-BLOCK-END
