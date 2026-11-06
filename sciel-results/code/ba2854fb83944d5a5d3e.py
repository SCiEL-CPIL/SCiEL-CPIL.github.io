import time
import random
import itertools

N = 16
DEADLINE = 105.0  # safety margin under the 150s limit

# Map each of the 28 local edges (i,j), 0<=i<j<=7, to a bit index 0..27
_pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)]
_pair_index = {p: k for k, p in enumerate(_pairs)}

def _gen_patterns():
    # Canonical cube edges on labels 0..7 (binary 000..111, Hamming distance 1)
    can = set()
    for a in range(8):
        for b in range(a + 1, 8):
            if bin(a ^ b).count('1') == 1:
                can.add((a, b))
    pats = set()
    for perm in itertools.permutations(range(8)):
        m = 0
        for (a, b) in can:
            x, y = perm[a], perm[b]
            if x > y:
                x, y = y, x
            m |= 1 << _pair_index[(x, y)]
        pats.add(m)
    return list(pats)

_PATTERNS = _gen_patterns()  # 840 distinct 28-bit Q3 edge-patterns

_ALL8 = list(itertools.combinations(range(N), 8))
NUM8 = len(_ALL8)  # 12870

# edge (u,v) -> list of 8-set indices that contain both endpoints
_EDGE_TO_8 = {}
for _k, _verts in enumerate(_ALL8):
    for _i in range(8):
        for _j in range(_i + 1, 8):
            u, v = _verts[_i], _verts[_j]
            if u > v:
                u, v = v, u
            _EDGE_TO_8.setdefault((u, v), []).append(_k)

_mask_cache = {}

def _mask_of(verts, adj):
    m = 0
    for i in range(8):
        vi = verts[i]
        ai = adj[vi]
        for j in range(i + 1, 8):
            vj = verts[j]
            if (ai >> vj) & 1:
                m |= 1 << _pair_index[(i, j)]
    return m

def _has_q3(mask):
    c = _mask_cache.get(mask)
    if c is not None:
        return c
    for p in _PATTERNS:
        if (mask & p) == p:
            _mask_cache[mask] = True
            return True
    _mask_cache[mask] = False
    return False

def _recompute(k, adj):
    return _has_q3(_mask_of(_ALL8[k], adj))

def _fallback():
    # Proven Q3-free: any Q3 needs 4 vertices on each bipartite side.
    return [[a, b] for a in range(3) for b in range(3, 16)]

def construct():
    t0 = time.time()
    random.seed(12345)

    def time_left():
        return DEADLINE - (time.time() - t0)

    # Start from the complete graph
    adj = [0] * N
    full = (1 << N) - 1
    for i in range(N):
        adj[i] = full ^ (1 << i)

    # Initial bad-8-set set
    bad = [False] * NUM8
    bad_set = set()
    for k in range(NUM8):
        if _recompute(k, adj):
            bad[k] = True
            bad_set.add(k)

    # Greedy removal: destroy every Q3
    while bad_set and time_left() > 20.0:
        k = random.choice(list(bad_set))
        verts = _ALL8[k]
        present = []
        for (i, j) in _pairs:
            u, v = verts[i], verts[j]
            if (adj[u] >> v) & 1:
                present.append((u, v) if u < v else (v, u))
        if not present:
            bad[k] = False
            bad_set.discard(k)
            continue
        u, v = random.choice(present)
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        for k2 in _EDGE_TO_8.get((u, v), ()):
            newb = _recompute(k2, adj)
            if newb and not bad[k2]:
                bad[k2] = True
                bad_set.add(k2)
            elif not newb and bad[k2]:
                bad[k2] = False
                bad_set.discard(k2)
        if time_left() < 20.0:
            break

    if bad_set:
        # Could not finish in time -> return the guaranteed-valid seed construction
        return _fallback()

    # Add-back local search: re-add any edge that does not recreate a Q3
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    random.shuffle(all_edges)
    for (u, v) in all_edges:
        if time_left() < 25.0:
            break
        if (adj[u] >> v) & 1:
            continue
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        created = False
        for k2 in _EDGE_TO_8.get((u, v), ()):
            if not bad[k2] and _recompute(k2, adj):
                created = True
                break
        if created:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)

    # New structural component: local search / hill-climbing on edge swaps
    # Try to improve the edge count by swapping one edge for another.
    # We maintain the bad set and only accept swaps that don't introduce new Q3s
    # and ideally remove some or keep the same.
    # Actually, a swap removes one edge and adds one. The net change in edges is 0.
    # To improve, we need to find a sequence of swaps that allows adding more edges later,
    # or simply try to add edges in a different order.
    # A better approach: Random restarts or simulated annealing?
    # Given time constraints, let's try a simple "shake" and re-add.
    # Remove a random set of edges, then re-run the add-back phase.
    
    best_adj = adj[:]
    best_edges_count = sum(bin(x).count('1') for x in adj) // 2

    if time_left() > 40.0:
        # Try a few restarts with different random seeds or perturbations
        for restart in range(5):
            if time_left() < 30.0:
                break
            
            # Perturb the current solution by removing a few random edges
            adj_pert = best_adj[:]
            num_remove = 5
            removed = []
            for _ in range(num_remove):
                # Pick a random present edge
                present_edges = [(i, j) for i in range(N) for j in range(i+1, N) if (adj_pert[i] >> j) & 1]
                if not present_edges:
                    break
                u, v = random.choice(present_edges)
                adj_pert[u] &= ~(1 << v)
                adj_pert[v] &= ~(1 << u)
                removed.append((u, v))
            
            # Recompute bad set for perturbed graph
            bad_pert = [False] * NUM8
            bad_set_pert = set()
            for k in range(NUM8):
                if _recompute(k, adj_pert):
                    bad_pert[k] = True
                    bad_set_pert.add(k)
            
            if bad_set_pert:
                # Need to remove edges to fix it, which is expensive.
                # Skip this restart if it's invalid.
                continue

            # Add-back phase
            all_edges_shuffled = all_edges[:]
            random.shuffle(all_edges_shuffled)
            for (u, v) in all_edges_shuffled:
                if time_left() < 30.0:
                    break
                if (adj_pert[u] >> v) & 1:
                    continue
                adj_pert[u] |= (1 << v)
                adj_pert[v] |= (1 << u)
                created = False
                for k2 in _EDGE_TO_8.get((u, v), ()):
                    if not bad_pert[k2] and _recompute(k2, adj_pert):
                        created = True
                        break
                if created:
                    adj_pert[u] &= ~(1 << v)
                    adj_pert[v] &= ~(1 << u)
                    # Update bad set if necessary? 
                    # The _recompute check above ensures no NEW Q3 is created.
                    # But wait, the bad_pert set might be stale if we added edges.
                    # Actually, if we add an edge and it doesn't create a Q3, 
                    # the graph remains Q3-free. So bad_pert remains accurate 
                    # (all false) as long as we start from a Q3-free graph.
                    # Wait, bad_pert tracks which 8-sets *have* a Q3.
                    # If the graph is Q3-free, all bad_pert[k] are False.
                    # So we don't need to update bad_pert during add-back, 
                    # we just need to check if adding an edge creates a Q3.
                    # The check `if not bad_pert[k2] and _recompute(...)` is slightly redundant 
                    # if bad_pert is all False, but safe.
                    # However, _recompute(k2, adj_pert) checks the current state.
                    # So this is correct.

            # Check if this perturbed solution is better
            current_count = sum(bin(x).count('1') for x in adj_pert) // 2
            if current_count > best_edges_count:
                best_adj = adj_pert[:]
                best_edges_count = current_count

    adj = best_adj

    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return edges
