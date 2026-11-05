import random
import time
import math

N = 16
TARGET_EDGES = 65
DEADLINE = 135.0  # safety margin before the 150s kill

def _score_from_adj(adj):
    """O(n^2) score: max deg(u)+deg(v)+deg(w) over all triangles.
    A triangle (u,v,w) exists iff exactly one of the three pairs is a non-edge,
    and the two edge-vertices are in the same part (since cross-part is always
    an edge in the K_{8,8}+1 base structure)."""
    deg = [0]*N
    for i in range(N):
        deg[i] = sum(adj[i])
    best = 0
    for i in range(N):
        for j in range(i+1, N):
            if not adj[i][j]:
                # i,j is the unique non-edge; third vertex w in same part as i,j
                for k in range(N):
                    if k != i and k != j and adj[i][k] and adj[j][k]:
                        s = deg[i] + deg[j] + deg[k]
                        if s > best:
                            best = s
    return best

def _score_from_deg(deg, part, extra):
    """Fast score using degree array and extra-edge info.
    part: list of part index (0 or 1) per vertex.
    extra: set of frozenset({u,v}) for internal (same-part) edges."""
    best = 0
    for e in extra:
        u, v = e
        p = part[u]
        # third vertex w must be in the other part
        for w in range(N):
            if part[w] != p:
                s = deg[u] + deg[v] + deg[w]
                if s > best:
                    best = s
    return best

def _build_state(part, extra):
    """Build adj matrix, deg, edges, nonedges from part assignment + extra edges."""
    adj = [[False]*N for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if part[i] != part[j]:
                adj[i][j] = adj[j][i] = True
    for e in extra:
        u, v = e
        adj[u][v] = adj[v][u] = True
    deg = [0]*N
    for i in range(N):
        deg[i] = sum(adj[i])
    edges = []
    nonedges = []
    for i in range(N):
        for j in range(i+1, N):
            if adj[i][j]:
                edges.append((i, j))
            else:
                nonedges.append((i, j))
    return adj, deg, edges, nonedges

def _init_state(rng, part_size=8):
    """Random part assignment + one extra edge."""
    part = [0]*8 + [1]*8
    rng.shuffle(part)
    # pick a random same-part pair for the extra edge
    same_part_pairs = []
    for i in range(N):
        for j in range(i+1, N):
            if part[i] == part[j]:
                same_part_pairs.append((i, j))
    u, v = rng.choice(same_part_pairs)
    extra = {frozenset((u, v))}
    return part, extra

def _search(rng, part, extra, t0, deadline, best_score, best_state):
    """Local search: try removing one extra edge and adding a different one,
    or swap a cross-part edge for an internal edge (keeping edge count).
    Actually, to keep exactly 65 edges and the K_{8,8}+1 structure,
    the only freedom is which internal edge is the 'extra'. But we can also
    move the partition boundary.
    
    Better: allow any edge swap that keeps 65 edges. But that breaks the
    clean structure. Instead, do two types of moves:
    1. Change the extra edge (remove current, add different same-part edge)
    2. Reassign one vertex to the other part (must adjust to keep 65 edges)
    
    For simplicity and speed, do move type 1 (change extra edge) plus
    occasional full re-partition."""
    best_local_score = best_score
    best_local_state = best_state
    part_cur = part[:]
    extra_cur = set(extra)
    
    # compute initial
    adj, deg, edges, nonedges = _build_state(part_cur, extra_cur)
    cur_score = _score_from_deg(deg, part_cur, extra_cur)
    
    if cur_score < best_local_score:
        best_local_score = cur_score
        best_local_state = (part_cur[:], set(extra_cur))
    
    # Collect all same-part pairs
    def same_part_pairs(p):
        sp = []
        for i in range(N):
            for j in range(i+1, N):
                if p[i] == p[j]:
                    sp.append((i, j))
        return sp
    
    # Simulated annealing over extra-edge choices
    # Also occasionally try moving a vertex between parts
    max_iters = 8000
    start_temp = 3.0
    end_temp = 0.05
    cooling = (end_temp / start_temp) ** (1.0 / max(1, max_iters - 1))
    temp = start_temp
    
    for it in range(max_iters):
        if time.time() - t0 > deadline:
            break
        
        # Move type: pick a new extra edge from same-part pairs
        sp = same_part_pairs(part_cur)
        new_e = rng.choice(sp)
        new_e_fs = frozenset(new_e)
        
        if new_e_fs == extra_cur:
            continue
        
        # Apply: remove old extra, add new
        old_e = next(iter(extra_cur))
        extra_cur.discard(old_e)
        extra_cur.add(new_e_fs)
        
        # Recompute degrees quickly: only the endpoints of old_e and new_e change
        # old_e endpoints lose 1, new_e endpoints gain 1
        ou, ov = old_e
        nu, nv = new_e
        deg[ou] -= 1
        deg[ov] -= 1
        deg[nu] += 1
        deg[nv] += 1
        
        new_score = _score_from_deg(deg, part_cur, extra_cur)
        
        if new_score < cur_score:
            accept = True
        elif new_score == cur_score:
            accept = rng.random() < 0.5
        else:
            delta = new_score - cur_score
            if delta > 8 or temp < 1e-9:
                accept = False
            else:
                accept = math.exp(-delta / temp) > rng.random()
        
        if accept:
            cur_score = new_score
            if new_score < best_local_score:
                best_local_score = new_score
                best_local_state = (part_cur[:], set(extra_cur))
        else:
            # Undo
            extra_cur.discard(new_e_fs)
            extra_cur.add(old_e)
            deg[ou] += 1
            deg[ov] += 1
            deg[nu] -= 1
            deg[nv] -= 1
        
        # Occasionally try moving a vertex between parts
        if rng.random() < 0.05:
            v = rng.randrange(N)
            old_part = part_cur[v]
            new_part = 1 - old_part
            # If we move v, the cross-part edges change: v now connects to old part,
            # and loses connection to new part. Edge count changes by 
            # (|old_part| - 1) - (|new_part| - 1) = |old_part| - |new_part|
            # To keep 65 edges, we'd need to adjust. This is complex.
            # Instead, skip this move for now; the extra-edge search is the main lever.
            pass
        
        temp *= cooling
    
    return best_local_score, best_local_state

def construct():
    t0 = time.time()
    rng = random.Random(12345)
    
    best_score = float('inf')
    best_state = None
    
    # Multiple restarts
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() - t0 > DEADLINE - 5:
            break
        
        part, extra = _init_state(rng)
        ls, lstate = _search(rng, part, extra, t0, DEADLINE - 5, best_score, best_state)
        
        if ls < best_score:
            best_score = ls
            best_state = lstate
    
    # Fallback
    if best_state is None:
        part = [0]*8 + [1]*8
        extra = {frozenset((0, 1))}
        best_state = (part, extra)
    
    part, extra = best_state
    adj, deg, edges, nonedges = _build_state(part, extra)
    
    # Verify edge count
    ec = _count_edges(adj)
    if ec != TARGET_EDGES:
        # Fallback to known-good
        adj = [[False]*N for _ in range(N)]
        for i in range(8):
            for j in range(8, 16):
                adj[i][j] = adj[j][i] = True
        adj[0][1] = adj[1][0] = True
    
    return {'n': N, 'edges': [[i, j] for i in range(N) for j in range(i+1, N) if adj[i][j]]}

def _count_edges(adj):
    return sum(sum(row) for row in adj) // 2
