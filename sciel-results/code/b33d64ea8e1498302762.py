"""Strong edge colouring via local search."""
import time
import random
from collections import defaultdict

_DEADLINE = 140.0  # safety margin before 150s kill


def _build_instance(n, d, seed):
    rng = random.Random(seed)
    degree = [0] * n
    edges = set()
    for _ in range(4 * n * d):
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u == v:
            continue
        e = (min(u, v), max(u, v))
        if degree[u] < d and degree[v] < d and e not in edges:
            edges.add(e)
            degree[u] += 1
            degree[v] += 1
    return sorted(edges)


def _precompute(n, edges):
    """Return adjacency dict and for each edge index its conflict edge indices."""
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    # edge -> (u, v); index
    m = len(edges)
    # For conflict: edges e1=(a,b), e2=(c,d) conflict if share vertex or
    # (a-c in adj) or (a-d in adj) or (b-c in adj) or (b-d in adj)
    conflict = [set() for _ in range(m)]
    for i in range(m):
        a, b = edges[i]
        for j in range(i + 1, m):
            c, d = edges[j]
            if a == c or a == d or b == c or b == d:
                conflict[i].add(j)
                conflict[j].add(i)
                continue
            # joined by an edge
            if c in adj[a] or d in adj[a] or c in adj[b] or d in adj[b]:
                conflict[i].add(j)
                conflict[j].add(i)
    return adj, conflict, m


def _greedy_init(edges, conflict, rng):
    m = len(edges)
    colour = [0] * m
    # order: random or by conflict degree
    order = list(range(m))
    rng.shuffle(order)
    used = [set() for _ in range(m)]  # not used; we do first-fit
    # first-fit: for each edge in order, pick smallest colour not used by any conflicting edge already coloured
    colour = [-1] * m
    colour_used_by_edge = [set() for _ in range(m)]  # colours present in conflict set
    for idx in order:
        taken = set()
        for j in conflict[idx]:
            if colour[j] != -1:
                taken.add(colour[j])
        c = 0
        while c in taken:
            c += 1
        colour[idx] = c
    # compress colours
    remap = {}
    nc = 0
    for c in colour:
        if c not in remap:
            remap[c] = nc
            nc += 1
    colour = [remap[c] for c in colour]
    return colour, nc


def _is_valid(colour, conflict):
    m = len(colour)
    by_colour = defaultdict(set)
    for i, c in enumerate(colour):
        by_colour[c].add(i)
    for c, members in by_colour.items():
        for i in members:
            for j in conflict[i]:
                if j in members:
                    return False
    return True


def _count_conflicts(colour, conflict):
    m = len(colour)
    by_colour = defaultdict(set)
    for i, c in enumerate(colour):
        by_colour[c].add(i)
    cnt = 0
    for c, members in by_colour.items():
        for i in members:
            for j in conflict[i]:
                if j in members and i < j:
                    cnt += 1
    return cnt


def _local_search(colour, conflict, rng, deadline):
    m = len(colour)
    best = list(colour)
    best_conf = _count_conflicts(best, conflict)
    if best_conf == 0:
        return best
    t = time.time()
    cur = list(colour)
    cur_conf = best_conf
    while time.time() < deadline:
        # pick a random conflicting pair and move one
        # find a conflicting edge
        # build member sets
        by_colour = defaultdict(list)
        for i, c in enumerate(cur):
            by_colour[c].append(i)
        # find a conflict
        found = None
        # sample
        for _ in range(50):
            i = rng.randrange(m)
            c = cur[i]
            for j in conflict[i]:
                if cur[j] == c:
                    found = (i, j)
                    break
            if found:
                break
        if found is None:
            # no conflict found by sampling; maybe none
            if _count_conflicts(cur, conflict) == 0:
                break
            # force pick
            i = rng.randrange(m)
            for j in conflict[i]:
                if cur[j] == cur[i]:
                    found = (i, j)
                    break
            if found is None:
                break
        i, j = found
        # move i to a new colour
        # candidate colours: current colours + new
        maxc = max(cur)
        # try to pick a colour that reduces conflicts
        best_move = None
        best_delta = 0
        # try current colours and maxc+1
        cands = set(cur) | {maxc + 1}
        for nc in cands:
            if nc == cur[i]:
                continue
            # delta = change in conflicts for i
            # conflicts involving i: for each j' in conflict[i], if cur[j']==nc then +1, if cur[j']==cur[i] then -1
            old_c = cur[i]
            d = 0
            for jp in conflict[i]:
                if cur[jp] == old_c:
                    d -= 1
                if cur[jp] == nc:
                    d += 1
            if d < best_delta:
                best_delta = d
                best_move = nc
        if best_move is None:
            best_move = maxc + 1
        cur[i] = best_move
        new_conf = cur_conf + best_delta
        if new_conf < cur_conf:
            cur_conf = new_conf
            if cur_conf == 0:
                break
            if cur_conf < best_conf:
                best_conf = cur_conf
                best = list(cur)
                if best_conf == 0:
                    break
        else:
            # accept with some probability (restart-ish)
            if rng.random() < 0.3:
                cur_conf = new_conf
            else:
                cur[i] = old_c
    return best


def _reduce_colours(colour, conflict, rng, deadline):
    """Given a valid colouring, try to reduce number of colours by moving edges."""
    m = len(colour)
    cur = list(colour)
    nc = max(cur) + 1
    t = time.time()
    while time.time() < deadline:
        improved = False
        # for each colour class from high to low, try to move all edges out
        for c in range(nc - 1, -1, -1):
            members = [i for i in range(m) if cur[i] == c]
            if not members:
                continue
            # try to move each member to a lower colour
            moved = 0
            for i in members:
                old = cur[i]
                # find a colour < nc-1 (i.e., not c) where i fits
                placed = False
                for nc2 in range(c):
                    # check i doesn't conflict with any edge of colour nc2
                    ok = True
                    for j in conflict[i]:
                        if cur[j] == nc2:
                            ok = False
                            break
                    if ok:
                        cur[i] = nc2
                        moved += 1
                        placed = True
                        break
                if not placed:
                    # try to put it in a brand new colour? no, we want reduce
                    pass
            # if all moved, drop colour c
            if moved == len(members):
                nc = c  # colours 0..c-1 remain
                improved = True
        if not improved:
            # try random moves to enable future reductions
            i = rng.randrange(m)
            old = cur[i]
            cands = [c for c in range(nc) if c != old]
            rng.shuffle(cands)
            for nc2 in cands:
                ok = True
                for j in conflict[i]:
                    if cur[j] == nc2:
                        ok = False
                        break
                if ok:
                    cur[i] = nc2
                    improved = True
                    break
            if not improved:
                break
    # compress
    remap = {}
    ncc = 0
    for c in cur:
        if c not in remap:
            remap[c] = ncc
            ncc += 1
    cur = [remap[c] for c in cur]
    return cur, ncc


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    out = []
    # Precompute per instance
    instances = []
    for spec in specs:
        edges = _build_instance(*spec)
        adj, conflict, m = _precompute(len(edges), edges)
        instances.append((edges, conflict, m))
    for (edges, conflict, m) in instances:
        if time.time() > deadline - 2:
            # fallback: greedy
            colour, nc = _greedy_init(edges, conflict, random.Random(0))
            out.append(colour)
            continue
        best = None
        best_nc = 10 ** 9
        rng = random.Random()
        # multiple restarts
        n_restarts = 200
        while time.time() < deadline - 1 and n_restarts > 0:
            n_restarts -= 1
            colour, nc = _greedy_init(edges, conflict, rng)
            # local search to fix conflicts
            colour = _local_search(colour, conflict, rng, deadline - 1)
            if _count_conflicts(colour, conflict) != 0:
                # still invalid; try harder or skip
                continue
            # reduce colours
            colour2, nc2 = _reduce_colours(colour, conflict, rng, deadline - 1)
            if nc2 < best_nc:
                best_nc = nc2
                best = colour2
        if best is None:
            # fallback greedy valid
            colour, nc = _greedy_init(edges, conflict, random.Random(0))
            # ensure valid by local search
            colour = _local_search(colour, conflict, random.Random(0), deadline - 1)
            best = colour
        out.append(best)
    return out
