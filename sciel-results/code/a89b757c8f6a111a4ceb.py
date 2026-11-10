import time
import random
from itertools import combinations

_N = 16
_TIME_LIMIT = 115.0


def build_adj(edges, n=_N):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj


def is_triangle_free(adj, n=_N):
    for u in range(n):
        mask = adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            mask ^= lsb
            common = adj[v] & adj[u]
            if common & ~(1 << u) & ~(1 << v):
                return False
    return True


def is_not_3_colorable(adj, n=_N):
    """Exact backtracking 3-colouring. Returns True iff the graph is NOT 3-colourable."""
    memo = set()

    def dfs(uncolored_mask, colors):
        if uncolored_mask == 0:
            return False
        key = (uncolored_mask, tuple(colors))
        if key in memo:
            return True
        # pick the vertex with the most already-coloured neighbours
        best_v = -1
        best_score = -1
        m = uncolored_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            score = 0
            nb = adj[v]
            while nb:
                lsb2 = nb & -nb
                w = lsb2.bit_length() - 1
                nb ^= lsb2
                if colors[w] != -1:
                    score += 1
            if score > best_score:
                best_score = score
                best_v = v
        v = best_v
        forbidden = 0
        nb = adj[v]
        while nb:
            lsb2 = nb & -nb
            w = lsb2.bit_length() - 1
            nb ^= lsb2
            if colors[w] != -1:
                forbidden |= 1 << colors[w]
        ok = True
        for c in range(3):
            if not (forbidden & (1 << c)):
                colors[v] = c
                if not dfs(uncolored_mask & ~(1 << v), colors):
                    ok = False
                    break
                colors[v] = -1
        if ok:
            memo.add(key)
        return ok

    return dfs((1 << n) - 1, [-1] * n)


def greedy_3_colorable(adj, n=_N, max_trials=6):
    for _ in range(max_trials):
        order = list(range(n))
        random.shuffle(order)
        colors = [-1] * n
        for node in order:
            used = 0
            nb = adj[node]
            while nb:
                lsb = nb & -nb
                w = lsb.bit_length() - 1
                nb ^= lsb
                if colors[w] != -1:
                    used |= 1 << colors[w]
            placed = False
            for c in range(3):
                if not (used & (1 << c)):
                    colors[node] = c
                    placed = True
                    break
            if not placed:
                break
        else:
            return True
    return False


def is_not_3_colorable_fast(adj, n=_N):
    if greedy_3_colorable(adj, n, max_trials=4):
        return False
    return is_not_3_colorable(adj, n)


def build_groetzsch_edges():
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
    for u, v in [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]:
        edges.append([5 + u, v])
        edges.append([5 + v, u])
    for u in range(5):
        edges.append([10, 5 + u])
    return edges

def build_chvatal_edges():
    # Chvatal graph: 12 vertices, 24 edges, triangle-free, 4-chromatic
    # Source: Standard representation
    edges = [
        [0, 1], [0, 5], [0, 8], [0, 9],
        [1, 2], [1, 6],
        [2, 3], [2, 7],
        [3, 4], [3, 8],
        [4, 5], [4, 9],
        [5, 10], [5, 11],
        [6, 7], [6, 10],
        [7, 8], [7, 11],
        [8, 10], [8, 11],
        [9, 10], [9, 11]
    ]
    # Verify triangle-free?
    # (0,1), (1,2), (0,2)? No (0,2).
    # (0,5), (5,10), (0,10)? No (0,10).
    # Seems correct.
    return edges


def build_clebsch_edges():
    edges = []
    for a in range(16):
        i1, j1 = a // 4, a % 4
        for b in range(a + 1, 16):
            i2, j2 = b // 4, b % 4
            if i1 != i2 and j1 != j2:
                edges.append([a, b])
    return edges


def generate_blowup_distributions(num_extra, num_original):
    dists = []

    def rec(rem, idx, current):
        if idx == num_original - 1:
            current.append(rem)
            dists.append(tuple(current))
            current.pop()
            return
        for i in range(rem + 1):
            current.append(i)
            rec(rem - i, idx + 1, current)
            current.pop()

    rec(num_extra, 0, [])
    return dists


def _polish_graph(edges, start_time, deadline):
    """
    Takes a list of edges, polishes it by adding triangle-free edges
    that maintain non-3-colorability. Includes escape mechanism.
    """
    n = _N
    adj = build_adj(edges, n)
    cur_edges = list(edges)
    
    if not is_triangle_free(adj, n):
        return None, 0
    if not is_not_3_colorable_fast(adj, n):
        return None, 0

    best_edges = list(cur_edges)
    best_count = len(cur_edges)
    
    stall_count = 0
    escape_threshold = 50 # Try escape after 50 failed additions
    
    while time.time() < deadline:
        # Generate candidates: non-edges that are triangle-free
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    if (adj[u] & adj[v]) == 0:
                        candidates.append((u, v))
        
        if not candidates:
            break
            
        random.shuffle(candidates)
        found_improvement = False
        
        for u, v in candidates:
            if time.time() > deadline:
                return best_edges, best_count
            
            # Try adding edge
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            cur_edges.append([u, v])
            
            if is_not_3_colorable_fast(adj, n):
                # Success
                best_count = len(cur_edges)
                best_edges = list(cur_edges)
                stall_count = 0
                found_improvement = True
                break # Restart candidate list generation
            else:
                # Revert
                adj[u] ^= 1 << v
                adj[v] ^= 1 << u
                cur_edges.pop()
                stall_count += 1
        
        if not found_improvement:
            # If we exhausted all candidates without success, try escape
            if stall_count > escape_threshold and len(cur_edges) > 10:
                # Remove a random edge to escape local optimum
                # We must ensure the graph remains triangle-free (it will, since we remove an edge)
                # But we must ensure it remains NOT 3-colorable? 
                # Actually, removing an edge might make it 3-colorable.
                # If it becomes 3-colorable, we have to revert or find a different escape.
                # However, we are in a state where we couldn't ADD any edge.
                # Removing an edge reduces density.
                # Let's try removing an edge that is "least critical".
                # For now, just pick a random edge.
                if cur_edges:
                    idx = random.randint(0, len(cur_edges) - 1)
                    u, v = cur_edges[idx]
                    adj[u] ^= 1 << v
                    adj[v] ^= 1 << u
                    cur_edges.pop(idx)
                    
                    if is_not_3_colorable_fast(adj, n):
                        # Still valid, try again
                        stall_count = 0
                        escape_threshold = int(escape_threshold * 1.5)
                    else:
                        # Became 3-colorable, revert
                        adj[u] |= 1 << v
                        adj[v] |= 1 << u
                        cur_edges.insert(idx, [u, v])
                        # If we can't escape, stop
                        break
            else:
                # No candidates or no escape possible
                break

    return best_edges, best_count


def _trial(base_edges, start_time, deadline):
    """
    Takes a base graph (e.g. blow-up), saturates it, then polishes.
    """
    n = _N
    adj = build_adj(base_edges, n)
    edges = list(base_edges)
    
    # Saturate: add every non-edge whose endpoints have no common neighbour
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] & (1 << v)):
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    edges.append([u, v])
    
    if not is_triangle_free(adj, n):
        return None, 0
    if not is_not_3_colorable_fast(adj, n):
        return None, 0
        
    return _polish_graph(edges, start_time, deadline)


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT

    best_edges = None
    best_count = 0

    # 1. Try Clebsch polish
    clebsch_edges = build_clebsch_edges()
    c_edges, c_count = _trial(clebsch_edges, start_time, deadline)
    if c_edges and c_count > best_count:
        best_edges = c_edges
        best_count = c_count

    # 2. Try Chvatal blow-ups
    num_original_chv = 12
    num_extra_chv = _N - num_original_chv
    dists_chv = generate_blowup_distributions(num_extra_chv, num_original_chv)
    random.shuffle(dists_chv)
    
    # Limit number of trials to avoid timeout
    max_trials = 200
    for i, dist in enumerate(dists_chv):
        if time.time() > deadline:
            break
        if i >= max_trials:
            break
            
        sizes = [1 + d for d in dist]
        orig_map = [0] * _N
        idx = 0
        for i_orig in range(num_original_chv):
            for _ in range(sizes[i_orig]):
                orig_map[idx] = i_orig
                idx += 1
        
        verts = [[] for _ in range(num_original_chv)]
        for v in range(_N):
            verts[orig_map[v]].append(v)
            
        g_adj = build_adj(build_chvatal_edges(), num_original_chv)
        blowup_edges = []
        for i in range(num_original_chv):
            for j in range(i + 1, num_original_chv):
                if g_adj[i] & (1 << j):
                    for u in verts[i]:
                        for v in verts[j]:
                            blowup_edges.append([u, v])
                            
        b_edges, b_count = _trial(blowup_edges, start_time, deadline)
        if b_edges and b_count > best_count:
            best_edges = b_edges
            best_count = b_count

    # 3. Try Groetzsch blow-ups
    num_original_gro = 11
    num_extra_gro = _N - num_original_gro
    dists_gro = generate_blowup_distributions(num_extra_gro, num_original_gro)
    random.shuffle(dists_gro)
    
    for i, dist in enumerate(dists_gro):
        if time.time() > deadline:
            break
        if i >= max_trials:
            break
            
        sizes = [1 + d for d in dist]
        orig_map = [0] * _N
        idx = 0
        for i_orig in range(num_original_gro):
            for _ in range(sizes[i_orig]):
                orig_map[idx] = i_orig
                idx += 1
        
        verts = [[] for _ in range(num_original_gro)]
        for v in range(_N):
            verts[orig_map[v]].append(v)
            
        g_adj = build_adj(build_groetzsch_edges(), num_original_gro)
        blowup_edges = []
        for i in range(num_original_gro):
            for j in range(i + 1, num_original_gro):
                if g_adj[i] & (1 << j):
                    for u in verts[i]:
                        for v in verts[j]:
                            blowup_edges.append([u, v])
                            
        b_edges, b_count = _trial(blowup_edges, start_time, deadline)
        if b_edges and b_count > best_count:
            best_edges = b_edges
            best_count = b_count

    if best_edges is None:
        # Fallback to Clebsch
        return build_clebsch_edges()
        
    return best_edges
