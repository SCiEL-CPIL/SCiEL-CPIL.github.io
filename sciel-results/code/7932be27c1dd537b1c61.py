import time
import random
from itertools import combinations

random.seed(42)
START_TIME = time.time()
TIME_LIMIT = 118

def build_adj(edges, n=16):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj

def is_triangle_free_bitmask(adj, n=16):
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

def is_not_3_colorable_bitmask(adj, n=16):
    colors = [-1] * n
    
    def dfs(uncolored_mask):
        if uncolored_mask == 0:
            return False
        
        # Heuristic: pick vertex with most colored neighbors
        best_v = -1
        best_score = -1
        m = uncolored_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            score = 0
            nb_mask = adj[v]
            mm = nb_mask
            while mm:
                lsb_nb = mm & -mm
                nb = lsb_nb.bit_length() - 1
                mm ^= lsb_nb
                if colors[nb] != -1:
                    score += 1
            if score > best_score:
                best_score = score
                best_v = v
        
        v = best_v
        
        forbidden = 0
        nb_mask = adj[v]
        mm = nb_mask
        while mm:
            lsb_nb = mm & -mm
            nb = lsb_nb.bit_length() - 1
            mm ^= lsb_nb
            if colors[nb] != -1:
                forbidden |= 1 << colors[nb]
        
        for c in range(3):
            if not (forbidden & (1 << c)):
                colors[v] = c
                if not dfs(uncolored_mask & ~(1 << v)):
                    return False
                colors[v] = -1
        
        return True
    
    initial_mask = (1 << n) - 1
    return dfs(initial_mask)

def greedy_3_colorable(adj, n=16, max_trials=20):
    for _ in range(max_trials):
        order = list(range(n))
        random.shuffle(order)
        colors = [-1] * n
        for node in order:
            used = 0
            nb_mask = adj[node]
            mm = nb_mask
            while mm:
                lsb = mm & -mm
                nb = lsb.bit_length() - 1
                mm ^= lsb
                if colors[nb] != -1:
                    used |= 1 << colors[nb]
            for c in range(3):
                if not (used & (1 << c)):
                    colors[node] = c
                    break
            else:
                break
        else:
            return True
    return False

def is_not_3_colorable_fast(adj, n=16):
    if greedy_3_colorable(adj, n, max_trials=15):
        return False
    return is_not_3_colorable_bitmask(adj, n)

def local_search_add_edges(adj, current_edges, n=16, deadline=None):
    """Try to add edges one at a time in a systematic way."""
    if deadline is None:
        deadline = START_TIME + TIME_LIMIT
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Find all non-edges that are triangle-safe
        candidates = []
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] & (1 << v)):
                    if (adj[u] & adj[v]) == 0:
                        candidates.append((u, v))
        
        if not candidates:
            break
            
        # Try candidates in random order
        random.shuffle(candidates)
        
        for u, v in candidates:
            if time.time() > deadline:
                break
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            current_edges.append([u, v])
            if is_not_3_colorable_fast(adj, n):
                improved = True
            else:
                adj[u] ^= 1 << v
                adj[v] ^= 1 << u
                current_edges.pop()
                
    return current_edges

def local_search_swaps(adj, current_edges, n=16, deadline=None, max_rounds=50):
    """Try swapping edges: remove one edge and add another to potentially increase density."""
    if deadline is None:
        deadline = START_TIME + TIME_LIMIT
    
    improved = False
    for _round in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Get current edges as set for quick lookup
        edge_set = set()
        for u, v in current_edges:
            edge_set.add((min(u,v), max(u,v)))
        
        # Find removable edges (edges whose removal keeps triangle-free, which is always true)
        # and addable edges (non-edges that are triangle-safe)
        removable = list(current_edges)
        addable = []
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] & (1 << v)):
                    if (adj[u] & adj[v]) == 0:
                        addable.append((u, v))
        
        if not removable or not addable:
            break
            
        random.shuffle(removable)
        random.shuffle(addable)
        
        found_swap = False
        for rem_u, rem_v in removable[:20]:  # Limit checks per round
            if time.time() > deadline:
                break
            if found_swap:
                break
            
            rem_key = (min(rem_u, rem_v), max(rem_u, rem_v))
            if rem_key not in edge_set:
                continue
                
            # Temporarily remove
            adj[rem_u] ^= 1 << rem_v
            adj[rem_v] ^= 1 << rem_u
            edge_set.remove(rem_key)
            # Remove from current_edges safely
            for i, e in enumerate(current_edges):
                if (e[0] == rem_u and e[1] == rem_v) or (e[0] == rem_v and e[1] == rem_u):
                    current_edges.pop(i)
                    break
            
            # Try adding
            for add_u, add_v in addable[:20]:
                if time.time() > deadline:
                    break
                if (adj[add_u] & (1 << add_v)):
                    continue
                if (adj[add_u] & adj[add_v]) != 0:
                    continue
                    
                adj[add_u] |= 1 << add_v
                adj[add_v] |= 1 << add_u
                current_edges.append([add_u, add_v])
                edge_set.add((min(add_u, add_v), max(add_u, add_v)))
                
                if is_not_3_colorable_fast(adj, n):
                    found_swap = True
                    improved = True
                    break
                else:
                    adj[add_u] ^= 1 << add_v
                    adj[add_v] ^= 1 << add_u
                    current_edges.pop()
                    edge_set.discard((min(add_u, add_v), max(add_u, add_v)))
            
            # Restore removed edge if no successful swap
            if not found_swap:
                adj[rem_u] |= 1 << rem_v
                adj[rem_v] |= 1 << rem_u
                edge_set.add(rem_key)
                current_edges.append([rem_u, rem_v])
                
        if not found_swap:
            break
            
    return current_edges, improved

def build_groetzsch_edges():
    edges = [[0,1],[1,2],[2,3],[3,4],[4,0]]
    for u, v in [[0,1],[1,2],[2,3],[3,4],[4,0]]:
        edges.append([5+u, v])
        edges.append([5+v, u])
    for u in range(5):
        edges.append([10, 5+u])
    return edges

def get_groetzsch_adj(n=11):
    edges = build_groetzsch_edges()
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj

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

def construct():
    global START_TIME, TIME_LIMIT
    START_TIME = time.time()
    TIME_LIMIT = 118
    
    n = 16
    num_original = 11
    num_extra = n - num_original
    
    g_adj = get_groetzsch_adj(num_original)
    
    # Generate all blow-up distributions
    dists = generate_blowup_distributions(num_extra, num_original)
    
    candidates = []
    
    for dist in dists:
        if time.time() - START_TIME > TIME_LIMIT - 10: # Reserve time for local search
            break
            
        sizes = [1 + d for d in dist]
        
        orig_map = [0] * n
        current_idx = 0
        for i in range(num_original):
            for _ in range(sizes[i]):
                orig_map[current_idx] = i
                current_idx += 1
        
        edges = []
        adj = [0] * n
        
        for i in range(num_original):
            for j in range(i+1, num_original):
                if g_adj[i] & (1 << j):
                    verts_i = [v for v in range(n) if orig_map[v] == i]
                    verts_j = [v for v in range(n) if orig_map[v] == j]
                    for u in verts_i:
                        for v in verts_j:
                            edges.append([u, v])
                            adj[u] |= 1 << v
                            adj[v] |= 1 << u
        
        # Triangle-free check (should always be true for blow-ups of triangle-free graphs)
        if not is_triangle_free_bitmask(adj, n):
            continue
            
        if is_not_3_colorable_fast(adj, n):
            candidates.append((len(edges), list(edges)))
            
    if not candidates:
        # Fallback
        return build_groetzsch_edges()

    # Sort candidates by edge count (descending)
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 5 candidates to run local search on
    top_candidates = candidates[:5]
    
    best_edges = top_candidates[0][1]
    best_count = top_candidates[0][0]
    
    # Run local search on each top candidate with a time budget
    time_per_candidate = (TIME_LIMIT - (time.time() - START_TIME)) / len(top_candidates)
    
    for count, edges in top_candidates:
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        adj = build_adj(edges, n)
        current_edges = list(edges)
        deadline = START_TIME + TIME_LIMIT - 5
        
        # Limit time for this candidate
        candidate_deadline = time.time() + time_per_candidate
        
        # Phase 1: Greedy edge addition (saturation)
        current_edges = local_search_add_edges(adj, current_edges, n, candidate_deadline)
        
        # Phase 2: Swap-based local search
        if time.time() < candidate_deadline:
            current_edges, improved = local_search_swaps(adj, current_edges, n, candidate_deadline, max_rounds=20)
            
        # Phase 3: Final saturation pass
        if time.time() < candidate_deadline:
            current_edges = local_search_add_edges(adj, current_edges, n, candidate_deadline)
            
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
            
    return best_edges
