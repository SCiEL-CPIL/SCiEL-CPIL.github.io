import random
import time
import sys

_START_TIME = time.time()
DEADLINE = 105.0  # seconds, safety margin before 120s

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return DEADLINE - _elapsed()

def _is_k4_free(adj, n=18):
    """Check no K4 in graph with adjacency bitmasks adj (list of ints)."""
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            common = adj[a] & adj[b]
            # Check if common contains an edge (i.e., two vertices in common are adjacent)
            c = common
            while c:
                lsb = c & -c
                x = lsb.bit_length() - 1
                c -= lsb
                if (adj[x] & common) != 0:
                    return False
    return True

def _max_triangle_free_induced(adj, n=18):
    """Exact branch-and-bound for largest triangle-free induced subgraph."""
    best = [0]
    
    # Precompute: for each pair (u,v), common neighbors = adj[u] & adj[v]
    # Only needed for edges, but computing for all pairs is cheap with n=18
    
    # Order vertices: try multiple orderings, use the one that gives best pruning
    # Heuristic: order by degree descending to find large solutions early
    
    def popcount(x):
        return bin(x).count('1')
    
    # Precompute common neighbor masks
    common = [[0] * n for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            c = adj[u] & adj[v]
            common[u][v] = c
            common[v][u] = c
    
    def bb(idx, current_mask, forbidden_mask):
        if _elapsed() > DEADLINE:
            return
        
        # Upper bound: current size + number of available vertices from idx onwards
        # available = vertices i >= idx not in forbidden_mask
        # Quick bound without full popcount:
        remaining_count = 0
        for i in range(idx, n):
            if not (forbidden_mask >> i) & 1:
                remaining_count += 1
        
        cur_sz = popcount(current_mask)
        if cur_sz + remaining_count <= best[0]:
            return
        
        if idx == n:
            if cur_sz > best[0]:
                best[0] = cur_sz
            return
        
        v = idx
        bit = 1 << v
        
        # Branch ordering: try including v first if it seems promising
        # (higher degree among remaining might lead to faster pruning)
        
        # Option 1: Include v (if not forbidden)
        if not (forbidden_mask & bit):
            new_forbidden = forbidden_mask
            cm = current_mask
            while cm:
                lsb = cm & -cm
                u = lsb.bit_length() - 1
                cm -= lsb
                if (adj[v] >> u) & 1:
                    new_forbidden |= (adj[v] & adj[u])
            
            bb(idx + 1, current_mask | bit, new_forbidden)
        
        # Option 2: Exclude v
        bb(idx + 1, current_mask, forbidden_mask)
    
    # Try vertex ordering by degree descending
    degrees = [popcount(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    
    # Build permutation: position p in recursion corresponds to vertex order[p]
    # We need to remap the adjacency and forbidden masks
    
    # Alternative: just use natural order but with better initial lower bound
    # Get a good initial lower bound via greedy
    
    # Greedy: repeatedly add vertex with fewest conflicts
    greedy_set = 0
    greedy_forbidden = 0
    remaining = (1 << n) - 1
    while remaining:
        # Find vertex in remaining not in greedy_forbidden
        candidates = remaining & ~greedy_forbidden
        if not candidates:
            break
        # Pick vertex with minimum degree within candidates (to minimize new forbiddens)
        best_v = -1
        best_score = n + 1
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            c -= lsb
            # Count how many new vertices would be forbidden
            new_forb = 0
            gm = greedy_set
            while gm:
                lsb2 = gm & -gm
                u = lsb2.bit_length() - 1
                gm -= lsb2
                if (adj[v] >> u) & 1:
                    new_forb |= (adj[v] & adj[u])
            cnt = popcount(new_forb & remaining)
            if cnt < best_score:
                best_score = cnt
                best_v = v
        if best_v == -1:
            break
        greedy_set |= (1 << best_v)
        remaining &= ~(1 << best_v)
        gm = greedy_set
        # Update greedy_forbidden
        # For each edge in greedy_set, add common neighbors
        # Incremental: for each u in greedy_set where u < best_v and adj[best_v][u]
        gm2 = greedy_set & ~(1 << best_v)
        while gm2:
            lsb2 = gm2 & -gm2
            u = lsb2.bit_length() - 1
            gm2 -= lsb2
            if (adj[best_v] >> u) & 1:
                greedy_forbidden |= (adj[best_v] & adj[u])
    
    best[0] = popcount(greedy_set)
    
    # Now run B&B with natural order
    bb(0, 0, 0)
    return best[0]

def construct():
    n = 18
    random.seed(42)
    
    # Generate multiple random K4-free graphs and pick the best
    # Strategy: start from random graphs and repair K4s, or use structured families
    
    def _random_k4_free():
        """Generate a random K4-free graph."""
        adj = [0] * n
        # Randomly add edges, then remove any that create K4s
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        # Remove edges to make K4-free: iteratively find K4s and remove an edge
        changed = True
        while changed and _time_left() > 10:
            changed = False
            for a in range(n):
                if _time_left() < 10:
                    break
                for b in range(a + 1, n):
                    if not (adj[a] >> b) & 1:
                        continue
                    common = adj[a] & adj[b]
                    c = common
                    while c:
                        lsb = c & -c
                        x = lsb.bit_length() - 1
                        c -= lsb
                        if (adj[x] & common) != 0:
                            # Found K4: a, b, x, and some y in common that's adjacent to x
                            # Remove edge (a,b)
                            adj[a] &= ~(1 << b)
                            adj[b] &= ~(1 << a)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
        return adj
    
    def _improve_adj(adj):
        """Local search to improve a K4-free graph."""
        current = adj[:]
        current_score = _max_triangle_free_induced(current, n)
        best = adj[:]
        best_score = current_score
        
        # Single-edge flips
        steps_since_improve = 0
        max_steps = 5000
        step = 0
        
        while step < max_steps and _time_left() > 15:
            step += 1
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            new_adj = current[:]
            if (new_adj[u] >> v) & 1:
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
            else:
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
            
            if not _is_k4_free(new_adj, n):
                continue
            
            new_score = _max_triangle_free_induced(new_adj, n)
            
            if new_score < best_score:
                best_score = new_score
                best = new_adj
                current = new_adj
                steps_since_improve = 0
            elif new_score == best_score and random.random() < 0.3:
                current = new_adj
            else:
                steps_since_improve += 1
            
            if steps_since_improve > 200:
                # Try a random restart from a perturbed version
                current = current[:]
                for _ in range(random.randint(1, 3)):
                    ru = random.randint(0, n - 1)
                    rv = random.randint(0, n - 1)
                    if ru != rv:
                        current[ru] ^= (1 << rv)
                        current[rv] ^= (1 << ru)
                if not _is_k4_free(current, n):
                    current = best[:]
                steps_since_improve = 0
        
        return best, best_score
    
    # Try several different starting points
    candidates = []
    
    # 1. Random K4-free graphs
    for i in range(5):
        if _time_left() < 20:
            break
        adj = _random_k4_free()
        candidates.append(adj)
    
    # 2. Structured: complement of a graph with small independence number
    # Try: random bipartite + some intra-part edges
    for i in range(3):
        if _time_left() < 20:
            break
        adj = [0] * n
        # Random bipartition
        parts = [set(), set()]
        for v in range(n):
            parts[random.randint(0, 1)].add(v)
        # Add all cross edges
        for a in parts[0]:
            for b in parts[1]:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
        # Add some random intra-part edges (keeping K4-free)
        for part in parts:
            plist = list(part)
            for i in range(len(plist)):
                for j in range(i + 1, len(plist)):
                    if random.random() < 0.3:
                        a, b = plist[i], plist[j]
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        # Make K4-free
        changed = True
        while changed and _time_left() > 15:
            changed = False
            for a in range(n):
                for b in range(a + 1, n):
                    if not (adj[a] >> b) & 1:
                        continue
                    common = adj[a] & adj[b]
                    c = common
                    while c:
                        lsb = c & -c
                        x = lsb.bit_length() - 1
                        c -= lsb
                        if (adj[x] & common) != 0:
                            adj[a] &= ~(1 << b)
                            adj[b] &= ~(1 << a)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
        candidates.append(adj)
    
    # 3. Turan-like: balanced complete multipartite with modifications
    adj_turan = [0] * n
    k = 3  # number of parts
    part_sizes = [6, 6, 6]
    start = 0
    parts = []
    for sz in part_sizes:
        parts.append(list(range(start, start + sz)))
        start += sz
    for i in range(k):
        for j in range(i + 1, k):
            for a in parts[i]:
                for b in parts[j]:
                    adj_turan[a] |= (1 << b)
                    adj_turan[b] |= (1 << a)
    candidates.append(adj_turan)
    
    # Evaluate and improve each candidate
    best_overall = None
    best_overall_score = 999
    
    for adj in candidates:
        if _time_left() < 20:
            break
        if not _is_k4_free(adj, n):
            continue
        improved, score = _improve_adj(adj)
        if score < best_overall_score:
            best_overall_score = score
            best_overall = improved
    
    if best_overall is None:
        # Fallback: use turan
        best_overall = adj_turan
        best_overall_score = _max_triangle_free_induced(adj_turan, n)
    
    # Final polish: more local search on the best
    if _time_left() > 20:
        best_overall, best_overall_score = _improve_adj(best_overall)
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_overall[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
