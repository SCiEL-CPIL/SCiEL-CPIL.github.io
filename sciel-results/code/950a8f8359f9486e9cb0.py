"""Triangle-free graph without independent 10-set, maximizing n (Ramsey R(3,10))."""
import time
import random
import sys
from itertools import combinations

_START = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START)


def _bit_count(x):
    return bin(x).count('1')


def _has_triangle_bitmask(adj, n):
    """Check if adjacency bitmask list has a triangle."""
    for i in range(n):
        ni = adj[i]
        j_mask = ni
        while j_mask:
            lsb = j_mask & -j_mask
            j = lsb.bit_length() - 1
            j_mask ^= lsb
            if j > i:
                if adj[i] & adj[j]:
                    return True
    return False


def _max_independent_set_size_bitmask(adj, n):
    """Find size of maximum independent set using branch and bound with bitmasks."""
    best = [0]
    full_mask = (1 << n) - 1

    def greedy_color_bound(available):
        """
        Upper bound on independent set size in G[available].
        Color the complement of G[available]. Each color class is a clique in G,
        so any independent set in G picks at most 1 from each color class.
        """
        uncolored = available
        num_colors = 0
        while uncolored:
            num_colors += 1
            # Greedily build a color class (clique in G = independent set in complement)
            # A vertex v can be in this color class if it's not adjacent to any already-chosen vertex in G
            # i.e., v is in the complement of the union of neighbors of chosen vertices
            color_class = 0
            candidates = uncolored
            while candidates:
                lsb = candidates & -candidates
                v = lsb.bit_length() - 1
                candidates ^= lsb
                # Check if v is non-adjacent to all vertices in color_class in G
                if color_class == 0:
                    color_class |= lsb
                else:
                    # v must not be adjacent to any vertex in color_class
                    ok = True
                    cc = color_class
                    while cc:
                        lsb_cc = cc & -cc
                        u = lsb_cc.bit_length() - 1
                        cc ^= lsb_cc
                        if adj[v] & (1 << u):
                            ok = False
                            break
                    if ok:
                        color_class |= lsb
            uncolored &= ~color_class
        return num_colors

    def bnb(available, current_size):
        if not available:
            if current_size > best[0]:
                best[0] = current_size
            return

        bound = greedy_color_bound(available)
        if current_size + bound <= best[0]:
            return

        # Pick vertex with max degree in G[available]
        v_mask = available
        max_deg = -1
        v = -1
        while v_mask:
            lsb = v_mask & -v_mask
            u = lsb.bit_length() - 1
            v_mask ^= lsb
            deg = _bit_count(adj[u] & available)
            if deg > max_deg:
                max_deg = deg
                v = u

        if v == -1:
            return

        # Branch 1: Include v
        new_avail_include = available & ~(1 << v) & ~adj[v]
        bnb(new_avail_include, current_size + 1)

        # Branch 2: Exclude v
        new_avail_exclude = available & ~(1 << v)
        bnb(new_avail_exclude, current_size)

    bnb(full_mask, 0)
    return best[0]


def _validate_bitmask(adj, n):
    """Check triangle-free and no independent 10-set."""
    if _has_triangle_bitmask(adj, n):
        return False
    if _max_independent_set_size_bitmask(adj, n) >= 10:
        return False
    return True


def _build_circulant_graph(n, shifts):
    """
    Build a circulant graph on n vertices with given shift set.
    Vertex i is connected to (i+s) mod n for each s in shifts.
    Returns adjacency bitmask list or None if not triangle-free.
    """
    adj = [0] * n
    for s in shifts:
        s = s % n
        if s == 0:
            continue
        for i in range(n):
            j = (i + s) % n
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)

    # Check triangle-free
    if _has_triangle_bitmask(adj, n):
        return None

    return adj


def _is_triangle_free_circulant(n, shifts):
    """
    Check if circulant graph with given shifts is triangle-free.
    Triangle exists if there are a, b in shifts such that a-b mod n is also in shifts.
    """
    shift_set = set(s % n for s in shifts)
    shift_set.discard(0)
    
    shifts_list = list(shift_set)
    for i in range(len(shifts_list)):
        for j in range(len(shifts_list)):
            if i == j:
                continue
            diff = (shifts_list[i] - shifts_list[j]) % n
            if diff in shift_set:
                return False
    return True


def _try_circulant_construction(n, max_shifts=5):
    """
    Try to find a valid circulant graph on n vertices.
    Returns (adj, alpha) if found, else None.
    """
    candidates = list(range(1, n // 2 + 1))
    
    # Try smaller shift sets first (sparser graphs may have better alpha properties)
    for size in range(1, min(max_shifts, len(candidates)) + 1):
        if _remaining_time() < 5:
            return None
        
        for combo in combinations(candidates, size):
            if _remaining_time() < 2:
                return None
            
            # Build the full shift set
            full_shifts = set()
            for s in combo:
                full_shifts.add(s)
                full_shifts.add(n - s)
            full_shifts.discard(0)
            
            if not _is_triangle_free_circulant(n, full_shifts):
                continue
            
            adj = _build_circulant_graph(n, full_shifts)
            if adj is None:
                continue
            
            alpha = _max_independent_set_size_bitmask(adj, n)
            if alpha < 10:
                return adj, alpha
    
    return None


def _greedy_triangle_free_bitmask(n, rng):
    """
    Build a triangle-free graph greedily using bitmasks.
    """
    adj = [0] * n
    
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edges)
    
    for i, j in edges:
        if _remaining_time() < 2:
            break
        if adj[i] & adj[j]:
            continue
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)
    
    return adj


def _local_search_bitmask(adj, n, rng, max_iter=2000):
    """
    Improve graph by adding edges to reduce independence number.
    """
    adj = adj[:]
    alpha = _max_independent_set_size_bitmask(adj, n)
    
    if alpha < 10:
        return adj, alpha
    
    for _ in range(max_iter):
        if _remaining_time() < 3:
            break
        
        addable = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    if not (adj[i] & adj[j]):
                        addable.append((i, j))
        
        if not addable:
            break
        
        improved = False
        for _ in range(10):
            i, j = rng.choice(addable)
            
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
            new_alpha = _max_independent_set_size_bitmask(adj, n)
            if new_alpha < alpha:
                alpha = new_alpha
                improved = True
                if alpha < 10:
                    break
            else:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
        
        if alpha < 10:
            break
        if not improved:
            pass

    return adj, alpha


def construct():
    """
    Construct a triangle-free graph on n vertices (1 <= n <= 41) with no independent set of 10.
    Maximize n.
    """
    best_adj = None
    best_n = 0
    best_alpha = float('inf')
    
    # Primary strategy: systematic circulant enumeration
    # Try n from high to low
    for n_target in range(40, 25, -1):
        if _remaining_time() < 10:
            break
        
        result = _try_circulant_construction(n_target, max_shifts=5)
        if result is not None:
            adj_res, alpha_res = result
            if alpha_res < best_alpha:
                best_adj = adj_res
                best_n = n_target
                best_alpha = alpha_res
            if alpha_res < 10:
                break
    
    # If circulant didn't work, try random greedy + local search as fallback
    if best_adj is None or best_n < 30:
        for n_target in range(40, 25, -1):
            if _remaining_time() < 8:
                break
            
            found = False
            for seed in range(30):
                if _remaining_time() < 3:
                    break
                
                rng = random.Random(seed)
                adj = _greedy_triangle_free_bitmask(n_target, rng)
                alpha = _max_independent_set_size_bitmask(adj, n_target)
                
                if alpha < 10:
                    if alpha < best_alpha:
                        best_adj = adj
                        best_n = n_target
                        best_alpha = alpha
                    found = True
                    break
                
                # Try local search if time permits
                if _remaining_time() > 5:
                    adj_ls, alpha_ls = _local_search_bitmask(adj, n_target, rng, max_iter=500)
                    if alpha_ls < 10:
                        if alpha_ls < best_alpha:
                            best_adj = adj_ls
                            best_n = n_target
                            best_alpha = alpha_ls
                        found = True
                        break
                    if alpha_ls < best_alpha:
                        best_adj = adj_ls
                        best_n = n_target
                        best_alpha = alpha_ls
            
            if found:
                break
    
    # Fallback: small known construction
    if best_adj is None:
        n = 26
        rng = random.Random(42)
        adj = _greedy_triangle_free_bitmask(n, rng)
        best_adj = adj
        best_n = n
    
    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
