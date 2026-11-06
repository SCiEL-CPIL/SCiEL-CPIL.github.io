import time
import random
from itertools import combinations
from collections import defaultdict

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560

# Precompute triple index lookup
_triple_idx = {}
for _i, _t in enumerate(ALL_TRIPLES):
    _triple_idx[_t] = _i

def _tidx(a, b, c):
    """Get index of triple {a,b,c} (all distinct)."""
    if a > b: a, b = b, a
    if b > c: b, c = c, b
    if a > b: a, b = b, a
    return _triple_idx[(a, b, c)]

# Precompute all K(2,2,2) configurations
def _precompute_k222s():
    k222_list = []
    seen = set()
    points = list(range(V))
    for a1 in range(V):
        for a2 in range(a1+1, V):
            for b1 in range(V):
                if b1 == a1 or b1 == a2:
                    continue
                for b2 in range(V):
                    if b2 == a1 or b2 == a2 or b2 == b1:
                        continue
                    for c1 in range(V):
                        if c1 in (a1, a2, b1, b2):
                            continue
                        for c2 in range(V):
                            if c2 in (a1, a2, b1, b2, c1):
                                continue
                            triples_in_k222 = [
                                _tidx(a1, b1, c1),
                                _tidx(a1, b1, c2),
                                _tidx(a1, b2, c1),
                                _tidx(a1, b2, c2),
                                _tidx(a2, b1, c1),
                                _tidx(a2, b1, c2),
                                _tidx(a2, b2, c1),
                                _tidx(a2, b2, c2)
                            ]
                            key = frozenset(triples_in_k222)
                            if key not in seen:
                                seen.add(key)
                                k222_list.append(tuple(sorted(triples_in_k222)))
    return k222_list

K222_LIST = _precompute_k222s()
N_K222 = len(K222_LIST)

# Build conflict graph: two triples conflict if they share a K(2,2,2)
# i.e., if they can be part of the same K(2,2,2)
# Actually, we need: two triples conflict if adding both could complete a K(2,2,2)
# More precisely: triple t1 and t2 conflict if there exists a K(2,2,2) containing both t1 and t2

# For each triple, find which K(2,2,2)s it belongs to
triple_to_k222 = [[] for _ in range(N_TRIPLES)]
for k_idx, k222 in enumerate(K222_LIST):
    for t_idx in k222:
        triple_to_k222[t_idx].append(k_idx)

# Two triples conflict if they share at least one K(2,2,2)
# Build adjacency as bitsets
N_WORDS = (N_TRIPLES + 63) // 64
adj = [[0] * N_WORDS for _ in range(N_TRIPLES)]

for k_idx, k222 in enumerate(K222_LIST):
    for i in range(len(k222)):
        for j in range(i+1, len(k222)):
            t1, t2 = k222[i], k222[j]
            adj[t1][t2 >> 6] |= (1 << (t2 & 63))
            adj[t2][t1 >> 6] |= (1 << (t1 & 63))

def _bitset_and(a, b):
    return [a[i] & b[i] for i in range(N_WORDS)]

def _bitset_or(a, b):
    return [a[i] | b[i] for i in range(N_WORDS)]

def _bitset_andnot(a, b):
    return [a[i] & ~b[i] for i in range(N_WORDS)]

def _bitset_popcount(a):
    return sum(bin(w).count('1') for w in a)

def _bitset_is_empty(a):
    return all(w == 0 for w in a)

def _bitset_first_bit(a):
    for i in range(N_WORDS):
        if a[i]:
            # Find first set bit
            bit = a[i] & (-a[i])
            pos = i * 64 + (bit.bit_length() - 1)
            return pos
    return -1

def _make_bitset(indices):
    bs = [0] * N_WORDS
    for idx in indices:
        bs[idx >> 6] |= (1 << (idx & 63))
    return bs

def _bitset_to_list(bs):
    result = []
    for i in range(N_WORDS):
        w = bs[i]
        while w:
            bit = w & (-w)
            result.append(i * 64 + (bit.bit_length() - 1))
            w ^= bit
    return result

# Greedy coloring for upper bound (Tomita's algorithm)
def _color_sort(P):
    """Color the vertices in P using greedy coloring, return order and colors."""
    # P is a bitset
    P_list = _bitset_to_list(P)
    if not P_list:
        return [], []
    
    # Greedy coloring
    color = {}
    order = []
    remaining = set(P_list)
    current_color = 1
    color_set = set()
    
    while remaining:
        # Pick vertex with smallest degree in remaining (or just first)
        # For speed, just pick the first one
        v = min(remaining)
        remaining.remove(v)
        color[v] = current_color
        order.append(v)
        # Remove v from consideration for this color
        color_set.add(v)
        # Continue adding vertices that don't conflict with color_set
        # Actually, greedy coloring: assign smallest available color
        # Let's do a simpler greedy: process in order, assign smallest color not used by neighbors
    
    # Better greedy coloring
    color = {}
    order = []
    for v in P_list:
        # Find smallest color not used by neighbors of v that are already colored
        used_colors = set()
        adj_v = adj[v]
        for u in P_list:
            if u > v and ((adj_v[u >> 6] >> (u & 63)) & 1):
                if u in color:
                    used_colors.add(color[u])
        c = 1
        while c in used_colors:
            c += 1
        color[v] = c
        order.append(v)
    
    # Sort by color descending
    order.sort(key=lambda x: color[x], reverse=True)
    max_color = max(color.values()) if color else 0
    return order, color, max_color

# Branch and bound for maximum independent set
def _max_independent_set(P, deadline, best_size_ref):
    """Find maximum independent set in the subgraph induced by P (bitset).
    Returns (size, vertex_list)."""
    P_size = _bitset_popcount(P)
    if P_size == 0:
        return 0, []
    if time.time() >= deadline:
        return 0, []
    
    # If P_size <= best_size, prune
    if P_size <= best_size_ref[0]:
        return 0, []
    
    # Get vertices in P
    P_list = _bitset_to_list(P)
    
    # Try each vertex in P
    best_size = 0
    best_set = []
    
    # Order vertices by degree (greedy)
    def degree_in_P(v):
        adj_v = adj[v]
        cnt = 0
        for u in P_list:
            if u != v and ((adj_v[u >> 6] >> (u & 63)) & 1):
                cnt += 1
        return cnt
    
    P_list_sorted = sorted(P_list, key=degree_in_P)
    
    for v in P_list_sorted:
        if time.time() >= deadline:
            break
        if P_size - 1 <= best_size:
            break
        
        # Include v: find independent set in P \ N[v]
        # N[v] = v and all neighbors of v
        Nv = [0] * N_WORDS
        Nv[v >> 6] |= (1 << (v & 63))
        adj_v = adj[v]
        for u in P_list:
            if u != v and ((adj_v[u >> 6] >> (u & 63)) & 1):
                Nv[u >> 6] |= (1 << (u & 63))
        
        P_new = _bitset_andnot(P, Nv)
        
        size, vertices = _max_independent_set(P_new, deadline, best_size_ref)
        size += 1
        if size > best_size:
            best_size = size
            best_set = [v] + vertices
            best_size_ref[0] = max(best_size_ref[0], best_size)
        
        # Exclude v
        P_excl = _bitset_andnot(P, _make_bitset([v]))
        P_size = _bitset_popcount(P_excl)
        P = P_excl
    
    return best_size, best_set

# Greedy independent set
def _greedy_is(P, deadline):
    """Greedy: repeatedly pick vertex with smallest degree in current P."""
    P_list = _bitset_to_list(P)
    if not P_list:
        return 0, []
    
    selected = []
    remaining = set(P_list)
    
    while remaining:
        if time.time() >= deadline:
            break
        # Pick vertex with smallest degree in remaining
        best_v = None
        best_deg = float('inf')
        for v in remaining:
            deg = 0
            adj_v = adj[v]
            for u in remaining:
                if u != v and ((adj_v[u >> 6] >> (u & 63)) & 1):
                    deg += 1
            if deg < best_deg:
                best_deg = deg
                best_v = v
            if deg == 0:
                break
        
        if best_v is None:
            break
        
        selected.append(best_v)
        remaining.remove(best_v)
        # Remove neighbors of best_v from remaining
        adj_v = adj[best_v]
        to_remove = set()
        for u in remaining:
            if ((adj_v[u >> 6] >> (u & 63)) & 1):
                to_remove.add(u)
        remaining -= to_remove
    
    return len(selected), selected

# Local search: try to improve by swapping
def _local_search(selected, deadline):
    """Try to improve the independent set by swaps."""
    selected_set = set(selected)
    improved = True
    iter_count = 0
    max_iters = 50
    
    while improved and iter_count < max_iters and time.time() < deadline:
        improved = False
        iter_count += 1
        
        # Try adding each unselected vertex
        unselected = [i for i in range(N_TRIPLES) if i not in selected_set]
        random.shuffle(unselected)
        
        for v in unselected:
            if time.time() >= deadline:
                break
            # Check if v can be added (no conflict with selected)
            adj_v = adj[v]
            conflict = False
            for u in selected_set:
                if ((adj_v[u >> 6] >> (u & 63)) & 1):
                    conflict = True
                    break
            if not conflict:
                selected_set.add(v)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps: remove one, add one
        for v in unselected:
            if time.time() >= deadline:
                break
            adj_v = adj[v]
            # Find conflicts
            conflicts = [u for u in selected_set if ((adj_v[u >> 6] >> (u & 63)) & 1)]
            if not conflicts:
                continue
            
            # Try removing each conflict
            for c in conflicts:
                if time.time() >= deadline:
                    break
                # Remove c, add v
                new_set = selected_set - {c} | {v}
                # Check if v has no conflict with new_set
                ok = True
                for u in new_set:
                    if u != v and ((adj_v[u >> 6] >> (u & 63)) & 1):
                        ok = False
                        break
                if ok:
                    selected_set = new_set
                    improved = True
                    break
            if improved:
                break
    
    return list(selected_set)

def _simulated_annealing(selected, deadline):
    """SA on the independent set."""
    selected_set = set(selected)
    T = 10.0
    best_set = set(selected)
    best_size = len(selected)
    
    steps = 0
    max_steps = 20000
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        if random.random() < 0.5:
            # Add a random unselected vertex
            unselected = [i for i in range(N_TRIPLES) if i not in selected_set]
            if not unselected:
                continue
            v = random.choice(unselected)
            adj_v = adj[v]
            conflicts = [u for u in selected_set if ((adj_v[u >> 6] >> (u & 63)) & 1)]
            
            if not conflicts:
                selected_set.add(v)
                if len(selected_set) > best_size:
                    best_size = len(selected_set)
                    best_set = set(selected_set)
            else:
                # Would need to remove conflicts
                delta = 1 - len(conflicts)
                if random.random() < min(1.0, 2.71828 ** (-delta / T)):
                    for c in conflicts:
                        selected_set.remove(c)
                    selected_set.add(v)
                    if len(selected_set) > best_size:
                        best_size = len(selected_set)
                        best_set = set(selected_set)
        else:
            # Remove a random selected vertex
            if not selected_set:
                continue
            v = random.choice(list(selected_set))
            selected_set.remove(v)
        
        T *= 0.999
        if T < 0.1:
            T = 0.1
    
    return list(best_set)

def construct():
    start = time.time()
    deadline = start + 115  # Safety margin
    
    best_selected = []
    best_size = 0
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Random greedy initialization
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        
        # Greedy: add triples in order if no conflict
        selected = []
        selected_set = set()
        for idx in order:
            if time.time() >= deadline:
                break
            adj_idx = adj[idx]
            conflict = False
            for u in selected_set:
                if ((adj_idx[u >> 6] >> (u & 63)) & 1):
                    conflict = True
                    break
            if not conflict:
                selected.append(idx)
                selected_set.add(idx)
        
        # Local search
        selected = _local_search(selected, deadline)
        
        # SA
        selected = _simulated_annealing(selected, deadline)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected
        
        # Try B&B on a subset if time permits
        if time.time() < deadline - 30 and best_size < 200:
            # Create a bitset of a random subset
            subset_size = min(100, N_TRIPLES)
            subset = random.sample(range(N_TRIPLES), subset_size)
            P = _make_bitset(subset)
            best_size_ref = [best_size]
            size, vertices = _max_independent_set(P, deadline, best_size_ref)
            if size > best_size:
                best_size = size
                best_selected = vertices
        
        if time.time() > deadline - 5:
            break
    
    result = [list(ALL_TRIPLES[i]) for i in best_selected]
    return result

random.seed(42)
