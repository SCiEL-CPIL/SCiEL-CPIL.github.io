import time
import random
from itertools import combinations

_START = time.time()
DEADLINE = 118.0  # 150s limit, ~32s safety margin

def _time_left():
    return DEADLINE - (time.time() - _START)

def _can_add_edge(adj, neighbors, u, v, num_colors):
    """Check if edge (u,v) can be added with some color, return the color or -1."""
    forbidden = set()
    for a in neighbors[u]:
        if a == v:
            continue
        ca = adj[u][a]
        for b in neighbors[v]:
            if b == u or b == a:
                continue
            if adj[a][b] != -1:
                cb = adj[a][b]
                cv = adj[b][v]
                if ca == cb or cb == cv or ca == cv:
                    return -1
                forbidden.add(ca)
                forbidden.add(cb)
                forbidden.add(cv)
    if len(forbidden) >= num_colors:
        return -1
    return min(c for c in range(num_colors) if c not in forbidden)

def _can_recolor(adj, neighbors, u, v, new_color, num_colors):
    """Check if edge (u,v) can be recolored to new_color."""
    # Check all 4-cycles containing (u,v)
    for a in neighbors[u]:
        if a == v: continue
        ca = adj[u][a]
        if ca == new_color:
            # If ca == new_color, then in cycle u-a-...-v-u, we need to ensure no conflict.
            # Actually, if u-a is ca, and we change u-v to new_color, we must check:
            # 1. No other edge in the cycle has new_color.
            # 2. No other edge in the cycle has ca? No, ca is on u-a.
            # The constraint is that all 4 edges in any 4-cycle must be distinct.
            # Cycle: u-a-b-v-u. Edges: (u,a)=ca, (a,b)=cb, (b,v)=cv, (v,u)=new_color.
            # Distinctness: ca != cb, cb != cv, cv != new_color, new_color != ca.
            # So if ca == new_color, it's immediately invalid for any such cycle where (a,b) and (b,v) exist.
            # Wait, if ca == new_color, then edge (u,a) and (u,v) have same color.
            # They are adjacent, but the constraint is on 4-cycles.
            # However, if there is a 4-cycle u-a-b-v-u, then ca (u-a) and new_color (u-v) are in the cycle.
            # If ca == new_color, the cycle is not rainbow.
            # So we must check if there exists ANY b such that (a,b) and (b,v) are edges.
            # If such b exists, re-coloring to new_color is invalid if ca == new_color.
            # But wait, if ca == new_color, it doesn't matter what cb, cv are. The cycle has two edges with same color.
            # So if there is ANY common neighbor b of a and v (other than u), and (a,b) and (b,v) exist, then it's invalid.
            # Actually, we just need to check all 4-cycles containing (u,v).
            pass
        for b in neighbors[v]:
            if b == u or b == a: continue
            if adj[a][b] != -1:
                cb = adj[a][b]
                cv = adj[b][v]
                # Cycle edges: (u,a)=ca, (a,b)=cb, (b,v)=cv, (v,u)=new_color
                if ca == cb or cb == cv or cv == new_color or new_color == ca:
                    return False
                # Also check if new_color conflicts with anything else? No, just these 4.
    return True

# Precompute all 4-cycles once (independent of graph)
def _precompute_cycles(n):
    cycles = []
    for combo in combinations(range(n), 4):
        a, b, c, d = combo
        # 3 ways to make a 4-cycle on 4 labeled vertices
        cycles.append(((a, b), (b, c), (c, d), (d, a)))
        cycles.append(((a, b), (b, d), (d, c), (c, a)))
        cycles.append(((a, c), (c, b), (b, d), (d, a)))
    return cycles

_CYCLES = _precompute_cycles(20)

def _validate(edges):
    """Return True iff every 4-cycle that is fully present is rainbow."""
    edge_set = {}
    for u, v, c in edges:
        edge_set[(u, v)] = c
        edge_set[(v, u)] = c
    for cyc in _CYCLES:
        cols = []
        ok = True
        for (u, v) in cyc:
            c = edge_set.get((u, v))
            if c is None:
                ok = False
                break
            cols.append(c)
        if ok and len(set(cols)) != 4:
            return False
    return True

def _greedy_build(seed, initial_edges, order_type='random'):
    """Build a valid coloured graph, optionally starting from initial_edges."""
    n = 20
    num_colors = 20
    rng = random.Random(seed)
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]

    adj = [[-1] * n for _ in range(n)]
    neighbors = [set() for _ in range(n)]
    edges = []

    if initial_edges:
        for u, v, c in initial_edges:
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            edges.append([u, v, c])
        existing = set((min(u, v), max(u, v)) for u, v, _ in edges)
        remaining = [e for e in all_edges if e not in existing]
    else:
        remaining = all_edges[:]

    if order_type == 'random':
        rng.shuffle(remaining)
    elif order_type == 'reverse':
        remaining.reverse()
    elif order_type == 'degree_heuristic':
        # Try edges between low-degree vertices first (they have fewer constraints)
        deg = [len(neighbors[i]) for i in range(n)]
        def edge_key(e):
            return deg[e[0]] + deg[e[1]]
        remaining.sort(key=edge_key)

    for (u, v) in remaining:
        color = _can_add_edge(adj, neighbors, u, v, num_colors)
        if color >= 0:
            adj[u][v] = color
            adj[v][u] = color
            neighbors[u].add(v)
            neighbors[v].add(u)
            edges.append([u, v, color])
    return edges

def _bipartite_seed(rng):
    """Build a K_{10,10}-style bipartite core with a linear colouring,
    removing edges to fix any non-rainbow 4-cycles."""
    n = 20
    A = list(range(10))
    B = list(range(10, 20))
    a = rng.randint(0, 19)
    b = rng.randint(1, 19)
    c0 = rng.randint(0, 19)

    def col(u, v):
        return (a * u + b * v + c0) % 20

    ec = {}
    for u in A:
        for v in B:
            ec[(u, v)] = col(u, v)
            ec[(v, u)] = col(u, v)

    bad = []
    for i in range(10):
        for j in range(i + 1, 10):
            u1, u2 = A[i], A[j]
            for k in range(10):
                for l in range(k + 1, 10):
                    v1, v2 = B[k], B[l]
                    cols = {ec[(u1, v1)], ec[(u2, v1)], ec[(u2, v2)], ec[(u1, v2)]}
                    if len(cols) < 4:
                        bad.append([(u1, v1), (u2, v1), (u2, v2), (u1, v2)])

    remove = set()
    rem = bad[:]
    while rem and _time_left() > 15.0:
        best_e, best_hits = None, 0
        cands = set()
        for cyc in rem:
            for e in cyc:
                if e not in remove:
                    cands.add(e)
        if not cands:
            break
        for e in cands:
            hits = sum(1 for cyc in rem if e in cyc)
            if hits > best_hits:
                best_hits = hits
                best_e = e
        if best_e is None:
            break
        remove.add(best_e)
        rem = [cyc for cyc in rem if best_e not in cyc]

    out = []
    for u in A:
        for v in B:
            if (u, v) not in remove:
                out.append([u, v, ec[(u, v)]])
    return out

def construct():
    rng = random.Random()
    best_edges = []
    best_count = 0

    # Phase 1: random greedy restarts (no seed)
    for ot in ('random', 'natural', 'reverse', 'degree_heuristic'):
        for i in range(40):
            if _time_left() < 25.0:
                break
            edges = _greedy_build(i * 7 + 1, None, ot)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges

    # Phase 2: bipartite-seeded greedy
    for i in range(15):
        if _time_left() < 25.0:
            break
        seed_edges = _bipartite_seed(rng)
        for ot in ('random', 'degree_heuristic'):
            edges = _greedy_build(1000 + i * 2 + (0 if ot == 'random' else 1), seed_edges, ot)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges

    # Phase 3: local search with smarter moves
    current = best_edges[:]
    current_adj = [[-1] * 20 for _ in range(20)]
    current_neighbors = [set() for _ in range(20)]
    for u, v, c in current:
        current_adj[u][v] = c
        current_adj[v][u] = c
        current_neighbors[u].add(v)
        current_neighbors[v].add(u)

    def rebuild_adj(edges):
        adj = [[-1] * 20 for _ in range(20)]
        neighbors = [set() for _ in range(20)]
        for u, v, c in edges:
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
        return adj, neighbors

    # Precompute missing edges list for fast access
    all_edges_list = [(u, v) for u in range(20) for v in range(u + 1, 20)]
    
    for it in range(5000):
        if _time_left() < 10.0:
            break
        
        # Strategy:
        # 1. Try to add a missing edge (if possible)
        # 2. If not, try to recolor an existing edge to allow adding a missing edge
        # 3. If not, try to recolor an existing edge to "clean up" (reduce constraint density)
        
        # Get current missing edges
        existing_set = set((min(u, v), max(u, v)) for u, v, _ in current)
        missing_edges = [e for e in all_edges_list if e not in existing_set]
        
        added = False
        
        # Attempt 1: Direct addition of a missing edge
        if missing_edges:
            # Sort missing edges by "potential" (e.g., random or by degree of endpoints)
            # For speed, just pick random subset or all if few
            if len(missing_edges) < 50:
                candidates = missing_edges
            else:
                candidates = random.sample(missing_edges, 50)
            
            for u, v in candidates:
                color = _can_add_edge(current_adj, current_neighbors, u, v, 20)
                if color >= 0:
                    current_adj[u][v] = color
                    current_adj[v][u] = color
                    current_neighbors[u].add(v)
                    current_neighbors[v].add(u)
                    current.append([u, v, color])
                    existing_set.add((min(u, v), max(u, v)))
                    missing_edges.remove((u, v)) # Update local list
                    added = True
                    if len(current) > best_count:
                        best_count = len(current)
                        best_edges = current[:]
                    break
        
        if not added:
            # Attempt 2: Recolor an existing edge to enable adding a missing edge
            # Pick a random missing edge that is "close" to being addable
            # Heuristic: Pick a missing edge (u,v) where _can_add_edge fails only due to 1-2 forbidden colors
            # For simplicity, pick a random missing edge and try to recolor neighbors
            
            if missing_edges:
                target_u, target_v = random.choice(missing_edges)
                
                # Identify edges that constrain (target_u, target_v)
                # These are edges (u, a) or (v, b) where a, b are common neighbors
                # Or more simply, just try to recolor random edges incident to u or v
                
                incident_u = [e for e in current if e[0] == target_u or e[1] == target_u]
                incident_v = [e for e in current if e[0] == target_v or e[1] == target_v]
                
                # Combine and shuffle
                candidates_to_recolor = incident_u + incident_v
                random.shuffle(candidates_to_recolor)
                
                improved = False
                for e in candidates_to_recolor[:10]: # Limit attempts
                    u, v, c = e
                    # Try to recolor this edge to a new color
                    # What new color? Try all 20 colors?
                    # We want a color that helps (target_u, target_v) be added.
                    # Ideally, the new color of (u,v) should not be in the forbidden set of (target_u, target_v)
                    # But (target_u, target_v) forbidden set depends on the graph state.
                    
                    # Let's just try to recolor to any valid color that is different from current
                    # and see if it allows adding (target_u, target_v)
                    
                    old_color = current_adj[u][v]
                    # Try a few random new colors
                    for _ in range(3):
                        new_color = random.randint(0, 19)
                        if new_color == old_color:
                            continue
                        if _can_recolor(current_adj, current_neighbors, u, v, new_color, 20):
                            # Apply recolor
                            current_adj[u][v] = new_color
                            current_adj[v][u] = new_color
                            # Update current list
                            for i, ce in enumerate(current):
                                if (ce[0] == u and ce[1] == v) or (ce[0] == v and ce[1] == u):
                                    current[i] = [u, v, new_color]
                                    break
                            
                            # Check if we can now add (target_u, target_v)
                            new_edge_color = _can_add_edge(current_adj, current_neighbors, target_u, target_v, 20)
                            if new_edge_color >= 0:
                                current_adj[target_u][target_v] = new_edge_color
                                current_adj[target_v][target_u] = new_edge_color
                                current_neighbors[target_u].add(target_v)
                                current_neighbors[target_v].add(target_u)
                                current.append([target_u, target_v, new_edge_color])
                                existing_set.add((min(target_u, target_v), max(target_u, target_v)))
                                if len(current) > best_count:
                                    best_count = len(current)
                                    best_edges = current[:]
                                improved = True
                                break
                            else:
                                # Revert recolor
                                current_adj[u][v] = old_color
                                current_adj[v][u] = old_color
                                for i, ce in enumerate(current):
                                    if (ce[0] == u and ce[1] == v) or (ce[0] == v and ce[1] == u):
                                        current[i] = [u, v, old_color]
                                        break
                    if improved:
                        break

    # Final safety: guarantee validity of what we return
    if not _validate(best_edges):
        # Fallback: rebuild a guaranteed-valid solution
        for ot in ('random', 'natural', 'reverse'):
            for i in range(20):
                if _time_left() < 2.0:
                    break
                e = _greedy_build(5000 + i, None, ot)
                if _validate(e) and len(e) > best_count:
                    best_count = len(e)
                    best_edges = e
        # If still invalid (shouldn't happen), return a trivial valid graph
        if not _validate(best_edges):
            best_edges = [[0, v, 0] for v in range(1, 20)]

    return best_edges
