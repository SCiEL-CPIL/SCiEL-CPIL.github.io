import time
import random
import math

N = 20
C = 20
DEADLINE = 145.0


def _validate(edges):
    """Return True iff every 4-cycle is rainbow."""
    if not edges:
        return True
    adj = {}
    neigh = [set() for _ in range(N)]
    for u, v, c in edges:
        adj[(u, v)] = c
        adj[(v, u)] = c
        neigh[u].add(v)
        neigh[v].add(u)

    for u in range(N):
        for v in neigh[u]:
            if u >= v:
                continue
            common = neigh[u] & neigh[v]
            common.discard(u)
            common.discard(v)
            common_list = list(common)
            for i in range(len(common_list)):
                x = common_list[i]
                for j in range(i + 1, len(common_list)):
                    y = common_list[j]
                    cols = {adj[(u, x)], adj[(x, v)], adj[(v, y)], adj[(y, u)]}
                    if len(cols) != 4:
                        return False
    return True


def _check_add(adj, neigh, u, v, c):
    """Check if adding edge (u,v) with color c maintains rainbow 4-cycles.
    Only considers 4-cycles that use the new edge (u,v): u-v-x-y-u."""
    # x in N(v), y in N(u), x!=u, y!=v, x!=y, edge(x,y) exists
    # Cycle: u-v-x-y-u
    # Colors: c, adj[v,x], adj[x,y], adj[y,u]
    for x in neigh[v]:
        if x == u:
            continue
        for y in neigh[u]:
            if y == v or y == x:
                continue
            if (x, y) not in adj and (y, x) not in adj:
                continue
            
            c_vx = adj[(v, x)]
            c_xy = adj[(x, y)] if (x, y) in adj else adj[(y, x)]
            c_yu = adj[(y, u)]
            
            # Check distinctness
            if c == c_vx or c == c_xy or c == c_yu:
                return False
            if c_vx == c_xy or c_vx == c_yu:
                return False
            if c_xy == c_yu:
                return False
    return True


def _check_swap(adj, neigh, u1, v1, u2, v2, new_c1, new_c2):
    """
    Check if swapping colours of edges (u1,v1) and (u2,v2) maintains validity.
    Edges (u1,v1) and (u2,v2) must exist.
    They must be distinct edges.
    new_c1 is the colour for (u1,v1), new_c2 for (u2,v2).
    """
    if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
        return False
    
    c1_old = adj[(u1, v1)]
    c2_old = adj[(u2, v2)]
    
    if new_c1 == new_c2:
        # If they swap to the same colour, and they share a vertex, it might be a proper edge coloring violation?
        # Actually, we only care about 4-cycles. But if they share a vertex, say u1=u2, then (u1,v1) and (u1,v2) would have same colour.
        # The problem statement doesn't explicitly forbid adjacent edges having the same colour, ONLY that 4-cycles are rainbow.
        # Wait, "edge-coloured ... so every 4-cycle is rainbow". It does NOT say "proper edge coloring".
        # So adjacent edges CAN have the same colour.
        # However, if (u1,v1) and (u2,v2) are in a 4-cycle together, they must have distinct colours.
        # Let's just check all 4-cycles involving u1, v1, u2, v2.
        pass

    # We need to check all 4-cycles that involve any of the 4 vertices u1, v1, u2, v2
    # and are affected by the colour change.
    # A 4-cycle is affected if it contains edge (u1,v1) or (u2,v2).
    
    # Helper to check a specific 4-cycle u-a-v-b-u with given colours
    def check_cycle(u, a, v, b, cu_a, ca_v, cv_b, cb_u):
        if len({cu_a, ca_v, cv_b, cb_u}) != 4:
            return False
        return True

    # Get neighbours
    # We will iterate over all 4-cycles that include (u1,v1) or (u2,v2)
    
    # Cycles including (u1, v1): u1-v1-x-y-u1
    # x in N(v1), y in N(u1), x!=u1, y!=v1, x!=y, edge(x,y)
    for x in neigh[v1]:
        if x == u1: continue
        for y in neigh[u1]:
            if y == v1 or y == x: continue
            if (x, y) not in adj and (y, x) not in adj: continue
            
            # Determine colours of the 4 edges
            # Edge (u1, v1) has new_c1
            # Edge (v1, x)
            if (v1, x) == (u2, v2) or (v1, x) == (v2, u2):
                c_vx = new_c2
            else:
                c_vx = adj[(v1, x)]
            
            # Edge (x, y)
            if (x, y) == (u2, v2) or (x, y) == (v2, u2):
                c_xy = new_c2
            elif (y, x) == (u2, v2) or (y, x) == (v2, u2):
                c_xy = new_c2
            else:
                c_xy = adj[(x, y)] if (x, y) in adj else adj[(y, x)]
            
            # Edge (y, u1)
            if (y, u1) == (u2, v2) or (y, u1) == (v2, u2):
                c_yu = new_c2
            else:
                c_yu = adj[(y, u1)]
                
            if not check_cycle(u1, v1, x, y, new_c1, c_vx, c_xy, c_yu):
                return False

    # Cycles including (u2, v2): u2-v2-x-y-u2
    for x in neigh[v2]:
        if x == u2: continue
        for y in neigh[u2]:
            if y == v2 or y == x: continue
            if (x, y) not in adj and (y, x) not in adj: continue
            
            # Edge (u2, v2) has new_c2
            # Edge (v2, x)
            if (v2, x) == (u1, v1) or (v2, x) == (v1, u1):
                c_vx = new_c1
            else:
                c_vx = adj[(v2, x)]
            
            # Edge (x, y)
            if (x, y) == (u1, v1) or (x, y) == (v1, u1):
                c_xy = new_c1
            elif (y, x) == (u1, v1) or (y, x) == (v1, u1):
                c_xy = new_c1
            else:
                c_xy = adj[(x, y)] if (x, y) in adj else adj[(y, x)]
            
            # Edge (y, u2)
            if (y, u2) == (u1, v1) or (y, u2) == (v1, u1):
                c_yu = new_c1
            else:
                c_yu = adj[(y, u2)]
                
            if not check_cycle(u2, v2, x, y, new_c2, c_vx, c_xy, c_yu):
                return False

    return True


def _greedy_build(order, rng, time_budget):
    """Greedy build."""
    adj = {}
    neigh = [set() for _ in range(N)]
    edges = []

    for (u, v) in order:
        if time.time() > time_budget:
            break
        
        # Try colours in random order
        cols = list(range(C))
        rng.shuffle(cols)
        
        placed = False
        for c in cols:
            if _check_add(adj, neigh, u, v, c):
                adj[(u, v)] = c
                adj[(v, u)] = c
                neigh[u].add(v)
                neigh[v].add(u)
                edges.append([u, v, c])
                placed = True
                break
    return edges, adj, neigh


def _local_search_with_swaps(edges, adj, neigh, rng, time_budget):
    """
    Local search that tries to add edges and also swaps colours of existing edges
    to resolve conflicts for new edges.
    """
    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    # Precompute existing edges set for quick lookup
    existing = set()
    for u, v, c in edges:
        existing.add((min(u, v), max(u, v)))
        
    # List of edges as tuples (u, v) for swapping
    # We need to be able to find edges quickly
    # adj is the source of truth for colours
    
    # We will iterate through missing edges and try to add them.
    # If we fail, we might try to swap colours.
    
    # To make swaps effective, we need to identify *why* an edge failed.
    # But _check_add just returns False.
    # Let's modify _check_add to return the conflicting cycle if it fails?
    # Or just try random swaps if stuck.
    
    # Strategy:
    # 1. Try to add all missing edges greedily.
    # 2. If no edges can be added, pick a random existing edge and a random valid swap (change colour to another unused/valid one) or swap with another edge.
    #    Actually, simple colour change: pick edge (u,v), try changing its colour to c' if valid.
    #    This is simpler than swapping two edges.
    
    def try_change_colour(u, v, new_c):
        # Check if changing (u,v) to new_c is valid
        # Must check all 4-cycles containing (u,v)
        old_c = adj[(u, v)]
        if new_c == old_c:
            return False
        
        # Temporarily assume new_c
        # Check cycles u-v-x-y-u
        for x in neigh[v]:
            if x == u: continue
            for y in neigh[u]:
                if y == v or y == x: continue
                if (x, y) not in adj and (y, x) not in adj: continue
                
                c_vx = adj[(v, x)]
                c_xy = adj[(x, y)] if (x, y) in adj else adj[(y, x)]
                c_yu = adj[(y, u)]
                
                # Colours: new_c, c_vx, c_xy, c_yu
                if len({new_c, c_vx, c_xy, c_yu}) != 4:
                    return False
        return True

    def do_change_colour(u, v, new_c):
        adj[(u, v)] = new_c
        adj[(v, u)] = new_c
        # Update edges list
        for i, e in enumerate(edges):
            if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                e[2] = new_c
                break

    # Main loop
    stuck_count = 0
    last_improvement = time.time()
    
    while time.time() < time_budget:
        # 1. Try to add missing edges
        missing = [(u, v) for (u, v) in all_pairs if (min(u, v), max(u, v)) not in existing]
        if not missing:
            break
            
        rng.shuffle(missing)
        added_any = False
        
        for u, v in missing:
            if time.time() > time_budget:
                break
            cols = list(range(C))
            rng.shuffle(cols)
            for c in cols:
                if _check_add(adj, neigh, u, v, c):
                    adj[(u, v)] = c
                    adj[(v, u)] = c
                    neigh[u].add(v)
                    neigh[v].add(u)
                    edges.append([u, v, c])
                    existing.add((min(u, v), max(u, v)))
                    added_any = True
                    stuck_count = 0
                    last_improvement = time.time()
                    break
            if added_any:
                # If we added one, restart the missing list scan? 
                # Or just continue? Restarting is safer to pick up new opportunities.
                # But it's expensive to rebuild missing list.
                # Let's just continue, but the next iteration will rebuild it.
                break
        
        if added_any:
            continue
            
        # 2. If no edges added, try local moves (colour changes)
        # Pick a random edge and try to change its colour
        if not edges:
            break
            
        idx = rng.randrange(len(edges))
        u, v, c = edges[idx]
        
        # Try changing colour of (u,v)
        cols = list(range(C))
        cols.remove(c)
        rng.shuffle(cols)
        
        moved = False
        for new_c in cols:
            if try_change_colour(u, v, new_c):
                do_change_colour(u, v, new_c)
                moved = True
                stuck_count = 0
                break
        
        if not moved:
            stuck_count += 1
            # If stuck for too long, maybe remove an edge?
            if stuck_count > 50:
                # Remove a random edge to break deadlock
                idx_rm = rng.randrange(len(edges))
                u_rm, v_rm, c_rm = edges.pop(idx_rm)
                del adj[(u_rm, v_rm)]
                del adj[(v_rm, u_rm)]
                neigh[u_rm].discard(v_rm)
                neigh[v_rm].discard(u_rm)
                existing.discard((min(u_rm, v_rm), max(u_rm, v_rm)))
                stuck_count = 0

    return edges


def construct():
    start = time.time()
    best_edges = []
    best_len = 0

    all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]

    # --- Phase 1: Multiple Greedy Constructions ---
    for trial in range(50):
        if time.time() > DEADLINE - 30:
            break
        rng = random.Random(trial * 137 + 42)
        order = all_pairs[:]
        rng.shuffle(order)
        
        edges, adj, neigh = _greedy_build(order, rng, DEADLINE - 30)
        
        if len(edges) > best_len:
            best_len = len(edges)
            best_edges = edges[:]
            
        # Early exit if very dense
        if best_len >= 180:
            break

    # --- Phase 2: Local Search with Swaps/Changes ---
    if best_edges and time.time() < DEADLINE - 10:
        adj = {}
        neigh = [set() for _ in range(N)]
        for u, v, c in best_edges:
            adj[(u, v)] = c
            adj[(v, u)] = c
            neigh[u].add(v)
            neigh[v].add(u)
            
        rng = random.Random(12345)
        budget = DEADLINE - 5
        improved = _local_search_with_swaps(best_edges[:], adj, neigh, rng, budget)
        
        if len(improved) > best_len:
            best_len = len(improved)
            best_edges = improved

    if not best_edges:
        return [[0, v, v] for v in range(1, N)]
    return best_edges


if __name__ == "__main__":
    res = construct()
    print("edges:", len(res))
    print("valid:", _validate(res))
