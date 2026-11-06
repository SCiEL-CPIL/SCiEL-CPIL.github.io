"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

DEADLINE_MARGIN = 10  # seconds before hard limit
HARD_LIMIT = 150      # seconds

def _make_paley(n):
    """Paley graph on n vertices (n must be prime ≡ 1 mod 4)."""
    squares = {x * x % n for x in range(1, n)}
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n in squares:
                edges.append([i, j])
    return {'n': n, 'edges': edges}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _has_k4(adj, n):
    """Check if graph has a K4."""
    for i in range(n):
        for j in range(i + 1, n):
            if j in adj[i]:
                for k in range(j + 1, n):
                    if k in adj[i] and k in adj[j]:
                        for l in range(k + 1, n):
                            if l in adj[i] and l in adj[j] and l in adj[k]:
                                return True
    return False

def _has_indep_6(adj, n):
    """Check if graph has an independent set of size 6."""
    comp = [set(range(n)) - adj[i] for i in range(n)]
    # Use complement: independent set in G = clique in complement
    # Try all 6-tuples
    for combo in combinations(range(n), 6):
        ok = True
        for a in range(5):
            if combo[a+1] in adj[combo[a]]:
                ok = False
                break
        if ok:
            return True
    return False

def _is_valid(n, edges):
    adj = _build_adj(n, edges)
    if _has_k4(adj, n):
        return False
    if _has_indep_6(adj, n):
        return False
    return True

def _try_extend(adj, n, new_vertex, time_limit):
    """
    Try to assign edges from new_vertex to existing vertices.
    Returns list of neighbors or None if impossible.
    Uses backtracking with pruning.
    """
    start = time.time()
    
    # For each existing vertex, determine constraints:
    # - If we connect new_vertex to v, then for any pair (u,w) where u,w are both neighbors of v in G,
    #   we cannot have new_vertex connected to all of {u,w,v} forming K4... actually:
    #   K4 constraint: no 4-clique. If new_vertex connects to v, and u,w are neighbors of v,
    #   then new_vertex cannot connect to both u and w if u~w (would form K4 with v).
    #   More precisely: {new, v, u, w} is K4 iff new~v, new~u, new~w, v~u, v~w, u~w.
    #   So if v~u and v~w and u~w, then new cannot connect to all of {v,u,w}.
    #   
    #   Simpler: if new connects to v, then among neighbors of v, new can connect to at most
    #   a set with no triangle containing v... wait, new+v+u+w forms K4 only if all 6 edges present.
    #   Since v~u and v~w are given, we need new~v, new~u, new~w, u~w.
    #   So if u~w and both are neighbors of v, new cannot connect to all three {v,u,w}.
    #
    # Independent set constraint: new + 5 others form independent set iff new is not connected to any of them.
    #   So if there exists a set S of 5 vertices that are mutually non-adjacent, and new is not connected to any of S,
    #   that's an independent 6-set. To prevent this, for every independent 5-set S, new must connect to at least one vertex in S.
    
    # Strategy: backtracking assignment
    # Variables: for each vertex 0..n-1, whether new connects to it.
    # Constraints:
    # 1. No K4: for any triangle {a,b,c} in G (a~b, b~c, c~a), new cannot connect to all of a,b,c.
    #    Also for any edge {a,b} in G, if new connects to a and b, then for any c adjacent to both a and b,
    #    new cannot connect to c. (Because {new,a,b,c} would be K4 if c~a and c~b.)
    #    More generally: if new connects to S, then S must be an independent set in G (no two vertices in S are adjacent).
    #    Wait no - S just needs to not contain a triangle that with new forms K4.
    #    Actually: {new} ∪ S is a clique iff S is a clique in G. So we need S to not contain any triangle.
    #    I.e., the neighborhood of new must be triangle-free in G.
    #
    # 2. No independent 6-set: for any independent 5-set S in G, new must be adjacent to at least one vertex in S.
    #    Equivalently: the non-neighbors of new must not contain an independent 5-set in G.
    
    # Let's use: N = set of vertices new connects to.
    # Constraint A: N is triangle-free in G (no triangle fully contained in N).
    # Constraint B: V \ N has no independent set of size 5 (i.e., complement of N has α ≤ 4).
    
    # Backtrack: decide for each vertex whether it's in N.
    # Order: process vertices in some order.
    
    # Precompute triangles in G
    triangles = []
    for i in range(n):
        for j in range(i + 1, n):
            if j in adj[i]:
                for k in range(j + 1, n):
                    if k in adj[i] and k in adj[j]:
                        triangles.append((i, j, k))
    
    # Precompute independent 5-sets (or at least check on the fly)
    # This could be expensive. Let's use a different approach.
    
    # Use randomized local search / backtracking with good heuristics.
    # Try multiple random initial assignments and do local repair.
    
    best = None
    for attempt in range(50):
        if time.time() - start > time_limit:
            break
        # Random initial assignment
        N = set()
        for v in range(n):
            if random.random() < 0.4:
                N.add(v)
        
        # Local search: try to fix violations
        for iteration in range(200):
            if time.time() - start > time_limit:
                break
            
            # Check constraint A: no triangle in N
            violation_a = None
            for (a, b, c) in triangles:
                if a in N and b in N and c in N:
                    violation_a = (a, b, c)
                    break
            
            if violation_a:
                # Remove one of the three from N (random choice)
                v = random.choice(violation_a)
                N.remove(v)
                continue
            
            # Check constraint B: complement of N has no independent 5-set
            # Complement = vertices not in N
            comp = [v for v in range(n) if v not in N]
            # Check if comp has independent set of size 5
            violation_b = None
            if len(comp) >= 5:
                # Sample some 5-subsets to check
                if len(comp) <= 20:
                    for combo in combinations(comp, 5):
                        ok = True
                        for idx in range(4):
                            if combo[idx+1] in adj[combo[idx]]:
                                ok = False
                                break
                        if ok:
                            violation_b = combo
                            break
                else:
                    # Random sampling
                    for _ in range(100):
                        combo = random.sample(comp, 5)
                        ok = True
                        for idx in range(4):
                            if combo[idx+1] in adj[combo[idx]]:
                                ok = False
                                break
                        if ok:
                            violation_b = combo
                            break
            
            if violation_b:
                # Add one of the five to N
                v = random.choice(violation_b)
                N.add(v)
                continue
            
            # No violations - we have a valid assignment
            best = N
            break
        
        if best is not None:
            break
    
    if best is not None:
        return sorted(best)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + HARD_LIMIT - DEADLINE_MARGIN
    
    # Start with Paley graph on 17 vertices
    best_result = _make_paley(17)
    best_n = 17
    
    # Try to extend
    for target_n in range(18, 36):
        if time.time() > deadline:
            break
        
        # Build adjacency from best_result
        adj = _build_adj(best_n, best_result['edges'])
        
        # Try to extend by one vertex
        new_neighbors = _try_extend(adj, best_n, target_n - 1, 5.0)
        
        if new_neighbors is not None:
            # Verify the new graph
            new_edges = best_result['edges'][:]
            for v in new_neighbors:
                new_edges.append([best_n, v])
            
            if _is_valid(target_n, new_edges):
                best_result = {'n': target_n, 'edges': new_edges}
                best_n = target_n
    
    # Final check
    if time.time() < deadline:
        if not _is_valid(best_n, best_result['edges']):
            # Fallback to known valid graph
            best_result = _make_paley(17)
    
    return best_result
# EVOLVE-BLOCK-END
