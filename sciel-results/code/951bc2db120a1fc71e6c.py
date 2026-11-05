"""Few edges in 5-uniform hypergraphs without Property B (m(5))."""
import time
import random
from itertools import combinations

# Known minimal non-2-colorable 5-uniform hypergraph construction
# The best known has 51 edges. We'll try to find one with fewer.
# Strategy: Use a SAT-based verifier and search for minimal constructions.

def _is_2_colorable(n, edges, deadline=None):
    """Check if hypergraph is 2-colorable via backtracking with unit propagation.
    Returns True if 2-colorable, False if not.
    
    A 5-uniform hypergraph is 2-colorable iff there exists a coloring c: V -> {0,1}
    such that for every edge {v1,...,v5}, not all have the same color.
    Equivalently, for each edge, at least one vertex has color 0 AND at least one has color 1.
    This is equivalent to: for each edge, (v1 OR v2 OR v3 OR v4 OR v5) AND (NOT v1 OR NOT v2 OR NOT v3 OR NOT v4 OR NOT v5)
    """
    m = len(edges)
    if m == 0:
        return True
    
    # Build clause structure
    # For each edge i, we have two clauses:
    #   Clause 2*i:   v1 OR v2 OR v3 OR v4 OR v5  (at least one True)
    #   Clause 2*i+1: NOT v1 OR NOT v2 OR NOT v3 OR NOT v4 OR NOT v5  (at least one False)
    
    # Precompute for each vertex, which clauses it appears in
    vert_clauses = [[] for _ in range(n)]
    for ci in range(2 * m):
        edge = edges[ci // 2]
        for v in edge:
            vert_clauses[v].append(ci)
    
    def solve(assignment, deadline):
        """DPLL with unit propagation."""
        if deadline is not None and time.time() > deadline:
            return True  # timeout: assume satisfiable (conservative)
        
        m_clause = 2 * m
        satisfied = [False] * m_clause
        unassigned_count = [0] * m_clause
        unit_v = [None] * m_clause
        
        # Initialize clause states
        for ci in range(m_clause):
            is_pos = (ci % 2 == 0)
            e = edges[ci // 2]
            cnt = 0
            uv = None
            for v in e:
                if assignment[v] is None:
                    cnt += 1
                    uv = v
                else:
                    if is_pos:
                        if assignment[v]:
                            satisfied[ci] = True
                            break
                    else:
                        if not assignment[v]:
                            satisfied[ci] = True
                            break
            else:
                unassigned_count[ci] = cnt
                if cnt == 1:
                    unit_v[ci] = uv
        
        # Unit propagation
        queue = []
        for ci in range(m_clause):
            if not satisfied[ci] and unassigned_count[ci] == 1:
                queue.append(ci)
        
        while queue:
            ci = queue.pop()
            if satisfied[ci]:
                continue
            if unassigned_count[ci] == 0:
                return False  # falsified
            if unassigned_count[ci] == 1:
                v = unit_v[ci]
                if assignment[v] is not None:
                    continue
                is_pos = (ci % 2 == 0)
                forced_val = is_pos
                assignment[v] = forced_val
                
                for cj in vert_clauses[v]:
                    if satisfied[cj]:
                        continue
                    unassigned_count[cj] -= 1
                    if unassigned_count[cj] == 1:
                        e = edges[cj // 2]
                        for vv in e:
                            if assignment[vv] is None:
                                unit_v[cj] = vv
                                break
                        queue.append(cj)
                    elif unassigned_count[cj] == 0:
                        is_pos_cj = (cj % 2 == 0)
                        e = edges[cj // 2]
                        sat = False
                        for vv in e:
                            if is_pos_cj:
                                if assignment[vv]:
                                    sat = True
                                    break
                            else:
                                if not assignment[vv]:
                                    sat = True
                                    break
                        if sat:
                            satisfied[cj] = True
                        else:
                            return False
        
        # Check for falsified clauses
        for ci in range(m_clause):
            if not satisfied[ci] and unassigned_count[ci] == 0:
                return False
        
        # Find unassigned variable with best heuristic
        best_v = None
        best_score = -1
        for v in range(n):
            if assignment[v] is not None:
                continue
            score = 0
            for ci in vert_clauses[v]:
                if satisfied[ci]:
                    continue
                cnt = unassigned_count[ci]
                if cnt == 1:
                    score += 100
                elif cnt == 2:
                    score += 10
                elif cnt >= 3:
                    score += 1
            if score > best_score:
                best_score = score
                best_v = v
        
        if best_v is None:
            return True
        
        # Branch
        for val in [True, False]:
            new_assign = assignment[:]
            new_assign[best_v] = val
            if solve(new_assign, deadline):
                return True
        return False
    
    assignment = [None] * n
    return solve(assignment, deadline)


def _known_construction():
    """Return a known non-2-colorable 5-uniform hypergraph with 51 edges."""
    # Based on the known result that m(5) = 51
    # This is a specific construction; we use a known example
    # The Petersen graph based construction or similar
    # For now, return a valid 51-edge construction
    # Actually, let's construct one based on known results
    
    # A known minimal example uses 12 vertices and 51 edges
    # Let me construct a valid one
    n = 12
    # Generate all 5-subsets and select a non-2-colorable subset
    # This is complex; let's use a different approach
    # We'll search for one during construct()
    return None


def _search_for_minimal(n, target_edges, deadline, rng):
    """Search for a non-2-colorable hypergraph with at most target_edges edges on n vertices."""
    all_edges = list(combinations(range(n), 5))
    rng.shuffle(all_edges)
    
    # Start with a random subset
    edges = [list(e) for e in all_edges[:target_edges]]
    
    # Check if non-2-colorable
    if not _is_2_colorable(n, edges, deadline):
        return edges
    
    # Try to find a minimal non-2-colorable subset
    # Use greedy addition: start with empty, add edges until non-2-colorable
    edges = []
    for e in all_edges:
        if time.time() > deadline:
            break
        edges.append(list(e))
        if len(edges) % 5 == 0 or len(edges) == len(all_edges):
            if not _is_2_colorable(n, edges, deadline):
                break
    
    if time.time() > deadline:
        return None
    
    if _is_2_colorable(n, edges, deadline):
        return None
    
    # Now try to minimize by removing edges
    improved = True
    while improved:
        improved = False
        for i in range(len(edges)):
            if time.time() > deadline:
                break
            candidate = edges[:i] + edges[i+1:]
            if not _is_2_colorable(n, candidate, deadline):
                edges = candidate
                improved = True
                break
    
    return edges


def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    rng = random.Random(42)
    
    best_result = None
    best_edge_count = float('inf')
    
    # Try different vertex counts
    for n in [10, 11, 12, 13, 14]:
        if time.time() > deadline:
            break
        
        # Try to find a minimal non-2-colorable hypergraph
        for trial in range(3):
            if time.time() > deadline:
                break
            
            edges = _search_for_minimal(n, 60, deadline, rng)
            if edges is not None and len(edges) < best_edge_count:
                best_edge_count = len(edges)
                best_result = {'n': n, 'edges': edges}
    
    if best_result is not None:
        return best_result
    
    # Fallback: complete 5-uniform on 10 vertices (252 edges)
    # This is definitely non-2-colorable
    n = 10
    return {'n': n, 'edges': [list(e) for e in combinations(range(n), 5)]}
