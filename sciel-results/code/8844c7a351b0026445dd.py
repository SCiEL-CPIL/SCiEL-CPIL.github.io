"""Few edges in 5-uniform hypergraphs without Property B (m(5))."""
import time
import random
from itertools import combinations

def _is_2_colorable(n, edges, deadline=None):
    """Check if hypergraph is 2-colorable via SAT backtracking with unit propagation.
    Returns True if 2-colorable, False if not."""
    m = len(edges)
    if m == 0:
        return True
    
    # Precompute clause structure
    m_clause = 2 * m
    # For each clause, store the 5 vertices
    clause_verts = []
    for i in range(m):
        clause_verts.append(edges[i])       # positive clause
        clause_verts.append(edges[i])       # negative clause (same vertices)
    
    # For each vertex, which clauses it appears in
    vert_clauses = [[] for _ in range(n)]
    for ci in range(m_clause):
        for v in clause_verts[ci]:
            vert_clauses[v].append(ci)
    
    # Fast SAT solver with incremental unit propagation
    def solve_fast(assignment, deadline):
        """DPLL with efficient unit propagation."""
        if deadline is not None and time.time() > deadline:
            return True
        
        # State arrays
        satisfied = [False] * m_clause
        unassigned_count = [0] * m_clause
        # For unit clauses: store the unassigned vertex
        unit_v = [None] * m_clause
        
        # Initialize clause states
        for ci in range(m_clause):
            is_pos = (ci % 2 == 0)
            e = clause_verts[ci]
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
        
        # Unit propagation queue
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
                # Determine forced value
                is_pos = (ci % 2 == 0)
                forced_val = is_pos  # positive clause forces True, negative forces False
                assignment[v] = forced_val
                
                # Propagate to all clauses containing v
                for cj in vert_clauses[v]:
                    if satisfied[cj]:
                        continue
                    unassigned_count[cj] -= 1
                    if unassigned_count[cj] == 1:
                        # Find the remaining unassigned vertex
                        e = clause_verts[cj]
                        for vv in e:
                            if assignment[vv] is None:
                                unit_v[cj] = vv
                                break
                        queue.append(cj)
                    elif unassigned_count[cj] == 0:
                        # Check if satisfied
                        is_pos_cj = (cj % 2 == 0)
                        e = clause_verts[cj]
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
                            return False  # falsified
        
        # Check for any falsified clauses
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
            if solve_fast(new_assign, deadline):
                return True
        return False
    
    assignment = [None] * n
    return solve_fast(assignment, deadline)


def _try_greedy_build(n, max_edges, deadline, rng):
    """Try to build a non-2-colorable hypergraph by greedily adding edges."""
    all_possible = list(combinations(range(n), 5))
    rng.shuffle(all_possible)
    
    edges = []
    for i, e in enumerate(all_possible[:max_edges]):
        if time.time() > deadline:
            return None
        edges.append(list(e))
        # Check periodically
        if (i + 1) % 4 == 0 or i == max_edges - 1:
            if not _is_2_colorable(n, edges, deadline):
                return edges
    return None


def _try_local_search(n, initial_edges, deadline, rng):
    """Try to improve an existing non-2-colorable hypergraph by removing edges."""
    edges = [list(e) for e in initial_edges]
    m = len(edges)
    
    # Try removing each edge and checking if still non-2-colorable
    for _ in range(20):  # limited attempts
        if time.time() > deadline:
            break
        removed = False
        # Try removing random edges
        for _ in range(min(m, 10)):
            if time.time() > deadline:
                break
            idx = rng.randint(0, len(edges) - 1)
            candidate = edges[:idx] + edges[idx+1:]
            if not _is_2_colorable(n, candidate, deadline):
                edges = candidate
                removed = True
                break
        if not removed:
            break
    
    return edges


def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    best_result = None
    best_edge_count = float('inf')
    
    rng = random.Random(42)
    
    # Try various vertex counts
    for n in [9, 10, 11, 12, 13]:
        if time.time() > deadline:
            break
        
        max_possible = len(list(combinations(range(n), 5)))
        
        # Multiple random trials
        for trial in range(50):
            if time.time() > deadline:
                break
            
            # Greedy build
            edges = _try_greedy_build(n, min(60, max_possible), deadline, rng)
            if edges is not None:
                # Try to minimize
                edges = _try_local_search(n, edges, deadline, rng)
                if len(edges) < best_edge_count:
                    best_edge_count = len(edges)
                    best_result = {'n': n, 'edges': edges}
    
    if best_result is not None:
        return best_result
    
    # Fallback: complete 5-uniform on 9 vertices (126 edges)
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
