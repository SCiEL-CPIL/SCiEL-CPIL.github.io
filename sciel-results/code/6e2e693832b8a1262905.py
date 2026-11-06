"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _pale_tournament(n):
    """Return adjacency list for Paley tournament on n vertices (n must be prime ≡ 3 mod 4)."""
    squares = {x * x % n for x in range(1, n)}
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n in squares:
                adj[i][j] = True
            else:
                adj[j][i] = True
    return adj

def _orientation_from_adj(adj):
    """Convert adjacency matrix to orientation list (one bit per pair i<j in lex order)."""
    n = len(adj)
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            orientation.append(0 if adj[i][j] else 1)
    return orientation

def _adj_from_orientation(n, orientation):
    """Convert orientation list to adjacency matrix."""
    adj = [[False] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if orientation[idx] == 0:
                adj[i][j] = True
            else:
                adj[j][i] = True
            idx += 1
    return adj

def _check_validity(n, adj):
    """Check if every set of 4 vertices has a common dominator."""
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        # Check if any vertex dominates all others in the quad
        for v in quad:
            if (adj[v][a] or v == a) and (adj[v][b] or v == b) and \
               (adj[v][c] or v == c) and (adj[v][d] or v == d):
                break
        else:
            return False
    return True

def _count_violations_fast(n, adj):
    """Count number of 4-sets without a common dominator."""
    violations = 0
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        found = False
        for v in quad:
            if (adj[v][a] or v == a) and (adj[v][b] or v == b) and \
               (adj[v][c] or v == c) and (adj[v][d] or v == d):
                found = True
                break
        if not found:
            violations += 1
    return violations

def _dominator_count(n, adj):
    """For each 4-set, count how many vertices dominate it. Returns dict mapping frozenset to count."""
    dom_counts = {}
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        count = 0
        for v in quad:
            if (adj[v][a] or v == a) and (adj[v][b] or v == b) and \
               (adj[v][c] or v == c) and (adj[v][d] or v == d):
                count += 1
        dom_counts[frozenset(quad)] = count
    return dom_counts

def _vertex_criticality(n, adj):
    """
    For each vertex, count how many 4-sets have this vertex as the UNIQUE dominator.
    If a vertex is the unique dominator of some 4-set, removing it breaks validity.
    """
    n_critical = [0] * n
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        dominators = []
        for v in quad:
            if (adj[v][a] or v == a) and (adj[v][b] or v == b) and \
               (adj[v][c] or v == c) and (adj[v][d] or v == d):
                dominators.append(v)
        if len(dominators) == 1:
            n_critical[dominators[0]] += 1
    return n_critical

def _remove_vertex(n, adj, vertex):
    """Remove a vertex and return new n and adj matrix."""
    new_n = n - 1
    new_adj = [[False] * new_n for _ in range(new_n)]
    for i in range(new_n):
        for j in range(i + 1, new_n):
            old_i = i if i < vertex else i + 1
            old_j = j if j < vertex else j + 1
            new_adj[i][j] = adj[old_i][old_j]
    return new_n, new_adj

def construct():
    """
    Try to find a valid tournament with fewer vertices than Paley(83).
    Strategy: Start from Paley(83), try to prune vertices that are not critical.
    Also try local search to flip edges and reduce violations.
    """
    deadline = time.time() + 140  # Safety margin
    
    # Start with Paley tournament on 83 vertices
    p = 83
    adj = _pale_tournament(p)
    n = p
    
    # Verify validity
    if not _check_validity(n, adj):
        # Should not happen, but return as is
        return {'n': n, 'orientation': _orientation_from_adj(adj)}
    
    # Try to prune vertices
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        n_critical = _vertex_criticality(n, adj)
        # Try to remove vertices in order of lowest criticality
        candidates = sorted(range(n), key=lambda v: n_critical[v])
        for v in candidates:
            if time.time() > deadline - 30:
                break
            if n_critical[v] > 0:
                continue
            # Try removing vertex v
            new_n, new_adj = _remove_vertex(n, adj, v)
            if _check_validity(new_n, new_adj):
                n = new_n
                adj = new_adj
                improved = True
                break
    
    # Now try local search to flip edges and see if we can reduce violations
    # Actually, we want to maintain validity while trying to find a smaller valid tournament
    # Let's try a different approach: use the current valid tournament and try to
    # perform edge flips that maintain validity, then try pruning again
    
    # For now, if we haven't made progress, try local search on the current solution
    # to see if we can find a better structure
    
    # Try random restarts with smaller initial tournaments
    for trial_n in [67, 63, 59, 55, 51, 47, 43, 39, 35, 31]:
        if time.time() > deadline - 10:
            break
        
        # Try random tournament
        best_adj = None
        best_violations = float('inf')
        
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            # Create random tournament
            rand_adj = [[False] * trial_n for _ in range(trial_n)]
            for i in range(trial_n):
                for j in range(i + 1, trial_n):
                    if random.random() < 0.5:
                        rand_adj[i][j] = True
                    else:
                        rand_adj[j][i] = True
            
            violations = _count_violations_fast(trial_n, rand_adj)
            if violations < best_violations:
                best_violations = violations
                best_adj = [row[:] for row in rand_adj]
            
            if best_violations == 0:
                # Found a valid tournament
                adj = best_adj
                n = trial_n
                return {'n': n, 'orientation': _orientation_from_adj(adj)}
            
            # Local search: try flipping edges to reduce violations
            for step in range(100):
                if time.time() > deadline - 10:
                    break
                if best_violations == 0:
                    break
                
                # Pick a random edge to flip
                i = random.randint(0, trial_n - 1)
                j = random.randint(0, trial_n - 1)
                if i == j:
                    continue
                
                old_val = best_adj[i][j]
                best_adj[i][j] = not old_val
                best_adj[j][i] = old_val
                
                new_violations = _count_violations_fast(trial_n, best_adj)
                if new_violations < best_violations:
                    best_violations = new_violations
                else:
                    # Revert
                    best_adj[i][j] = old_val
                    best_adj[j][i] = not old_val
    
    # Return the best solution found
    return {'n': n, 'orientation': _orientation_from_adj(adj)}
# EVOLVE-BLOCK-END
