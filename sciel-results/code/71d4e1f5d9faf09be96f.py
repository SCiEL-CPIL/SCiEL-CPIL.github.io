import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
_N = 15
_ALL_EDGES = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _norm(e):
    a, b = e
    return (a, b) if a < b else (b, a)

def _build_adj_deg(edges):
    adj = [set() for _ in range(_N)]
    deg = [0] * _N
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
    return adj, deg

def _is_valid(adj, deg, edges):
    for u, v in edges:
        for s in adj[u]:
            if s == v:
                continue
            ds = deg[s]
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if ds == deg[t]:
                    return False
    return True

def _try_add_edge(adj, deg, g, edge):
    """Try to add edge to g, return (new_g, new_adj, new_deg) if valid, else None."""
    u, v = edge
    new_g = g | {edge}
    new_adj = [set(a) for a in adj]
    new_deg = deg[:]
    new_adj[u].add(v)
    new_adj[v].add(u)
    new_deg[u] += 1
    new_deg[v] += 1
    if _is_valid(new_adj, new_deg, new_g):
        return new_g, new_adj, new_deg
    return None

def _construct_from_degree_sequence(target_degrees, rng):
    """
    Construct a graph matching target_degrees (sorted in non-increasing order).
    Uses Havel-Hakimi style greedy assignment.
    Returns set of edges if successful, else None.
    """
    n = _N
    # target_degrees should be sorted non-increasing
    remaining = list(target_degrees)
    vertices = list(range(n))
    
    # We'll build adjacency incrementally
    adj = [set() for _ in range(n)]
    current_deg = [0] * n
    edges = set()
    
    # Process vertices in order of target degree (highest first)
    # For each vertex, connect it to the highest-degree remaining vertices it can
    order = sorted(range(n), key=lambda i: -target_degrees[i])
    
    for vi in order:
        need = target_degrees[vi] - current_deg[vi]
        if need < 0:
            return None  # Already exceeded target
        
        if need == 0:
            continue
        
        # Candidates: vertices that haven't been fully processed yet
        # and have remaining capacity
        candidates = []
        for vj in range(n):
            if vj == vi:
                continue
            if vj in adj[vi]:
                continue  # Already connected
            # Check if vj has remaining capacity
            if current_deg[vj] < target_degrees[vj]:
                candidates.append(vj)
        
        # Sort candidates by target degree (descending) to prefer high-degree vertices
        candidates.sort(key=lambda x: -target_degrees[x])
        
        if len(candidates) < need:
            return None  # Not enough candidates
        
        # Connect to the top 'need' candidates
        for vj in candidates[:need]:
            adj[vi].add(vj)
            adj[vj].add(vi)
            current_deg[vi] += 1
            current_deg[vj] += 1
            edges.add(_norm((vi, vj)))
    
    # Verify the degrees match
    for i in range(n):
        if current_deg[i] != target_degrees[i]:
            return None
    
    # Verify validity
    if not _is_valid(adj, current_deg, edges):
        return None
    
    return edges

def _optimize_edges(g, adj, deg, rng, deadline):
    """Local search: try to add edges, and swap moves."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding single edges
        candidates = [e for e in _ALL_EDGES if e not in g]
        rng.shuffle(candidates)
        
        for e in candidates:
            if time.time() > deadline:
                break
            result = _try_add_edge(adj, deg, g, e)
            if result:
                g, adj, deg = result
                improved = True
                break
        
        # Try swap moves: remove one edge, then greedily add
        if not improved and len(g) < _N * (_N - 1) // 2:
            current_edges = list(g)
            if current_edges:
                rng.shuffle(current_edges)
                for e_out in current_edges[:20]:
                    if time.time() > deadline:
                        break
                    
                    g_temp = g - {e_out}
                    adj_temp, deg_temp = _build_adj_deg(g_temp)
                    
                    candidates_add = [e for e in _ALL_EDGES if e not in g_temp]
                    rng.shuffle(candidates_add)
                    
                    for e_in in candidates_add:
                        if time.time() > deadline:
                            break
                        result = _try_add_edge(adj_temp, deg_temp, g_temp, e_in)
                        if result:
                            g_temp, adj_temp, deg_temp = result
                    
                    if len(g_temp) > len(g):
                        g, adj, deg = g_temp, adj_temp, deg_temp
                        improved = True
                        break
    
    return g, adj, deg

def _generate_degree_sequences(rng, count=200):
    """Generate diverse degree sequences to try."""
    sequences = []
    
    # All distinct degrees: choose 15 distinct values from 0..14
    # The sum must be even (handshake lemma)
    # Try many combinations
    
    # Strategy 1: Use all degrees 0-14 (sum = 105, odd - invalid)
    # So we need to adjust: e.g., use {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14} but sum=105 is odd
    # We need sum to be even. So we can't use exactly 0..14.
    
    # Strategy: Pick 15 distinct values from 0..14 with even sum
    # Since 0+1+...+14 = 105 (odd), we need to change one value or use a different set
    # Actually we can only use values 0..14, and need 15 distinct values, so it MUST be 0..14
    # But 105 is odd, so no graph exists with all distinct degrees 0..14!
    
    # So we need at least two vertices with the same degree, but arranged so that
    # no path has endpoints with equal degree.
    
    # Alternative approach: Allow some equal degrees but ensure structural separation.
    # For example, if vertices with degree d are not in position to be path endpoints together.
    
    # Let's try: degrees that are mostly distinct but with some duplicates
    # Key insight: if two vertices have the same degree, they must not appear as
    # endpoints of any 3-edge path. This means they should be "far apart" structurally.
    
    # A simpler approach: use a star-like structure where high-degree vertices
    # are isolated from each other in terms of path endpoints.
    
    # Let's just try many random degree sequences and see which are graphical
    for _ in range(count):
        # Generate random distinct-ish degrees
        base = list(range(_N))
        rng.shuffle(base)
        # Make sure sum is even
        seq = sorted(base, reverse=True)
        s = sum(seq)
        if s % 2 != 0:
            # Adjust: decrease the largest odd-degree vertex by 1, or increase smallest
            # Find a vertex we can adjust
            for i in range(_N):
                if seq[i] > 0 and (s - 1) % 2 == 0:
                    seq[i] -= 1
                    # Re-sort
                    seq = sorted(seq, reverse=True)
                    break
            else:
                continue
        
        # Check if graphical using Havel-Hakimi
        if _is_graphical(seq):
            sequences.append(seq)
    
    return sequences

def _is_graphical(seq):
    """Check if a degree sequence is graphical using Havel-Hakimi."""
    seq = sorted(seq, reverse=True)
    if sum(seq) % 2 != 0:
        return False
    if any(d < 0 or d > _N - 1 for d in seq):
        return False
    
    while seq:
        if not seq:
            break
        d = seq[0]
        seq = seq[1:]
        if d > len(seq):
            return False
        for i in range(d):
            seq[i] -= 1
            if seq[i] < 0:
                return False
        seq = sorted(seq, reverse=True)
    
    return True

def _solve_once(rng, deadline):
    """One run of the search using degree-sequence-guided construction."""
    best_g = None
    best_size = -1
    
    # Phase 1: Try to construct from various degree sequences
    degree_seqs = _generate_degree_sequences(rng, count=100)
    
    for seq in degree_seqs:
        if time.time() > deadline:
            break
        g = _construct_from_degree_sequence(seq, rng)
        if g is not None:
            if len(g) > best_size:
                best_size = len(g)
                best_g = g
    
    # If no construction worked, fall back to greedy removal
    if best_g is None:
        g = set(_ALL_EDGES)
        adj, deg = _build_adj_deg(g)
        
        for _ in range(500):
            if time.time() > deadline:
                break
            if _is_valid(adj, deg, g):
                break
            
            violation_count = {}
            for u, v in g:
                for s in adj[u]:
                    if s == v:
                        continue
                    ds = deg[s]
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if ds == deg[t]:
                            e1 = _norm((s, u))
                            e2 = _norm((u, v))
                            e3 = _norm((v, t))
                            violation_count[e1] = violation_count.get(e1, 0) + 1
                            violation_count[e2] = violation_count.get(e2, 0) + 1
                            violation_count[e3] = violation_count.get(e3, 0) + 1
            
            if not violation_count:
                break
            
            best_edge_to_remove = max(violation_count.items(), key=lambda x: x[1])[0]
            g.discard(best_edge_to_remove)
            adj, deg = _build_adj_deg(g)
        
        if len(g) > best_size:
            best_size = len(g)
            best_g = g
    
    # Phase 2: Local search optimization
    if best_g is not None:
        adj, deg = _build_adj_deg(best_g)
        best_g, adj, deg = _optimize_edges(best_g, adj, deg, rng, deadline)
    
    return best_g

def construct():
    start_time = time.time()
    deadline = 85.0
    
    best_g = None
    best_size = -1
    
    restart_count = 0
    while time.time() - start_time < deadline:
        rng = random.Random(12345 + 7919 * restart_count)
        
        g = _solve_once(rng, deadline)
        
        if g is not None and g:
            adj, deg = _build_adj_deg(g)
            if _is_valid(adj, deg, g):
                if len(g) > best_size:
                    best_size = len(g)
                    best_g = g
        
        restart_count += 1
        if restart_count > 100:
            break
    
    if best_g is None:
        best_g = {(0, i) for i in range(1, _N)}
    
    result = []
    for u, v in best_g:
        result.append([int(u), int(v)])
    
    return result
# EVOLVE-BLOCK-END
