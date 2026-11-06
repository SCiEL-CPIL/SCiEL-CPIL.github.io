"""Model-and-optimize: parameterize near-Turán graphs, optimize partition sizes,
then hill-climb with incremental triangle tracking and multi-restart SA."""
import random
import time
import math

N = 16
TARGET_EDGES = 65
DEADLINE = 145.0  # safety margin before 150s kill


def _build_adj_from_edges(edge_set):
    """Build bitset adjacency from a set of edges."""
    adj_bits = [0] * N
    for (u, v) in edge_set:
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
    return adj_bits


def _compute_degrees(adj_bits):
    """Compute degrees from bitset adjacency."""
    deg = [0] * N
    full_mask = (1 << N) - 1
    for i in range(N):
        deg[i] = bin(adj_bits[i] & full_mask).count('1')
    return deg


def _enumerate_triangles(adj_bits):
    """Enumerate all triangles. Returns list of (i,j,k) with i<j<k."""
    tris = []
    full_mask = (1 << N) - 1
    for i in range(N):
        for j in range(i + 1, N):
            if not (adj_bits[i] >> j) & 1:
                continue
            common = adj_bits[i] & adj_bits[j]
            mask = full_mask ^ ((1 << (j + 1)) - 1)
            common &= mask
            c = common
            while c:
                k = (c & (-c)).bit_length() - 1
                tris.append((i, j, k))
                c &= c - 1
    return tris


def _max_tri_deg_sum(adj_bits, deg, tris):
    """Compute max degree sum over all triangles."""
    if not tris:
        return 0
    best = 0
    for (i, j, k) in tris:
        s = deg[i] + deg[j] + deg[k]
        if s > best:
            best = s
    return best


def _score_graph(adj_bits):
    """Full score computation."""
    deg = _compute_degrees(adj_bits)
    tris = _enumerate_triangles(adj_bits)
    score = _max_tri_deg_sum(adj_bits, deg, tris)
    return score, deg, tris


def _build_turan_plus_one():
    """Build K_{8,8} plus one internal edge in one part."""
    adj_bits = [0] * N
    for i in range(8):
        for j in range(8, 16):
            adj_bits[i] |= (1 << j)
            adj_bits[j] |= (1 << i)
    # One internal edge in A: (0,1)
    adj_bits[0] |= (1 << 1)
    adj_bits[1] |= (1 << 0)
    
    deg = _compute_degrees(adj_bits)
    tris = _enumerate_triangles(adj_bits)
    score = _max_tri_deg_sum(adj_bits, deg, tris)
    return adj_bits, deg, tris, score


def _apply_swap(adj_bits, deg, e_remove, e_add):
    """
    Apply an edge swap with incremental triangle score update.
    Returns (new_adj_bits, new_deg, new_score, new_tris).
    """
    u, v = e_remove
    x, y = e_add
    
    new_adj_bits = adj_bits[:]
    new_deg = deg[:]
    
    # Remove edge (u,v)
    new_adj_bits[u] &= ~(1 << v)
    new_adj_bits[v] &= ~(1 << u)
    new_deg[u] -= 1
    new_deg[v] -= 1
    
    # Add edge (x,y)
    new_adj_bits[x] |= (1 << y)
    new_adj_bits[y] |= (1 << x)
    new_deg[x] += 1
    new_deg[y] += 1
    
    # Incremental triangle score computation
    # Only triangles involving {u,v,x,y} can change
    affected = set()
    for node in (u, v, x, y):
        affected.add(node)
    
    full_mask = (1 << N) - 1
    
    # Collect all triangles involving affected vertices in the NEW graph
    # and compute their degree sums using new_deg
    tri_sums = {}
    for a in affected:
        for b in range(a + 1, N):
            if not (new_adj_bits[a] >> b) & 1:
                continue
            common = new_adj_bits[a] & new_adj_bits[b]
            # Only consider c > b to avoid duplicates
            mask = full_mask ^ ((1 << (b + 1)) - 1)
            common &= mask
            c = common
            while c:
                k = (c & (-c)).bit_length() - 1
                tri_key = (a, b, k)
                tri_sums[tri_key] = new_deg[a] + new_deg[b] + new_deg[k]
                c &= c - 1
    
    # Also need triangles where affected vertex is the largest index
    # The above loop with a < b < k handles all triangles where at least one
    # affected vertex is the smallest. But we might miss triangles where
    # affected vertex is in middle or largest position.
    # Actually, for each triangle (i,j,k) with i<j<k, it's counted when
    # a=i. So we need to check all triangles where ANY of i,j,k is in affected.
    # Let me redo: enumerate all triangles and check if any vertex is affected.
    
    # Better approach: enumerate all triangles in new graph, but only compute
    # score for those with at least one affected vertex, plus we need the max
    # over ALL triangles. For unchanged triangles, their scores are unchanged
    # UNLESS their degrees changed (which only happens for affected vertices).
    # So triangles not involving any affected vertex have unchanged scores.
    # Triangles involving affected vertices may have changed scores.
    
    # We need max over:
    # 1. All triangles not involving affected vertices (unchanged scores)
    # 2. All triangles involving affected vertices (recomputed scores)
    
    # For (1), we can't easily get the max without enumerating all.
    # For N=16, let's just enumerate all triangles but only compute the
    # affected ones' scores explicitly, and track the global max.
    
    all_tris = []
    for i in range(N):
        for j in range(i + 1, N):
            if not (new_adj_bits[i] >> j) & 1:
                continue
            common = new_adj_bits[i] & new_adj_bits[j]
            mask = full_mask ^ ((1 << (j + 1)) - 1)
            common &= mask
            c = common
            while c:
                k = (c & (-c)).bit_length() - 1
                all_tris.append((i, j, k))
                c &= c - 1
    
    new_score = 0
    for (i, j, k) in all_tris:
        s = new_deg[i] + new_deg[j] + new_deg[k]
        if s > new_score:
            new_score = s
    
    new_tris = all_tris
    
    return new_adj_bits, new_deg, new_score, new_tris


def _try_internal_edges(a_size, rng, n_trials=20):
    """Try random internal edge placements for a given bipartition size."""
    b_size = N - a_size
    cross = a_size * b_size
    internal_needed = TARGET_EDGES - cross
    if internal_needed < 0:
        return None
    max_internal = a_size * (a_size - 1) // 2 + b_size * (b_size - 1) // 2
    if internal_needed > max_internal:
        return None
    
    # Build cross edges
    adj_base = [0] * N
    for i in range(a_size):
        for j in range(a_size, N):
            adj_base[i] |= (1 << j)
            adj_base[j] |= (1 << i)
    
    internal_candidates_a = [(i, j) for i in range(a_size) for j in range(i + 1, a_size)]
    internal_candidates_b = [(i, j) for i in range(a_size, N) for j in range(i + 1, N)]
    all_internal = internal_candidates_a + internal_candidates_b
    
    best_result = None
    best_score = float('inf')
    
    for _ in range(n_trials):
        if internal_needed > len(all_internal):
            break
        chosen = rng.sample(all_internal, internal_needed)
        
        adj_try = adj_base[:]
        for (i, j) in chosen:
            adj_try[i] |= (1 << j)
            adj_try[j] |= (1 << i)
        
        deg_try = _compute_degrees(adj_try)
        tris_try = _enumerate_triangles(adj_try)
        score_try = _max_tri_deg_sum(adj_try, deg_try, tris_try)
        
        if score_try < best_score:
            best_score = score_try
            best_result = (adj_try, deg_try, tris_try, score_try)
    
    return best_result


def construct():
    t0 = time.time()
    rng = random.Random(42)
    
    def elapsed():
        return time.time() - t0
    
    # Phase 1: Try different initial structures
    best_adj_bits, best_deg, best_tris, best_score = _build_turan_plus_one()
    
    # Try various bipartition sizes
    for a_size in range(5, 12):
        if elapsed() > DEADLINE - 60:
            break
        result = _try_internal_edges(a_size, rng, n_trials=30)
        if result and result[3] < best_score:
            best_adj_bits, best_deg, best_tris, best_score = result
    
    # Also try some random non-bipartite starts
    for trial in range(10):
        if elapsed() > DEADLINE - 60:
            break
        # Start from random graph with 65 edges
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        random_edges = set(rng.sample(all_pairs, TARGET_EDGES))
        adj_try = _build_adj_from_edges(random_edges)
        deg_try = _compute_degrees(adj_try)
        tris_try = _enumerate_triangles(adj_try)
        score_try = _max_tri_deg_sum(adj_try, deg_try, tris_try)
        if score_try < best_score:
            best_adj_bits, best_deg, best_tris, best_score = (
                adj_try, deg_try, tris_try, score_try)
    
    # Phase 2: Simulated Annealing with multiple restarts
    time_budget_sa = DEADLINE - elapsed() - 15.0  # reserve 15s for final polish
    if time_budget_sa > 5.0:
        # Determine number of SA restarts based on available time
        n_restarts = max(1, int(time_budget_sa / 20.0))
        time_per_restart = time_budget_sa / n_restarts
        
        for restart in range(n_restarts):
            if elapsed() > DEADLINE - 15.0:
                break
            
            # For restarts > 0, perturb the best solution
            if restart == 0:
                adj_bits = best_adj_bits[:]
                deg = best_deg[:]
                tris = best_tris[:]
                cur_score = best_score
            else:
                # Perturb: do k random swaps
                adj_bits = best_adj_bits[:]
                deg = _compute_degrees(adj_bits)
                tris = _enumerate_triangles(adj_bits)
                cur_score = _max_tri_deg_sum(adj_bits, deg, tris)
                
                all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
                edges = [(i, j) for (i, j) in all_pairs if (adj_bits[i] >> j) & 1]
                nonedges = [(i, j) for (i, j) in all_pairs if not ((adj_bits[i] >> j) & 1)]
                
                for _ in range(5):
                    if not edges or not nonedges:
                        break
                    e_remove = edges[rng.randrange(len(edges))]
                    e_add = nonedges[rng.randrange(len(nonedges))]
                    new_adj, new_deg, new_score, new_tris = _apply_swap(
                        adj_bits, deg, e_remove, e_add)
                    adj_bits, deg, tris = new_adj, new_deg, new_tris
                    cur_score = new_score
                    edges.remove(e_remove)
                    nonedges.append(e_remove)
                    nonedges.remove(e_add)
                    edges.append(e_add)
            
            # SA parameters
            start_temp = 3.0
            end_temp = 0.01
            max_iters = 50000
            cooling = (end_temp / start_temp) ** (1.0 / max(1, max_iters - 1))
            temp = start_temp
            
            all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
            edges = [(i, j) for (i, j) in all_pairs if (adj_bits[i] >> j) & 1]
            nonedges = [(i, j) for (i, j) in all_pairs if not ((adj_bits[i] >> j) & 1)]
            
            iters = 0
            deadline_inner = DEADLINE - 15.0
            while iters < max_iters:
                if time.time() > deadline_inner:
                    break
                
                if not edges or not nonedges:
                    break
                
                e_remove = edges[rng.randrange(len(edges))]
                e_add = nonedges[rng.randrange(len(nonedges))]
                
                new_adj_bits, new_deg, new_score, new_tris = _apply_swap(
                    adj_bits, deg, e_remove, e_add)
                
                # Acceptance
                if new_score < cur_score:
                    accept = True
                elif new_score == cur_score:
                    accept = rng.random() < 0.3
                else:
                    delta = new_score - cur_score
                    if temp < 1e-10:
                        accept = False
                    else:
                        accept = math.exp(-delta / temp) > rng.random()
                
                if accept:
                    adj_bits = new_adj_bits
                    deg = new_deg
                    tris = new_tris
                    cur_score = new_score
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_adj_bits = adj_bits[:]
                        best_deg = deg[:]
                        best_tris = tris[:]
                    
                    # O(1) swap using index tracking
                    # Find positions
                    er_idx = edges.index(e_remove)
                    edges[er_idx] = e_add
                    nonedges.remove(e_add)
                    nonedges.append(e_remove)
                    # Fix: we need to move e_remove from edges to nonedges and e_add from nonedges to edges
                    # Actually simpler: just rebuild if needed, but for N=16 with 65 edges,
                    # list operations are cheap. Let's use a different approach.
                    
                iters += 1
                temp = max(end_temp, temp * cooling)
    
    # Phase 3: Deterministic hill-climbing polish (multiple passes)
    if elapsed() < DEADLINE - 3.0:
        adj_bits = best_adj_bits[:]
        deg = best_deg[:]
        tris = best_tris[:]
        cur_score = best_score
        
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        edges = [(i, j) for (i, j) in all_pairs if (adj_bits[i] >> j) & 1]
        nonedges = [(i, j) for (i, j) in all_pairs if not ((adj_bits[i] >> j) & 1)]
        
        # Multiple passes of best-improvement hill climbing
        for polish_pass in range(8):
            if elapsed() > DEADLINE - 3.0:
                break
            
            improved = False
            # Try all edge swaps, keep best improvement
            best_swap = None
            best_new_score = cur_score
            best_new_adj = None
            best_new_deg = None
            best_new_tris = None
            
            for e_remove in edges:
                if elapsed() > DEADLINE - 3.0:
                    break
                for e_add in nonedges:
                    new_adj_bits, new_deg, new_score, new_tris = _apply_swap(
                        adj_bits, deg, e_remove, e_add)
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_swap = (e_remove, e_add)
                        best_new_adj = new_adj_bits
                        best_new_deg = new_deg
                        best_new_tris = new_tris
            
            if best_swap is not None and best_new_score < cur_score:
                e_remove, e_add = best_swap
                adj_bits = best_new_adj
                deg = best_new_deg
                tris = best_new_tris
                cur_score = best_new_score
                
                if cur_score < best_score:
                    best_score = cur_score
                    best_adj_bits = adj_bits[:]
                    best_deg = deg[:]
                    best_tris = tris[:]
                
                # O(1) swap using last-element trick
                er_pos = edges.index(e_remove)
                edges[er_pos] = edges[-1]
                edges.pop()
                ea_pos = nonedges.index(e_add)
                nonedges[ea_pos] = nonedges[-1]
                nonedges.pop()
                nonedges.append(e_remove)
                edges.append(e_add)
                improved = True
            
            if not improved:
                break
    
    # Final exhaustive 1-swap local search
    if elapsed() < DEADLINE - 2.0:
        adj_bits = best_adj_bits[:]
        deg = _compute_degrees(adj_bits)
        cur_score = _max_tri_deg_sum(adj_bits, deg, _enumerate_triangles(adj_bits))
        
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        edges = [(i, j) for (i, j) in all_pairs if (adj_bits[i] >> j) & 1]
        nonedges = [(i, j) for (i, j) in all_pairs if not ((adj_bits[i] >> j) & 1)]
        
        # Try every possible single swap
        for e_remove in edges:
            if elapsed() > DEADLINE - 2.0:
                break
            for e_add in nonedges:
                new_adj_bits, new_deg, new_score, _ = _apply_swap(
                    adj_bits, deg, e_remove, e_add)
                if new_score < cur_score:
                    # Apply this swap and restart the search
                    adj_bits = new_adj_bits
                    deg = new_deg
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_adj_bits = adj_bits[:]
                        best_deg = deg[:]
                    # Rebuild edge lists
                    edges = [(i, j) for (i, j) in all_pairs if (adj_bits[i] >> j) & 1]
                    nonedges = [(i, j) for (i, j) in all_pairs if not ((adj_bits[i] >> j) & 1)]
                    break
    
    # Final verification
    final_deg = _compute_degrees(best_adj_bits)
    final_tris = _enumerate_triangles(best_adj_bits)
    final_score = _max_tri_deg_sum(best_adj_bits, final_deg, final_tris)
    
    result_edges = [[i, j] for i in range(N) for j in range(i + 1, N)
                    if (best_adj_bits[i] >> j) & 1]
    
    # Verify edge count
    assert len(result_edges) == TARGET_EDGES, f"Expected {TARGET_EDGES} edges, got {len(result_edges)}"
    
    return {'n': N, 'edges': result_edges}
