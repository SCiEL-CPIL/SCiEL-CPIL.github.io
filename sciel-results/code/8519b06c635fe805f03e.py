"""Hexagon-free graph construction on 40 vertices using QCP relaxation."""
import time
import random
import numpy as np

N = 40
NPAIRS = N * (N - 1) // 2  # 780

def _pair_index(u, v):
    if u > v:
        u, v = v, u
    return u * (N - 1) - u * (u + 1) // 2 + v - u - 1

def _index_to_pair(idx):
    # Inverse of _pair_index
    # idx = u*(N-1) - u*(u+1)//2 + v - u - 1
    # Let f(u) = u*(N-1) - u*(u+1)//2
    # We need largest u such that f(u) <= idx
    # f(u) = u*N - u - u*(u+1)//2 = u*(2N-2-u-1)//2 ... let's just precompute
    # Actually let's just use a lookup or simple loop (N is small)
    for u in range(N):
        base = u * (N - 1) - u * (u + 1) // 2
        if base <= idx < base + (N - 1 - u):
            v = u + 1 + (idx - base)
            return u, v
    return 0, 1  # fallback

# Precompute pair mapping
_pair_list = []
_idx_map = {}
for u in range(N):
    for v in range(u + 1, N):
        idx = _pair_index(u, v)
        _pair_list.append((u, v))
        _idx_map[idx] = (u, v)

def _build_adj_from_edges(edges):
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _has_c6_adj(adj):
    """Check if graph has a 6-cycle using set-based enumeration."""
    for u in range(N):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w == u:
                    continue
                for x in adj[w]:
                    if x == u or x == v:
                        continue
                    for y in adj[x]:
                        if y == u or y == v or y == w:
                            continue
                        for z in adj[y]:
                            if z == u or z == v or z == w or z == x:
                                continue
                            if u in adj[z]:
                                return True
    return False

def _incremental_c6_check(adj, u, v):
    """Check if adding edge (u,v) creates a C6."""
    for w in adj[v]:
        if w == u:
            continue
        for x in adj[w]:
            if x == u or x == v or x == w:
                continue
            for y in adj[x]:
                if y == u or y == v or y == w or y == x:
                    continue
                for z in adj[y]:
                    if z == u or z == v or z == w or z == x or z == y:
                        continue
                    if u in adj[z]:
                        return True
    return False

def _seed_graph_8K5():
    """8 disjoint K5s: 100 edges."""
    edges = []
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append([base + a, base + b])
    return edges

def _edges_to_adj_matrix(edges):
    """Convert edge list to NxN adjacency matrix."""
    A = np.zeros((N, N), dtype=np.float64)
    for u, v in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0
    return A

def _c6_penalty(A):
    """
    Compute a penalty for C6-freeness using trace(A^6) and corrections.
    For a simple graph, the number of 6-cycles can be computed from A^6.
    trace(A^6) = sum of (A^6)[i,i]
    Each 6-cycle contributes 12 to trace(A^6) (6 starting points * 2 directions).
    But we also get contributions from:
    - 6-walks that aren't simple cycles (revisit vertices)
    - 3-cycles traversed twice (contribute 2*3=6 per triangle? No...)
    
    Actually, let's use a simpler penalty: count the number of length-6 closed walks
    that could form C6s. For the QCP relaxation, we want a smooth proxy.
    
    A simpler approach: the number of (not necessarily induced) 6-cycles is related to
    trace(A^6) but contaminated. Let's use a different penalty:
    
    For each pair of vertices (u,v) that are NOT adjacent, count paths of length 5 from u to v.
    If any such path exists, adding (u,v) would create a C6.
    But in the relaxation, we want to penalize configurations where many such paths exist.
    
    Alternative: Use the trace formula. For a C6-free graph, we want trace(A^6) to be
    as small as possible relative to the number of edges. But this is complicated.
    
    Let's use a direct penalty: for each potential C6 (u,v,w,x,y,z), the product of
    edge indicators A[u,v]*A[v,w]*A[w,x]*A[x,y]*A[y,z]*A[z,u] should be 0.
    Summing over all such products gives the penalty. But enumerating all 6-tuples is O(N^6).
    
    For N=40, N^6 = 4096 million - too slow per iteration.
    
    Better: Use matrix multiplication. The number of closed walks of length 6 is trace(A^6).
    For a C6-free graph with no shorter cycles, trace(A^6) = 0. But we may have shorter cycles.
    
    Let's use a hybrid: minimize trace(A^6) - 2*trace(A^3)^2/3 or similar correction.
    Actually, let's just use trace(A^6) as a proxy and rely on the fact that for dense
    C6-free graphs, this will be relatively small.
    
    Even better: use the number of 6-cycles directly via the formula:
    n_C6 = (trace(A^6) - 2*sum_{i<j} A[i,j]^6 * something - ...) / 12
    This is getting complex. Let's use a simpler approach.
    
    For the QCP, let's use: penalty = trace(A^6) / 12
    This overcounts (includes non-simple walks) but is a smooth proxy.
    """
    A2 = A @ A
    A3 = A2 @ A
    A6 = A3 @ A3
    return np.trace(A6)

def _qcp_optimize(A_init, time_limit, lr=0.05, penalty_weight=1.0):
    """
    Projected gradient ascent to maximize edges subject to C6 penalty.
    """
    start = time.time()
    A = A_init.copy()
    best_A = A.copy()
    best_score = -np.inf
    
    while time.time() - start < time_limit:
        # Compute gradient of objective: maximize sum(A) - penalty_weight * trace(A^6)/12
        # d/dA_{ij} sum(A) = 1
        # d/dA_{ij} trace(A^6) = 6 * (A^5)[j,i]  (by matrix calculus)
        A2 = A @ A
        A3 = A2 @ A
        A5 = A3 @ A2
        
        # Gradient of penalty w.r.t. A
        d_penalty = 6.0 * A5.T  # d trace(A^6)/dA_{ij} = 6*(A^5)_{ji}
        
        # Gradient of objective
        grad = np.ones_like(A) - penalty_weight * d_penalty / 12.0
        
        # Projected gradient ascent
        A_new = A + lr * grad
        # Project: clip to [0,1], make symmetric, zero diagonal
        A_new = np.clip(A_new, 0.0, 1.0)
        A_new = (A_new + A_new.T) / 2.0
        np.fill_diagonal(A_new, 0.0)
        
        A = A_new
        
        # Evaluate current solution
        edge_count = np.sum(A) / 2.0  # Each edge counted twice
        penalty = _c6_penalty(A)
        score = edge_count - penalty_weight * penalty / 12.0
        
        if score > best_score:
            best_score = score
            best_A = A.copy()
        
        # Adaptive learning rate
        if time.time() - start > time_limit * 0.5:
            lr *= 0.95
    
    return best_A

def _round_to_graph(A, threshold=0.5):
    """Round the relaxed solution to a valid graph."""
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] >= threshold:
                edges.append([i, j])
    return edges

def _polish_graph(edges, time_limit):
    """Polish a graph by trying to add edges and doing swaps."""
    adj = _build_adj_from_edges(edges)
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    current_count = len(edge_set)
    best_count = current_count
    best_edges = list(edge_set)
    
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    random.shuffle(all_pairs)
    
    start = time.time()
    
    # Try to add edges
    for u, v in all_pairs:
        if time.time() - start > time_limit:
            break
        if (min(u,v), max(u,v)) in edge_set:
            continue
        if not _incremental_c6_check(adj, u, v):
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u,v), max(u,v)))
            current_count += 1
            if current_count > best_count:
                best_count = current_count
                best_edges = list(edge_set)
    
    # Local search with swaps
    remaining = time_limit - (time.time() - start)
    if remaining > 1:
        edge_list = list(edge_set)
        while time.time() - start < time_limit - 1:
            progress = False
            # Try random additions
            for _ in range(20):
                u, v = random.choice(all_pairs)
                if (min(u,v), max(u,v)) in edge_set:
                    continue
                if not _incremental_c6_check(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    edge_set.add((min(u,v), max(u,v)))
                    edge_list.append((min(u,v), max(u,v)))
                    current_count += 1
                    progress = True
                    if current_count > best_count:
                        best_count = current_count
                        best_edges = list(edge_set)
                    break
            
            if not progress and edge_list:
                # Swap
                idx = random.randint(0, len(edge_list) - 1)
                u, v = edge_list[idx]
                adj[u].discard(v)
                adj[v].discard(u)
                edge_set.discard((u, v))
                edge_list.pop(idx)
                current_count -= 1
                
                added = False
                for _ in range(20):
                    a, b = random.choice(all_pairs)
                    if (min(a,b), max(a,b)) in edge_set:
                        continue
                    if not _incremental_c6_check(adj, a, b):
                        adj[a].add(b)
                        adj[b].add(a)
                        edge_set.add((min(a,b), max(a,b)))
                        edge_list.append((min(a,b), max(a,b)))
                        current_count += 1
                        added = True
                        if current_count > best_count:
                            best_count = current_count
                            best_edges = list(edge_set)
                        break
                
                if not added:
                    adj[u].add(v)
                    adj[v].add(u)
                    edge_set.add((u, v))
                    edge_list.append((u, v))
                    current_count += 1
    
    return [[u, v] for u, v in best_edges], best_count

def construct():
    n = N
    deadline = 135  # Safety margin before 150s kill
    start_time = time.time()
    
    best_edges = _seed_graph_8K5()
    best_count = len(best_edges)
    
    # Start from seed graph
    seed_edges = _seed_graph_8K5()
    A_init = _edges_to_adj_matrix(seed_edges)
    
    # QCP optimization with multiple penalty weights and learning rates
    configs = [
        (0.05, 0.5, 60),
        (0.1, 1.0, 50),
        (0.03, 0.3, 55),
        (0.08, 2.0, 45),
    ]
    
    for lr, pw, t_alloc in configs:
        elapsed = time.time() - start_time
        remaining = deadline - 10 - elapsed
        if remaining < 5:
            break
        
        t_use = min(t_alloc, remaining)
        A_relaxed = _qcp_optimize(A_init, t_use, lr=lr, penalty_weight=pw)
        
        # Round to graph
        edges = _round_to_graph(A_relaxed, threshold=0.5)
        
        # Polish
        elapsed = time.time() - start_time
        remaining = deadline - 5 - elapsed
        if remaining < 2:
            break
        polished, count = _polish_graph(edges, min(15, remaining))
        
        if count > best_count:
            best_count = count
            best_edges = polished
    
    # Also try from a random dense start
    elapsed = time.time() - start_time
    remaining = deadline - 5 - elapsed
    if remaining > 10:
        # Random start with moderate density
        random.seed(42)
        random_edges = []
        for u in range(N):
            for v in range(u + 1, N):
                if random.random() < 0.3:
                    random_edges.append([u, v])
        
        A_rand = _edges_to_adj_matrix(random_edges)
        A_relaxed = _qcp_optimize(A_rand, min(30, remaining - 5), lr=0.08, penalty_weight=1.0)
        edges = _round_to_graph(A_relaxed, threshold=0.5)
        
        elapsed = time.time() - start_time
        remaining = deadline - 3 - elapsed
        if remaining > 2:
            polished, count = _polish_graph(edges, min(10, remaining))
            if count > best_count:
                best_count = count
                best_edges = polished
    
    # Final verification
    adj = _build_adj_from_edges(best_edges)
    if _has_c6_adj(adj):
        # Fallback to seed
        return _seed_graph_8K5()
    
    return best_edges
