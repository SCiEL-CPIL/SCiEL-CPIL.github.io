import time
import random
import itertools
from collections import defaultdict

n = 16

# Q3 edges (canonical, vertices 0..7)
_Q3 = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]

# All 8-vertex subsets of {0..15} and whether they induce a Q3
# Precompute: for each 8-subset, check if the induced subgraph contains Q3 as a subgraph
# Actually we need: for a given 8-vertex set, does it contain Q3 as a subgraph?
# We'll check by trying all 8! / 6 = 560 isomorphisms... that's too many.
# Better: for each 8-vertex set, check all ways to map Q3 vertices to these 8 vertices.
# Q3 has 8 vertices. An embedding of Q3 into an 8-vertex set is a bijection from {0..7} to the 8 vertices.
# There are 8! = 40320 such bijections, but Q3 has automorphisms. The automorphism group of Q3 has size 48.
# So there are 40320/48 = 840 distinct embeddings.
# For each 8-subset (C(16,8) = 12870), checking 840 embeddings is ~10.8M checks.
# Each check is 12 edge lookups. That's ~130M operations. Might be too slow in Python.

# Alternative: precompute all Q3 subgraphs (as edge sets) on K16.
# A Q3 subgraph is determined by choosing 8 vertices and an isomorphism class of embedding.
# Two embeddings are the same subgraph if they use the same 8 edges.
# Let's enumerate all distinct edge sets that form a Q3 on K16.

def _generate_all_q3_edge_sets():
    """Generate all distinct sets of 12 edges that form a Q3 on K16."""
    seen = set()
    result = []
    for vertex_set in itertools.combinations(range(16), 8):
        vs = set(vertex_set)
        # Try all permutations of the 8 Q3 vertices assigned to these 8 graph vertices
        # To reduce work, we can use the fact that Q3 is vertex-transitive and edge-transitive.
        # We'll just try all 8! permutations and deduplicate by edge set.
        for perm in itertools.permutations(range(8)):
            # perm[i] is the Q3 vertex assigned to vertex_set[i]
            # Actually: graph_vertex = vertex_set[i], q3_vertex = perm[i]
            # So edge (a,b) in Q3 corresponds to edge (vertex_set[inv_perm[a]], vertex_set[inv_perm[b]])
            # Let's build a mapping: q3_vertex -> graph_vertex
            q2g = [0]*8
            for i in range(8):
                q2g[perm[i]] = vertex_set[i]
            edge_set = frozenset()
            for a, b in _Q3:
                u, v = q2g[a], q2g[b]
                edge_set = edge_set | frozenset([u, v])
            if edge_set not in seen:
                seen.add(edge_set)
                result.append(edge_set)
    return result

# This is too slow (12870 * 40320 = 519M iterations). We need a smarter approach.

# Better approach: for the conflict graph, we don't need to precompute all Q3s.
# Instead, for a given set of edges, we check if it contains a Q3.
# For the branch-and-bound, we need: given a partial edge set S, which additional edges
# would create a Q3? This is hard to compute incrementally.

# Let's use a different strategy: 
# 1. Use randomized greedy to get a good initial solution.
# 2. Then use local search (edge swaps) to improve.
# 3. Use multiple restarts.

# Q3 detection: given an adjacency bitmask, check if Q3 exists.
# We'll use the method from the current program but optimize it.

def _bits(mask):
    while mask:
        lsb = mask & (-mask)
        yield lsb.bit_length() - 1
        mask ^= lsb

def _check_role(adj, n, a0, a1):
    """a0 is Q3-vertex 0, a1 is Q3-vertex 1 (edge (0,1))."""
    n0 = adj[a0] & ~ (1 << a1)
    n1 = adj[a1] & ~ (1 << a0)
    for a2 in _bits(n0):
        for a4 in _bits(n0):
            if a4 == a2:
                continue
            common6 = adj[a2] & adj[a4]
            common6 &= ~ (1 << a0)
            common6 &= ~ (1 << a1)
            for a6 in _bits(common6):
                if a6 == a2 or a6 == a4:
                    continue
                for a3 in _bits(n1):
                    for a5 in _bits(n1):
                        if a5 == a3:
                            continue
                        common7 = adj[a3] & adj[a5]
                        common7 &= ~ (1 << a0)
                        common7 &= ~ (1 << a1)
                        common7 &= ~ (1 << a6)
                        for a7 in _bits(common7):
                            if a7 == a3 or a7 == a5 or a7 == a6:
                                continue
                            verts = [a0,a1,a2,a4,a3,a5,a6,a7]
                            if len(set(verts)) == 8:
                                if not (adj[a0] >> a2) & 1: continue
                                if not (adj[a0] >> a4) & 1: continue
                                if not (adj[a1] >> a3) & 1: continue
                                if not (adj[a1] >> a5) & 1: continue
                                if not (adj[a2] >> a3) & 1: continue
                                if not (adj[a2] >> a6) & 1: continue
                                if not (adj[a3] >> a7) & 1: continue
                                if not (adj[a4] >> a5) & 1: continue
                                if not (adj[a4] >> a6) & 1: continue
                                if not (adj[a5] >> a7) & 1: continue
                                if not (adj[a6] >> a7) & 1: continue
                                return True
    return False

def _edge_makes_q3(adj, n, u, v):
    if _check_role(adj, n, u, v):
        return True
    if _check_role(adj, n, v, u):
        return True
    return False

def _count_edges(adj, n):
    return sum(bin(adj[i]).count('1') for i in range(n)) // 2

def _edge_list_from_adj(adj, n):
    es = []
    for u in range(n):
        for v in range(u+1, n):
            if (adj[u] >> v) & 1:
                es.append([u, v])
    return es

def _greedy_build(adj, n, order, deadline):
    """Add edges in the given order if they don't create a Q3."""
    count = 0
    for u, v in order:
        if time.time() > deadline:
            break
        if (adj[u] >> v) & 1:
            continue
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if _edge_makes_q3(adj, n, u, v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            count += 1
    return count

def _local_search(adj, n, deadline, max_iter=5000):
    """Try edge swaps (remove one, add another) to increase edge count."""
    count = _count_edges(adj, n)
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter and time.time() < deadline:
        improved = False
        iter_count += 1
        # Find all missing edges
        missing = []
        for u in range(n):
            for v in range(u+1, n):
                if not ((adj[u] >> v) & 1):
                    missing.append((u, v))
        random.shuffle(missing)
        # Try to add each missing edge
        for u, v in missing:
            if time.time() > deadline:
                return _count_edges(adj, n)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if _edge_makes_q3(adj, n, u, v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                count += 1
                improved = True
                break
        if improved:
            continue
        # Try edge swaps: remove an existing edge, add a missing edge
        existing = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    existing.append((u, v))
        random.shuffle(existing)
        for eu, ev in existing[:10]:
            if time.time() > deadline:
                return _count_edges(adj, n)
            # Remove (eu, ev)
            adj[eu] &= ~(1 << ev)
            adj[ev] &= ~(1 << eu)
            # Try to add a missing edge
            for u, v in missing:
                if (u == eu and v == ev) or (u == ev and v == eu):
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if not _edge_makes_q3(adj, n, u, v):
                    # Also need to check the whole graph doesn't have Q3
                    # Actually we only checked the new edge. But removing (eu,ev) might have broken a Q3,
                    # and adding (u,v) might create a new one. We should verify no Q3 exists.
                    # For efficiency, we'll just check if the graph is Q3-free.
                    if _is_q3_free(adj, n):
                        count = _count_edges(adj, n)
                        improved = True
                        break
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
            if improved:
                break
            # Restore (eu, ev)
            adj[eu] |= (1 << ev)
            adj[ev] |= (1 << eu)
        if not improved:
            break
    return _count_edges(adj, n)

def _is_q3_free(adj, n):
    """Check if the graph is Q3-free."""
    for u in range(n):
        au = adj[u]
        v = au
        while v:
            lsb = v & (-v)
            vv = lsb.bit_length() - 1
            v ^= lsb
            if vv <= u:
                continue
            if _edge_makes_q3(adj, n, u, vv):
                return False
    return True

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin

    best_edges = None
    best_count = -1
    rng = random.Random(42)

    # Pre-generate some edge orders
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Strategy 1: Randomized greedy from empty graph
    for trial in range(50):
        if time.time() > deadline - 5.0:
            break
        adj = [0] * n
        order = all_edges[:]
        rng.shuffle(order)
        cnt = _greedy_build(adj, n, order, deadline - 5.0)
        if cnt > best_count:
            best_count = cnt
            best_edges = _edge_list_from_adj(adj, n)
    
    # Strategy 2: Local search on the best solution
    if best_edges is not None:
        adj = [0] * n
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        cnt = _local_search(adj, n, deadline - 5.0)
        if cnt > best_count:
            best_count = cnt
            best_edges = _edge_list_from_adj(adj, n)

    # Strategy 3: More randomized greedy with different seeds
    for trial in range(100):
        if time.time() > deadline - 5.0:
            break
        adj = [0] * n
        order = all_edges[:]
        rng.shuffle(order)
        cnt = _greedy_build(adj, n, order, deadline - 5.0)
        if cnt > best_count:
            best_count = cnt
            best_edges = _edge_list_from_adj(adj, n)
        # Quick local search
        if time.time() < deadline - 5.0:
            cnt2 = _local_search(adj, n, deadline - 5.0, max_iter=2000)
            if cnt2 > best_count:
                best_count = cnt2
                best_edges = _edge_list_from_adj(adj, n)

    # Final: try more local search on the best
    if best_edges is not None and time.time() < deadline - 5.0:
        adj = [0] * n
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        cnt = _local_search(adj, n, deadline - 5.0, max_iter=10000)
        if cnt > best_count:
            best_count = cnt
            best_edges = _edge_list_from_adj(adj, n)

    if best_edges is None:
        # Fallback: return K3,13
        best_edges = [[a, b] for a in range(3) for b in range(3, 16)]
    
    return best_edges
