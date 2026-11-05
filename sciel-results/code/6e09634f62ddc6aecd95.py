import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin for 150s limit

    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    full_mask = (1 << n) - 1

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v = adj[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    edges.append([u, vv])
                v ^= low
        return edges

    def is_valid(adj):
        """Check if no 3 vertices have 3 common neighbors (K3,3-free)."""
        for i in range(n):
            ai = adj[i]
            for j in range(i+1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j+1, n):
                    cijk = cij & adj[k]
                    if cijk.bit_count() >= 3:
                        return False
        return True

    def find_violation(adj):
        """Find a K3,3 violation. Returns (i,j,k) or None."""
        for i in range(n):
            ai = adj[i]
            for j in range(i+1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j+1, n):
                    cijk = cij & adj[k]
                    if cijk.bit_count() >= 3:
                        return (i, j, k)
        return None

    def make_valid(adj):
        """Remove edges to make the graph valid."""
        adj = adj[:]
        iterations = 0
        while iterations < 1000:
            viol = find_violation(adj)
            if viol is None:
                break
            i, j, k = viol
            # Find common neighbors of i, j, k
            cij = adj[i] & adj[j]
            cijk = cij & adj[k]
            # Remove an edge: pick a common neighbor and remove its edge to one of i,j,k
            cn = cijk
            if cn == 0:
                break
            # Pick a random common neighbor
            bits = []
            v = cn
            while v:
                low = v & (-v)
                bits.append(low.bit_length() - 1)
                v ^= low
            x = bits[0]
            # Remove edge (i, x)
            adj[i] &= ~(1 << x)
            adj[x] &= ~(1 << i)
            iterations += 1
        return adj

    def is_valid_fast(adj):
        """Faster validity check: only check triples where pairwise common >= 3."""
        for i in range(n):
            ai = adj[i]
            for j in range(i+1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3:
                    continue
                for k in range(j+1, n):
                    cijk = cij & adj[k]
                    if cijk.bit_count() >= 3:
                        return False
        return True

    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free.
        Only need to check triples that include u or v as one of the three
        common-neighbor vertices, and where the new edge creates a 4th common neighbor.
        Actually simpler: adding edge (u,v) can only create violations where
        u and v are both in the triple, or one of them is in the triple.
        A K3,3 has 3 vertices on each side. The new edge (u,v) means u and v
        are now adjacent. A violation involving this edge would need 3 common
        neighbors of some triple. The new edge doesn't directly create common
        neighbors - it just adds an edge. The constraint is about common neighbors
        of triples, not about edges between the triple vertices.
        
        Wait - re-reading: "no three vertices have three common neighbours".
        This means for any 3 vertices {a,b,c}, |N(a) ∩ N(b) ∩ N(c)| <= 2.
        Adding edge (u,v) changes N(u) and N(v). So we need to check triples
        where u or v is one of the three vertices.
        """
        # Triples containing u: (u, j, k) for j < k, j != u, k != u
        # Need |N(u) ∩ N(j) ∩ N(k)| <= 2 after adding v to N(u)
        # N(u) gains v, so check if v is in N(j) ∩ N(k) and |N(u) ∩ N(j) ∩ N(k)| was already 2
        # i.e., v in N(j) ∩ N(k) and (N(u) ∩ N(j) ∩ N(k)) already has size 2
        # Similarly for triples containing v
        
        # Check triples (u, j, k): adding v to N(u) means we need
        # v in N(j) ∩ N(k) AND |N(u) ∩ N(j) ∩ N(k)| >= 2 (before adding)
        nu = adj[u]
        for j in range(n):
            if j == u:
                continue
            nj = adj[j]
            if not (nu & (1 << j)):  # u-j not an edge, but that's fine
                pass
            # v must be neighbor of j for v to be in N(j)
            if not (nj & (1 << v)):
                continue
            for k in range(n):
                if k == u or k == j:
                    continue
                nk = adj[k]
                if not (nk & (1 << v)):
                    continue
                # v is in N(j) ∩ N(k). Check if |N(u) ∩ N(j) ∩ N(k)| >= 2
                common = nu & nj & nk
                if common.bit_count() >= 2:
                    return False
        
        # Check triples (v, j, k): adding u to N(v)
        nv = adj[v]
        for j in range(n):
            if j == v:
                continue
            nj = adj[j]
            if not (nj & (1 << u)):
                continue
            for k in range(n):
                if k == v or k == j:
                    continue
                nk = adj[k]
                if not (nk & (1 << u)):
                    continue
                common = nv & nj & nk
                if common.bit_count() >= 2:
                    return False
        
        return True

    def greedy_add(adj):
        """Try to add edges greedily until no more can be added."""
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for u, v in all_pairs:
                if adj[u] & (1 << v):
                    continue
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
                    break
        return adj

    def conflict_based_repair(adj):
        """Hypergraph approach: find all K3,3 violations, build conflict structure,
        and remove edges to break all violations using a greedy MIS-like heuristic."""
        adj = adj[:]
        for iteration in range(200):
            if time.time() > deadline - 20:
                break
            # Find all violations
            violations = []
            for i in range(n):
                ai = adj[i]
                for j in range(i+1, n):
                    cij = ai & adj[j]
                    if cij.bit_count() < 3:
                        continue
                    for k in range(j+1, n):
                        cijk = cij & adj[k]
                        if cijk.bit_count() >= 3:
                            # Find the 3+ common neighbors
                            cn = cijk
                            common_neighbors = []
                            v = cn
                            while v:
                                low = v & (-v)
                                common_neighbors.append(low.bit_length() - 1)
                                v ^= low
                            violations.append((i, j, k, common_neighbors))
            
            if not violations:
                break
            
            # Build conflict: each edge removal can fix multiple violations
            # For each violation (i,j,k, C), removing edge (x, y) where x in {i,j,k} and y in C
            # fixes this violation if y is no longer a common neighbor
            # Actually, removing edge (x,y) where x in {i,j,k} and y in C removes y from
            # N(x), so y is no longer in N(i) ∩ N(j) ∩ N(k) if x is one of i,j,k.
            
            # Greedy: pick the edge whose removal fixes the most violations
            edge_fix_count = {}
            for (i, j, k, cn) in violations:
                for x in [i, j, k]:
                    for y in cn:
                        if y == x:
                            continue
                        key = (min(x,y), max(x,y))
                        edge_fix_count[key] = edge_fix_count.get(key, 0) + 1
            
            if not edge_fix_count:
                break
            
            # Pick edge with max fix count
            best_edge = max(edge_fix_count, key=edge_fix_count.get)
            u, v = best_edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        
        return adj

    def local_search(adj, max_iter=2000):
        """Local search: try adding edges, and if blocked, try swaps (1-1 and 2-2)."""
        adj = adj[:]
        iters = 0
        while iters < max_iter and time.time() < deadline - 15:
            # Try to add an edge
            added = False
            for u, v in all_pairs:
                if adj[u] & (1 << v):
                    continue
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added = True
                    break
            if added:
                iters += 1
                continue
            
            # Try a 1-1 swap: remove one edge, add another
            edges_in = [(i, j) for i, j in all_pairs if adj[i] & (1 << j)]
            edges_out = [(i, j) for i, j in all_pairs if not (adj[i] & (1 << j))]
            
            if not edges_out:
                break
            
            random.shuffle(edges_in)
            random.shuffle(edges_out)
            
            swapped = False
            for e_rem in edges_in[:25]:
                if swapped:
                    break
                for e_add in edges_out[:25]:
                    (u1, v1) = e_rem
                    (u2, v2) = e_add
                    if (u1, v1) == (u2, v2):
                        continue
                    # Check if swap is valid
                    old_u1, old_v1 = adj[u1], adj[v1]
                    old_u2, old_v2 = adj[u2], adj[v2]
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    if is_valid_fast(adj):
                        swapped = True
                        break
                    else:
                        adj[u1], adj[v1] = old_u1, old_v1
                        adj[u2], adj[v2] = old_u2, old_v2
                if not swapped:
                    continue
            
            if swapped:
                iters += 1
                continue
            
            # Try a 2-2 swap: remove two edges, add two edges
            if len(edges_in) >= 2 and len(edges_out) >= 2:
                two_swapped = False
                for idx1 in range(min(15, len(edges_in))):
                    if two_swapped:
                        break
                    for idx2 in range(idx1+1, min(15, len(edges_in))):
                        if two_swapped:
                            break
                        e_rem1 = edges_in[idx1]
                        e_rem2 = edges_in[idx2]
                        # Don't share vertices
                        if len(set([e_rem1[0], e_rem1[1], e_rem2[0], e_rem2[1]])) < 4:
                            continue
                        for idx3 in range(min(15, len(edges_out))):
                            if two_swapped:
                                break
                            for idx4 in range(idx3+1, min(15, len(edges_out))):
                                if two_swapped:
                                    break
                                e_add1 = edges_out[idx3]
                                e_add2 = edges_out[idx4]
                                if len(set([e_add1[0], e_add1[1], e_add2[0], e_add2[1]])) < 4:
                                    continue
                                u1, v1 = e_rem1
                                u2, v2 = e_rem2
                                u3, v3 = e_add1
                                u4, v4 = e_add2
                                old = (adj[u1], adj[v1], adj[u2], adj[v2], adj[u3], adj[v3], adj[u4], adj[v4])
                                adj[u1] &= ~(1 << v1)
                                adj[v1] &= ~(1 << u1)
                                adj[u2] &= ~(1 << v2)
                                adj[v2] &= ~(1 << u2)
                                adj[u3] |= (1 << v3)
                                adj[v3] |= (1 << u3)
                                adj[u4] |= (1 << v4)
                                adj[v4] |= (1 << u4)
                                if is_valid_fast(adj):
                                    two_swapped = True
                                    break
                                else:
                                    (adj[u1], adj[v1], adj[u2], adj[v2], adj[u3], adj[v3], adj[u4], adj[v4]) = old
                        if two_swapped:
                            break
                    if two_swapped:
                        break
                if two_swapped:
                    iters += 1
                    continue
            
            if not swapped and not two_swapped:
                break
        
        return adj

    # Best solution tracking
    best_adj = [0] * n
    best_count = 0
    best_edges = []

    def update_best(adj):
        nonlocal best_adj, best_count, best_edges
        c = count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]
            best_edges = adj_to_edges(adj)

    # Strategy 1: Start from complete graph, make valid, then improve
    adj = [(full_mask ^ (1 << i)) for i in range(n)]
    adj = make_valid(adj)
    adj = conflict_based_repair(adj)
    adj = greedy_add(adj)
    adj = local_search(adj)
    update_best(adj)
    
    # Strategy 1b: Direct conflict-based repair from complete graph
    adj = [(full_mask ^ (1 << i)) for i in range(n)]
    adj = conflict_based_repair(adj)
    adj = greedy_add(adj)
    adj = local_search(adj, max_iter=3000)
    update_best(adj)

    # Strategy 2: Try various structured initializations
    # Circulant graphs
    for s in range(5, 11):
        if time.time() > deadline - 30:
            break
        for trial in range(3):
            if time.time() > deadline - 30:
                break
            random.seed(42 + s * 10 + trial)
            # Random symmetric connection set of size s
            offsets = set()
            while len(offsets) < s:
                d = random.randint(1, n - 1)
                offsets.add(d)
                offsets.add(n - d)
            offsets.discard(0)
            
            adj = [0] * n
            for u in range(n):
                for d in offsets:
                    v = (u + d) % n
                    if v > u:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            if not is_valid_fast(adj):
                adj = make_valid(adj)
                adj = conflict_based_repair(adj)
            adj = greedy_add(adj)
            adj = local_search(adj)
            update_best(adj)

    # Strategy 3: Random dense graphs with local search
    for trial in range(8):
        if time.time() > deadline - 20:
            break
        random.seed(1000 + trial * 7)
        adj = [0] * n
        p = 0.5 + trial * 0.05  # increasing density
        for i, j in all_pairs:
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        adj = make_valid(adj)
        adj = conflict_based_repair(adj)
        adj = greedy_add(adj)
        adj = local_search(adj)
        update_best(adj)

    # Strategy 4: Start from best and do more local search
    if best_edges:
        adj = best_adj[:]
        adj = local_search(adj, max_iter=5000)
        update_best(adj)
        
        # Try more aggressive swaps with multiple restarts
        adj = best_adj[:]
        for _ in range(8):
            if time.time() > deadline - 5:
                break
            adj = local_search(adj, max_iter=2000)
            update_best(adj)
        
        # Final polish: try all possible single-edge additions
        adj = best_adj[:]
        for u, v in all_pairs:
            if time.time() > deadline - 2:
                break
            if not (adj[u] & (1 << v)):
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        update_best(adj)

    # Fallback
    if not best_edges:
        return [[i, i+1] for i in range(19)]

    return best_edges
