import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 148  # safety margin for 150s limit

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
        """Check if no 3 vertices have 3 common neighbours (K3,3-free)."""
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

    def can_add_edge(adj, u, v):
        """Fast check: can we add edge (u,v) without creating K3,3?"""
        # Adding edge (u,v) creates K3,3 if there exist a,b such that
        # u,v,a,b have >=3 common neighbors, or u,a,b have >=3 common neighbors including v, etc.
        # More precisely: after adding (u,v), check all triples that include u or v.
        # A triple (x,y,z) has >=3 common neighbors iff |N(x)&N(y)&N(z)| >= 3.
        # When we add (u,v), the only triples whose common neighbor set changes are those containing u or v.
        
        # For triples containing both u and v: (u,v,w) for any w
        # N(u)&N(v)&N(w) - since we're adding (u,v), u is now in N(v) and v in N(u)
        # So N(u)&N(v)&N(w) = (N(u) ∪ {v}) & (N(v) ∪ {u}) & N(w)
        # = (N(u)&N(v)&N(w)) ∪ (N(u)&{u}&N(w)) ∪ ({v}&N(v)&N(w)) ∪ ({v}&{u}&N(w))
        # The last three terms: u in N(w)? v in N(w)? u=v? No.
        # Actually simpler: just compute directly.
        
        nu = adj[u] | (1 << v)
        nv = adj[v] | (1 << u)
        
        # Check triples (u, v, w) for all w
        for w in range(n):
            if w == u or w == v:
                continue
            common = nu & nv & adj[w]
            if common.bit_count() >= 3:
                return False
        
        # Check triples (u, w1, w2) for all w1 < w2, w1 != v, w2 != v
        # N(u)&N(w1)&N(w2) with new N(u) = adj[u] | (1<<v)
        # = (adj[u]&adj[w1]&adj[w2]) | ({v}&adj[w1]&adj[w2])
        # The first part is already < 3 (graph was valid). So we only need to check if v is in adj[w1]&adj[w2]
        # AND if that makes the count >= 3.
        # Actually: new_common = (adj[u]&adj[w1]&adj[w2]) | ((1<<v) & adj[w1] & adj[w2])
        # = old_common | (v_in_w1w2 ? (1<<v) : 0)
        # So if old_common had exactly 2 bits and v is in both w1 and w2's neighbors, we get 3.
        
        for w1 in range(n):
            if w1 == u:
                continue
            for w2 in range(w1+1, n):
                if w2 == u or w2 == v:
                    continue
                old_common = adj[u] & adj[w1] & adj[w2]
                if old_common.bit_count() >= 3:
                    return False  # Shouldn't happen if graph was valid
                if old_common.bit_count() == 2:
                    # Check if v is a common neighbor of w1 and w2
                    if (adj[w1] & (1 << v)) and (adj[w2] & (1 << v)):
                        return False
        
        return True

    def make_valid(adj):
        """Remove edges to make the graph valid."""
        adj = adj[:]
        iterations = 0
        while iterations < 2000:
            viol = find_violation(adj)
            if viol is None:
                break
            i, j, k = viol
            cijk = adj[i] & adj[j] & adj[k]
            cn = cijk
            if cn == 0:
                break
            # Pick a common neighbor and remove an edge
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

    def local_search(adj, max_iter=5000):
        """Local search with simulated annealing: try adding edges, 1-swaps, 2-swaps."""
        adj = adj[:]
        iters = 0
        no_improve = 0
        best_local = count_edges(adj)
        
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
                    no_improve = 0
                    best_local = count_edges(adj)
                    break
            
            if added:
                iters += 1
                continue
            
            no_improve += 1
            
            # Try a 1-swap: remove one edge, add another
            edges_in = [(i, j) for i, j in all_pairs if adj[i] & (1 << j)]
            edges_out = [(i, j) for i, j in all_pairs if not (adj[i] & (1 << j))]
            
            if not edges_out:
                break
            
            random.shuffle(edges_in)
            random.shuffle(edges_out)
            
            swapped = False
            for e_rem in edges_in[:80]:
                if swapped:
                    break
                for e_add in edges_out[:80]:
                    (u1, v1) = e_rem
                    (u2, v2) = e_add
                    if (u1, v1) == (u2, v2):
                        continue
                    # Check if removing e_rem and adding e_add is valid
                    old_u1, old_v1, old_u2, old_v2 = adj[u1], adj[v1], adj[u2], adj[v2]
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    if is_valid(adj):
                        swapped = True
                        no_improve = 0
                        best_local = max(best_local, count_edges(adj))
                        break
                    else:
                        adj[u1], adj[v1], adj[u2], adj[v2] = old_u1, old_v1, old_u2, old_v2
            
            if swapped:
                iters += 1
                continue
            
            # Try a 2-swap: remove two edges, add two different edges
            if len(edges_in) >= 2 and len(edges_out) >= 2:
                sample_in = random.sample(edges_in, min(30, len(edges_in)))
                sample_out = random.sample(edges_out, min(30, len(edges_out)))
                
                two_swapped = False
                for i_rem in range(len(sample_in)):
                    if two_swapped:
                        break
                    for j_rem in range(i_rem + 1, len(sample_in)):
                        if two_swapped:
                            break
                        e_rem1 = sample_in[i_rem]
                        e_rem2 = sample_in[j_rem]
                        (u1, v1) = e_rem1
                        (u2, v2) = e_rem2
                        
                        old_u1, old_v1, old_u2, old_v2 = adj[u1], adj[v1], adj[u2], adj[v2]
                        
                        # Remove e_rem1 and e_rem2
                        adj[u1] &= ~(1 << v1)
                        adj[v1] &= ~(1 << u1)
                        adj[u2] &= ~(1 << v2)
                        adj[v2] &= ~(1 << u2)
                        
                        for i_add in range(len(sample_out)):
                            if two_swapped:
                                break
                            for j_add in range(i_add + 1, len(sample_out)):
                                if two_swapped:
                                    break
                                e_add1 = sample_out[i_add]
                                e_add2 = sample_out[j_add]
                                (w1, x1) = e_add1
                                (w2, x2) = e_add2
                                
                                if (e_add1 == e_rem1 or e_add1 == e_rem2 or e_add2 == e_rem1 or e_add2 == e_rem2):
                                    continue
                                
                                adj[w1] |= (1 << x1)
                                adj[x1] |= (1 << w1)
                                adj[w2] |= (1 << x2)
                                adj[x2] |= (1 << w2)
                                
                                if is_valid(adj):
                                    two_swapped = True
                                    no_improve = 0
                                    best_local = max(best_local, count_edges(adj))
                                    break
                                else:
                                    adj[w1] &= ~(1 << x1)
                                    adj[x1] &= ~(1 << w1)
                                    adj[w2] &= ~(1 << x2)
                                    adj[x2] &= ~(1 << w2)
                        
                        if two_swapped:
                            break
                        else:
                            adj[u1], adj[v1], adj[u2], adj[v2] = old_u1, old_v1, old_u2, old_v2
                
                if two_swapped:
                    iters += 1
                    continue
            
            # If stuck for too long, try a random restart perturbation
            if no_improve > 200:
                # Randomly remove a few edges to escape local minimum
                for _ in range(random.randint(2, 5)):
                    if not edges_in:
                        break
                    u, v = random.choice(edges_in)
                    if adj[u] & (1 << v):
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                no_improve = 0
                iters += 1
                continue
            
            if no_improve > 500:
                break
            
            iters += 1
        
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
    adj = greedy_add(adj)
    adj = local_search(adj, max_iter=3000)
    update_best(adj)

    # Strategy 2: Random dense graphs with local search
    for trial in range(15):
        if time.time() > deadline - 30:
            break
        random.seed(2000 + trial * 13)
        adj = [0] * n
        p = 0.4 + trial * 0.03
        for i, j in all_pairs:
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        adj = make_valid(adj)
        adj = greedy_add(adj)
        adj = local_search(adj, max_iter=2000)
        update_best(adj)

    # Strategy 3: Structured initializations (circulant-like)
    for s in range(6, 14):
        if time.time() > deadline - 25:
            break
        for trial in range(3):
            if time.time() > deadline - 25:
                break
            random.seed(5000 + s * 10 + trial)
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
            
            if not is_valid(adj):
                adj = make_valid(adj)
            adj = greedy_add(adj)
            adj = local_search(adj, max_iter=2000)
            update_best(adj)

    # Strategy 4: Final polish on best solution
    if best_edges:
        adj = best_adj[:]
        adj = local_search(adj, max_iter=10000)
        update_best(adj)
        
        # More aggressive restarts from best
        for _ in range(8):
            if time.time() > deadline - 5:
                break
            adj = best_adj[:]
            # Perturb: remove a few random edges
            edges_in = [(i, j) for i, j in all_pairs if adj[i] & (1 << j)]
            for _ in range(random.randint(3, 8)):
                if not edges_in:
                    break
                u, v = random.choice(edges_in)
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    edges_in = [(i, j) for i, j in all_pairs if adj[i] & (1 << j)]
            adj = greedy_add(adj)
            adj = local_search(adj, max_iter=5000)
            update_best(adj)

    # Fallback
    if not best_edges:
        return [[i, i+1] for i in range(19)]

    return best_edges
