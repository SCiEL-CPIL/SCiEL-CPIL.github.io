import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges_total = len(all_edges)

    # Build conflict graph: two edges conflict if they can form K2,3
    # Edge e=(u,v) conflicts with edge f=(x,y) if:
    # - They share exactly one endpoint, and the other two endpoints have >=2 common neighbors
    # - They are disjoint, and {u,v} and {x,y} have >=3 common neighbors
    # Actually, K2,3 requires two vertices with 3 common neighbors.
    # Two edges create a K2,3 violation if:
    # 1. They share a vertex w, and the other endpoints u and v have >=2 common neighbors (not counting each other's edges)
    # 2. They are disjoint, and the 4 endpoints form a situation where some pair has 3+ common neighbors
    
    # Simpler: build conflict graph where edge i conflicts with edge j if adding both creates a K2,3
    # Two edges (u,v) and (x,y) conflict if:
    # - They share exactly one vertex, say w. Then u and v (the non-shared endpoints) 
    #   need to have at least 2 common neighbors in the remaining graph, OR
    # - They are disjoint. Then we need to check if any pair among {u,v,x,y} has >=3 common neighbors
    #   that includes vertices from both edges.
    
    # Actually, the cleanest way: two edges conflict if there exist two vertices a,b 
    # such that both edges are in N(a) ∩ N(b), and |N(a) ∩ N(b)| >= 3.
    # But that's circular. Let me think differently.
    
    # A K2,3 subgraph consists of two vertices {a,b} with three common neighbors {c,d,e}.
    # The edges forming the K2,3 are: a-c, a-d, a-e, b-c, b-d, b-e.
    # So two edges e1 and e2 conflict if they can be part of the same K2,3.
    # Two edges are in the same K2,3 if:
    # - They share a vertex w, and the other endpoints u,v satisfy: there exist vertices a,b 
    #   such that w is one of {a,b}, and u,v are among the three common neighbors.
    #   This means u and v have at least 2 common neighbors (one being w, and at least one more).
    #   Wait, if they share vertex w, then for them to be in a K2,3, we need another vertex x 
    #   such that x is also adjacent to both u and v. So u and v need at least 2 common neighbors 
    #   (w and x), and we need a third common neighbor y for the K2,3.
    #   Actually no: if edges are (w,u) and (w,v), they're in a K2,3 if there exist x,y such that 
    #   x,y are both adjacent to u and v, AND one of {w,x,y} pairs with another to form the K2,3.
    #   The K2,3 would be: vertices {a,b} with common neighbors {c,d,e}.
    #   If edges are (w,u) and (w,v), then w is one of {a,b}, and u,v are two of {c,d,e}.
    #   We need a third common neighbor of u and v (besides w). So u and v need >= 2 common neighbors.
    
    # Let me just build the conflict graph by checking all pairs.
    
    # For each pair of edges, check if they can be in a K2,3 subgraph.
    # Two edges e1=(u,v) and e2=(x,y) can be in a K2,3 if:
    # Case 1: They share exactly one vertex, say w.
    #   Then u and v (non-shared) need at least 2 common neighbors in the full graph.
    #   (One is w, need at least one more, say z. Then {w,z} are the two vertices of K2,3,
    #    and {u,v,?} are three common neighbors. We need a third common neighbor of u and v.)
    #   Actually: if edges are (w,u) and (w,v), and u,v have common neighbors {w, z1, z2}, 
    #   then K2,3 exists with bipartition {z1, z2} and {u, v, w} if z1,z2 are both adjacent to u,v,w.
    #   Wait, z1 and z2 need to be adjacent to all of u,v,w. But w is only adjacent to u and v (and others).
    #   Let me re-examine: K2,3 has vertices {a,b} on one side, {c,d,e} on other. Edges are all a-c,a-d,a-e,b-c,b-d,b-e.
    #   If two edges are (w,u) and (w,v), then w is on one side, u,v on the other. 
    #   We need another vertex x on w's side such that x is adjacent to u,v, and a third vertex y 
    #   on u,v's side adjacent to both w and x. So u,v need a common neighbor x (besides w), 
    #   and x needs to be adjacent to w. And we need a third common neighbor y of u,v adjacent to w.
    #   This is getting complicated. Let me just check: does adding both edges create a K2,3?
    #   I'll check all pairs of vertices for having 3+ common neighbors after adding both edges.
    
    # For efficiency, I'll precompute the conflict graph.
    # Two edges conflict if they cannot both be in a K2,3-free graph.
    
    # Let's build the conflict graph properly.
    # Edge i = (u,v), edge j = (x,y).
    # They conflict if there exist vertices a,b such that:
    # - Both edges have one endpoint in {a,b} and one in the complement, OR
    # - More precisely, the union of the two edges is a subset of some K2,3.
    
    # A K2,3 has 6 edges. Two edges are in the same K2,3 if:
    # - They share a vertex w, and the other endpoints u,v have at least 2 common neighbors (so there's a K2,3 with w and one of those neighbors as the two "center" vertices)
    # - OR they are disjoint, and the four endpoints have the property that some pair has 3+ common neighbors including the other two.
    
    # Let me just compute it directly for each pair.
    
    # Precompute adjacency for the complete graph (all possible edges present)
    # For conflict detection, we need to know: given two specific edges, can they coexist in a K2,3-free graph?
    # Answer: NO if there exist vertices a,b such that both edges are "between" a,b and their common neighbors,
    # and a,b have 3+ common neighbors total.
    
    # Simpler approach: two edges e1=(u,v) and e2=(x,y) conflict if:
    # There exist distinct vertices a,b such that:
    # - {a,b} are the two "left" vertices of K2,3
    # - u,v,x,y are among the "right" vertices (the 3 common neighbors)
    # - Both e1 and e2 connect a left vertex to a right vertex
    # - The three right vertices are all adjacent to both a and b
    
    # For two specific edges to be in a K2,3:
    # Case 1: They share a vertex w.
    #   Let e1=(w,u), e2=(w,v). Then w is a left vertex. u,v are right vertices.
    #   We need a second left vertex x such that x is adjacent to u and v.
    #   And we need a third right vertex y adjacent to both w and x.
    #   So: u,v have a common neighbor x != w, and x is adjacent to w, and there's y adjacent to w and x, y != u,v.
    #   Actually, we just need: u and v have at least 2 common neighbors (one is w, need one more x),
    #   and x is adjacent to w (which it is if x is a common neighbor of u,v and w is... wait, x is adjacent to u and v, not necessarily w).
    #   Hmm. Let me just check: if we have edges (w,u) and (w,v), when do they form part of a K2,3?
    #   K2,3: left={a,b}, right={c,d,e}. Edges: a-c,a-d,a-e,b-c,b-d,b-e.
    #   If w=a, u=c, v=d, then we need b adjacent to c,d, and e adjacent to a,b.
    #   So: c,d have a common neighbor b (b != a), and a,b have a common neighbor e (e != c,d).
    #   So u,v need a common neighbor b != w, and w,b need a common neighbor e != u,v.
    #   
    # Case 2: Disjoint edges e1=(u,v), e2=(x,y).
    #   They could be: u=a, v=c, x=b, y=d. Then we need e adjacent to a,b and c,d adjacent to a,b.
    #   So: c,d have common neighbors a,b (which they do if edges exist), and a,b have common neighbor e.
    #   Also need: a adjacent to d, b adjacent to c (for the K2,3 edges).
    #   So: a,b have a common neighbor e, a is adjacent to d, b is adjacent to c.
    #   But in our graph, a=u, b=x, c=v, d=y. So we need: u,x have common neighbor e, u adjacent to y, x adjacent to v.
    #   But u-y and x-v are NOT the edges we're considering (we have u-v and x-y).
    #   So this case requires additional edges to exist.
    
    # This is getting complex. Let me just build the conflict graph by checking: 
    # for each pair of edges, simulate adding both to an empty graph and check if any K2,3 exists.
    # But that's O(m^2 * n^2) which might be too slow.
    
    # Alternative: use the fact that K2,3-free means no two vertices have 3+ common neighbors.
    # Two edges conflict if they force some pair to have 3+ common neighbors.
    # Edge (u,v) and edge (x,y):
    # - If they share vertex w: say (w,u) and (w,v). Then for any z adjacent to both u and v (z != w), 
    #   the pair (w,z) has common neighbors including u,v. If there's a third, we have K2,3.
    #   So (w,u) and (w,v) conflict if u,v have 2+ common neighbors (besides each other's contribution).
    #   Actually, if u,v have common neighbors {w, z1, z2}, then (w,z1) has common neighbors {u,v,?}. 
    #   We need 3 common neighbors for some pair. Pair (w,z1): common neighbors are vertices adjacent to both w and z1.
    #   u is adjacent to w and z1? u is adjacent to w (edge w-u) and z1 (if z1 is common neighbor of u,v). 
    #   So u is a common neighbor of w and z1. Similarly v. And z2 if z2 adjacent to w.
    #   So if u,v have 2+ common neighbors (including w), then w and any other common neighbor z have 2+ common neighbors (u,v),
    #   and if there's a third, K2,3.
    
    # I think the cleanest is: precompute for each pair of edges whether they conflict by checking
    # all pairs of vertices for 3+ common neighbors in the subgraph induced by just those two edges 
    # plus the necessary connections. But since we're building the conflict graph for the purpose of 
    # finding an independent set, we need: two edges conflict if they CANNOT both be in a K2,3-free graph.
    # This happens if and only if there exist vertices a,b such that both edges are "witnesses" 
    # for a,b having 3+ common neighbors.
    
    # Let me just do a direct check for each pair.
    
    # For edges (u,v) and (x,y), they conflict if there exist distinct a,b such that:
    # - a is adjacent to v and y (or some combination) such that a,b have 3 common neighbors
    # This is still complex. Let me just use a simpler sufficient condition and hope it's tight enough.
    
    # Actually, for the branch-and-bound to work, I need the EXACT conflict graph.
    # Let me build it by: for each pair of edges, check if adding both to an empty graph 
    # creates a K2,3. Since the graph only has 2 edges, a K2,3 can only exist if the 2 edges 
    # are part of a K2,3 structure. With only 2 edges, we can't have a full K2,3 (which needs 6 edges).
    # So two edges NEVER conflict by themselves!
    
    # Wait, that's not right either. The conflict graph for MIS should represent: 
    # two edges conflict if they cannot both be in a K2,3-free graph. But with only 2 edges, 
    # any pair can be in a K2,3-free graph (since 2 edges can't form K2,3). 
    # The conflicts arise when we consider the full set of edges chosen.
    
    # I think I'm overcomplicating this. Let me reconsider the problem.
    # We want to find a maximum set of edges such that no two vertices have 3+ common neighbors.
    # This is NOT a standard independent set problem because the constraint is global, not pairwise.
    # Two edges don't directly conflict; rather, a SET of edges conflicts if it creates a K2,3.
    
    # So branch-and-bound on a conflict graph won't directly work unless I reformulate.
    # Let me stick with the simulated annealing approach but improve it.
    
    # I'll use the existing approach but with better initialization and more aggressive search.
    
    def get_edges_from_mask(mask):
        edges = []
        for i in range(num_edges_total):
            if (mask >> i) & 1:
                edges.append(list(all_edges[i]))
        return edges

    def init_adj(mask):
        adj = [0] * n
        for i in range(num_edges_total):
            if (mask >> i) & 1:
                u, v = all_edges[i]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def count_violations(adj):
        v = 0
        for u in range(n):
            for w in range(u + 1, n):
                if (adj[u] & adj[w]).bit_count() >= 3:
                    v += 1
        return v

    def can_add(u, v, adj):
        if (adj[u] & adj[v]).bit_count() >= 3:
            return False
        for w in range(n):
            if w == u or w == v:
                continue
            if (adj[w] >> v) & 1:
                if (adj[u] & adj[w]).bit_count() >= 2:
                    return False
            if (adj[w] >> u) & 1:
                if (adj[v] & adj[w]).bit_count() >= 2:
                    return False
        return True

    def greedy_fill(adj):
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if (adj[u] >> v) & 1:
                    continue
                if can_add(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True

    def adj_to_mask(adj):
        mask = 0
        for i in range(num_edges_total):
            u, v = all_edges[i]
            if (adj[u] >> v) & 1:
                mask |= (1 << i)
        return mask

    best_mask = 0
    best_edge_count = 0

    # Use multiple random restarts with simulated annealing
    num_runs = 30
    for run in range(num_runs):
        if time.time() > deadline - 3.0:
            break

        # Random initialization with varying density
        p = random.uniform(0.10, 0.45)
        current_mask = 0
        for i in range(num_edges_total):
            if random.random() < p:
                current_mask |= (1 << i)

        adj = init_adj(current_mask)
        violations = count_violations(adj)
        edge_count = bin(current_mask).count('1')

        if violations == 0 and edge_count > best_edge_count:
            best_edge_count = edge_count
            best_mask = current_mask

        T = 3.0
        cooling = 0.997
        min_T = 0.03
        steps = 0
        max_steps = 50000

        while T > min_T and steps < max_steps:
            if time.time() > deadline - 3.0:
                break

            idx = random.randint(0, num_edges_total - 1)
            u, v = all_edges[idx]
            is_present = (current_mask >> idx) & 1

            if is_present:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                new_violations = count_violations(adj)
                new_edge_count = edge_count - 1
                old_energy = violations * 100 - edge_count
                new_energy = new_violations * 100 - new_edge_count
                delta_energy = new_energy - old_energy
                if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                    current_mask ^= (1 << idx)
                    violations = new_violations
                    edge_count = new_edge_count
                    if violations == 0 and edge_count > best_edge_count:
                        best_edge_count = edge_count
                        best_mask = current_mask
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            else:
                if can_add(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    new_violations = violations
                    new_edge_count = edge_count + 1
                    old_energy = violations * 100 - edge_count
                    new_energy = new_violations * 100 - new_edge_count
                    delta_energy = new_energy - old_energy
                    if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                        current_mask |= (1 << idx)
                        violations = new_violations
                        edge_count = new_edge_count
                        if violations == 0 and edge_count > best_edge_count:
                            best_edge_count = edge_count
                            best_mask = current_mask
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    new_violations = violations + 1
                    new_edge_count = edge_count + 1
                    old_energy = violations * 100 - edge_count
                    new_energy = new_violations * 100 - new_edge_count
                    delta_energy = new_energy - old_energy
                    if random.random() < math.exp(-delta_energy / T):
                        current_mask |= (1 << idx)
                        violations = new_violations
                        edge_count = new_edge_count
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)

            T *= cooling
            steps += 1

        # Try to repair if there are violations
        if violations > 0:
            for _ in range(300):
                if time.time() > deadline - 2.0:
                    break
                found = False
                for a in range(n):
                    for b in range(a + 1, n):
                        if (adj[a] & adj[b]).bit_count() >= 3:
                            # Find an edge to remove that involves a or b
                            candidates = []
                            for i in range(num_edges_total):
                                eu, ev = all_edges[i]
                                if (current_mask >> i) & 1:
                                    if eu == a or eu == b or ev == a or ev == b:
                                        candidates.append(i)
                            if candidates:
                                rem_idx = random.choice(candidates)
                                ru, rv = all_edges[rem_idx]
                                adj[ru] &= ~(1 << rv)
                                adj[rv] &= ~(1 << ru)
                                current_mask ^= (1 << rem_idx)
                                violations = count_violations(adj)
                                edge_count = bin(current_mask).count('1')
                                found = True
                                break
                    if found:
                        break
                if violations == 0:
                    break

        if violations == 0:
            greedy_fill(adj)
            fill_mask = adj_to_mask(adj)
            fill_count = bin(fill_mask).count('1')
            if fill_count > best_edge_count:
                best_edge_count = fill_count
                best_mask = fill_mask

    if best_mask != 0:
        adj = init_adj(best_mask)
        greedy_fill(adj)
        best_mask = adj_to_mask(adj)

    return get_edges_from_mask(best_mask)

def _construct_greedy():
    n = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.seed(42)
    shuffled = all_edges[:]
    random.shuffle(shuffled)
    adj = [0] * n
    edges = []
    for u, v in shuffled:
        if (adj[u] & adj[v]).bit_count() < 3:
            ok = True
            for w in range(n):
                if w == u or w == v:
                    continue
                if (adj[u] & adj[w]).bit_count() + (1 if (adj[w] & (1 << v)) else 0) >= 3:
                    ok = False
                    break
                if ok:
                    if (adj[v] & adj[w]).bit_count() + (1 if (adj[w] & (1 << u)) else 0) >= 3:
                        ok = False
                        break
            if ok:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
    return edges

def _greedy_construct(n, all_edges, seed):
    pass

def _can_add_edge_fast(adj, u, v):
    pass

def _local_search(n, edges, all_edges, deadline):
    pass
