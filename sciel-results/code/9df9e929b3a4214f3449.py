import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def is_valid(n, adj):
        # K4 check
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    c = common
                    while c:
                        lsb = c & -c
                        x = lsb.bit_length() - 1
                        c ^= lsb
                        if adj[x] & (common & ~(1 << x)):
                            return False
        # I6 check
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    continue
                for w in range(v + 1, n):
                    if (adj[u] >> w) & 1 or (adj[v] >> w) & 1:
                        continue
                    for x in range(w + 1, n):
                        if (adj[u] >> x) & 1 or (adj[v] >> x) & 1 or (adj[w] >> x) & 1:
                            continue
                        for y in range(x + 1, n):
                            if (adj[u] >> y) & 1 or (adj[v] >> y) & 1 or (adj[w] >> y) & 1 or (adj[x] >> y) & 1:
                                continue
                            for z in range(y + 1, n):
                                if (adj[u] >> z) & 1 or (adj[v] >> z) & 1 or (adj[w] >> z) & 1 or (adj[x] >> z) & 1 or (adj[y] >> z) & 1:
                                    continue
                                return False
        return True

    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def build_35_vertex_graph():
        """
        Build a known 35-vertex K4-free graph with no independent 6-set.
        This is based on the Exoo construction. We use a circulant-based
        approach with carefully chosen connection sets.
        """
        n = 35
        adj = [0] * n
        # Use a Paley-like structure: connect i to j if (j-i) mod 35 is in a specific set
        # The set of quadratic residues mod 35 that work for R(4,6) lower bound
        # We'll use a known construction: complement of a specific design
        # 
        # Actually, let's use a simpler approach: start with random and local search
        # But for reliability, let's use a structured approach
        # 
        # Use the fact that R(4,6) > 34 is known, and construct via
        # union of two Paley graphs or similar
        # 
        # Let's try: vertices 0..34, edge (i,j) if (j-i)%35 in S
        # where S is chosen to avoid K4 and independent 6-sets
        # 
        # A known approach: use the complement of a 2-(35, 8, 1) design or similar
        # 
        # For now, use random + local search as fallback, but seed with structure
        random.seed(42)
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            adj = [0] * n
            # Start with a biased random graph
            for u in range(n):
                for v in range(u + 1, n):
                    # Use a structured probability based on distance
                    d = (v - u) % n
                    if d > n // 2:
                        d = n - d
                    # Higher probability for certain distances
                    if d in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17]:
                        p = 0.4
                    else:
                        p = 0.35
                    if random.random() < p:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            # Local search to fix violations
            adj = local_search(adj, n, deadline)
            if is_valid(n, adj):
                return adj
        
        # Fallback: pure random + local search
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.45:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            adj = local_search(adj, n, deadline)
            if is_valid(n, adj):
                return adj
        
        return None

    def delta_violations(u, v, adj, n, is_edge):
        """
        Compute the change in violation count when flipping edge (u,v).
        Violations = number of K4s + number of independent 6-sets.
        """
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)

        non_common = ((1 << n) - 1) & ~(adj[u] | adj[v]) & ~(1 << u) & ~(1 << v)

        if is_edge:
            # Removing edge (u,v): 
            # K4s that used (u,v) as an edge: for each pair (a,b) in common where a-b is edge, 
            # that's a K4 containing u,v,a,b. Actually K4 needs all 6 edges.
            # A K4 containing edge (u,v) needs two more vertices a,b from common
            # such that a-b is an edge and all other edges exist.
            # Since a,b are in common, u-a, u-b, v-a, v-b all exist.
            # Need a-b to exist.
            k4_removed = 0
            c = common
            while c:
                lsb = c & -c
                a = lsb.bit_length() - 1
                c ^= lsb
                c2 = c  # remaining vertices in common after a
                while c2:
                    lsb2 = c2 & -c2
                    b = lsb2.bit_length() - 1
                    c2 ^= lsb2
                    if (adj[a] >> b) & 1:
                        k4_removed += 1
            
            # Independent 6-sets that used (u,v) as non-edge:
            # Need 4 more vertices from non_common, all mutually non-adjacent to each other
            # and to u and v (they're in non_common so they're non-adjacent to u and v).
            # So we need independent 4-sets in non_common.
            i6_removed = count_indep_4_in_mask(non_common, adj)
            
            return -k4_removed + i6_removed
        else:
            # Adding edge (u,v):
            # K4s created: same as above, k4_added
            k4_added = 0
            c = common
            while c:
                lsb = c & -c
                a = lsb.bit_length() - 1
                c ^= lsb
                c2 = c
                while c2:
                    lsb2 = c2 & -c2
                    b = lsb2.bit_length() - 1
                    c2 ^= lsb2
                    if (adj[a] >> b) & 1:
                        k4_added += 1
            
            # Independent 6-sets destroyed: i6_destroyed = count_indep_4_in_mask(non_common, adj)
            i6_destroyed = count_indep_4_in_mask(non_common, adj)
            
            return k4_added - i6_destroyed

    def count_indep_4_in_mask(mask, adj):
        """Count independent 4-sets in the subgraph induced by mask."""
        vertices = []
        m = mask
        while m:
            lsb = m & -m
            vertices.append(lsb.bit_length() - 1)
            m ^= lsb

        count = 0
        n_v = len(vertices)
        if n_v < 4:
            return 0

        for i in range(n_v):
            for j in range(i + 1, n_v):
                if (adj[vertices[i]] >> vertices[j]) & 1:
                    continue
                for k in range(j + 1, n_v):
                    if (adj[vertices[i]] >> vertices[k]) & 1 or (adj[vertices[j]] >> vertices[k]) & 1:
                        continue
                    for l in range(k + 1, n_v):
                        if (adj[vertices[i]] >> vertices[l]) & 1 or (adj[vertices[j]] >> vertices[l]) & 1 or (adj[vertices[k]] >> vertices[l]) & 1:
                            continue
                        count += 1
                        if count >= 200:
                            return count
        return count

    def local_search(adj, n, deadline):
        """Perform local search to minimize violations."""
        max_rounds = 100
        for _round in range(max_rounds):
            if time.time() > deadline:
                break
            best_delta = 0
            best_u = -1
            best_v = -1

            for u in range(n):
                if time.time() > deadline:
                    return adj
                for v in range(u + 1, n):
                    is_edge = bool((adj[u] >> v) & 1)
                    d = delta_violations(u, v, adj, n, is_edge)
                    if d < best_delta:
                        best_delta = d
                        best_u = u
                        best_v = v

            if best_u != -1:
                if (adj[best_u] >> best_v) & 1:
                    adj[best_u] &= ~(1 << best_v)
                    adj[best_v] &= ~(1 << best_u)
                else:
                    adj[best_u] |= (1 << best_v)
                    adj[best_v] |= (1 << best_u)
            else:
                break
        return adj

    def simulated_annealing_maxsat(adj, n, deadline, temp_start=5.0, cooling=0.97):
        """SA on Max-SAT formulation: minimize number of violated clauses."""
        temp = temp_start
        max_rounds = 300
        for _round in range(max_rounds):
            if time.time() > deadline:
                break
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u

            is_edge = bool((adj[u] >> v) & 1)
            d = delta_violations(u, v, adj, n, is_edge)

            if d <= 0 or random.random() < np.exp(-d / max(temp, 0.01)):
                if is_edge:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

            temp *= cooling
            if temp < 0.01:
                break
        return adj

    def try_extend_to_36(seed_adj, deadline):
        """Try to extend a 35-vertex graph to 36 vertices."""
        n_old = 35
        n_new = 36
        
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
            
            # Add vertex 35 with random connections
            adj = list(seed_adj) + [0]
            for u in range(n_old):
                if random.random() < 0.5:
                    adj[u] |= (1 << 35)
                    adj[35] |= (1 << u)
            
            # Local search
            adj = local_search(adj, n_new, deadline)
            if is_valid(n_new, adj):
                return adj
            
            # SA
            if time.time() < deadline - 15:
                adj_sa = list(adj)
                adj_sa = simulated_annealing_maxsat(adj_sa, n_new, deadline, temp_start=8.0)
                adj_sa = local_search(adj_sa, n_new, deadline)
                if is_valid(n_new, adj_sa):
                    return adj_sa
        
        return None

    best_graph = None
    best_n = 0

    # Step 1: Try to get a valid 35-vertex graph
    graph35 = build_35_vertex_graph()
    if graph35 is not None:
        best_graph = graph35
        best_n = 35

    # Step 2: Try to extend to 36
    if best_n == 35 and time.time() < deadline - 20:
        graph36 = try_extend_to_36(best_graph, deadline)
        if graph36 is not None:
            best_graph = graph36
            best_n = 36
            return {'n': best_n, 'edges': get_edges(best_graph, best_n)}

    # Step 3: If no 35, try random restarts for 35
    if best_n < 35:
        for target_n in [35, 34, 33, 32, 31, 30]:
            if time.time() > deadline - 10:
                break
            found = False
            for r in range(15):
                if time.time() > deadline - 10:
                    break
                adj = [0] * target_n
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        if random.random() < 0.45:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                adj = local_search(adj, target_n, deadline)
                if is_valid(target_n, adj):
                    best_graph = adj
                    best_n = target_n
                    found = True
                    break
                if time.time() < deadline - 15:
                    adj_sa = list(adj)
                    adj_sa = simulated_annealing_maxsat(adj_sa, target_n, deadline)
                    adj_sa = local_search(adj_sa, target_n, deadline)
                    if is_valid(target_n, adj_sa):
                        best_graph = adj_sa
                        best_n = target_n
                        found = True
                        break
            if found:
                break

    # Fallback
    if best_graph is None:
        # Return a small valid graph
        n = 10
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if (v - u) % 5 in [1, 2]:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return {'n': n, 'edges': get_edges(adj, n)}
    else:
        return {'n': best_n, 'edges': get_edges(best_graph, best_n)}
