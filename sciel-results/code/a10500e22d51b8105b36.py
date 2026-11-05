import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n = 18
    full_mask = (1 << n) - 1

    def popcount(x):
        return bin(x).count('1')

    def is_triangle_free(mask, adj):
        """Check if the induced subgraph on mask is triangle-free."""
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if adj[v] & (mask ^ lsb) & adj[v]:
                # Check if any two neighbors of v in mask are adjacent
                neighbors_in_mask = adj[v] & (mask ^ lsb)
                if neighbors_in_mask:
                    # For each neighbor, check if it has another neighbor in mask
                    nm = neighbors_in_mask
                    while nm:
                        lsb2 = nm & -nm
                        w = lsb2.bit_length() - 1
                        if adj[w] & (mask ^ lsb) & (mask ^ lsb2):
                            return False
                        nm ^= lsb2
            m ^= lsb
        return True

    def largest_triangle_free(adj, time_limit=None):
        """Find the size of the largest triangle-free induced subgraph using BnB."""
        best = [0]
        
        # Greedy lower bound: build a triangle-free set
        # Start with empty set, add vertices that don't create triangles
        tf_set = 0
        remaining = full_mask
        while remaining:
            # Find vertex with min degree in remaining
            best_v = -1
            best_deg = n + 1
            m = remaining
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = popcount(adj[v] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                m ^= lsb
            # Check if adding best_v keeps triangle-free
            if not (adj[best_v] & tf_set):
                # No edges from best_v to tf_set, so no triangle
                tf_set |= (1 << best_v)
            else:
                # Check if adding best_v creates a triangle
                neighbors_in_tf = adj[best_v] & tf_set
                creates_triangle = False
                nm = neighbors_in_tf
                while nm:
                    lsb = nm & -nm
                    w = lsb.bit_length() - 1
                    if adj[w] & tf_set & (tf_set ^ lsb):
                        creates_triangle = True
                        break
                    nm ^= lsb
                if not creates_triangle:
                    tf_set |= (1 << best_v)
            remaining &= ~(1 << best_v)
        best[0] = popcount(tf_set)
        
        # Better greedy: try multiple orderings
        for seed in range(5):
            random.seed(seed + 1000)
            tf_set2 = 0
            remaining2 = full_mask
            verts = list(range(n))
            random.shuffle(verts)
            for v in verts:
                if remaining2 & (1 << v):
                    # Check if adding v keeps triangle-free
                    neighbors_in_tf = adj[v] & tf_set2
                    creates_triangle = False
                    nm = neighbors_in_tf
                    while nm:
                        lsb = nm & -nm
                        w = lsb.bit_length() - 1
                        if adj[w] & tf_set2 & (tf_set2 ^ lsb):
                            creates_triangle = True
                            break
                        nm ^= lsb
                    if not creates_triangle:
                        tf_set2 |= (1 << v)
            if popcount(tf_set2) > best[0]:
                best[0] = popcount(tf_set2)
            random.seed()  # Reset
        
        def bnb(S_mask, C_mask, depth=0):
            if time_limit and time.time() > time_limit:
                return
            sz = popcount(S_mask)
            if sz + popcount(C_mask) <= best[0]:
                return
            
            if not C_mask:
                if sz > best[0]:
                    best[0] = sz
                return
            
            # Upper bound: sz + popcount(C_mask)
            # Select vertex with max degree in C
            max_deg = -1
            chosen_v = -1
            c = C_mask
            while c:
                lsb = c & -c
                vid = lsb.bit_length() - 1
                deg_in_c = popcount(adj[vid] & C_mask)
                if deg_in_c > max_deg:
                    max_deg = deg_in_c
                    chosen_v = vid
                c ^= lsb
            
            v = chosen_v
            v_bit = 1 << v
            
            # Branch: exclude v
            bnb(S_mask, C_mask & ~v_bit, depth + 1)
            
            # Branch: include v (check if it maintains triangle-free)
            # v can be added if no two neighbors of v in S_mask are adjacent
            neighbors_in_S = adj[v] & S_mask
            valid = True
            nm = neighbors_in_S
            while nm:
                lsb = nm & -nm
                w = lsb.bit_length() - 1
                if adj[w] & S_mask & (S_mask ^ lsb):
                    valid = False
                    break
                nm ^= lsb
            
            if valid:
                bnb(S_mask | v_bit, C_mask & ~v_bit, depth + 1)
        
        bnb(0, full_mask)
        return best[0]

    def turan_graph_k3():
        """Construct Turán graph T(18,3) - 3-partite complete graph."""
        adj = [0] * n
        # Partition vertices into 3 groups of 6
        groups = [[i for i in range(6)], [i for i in range(6, 12)], [i for i in range(12, 18)]]
        for i in range(3):
            for j in range(i + 1, 3):
                for u in groups[i]:
                    for v in groups[j]:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        return adj

    def is_k4_free(adj):
        """Check if graph is K4-free."""
        for i in range(n):
            for j in range(i + 1, n):
                if not ((adj[i] >> j) & 1):
                    continue
                common = adj[i] & adj[j]
                # Check if any two common neighbors are adjacent
                c = common
                while c:
                    lsb1 = c & -c
                    u = lsb1.bit_length() - 1
                    c2 = c & ~lsb1
                    while c2:
                        lsb2 = c2 & -c2
                        v = lsb2.bit_length() - 1
                        if (adj[u] >> v) & 1:
                            return False
                        c2 ^= lsb2
                    c = c & ~lsb1
        return True

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K4-free."""
        if (adj[u] >> v) & 1:
            return True
        common = adj[u] & adj[v]
        # Check if common neighbors form an independent set
        c = common
        while c:
            lsb1 = c & -c
            a = lsb1.bit_length() - 1
            c2 = c & ~lsb1
            while c2:
                lsb2 = c2 & -c2
                b = lsb2.bit_length() - 1
                if (adj[a] >> b) & 1:
                    return False
                c2 ^= lsb2
            c = c & ~lsb1
        return True

    def random_k4_free_adj():
        """Generate a random K4-free graph."""
        adj = [0] * n
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    # Start with Turán graph
    best_adj = turan_graph_k3()
    best_score = largest_triangle_free(best_adj, time_limit=deadline - 5.0)
    print(f"Initial Turán: {best_score}")

    # Local search: try edge additions and swaps
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    improved = True
    iterations = 0
    while improved and time.time() < deadline - 10.0:
        improved = False
        iterations += 1
        random.shuffle(all_edges)
        
        # Try adding edges
        for u, v in all_edges:
            if time.time() > deadline - 10.0:
                break
            if not ((best_adj[u] >> v) & 1):
                if can_add_edge(best_adj, u, v):
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
                    new_score = largest_triangle_free(best_adj, time_limit=deadline - 10.0)
                    if new_score < best_score:
                        best_score = new_score
                        improved = True
                        print(f"Iter {iterations}: Add edge ({u},{v}) -> {best_score}")
                    else:
                        best_adj[u] &= ~(1 << v)
                        best_adj[v] &= ~(1 << u)
        
        # Try edge swaps: remove one edge, add another
        if time.time() < deadline - 10.0:
            current_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if (best_adj[i] >> j) & 1]
            random.shuffle(current_edges)
            for rem_u, rem_v in current_edges:
                if time.time() > deadline - 10.0:
                    break
                # Remove edge
                best_adj[rem_u] &= ~(1 << rem_v)
                best_adj[rem_v] &= ~(1 << rem_u)
                
                # Try adding a different edge
                for add_u, add_v in all_edges:
                    if time.time() > deadline - 10.0:
                        break
                    if (add_u, add_v) == (rem_u, rem_v) or (add_u, add_v) == (rem_v, rem_u):
                        continue
                    if not ((best_adj[add_u] >> add_v) & 1):
                        if can_add_edge(best_adj, add_u, add_v):
                            best_adj[add_u] |= (1 << add_v)
                            best_adj[add_v] |= (1 << add_u)
                            new_score = largest_triangle_free(best_adj, time_limit=deadline - 10.0)
                            if new_score < best_score:
                                best_score = new_score
                                improved = True
                                print(f"Iter {iterations}: Swap ({rem_u},{rem_v})->({add_u},{add_v}) -> {best_score}")
                                break
                            else:
                                best_adj[add_u] &= ~(1 << add_v)
                                best_adj[add_v] &= ~(1 << add_u)
                
                # Restore removed edge if no improvement
                if not improved:
                    best_adj[rem_u] |= (1 << rem_v)
                    best_adj[rem_v] |= (1 << rem_u)
                else:
                    # Edge was removed and a new one added, keep state
                    break

    # Try some random restarts if time allows
    if time.time() < deadline - 15.0:
        for _ in range(3):
            if time.time() > deadline - 10.0:
                break
            adj = random_k4_free_adj()
            score = largest_triangle_free(adj, time_limit=deadline - 10.0)
            if score < best_score:
                best_score = score
                best_adj = adj[:]
                print(f"Random restart: {best_score}")

    # Final evaluation
    final_score = largest_triangle_free(best_adj, time_limit=deadline - 2.0)
    if final_score < best_score:
        best_score = final_score

    return get_edge_list(best_adj)
