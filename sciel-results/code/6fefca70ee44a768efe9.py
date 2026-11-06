"""Construct C6-free graph on 40 vertices with maximum edges."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit

    n = 40
    full_mask = (1 << n) - 1
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def has_c6_fast(adj, u, v):
        """
        Check if adding edge (u,v) creates a C6.
        C6: u-a-b-c-d-v-u
        a in N(u)\{v}, d in N(v)\{u}, a!=d, and dist(a,d)=2 in G\{u,v}.
        """
        if (adj[u] >> v) & 1:
            return True

        nu_mask = adj[u] & ~(1 << v)
        nv_mask = adj[v] & ~(1 << u)

        a = nu_mask
        while a:
            lb_a = a & (-a)
            ai = lb_a.bit_length() - 1
            a ^= lb_a

            d = nv_mask & ~lb_a
            while d:
                lb_d = d & (-d)
                di = lb_d.bit_length() - 1
                d ^= lb_d

                # b in N(a) \ {u, v, d, a}
                b_cand = adj[ai] & ~(1 << u) & ~(1 << v) & ~lb_d & ~lb_a
                # c in N(d) \ {u, v, a, d}
                c_cand = adj[di] & ~(1 << u) & ~(1 << v) & ~lb_a & ~lb_d

                # Check if any neighbor of b is in c_cand
                bm = b_cand
                while bm:
                    lb_b = bm & (-bm)
                    bi = lb_b.bit_length() - 1
                    bm ^= lb_b
                    if adj[bi] & c_cand:
                        return True
        return False

    def try_add_edge(adj, u, v):
        if not has_c6_fast(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            return True
        return False

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def local_search(adj):
        """
        Iterative local search: try to add any missing edge that doesn't create C6.
        """
        improved = True
        while improved:
            if time.time() > deadline:
                return adj
            improved = False
            for u, v in all_pairs:
                if not ((adj[u] >> v) & 1):
                    if try_add_edge(adj, u, v):
                        improved = True
        return adj

    def find_c6_and_remove(adj):
        """
        Find a C6 in the graph and remove one of its edges.
        Returns True if a C6 was found and an edge removed.
        """
        # Check all pairs of non-adjacent vertices (u, v) as potential ends of the path
        for u in range(n):
            nu = adj[u]
            # v must be non-adjacent to u
            nv_candidates = full_mask & ~nu & ~(1 << u)
            
            v_mask = nv_candidates
            while v_mask:
                lb_v = v_mask & (-v_mask)
                vi = lb_v.bit_length() - 1
                v_mask ^= lb_v
                
                # Path u-a-b-c-d-v
                a_mask = nu
                while a_mask:
                    lb_a = a_mask & (-a_mask)
                    ai = lb_a.bit_length() - 1
                    a_mask ^= lb_a

                    d_mask = adj[vi] & ~lb_a
                    while d_mask:
                        lb_d = d_mask & (-d_mask)
                        di = lb_d.bit_length() - 1
                        d_mask ^= lb_d

                        b_cand = adj[ai] & ~(1 << u) & ~(1 << vi) & ~lb_d & ~lb_a
                        c_cand = adj[di] & ~(1 << u) & ~(1 << vi) & ~lb_a & ~lb_d

                        bm = b_cand
                        while bm:
                            lb_b = bm & (-bm)
                            bi = lb_b.bit_length() - 1
                            bm ^= lb_b
                            if adj[bi] & c_cand:
                                # Found C6: u-ai-bi-?-di-vi
                                # Remove a random edge from the cycle
                                if random.random() < 0.5:
                                    remove_edge(adj, u, ai)
                                else:
                                    remove_edge(adj, di, vi)
                                return True
        return False

    random.seed(42)
    best_adj = None
    best_count = 0
    n_restarts = 50

    for restart in range(n_restarts):
        if time.time() > deadline:
            break

        adj = [0] * n

        # Random initialization with density ~0.5
        random.seed(1000 + restart)
        for u, v in all_pairs:
            if random.random() < 0.5:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

        # Prune C6s
        while True:
            if time.time() > deadline:
                break
            if not find_c6_and_remove(adj):
                break

        # Local search to fill in missing edges
        adj = local_search(adj)

        current_count = count_edges(adj)
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]

        if best_count >= 105:
            break

    # Iterated Local Search (ILS) phase
    if best_adj is not None:
        current_adj = best_adj[:]
        current_count = best_count
        
        # Simulated annealing style perturbation
        temp = 1.0
        cooling = 0.995
        
        while time.time() < deadline:
            # Perturb: remove k random edges
            k = random.randint(2, 5)
            adj = current_adj[:]
            current_edges = get_edges(adj)
            if len(current_edges) > k:
                removed = random.sample(current_edges, k)
                for u, v in removed:
                    remove_edge(adj, u, v)

            # Repair: greedy add
            shuffled = all_pairs[:]
            random.shuffle(shuffled)
            for u, v in shuffled:
                if time.time() > deadline:
                    break
                if not ((adj[u] >> v) & 1):
                    try_add_edge(adj, u, v)

            # Local search
            adj = local_search(adj)

            new_count = count_edges(adj)
            
            # Accept if better or with probability exp((new-old)/T)
            if new_count > current_count:
                current_adj = adj
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
                temp = 1.0 # Reset temp on improvement
            else:
                # Accept worse solution to escape local optima
                delta = new_count - current_count
                if delta < 0:
                    prob = np.exp(delta / temp)
                    if random.random() < prob:
                        current_adj = adj
                        current_count = new_count
                temp *= cooling

    return get_edges(best_adj)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Edges: {len(edges)}, Time: {t1-t0:.2f}s")
