import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [[False] * n for _ in range(n)]
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u][v] = True
                    adj[v][u] = True
                    edges.append((u, v))
        return adj, edges

    def bron_kerbosch(adj, n):
        cliques = []
        
        def bk(R, P, X):
            if time.time() > deadline:
                raise TimeoutError("BK Timeout")
                
            if P == 0 and X == 0:
                if R != 0:
                    # Check size >= 2
                    # Count bits
                    if bin(R).count('1') >= 2:
                        cliques.append([i for i in range(n) if (R >> i) & 1])
                return
            
            px = P | X
            if px == 0:
                return
            
            # Pivot u
            u_bit = px & (-px)
            u = u_bit.bit_length() - 1
            
            N_u = 0
            for i in range(n):
                if adj[u][i]:
                    N_u |= (1 << i)
            
            candidates = P & (~N_u)
            
            while candidates:
                v_bit = candidates & (-candidates)
                v = v_bit.bit_length() - 1
                candidates ^= v_bit
                
                N_v = 0
                for i in range(n):
                    if adj[v][i]:
                        N_v |= (1 << i)
                
                bk(R | v_bit, P & N_v, X & N_v)
                
                P &= ~v_bit
                X |= v_bit

        all_vertices = (1 << n) - 1
        bk(0, all_vertices, 0)
        return cliques

    def solve_instance(n, p, seed):
        adj, edges = random_graph(n, p, seed)
        
        if not edges:
            return []

        try:
            cliques = bron_kerbosch(adj, n)
        except TimeoutError:
            # If BK times out, we can't guarantee correctness.
            # Fallback: Return all non-isolated vertices?
            # Or just return empty? Empty is invalid if there are cliques.
            # Given N=40, BK should not timeout.
            # But if it does, we return a safe (though large) solution.
            touched = set()
            for u, v in edges:
                touched.add(u)
                touched.add(v)
            return sorted(touched)

        if not cliques:
            return []

        k = len(cliques)
        all_cliques_mask = (1 << k) - 1
        
        vertex_mask = [0] * n
        for i, clique in enumerate(cliques):
            bit = 1 << i
            for v in clique:
                vertex_mask[v] |= bit

        # Greedy Initialization
        def greedy_cover():
            covered = 0
            size = 0
            avail = [True] * n
            masks = vertex_mask[:]
            selected = []
            
            while covered != all_cliques_mask:
                best_v = -1
                best_new = 0
                for v in range(n):
                    if not avail[v]:
                        continue
                    new_cov = masks[v] & (~covered)
                    cnt = bin(new_cov).count('1')
                    if cnt > best_new:
                        best_new = cnt
                        best_v = v
                
                if best_v == -1:
                    break
                
                size += 1
                covered |= masks[best_v]
                avail[best_v] = False
                selected.append(best_v)
                
            return size, selected

        init_size, init_sol = greedy_cover()
        best_size = init_size
        best_solution = init_sol

        # Dominance Pruning
        dominated = [False] * n
        for i in range(n):
            if vertex_mask[i] == 0:
                dominated[i] = True
                continue
            for j in range(n):
                if i == j:
                    continue
                if (vertex_mask[i] & ~vertex_mask[j]) == 0:
                    # i is subset of j
                    # If j is not 0, i is dominated
                    if vertex_mask[j] != 0:
                        dominated[i] = True
                        break
        
        candidate_vertices = [v for v in range(n) if not dominated[v]]
        candidate_vertices.sort(key=lambda v: bin(vertex_mask[v]).count('1'), reverse=True)

        def search(covered, size, path, start_idx):
            nonlocal best_size, best_solution
            
            if time.time() > deadline:
                raise TimeoutError("Search Timeout")
                
            if size >= best_size:
                return
                
            if covered == all_cliques_mask:
                best_size = size
                best_solution = path[:]
                return

            uncovered = all_cliques_mask & (~covered)
            
            # Lower Bound: 
            # If we can't cover any more, return
            # Simple LB: 1
            # Better: Check if any candidate covers anything
            has_coverage = False
            for i in range(start_idx, len(candidate_vertices)):
                v = candidate_vertices[i]
                if vertex_mask[v] & uncovered:
                    has_coverage = True
                    break
            if not has_coverage:
                return

            for i in range(start_idx, len(candidate_vertices)):
                v = candidate_vertices[i]
                mask_v = vertex_mask[v]
                new_cov = mask_v & uncovered
                
                if new_cov == 0:
                    continue
                
                search(covered | new_cov, size + 1, path + [v], i + 1)
                
                if best_size == size + 1:
                    break

        try:
            search(0, 0, [], 0)
        except TimeoutError:
            pass # Use greedy

        return best_solution

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        sol = solve_instance(n, p, seed)
        results.append(sol)

    return results
