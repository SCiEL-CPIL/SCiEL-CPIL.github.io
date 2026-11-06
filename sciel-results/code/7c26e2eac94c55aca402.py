import time
import random

def construct():
    t0 = time.time()
    TOTAL_DEADLINE = t0 + 140.0  # Safety margin before 150s kill

    def gen_edges(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def greedy_partition(n, adj, edges, seed, deadline):
        rng = random.Random(seed)
        remaining = set(edges)
        cliques = []
        
        while remaining and time.time() < deadline:
            # Calculate degrees in the remaining graph
            degs = [0] * n
            for u, v in remaining:
                degs[u] += 1
                degs[v] += 1
            
            # Find top 5 vertices by degree
            top_vertices = sorted(range(n), key=lambda x: degs[x], reverse=True)[:5]
            top_vertices = [v for v in top_vertices if degs[v] > 0]
            
            if not top_vertices:
                break

            best_clique = None
            best_coverage = 0
            
            for v_start in top_vertices:
                # Find a maximal clique containing v_start using greedy expansion
                C = {v_start}
                candidates = set(adj[v_start])
                
                # Shuffle candidates for randomness
                candidates_list = list(candidates)
                rng.shuffle(candidates_list)
                
                # Greedy expansion loop
                while candidates_list:
                    best_cand = None
                    best_score = -1
                    
                    for v in candidates_list:
                        # Check if v is connected to all vertices in current clique C
                        is_valid = True
                        for u in C:
                            if v not in adj[u]:
                                is_valid = False
                                break
                        if not is_valid:
                            continue
                        
                        # Score: degree in remaining graph + small random noise
                        score = degs[v] + rng.random() * 0.1
                        
                        if score > best_score:
                            best_score = score
                            best_cand = v
                    
                    if best_cand is None:
                        break
                    
                    C.add(best_cand)
                    candidates_list.remove(best_cand)
                    # Filter candidates to those connected to best_cand
                    candidates_list = [v for v in candidates_list if v in adj[best_cand]]
                
                # Calculate how many remaining edges this clique covers
                coverage = 0
                C_list = list(C)
                for i in range(len(C_list)):
                    for j in range(i + 1, len(C_list)):
                        u, v = C_list[i], C_list[j]
                        e = (u, v) if u < v else (v, u)
                        if e in remaining:
                            coverage += 1
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_clique = C

            if best_clique is None or best_coverage == 0:
                # Fallback: pick a single edge
                e = next(iter(remaining))
                best_clique = {e[0], e[1]}
            
            cliques.append(list(best_clique))
            
            # Remove edges covered by best_clique from remaining
            best_clique_list = list(best_clique)
            for i in range(len(best_clique_list)):
                for j in range(i + 1, len(best_clique_list)):
                    u, v = best_clique_list[i], best_clique_list[j]
                    e = (u, v) if u < v else (v, u)
                    if e in remaining:
                        remaining.discard(e)
                        
        # Add any leftover edges as individual cliques
        for e in remaining:
            cliques.append([e[0], e[1]])
            
        return cliques

    def merge_pass(cliques, adj, deadline):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            T = len(cliques)
            for i in range(T):
                if changed:
                    break
                for j in range(i + 1, T):
                    if changed:
                        break
                    Ci = set(cliques[i])
                    Cj = set(cliques[j])
                    Union = Ci | Cj
                    # Check if Union is a clique
                    is_clique = True
                    for u in Union:
                        for v in Union:
                            if u != v and v not in adj[u]:
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    if is_clique:
                        cliques[i] = list(Union)
                        del cliques[j]
                        changed = True
                        break
        return cliques

    def solve_instance(n, p, seed, budget):
        edges = gen_edges(n, p, seed)
        adj = build_adj(n, edges)
        
        best_partition = None
        best_count = float('inf')
        
        start_time = time.time()
        attempt = 0
        while time.time() < budget:
            attempt += 1
            # Use a slightly earlier deadline for the greedy part to leave time for merging
            partition = greedy_partition(n, adj, edges, seed + attempt, budget - 1.0)
            partition = merge_pass(partition, adj, budget - 0.5)
            
            if len(partition) < best_count:
                best_count = len(partition)
                best_partition = partition
                if best_count == 1:
                    break
        
        if best_partition is None:
            best_partition = [[u, v] for u, v in edges]
            
        return best_partition

    # Time allocation
    # n=32 is easier, n=64 is harder.
    # Total budget: 140s.
    # Give 25s to n=32, 110s to n=64, 5s overhead.
    
    t1_deadline = t0 + 25.0
    t2_deadline = TOTAL_DEADLINE - 1.0
    
    res1 = solve_instance(32, 0.6, 601, t1_deadline)
    res2 = solve_instance(64, 0.55, 602, t2_deadline)
    
    return [res1, res2]
