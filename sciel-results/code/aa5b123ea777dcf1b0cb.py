import time
import random

def construct():
        start_time = time.time()
        deadline = start_time + 110.0 # 10s margin
        
        n = 16
        target_edges = 65
        
        # Precompute all triangles (triples)
        all_triples = [(u, v, w) for u in range(n) for v in range(u+1, n) for w in range(v+1, n)]
        
        def get_cost(adj, deg):
            max_sum = 0
            for u, v, w in all_triples:
                if adj[u][v] and adj[v][w] and adj[u][w]:
                    s = deg[u] + deg[v] + deg[w]
                    if s > max_sum:
                        max_sum = s
            return max_sum

        def make_adj(edges):
            adj = [[False]*n for _ in range(n)]
            deg = [0]*n
            for u, v in edges:
                adj[u][v] = True
                adj[v][u] = True
                deg[u] += 1
                deg[v] += 1
            return adj, deg

        def random_swap(adj, deg, edges):
            # Pick random edge to remove
            e1 = random.choice(list(edges))
            # Pick random non-edge to add
            # Generate random non-edge
            while True:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if u > v: u, v = v, u
                if not adj[u][v]:
                    e2 = (u, v)
                    break
            
            # Apply swap
            edges.remove(e1)
            edges.add(e2)
            
            u1, v1 = e1
            u2, v2 = e2
            
            adj[u1][v1] = False
            adj[v1][u1] = False
            deg[u1] -= 1
            deg[v1] -= 1
            
            adj[u2][v2] = True
            adj[v2][u2] = True
            deg[u2] += 1
            deg[v2] += 1
            
            return adj, deg, edges

        def simulate_annealing(initial_edges, iterations=50000):
            edges = set(initial_edges)
            adj, deg = make_adj(edges)
            current_cost = get_cost(adj, deg)
            best_cost = current_cost
            best_edges = set(edges)
            
            T = 10.0
            T_min = 0.1
            cooling = 0.999
            
            for i in range(iterations):
                if time.time() > deadline:
                    break
                
                new_adj, new_deg, new_edges = random_swap(adj, deg, edges)
                new_cost = get_cost(new_adj, new_deg)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    edges = new_edges
                    adj = new_adj
                    deg = new_deg
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_edges = set(edges)
                else:
                    # Accept with probability
                    if random.random() < pow(2.71828, -(new_cost - current_cost) / T):
                        current_cost = new_cost
                        edges = new_edges
                        adj = new_adj
                        deg = new_deg
                
                T = max(T_min, T * cooling)
            
            return best_edges, best_cost

        # Initial Graph 1: K_{8,8} + 1 edge
        init_edges_1 = set([(u, v) for u in range(8) for v in range(8, 16)])
        init_edges_1.add((0, 1))
        
        best_edges_overall = init_edges_1
        best_score_overall = 26 # Seed score
        
        # Run SA 1
        e1, s1 = simulate_annealing(init_edges_1)
        if s1 < best_score_overall:
            best_score_overall = s1
            best_edges_overall = e1
            
        # Initial Graph 2: Random
        all_possible_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_possible_edges)
        init_edges_2 = set(all_possible_edges[:target_edges])
        
        # Run SA 2
        e2, s2 = simulate_annealing(init_edges_2)
        if s2 < best_score_overall:
            best_score_overall = s2
            best_edges_overall = e2
            
        # Convert best_edges to list of lists
        result_edges = [[u, v] for u, v in best_edges_overall]
        
        return {'n': 16, 'edges': result_edges}
