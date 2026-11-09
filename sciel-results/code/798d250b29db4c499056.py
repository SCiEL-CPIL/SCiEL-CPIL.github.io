import random
import time

def construct():
    n = 16
    target_edges = 65
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def get_score_and_worst(edges_set):
        adj = [0] * n
        for u, v in edges_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        deg = [adj[i].bit_count() for i in range(n)]
        
        max_sum = 0
        worst_tri = None
        for u, v in edges_set:
            common = adj[u] & adj[v]
            c = common
            while c:
                lsb = c & -c
                w = lsb.bit_length() - 1
                s = deg[u] + deg[v] + deg[w]
                if s > max_sum:
                    max_sum = s
                    worst_tri = (u, v, w)
                c ^= lsb
        return max_sum, worst_tri

    def random_graph():
        edges = set()
        while len(edges) < target_edges:
            edges.add(random.choice(all_pairs))
        return edges

    best_score = 999
    best_edges = None
    
    start_time = time.time()
    timeout = 140
    
    while time.time() - start_time < timeout:
        current_edges = random_graph()
        current_score, current_worst = get_score_and_worst(current_edges)
        
        if current_score < best_score:
            best_score = current_score
            best_edges = set(current_edges)
        
        T = 5.0
        T_min = 0.01
        alpha = 0.999
        
        inner_start = time.time()
        while time.time() - inner_start < 8: # 8s per restart to allow more restarts
            if time.time() - start_time > timeout:
                break
            
            # Move selection
            # 80% random, 20% targeted
            if random.random() < 0.2 and current_worst:
                # Pick an edge from the worst triangle
                u, v, w = current_worst
                # Edges are (u,v), (u,w), (v,w)
                # Ensure they are in the set (they must be)
                possible_edges = [(u, v), (u, w), (v, w)]
                e_to_remove = random.choice(possible_edges)
            else:
                e_to_remove = random.choice(list(current_edges))
            
            while True:
                u, v = random.choice(all_pairs)
                if (u, v) not in current_edges:
                    e_to_add = (u, v)
                    break
            
            new_edges = set(current_edges)
            new_edges.remove(e_to_remove)
            new_edges.add(e_to_add)
            
            new_score, new_worst = get_score_and_worst(new_edges)
            
            if new_score < current_score or random.random() < pow(2.718281828, (current_score - new_score) / T):
                current_edges = new_edges
                current_score = new_score
                current_worst = new_worst
                if current_score < best_score:
                    best_score = current_score
                    best_edges = set(current_edges)
            
            T *= alpha
            if T < T_min:
                T = T_min

    edge_list = [list(e) for e in best_edges]
    return {'n': n, 'edges': edge_list}
