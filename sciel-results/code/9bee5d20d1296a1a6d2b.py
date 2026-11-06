import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n = 16
    target_edges = 65
    
    def get_adjacency(edges_set):
        adj = [set() for _ in range(n)]
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def get_degrees(adj):
        return [len(adj[i]) for i in range(n)]

    def calculate_score(adj, degrees):
        max_sum = 0
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    common = adj[u].intersection(adj[v])
                    for w in common:
                        s = degrees[u] + degrees[v] + degrees[w]
                        if s > max_sum:
                            max_sum = s
        return max_sum

    def optimize(initial_edges, time_limit):
        edges = set(initial_edges)
        adj = get_adjacency(edges)
        degrees = get_degrees(adj)
        current_score = calculate_score(adj, degrees)
        best_score = current_score
        best_edges = set(edges)
        
        while time.time() < time_limit:
            # Identify the worst triangle
            worst_triangle = None
            worst_sum = -1
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        common = adj[u].intersection(adj[v])
                        for w in common:
                            s = degrees[u] + degrees[v] + degrees[w]
                            if s > worst_sum:
                                worst_sum = s
                                worst_triangle = (u, v, w)
            
            if worst_triangle is None:
                break
                
            # Try to improve by removing an edge from the worst triangle
            # and adding a random non-edge
            improved = False
            u, v, w = worst_triangle
            
            # Candidates for removal: (u,v), (v,w), (u,w)
            candidates = [(u, v), (v, w), (u, w)]
            random.shuffle(candidates)
            
            for rem_edge in candidates:
                if rem_edge[0] > rem_edge[1]:
                    rem_edge = (rem_edge[1], rem_edge[0])
                
                # Try to find a good edge to add
                for _ in range(20):
                    x = random.randint(0, n-1)
                    y = random.randint(0, n-1)
                    if x == y:
                        continue
                    if x > y:
                        x, y = y, x
                    if (x, y) in edges:
                        continue
                    if (x, y) == rem_edge:
                        continue
                    
                    edges.discard(rem_edge)
                    edges.add((x, y))
                    
                    new_adj = get_adjacency(edges)
                    new_degrees = get_degrees(new_adj)
                    new_score = calculate_score(new_adj, new_degrees)
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_edges = set(edges)
                        current_score = new_score
                        adj = new_adj
                        degrees = new_degrees
                        improved = True
                        break
                    else:
                        edges.discard((x, y))
                        edges.add(rem_edge)
                        
                if improved:
                    break
                    
            if not improved:
                # Random restart or random walk to escape local minimum
                # Try a few random swaps regardless of score to jump out
                for _ in range(5):
                    rem = random.choice(list(edges))
                    x = random.randint(0, n-1)
                    y = random.randint(0, n-1)
                    if x == y: continue
                    if x > y: x, y = y, x
                    if (x, y) in edges or (x, y) == rem: continue
                    
                    edges.discard(rem)
                    edges.add((x, y))
                    new_adj = get_adjacency(edges)
                    new_degrees = get_degrees(new_adj)
                    new_score = calculate_score(new_adj, new_degrees)
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_edges = set(edges)
                        current_score = new_score
                        adj = new_adj
                        degrees = new_degrees
                    else:
                        # Revert
                        edges.discard((x, y))
                        edges.add(rem)

        return best_edges, best_score

    # Strategy 1: K_{8,8} + 1
    edges_k88 = set()
    for u in range(8):
        for v in range(8, 16):
            edges_k88.add((u, v))
    edges_k88.add((0, 1))
    
    best_edges_1, score_1 = optimize(edges_k88, deadline - 5)
    
    # Strategy 2: Random Graph
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_possible)
    edges_rand = set(all_possible[:target_edges])
    
    best_edges_2, score_2 = optimize(edges_rand, deadline)
    
    if score_1 <= score_2:
        final_edges = best_edges_1
    else:
        final_edges = best_edges_2
        
    return {'n': n, 'edges': list(final_edges)}
