import random
import itertools

def construct():
    n = 16
    # Start with K_8,8 + 1 edge
    edges = set()
    for u in range(8):
        for v in range(8, 16):
            edges.add((u, v))
    edges.add((0, 1))
    
    def get_degree_sum(edges_set):
        deg = [0] * n
        for u, v in edges_set:
            deg[u] += 1
            deg[v] += 1
        max_sum = 0
        # Find triangles
        adj = [set() for _ in range(n)]
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    for w in adj[v]:
                        if w > v and w in adj[u]:
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    current_score = get_degree_sum(edges)
    
    # Local search
    for _ in range(1000):
        # Pick a random edge to remove
        e_to_remove = random.choice(list(edges))
        # Pick a random edge to add
        # Generate all possible edges
        all_edges = set()
        for i in range(n):
            for j in range(i+1, n):
                all_edges.add((i, j))
        possible_adds = [e for e in all_edges if e not in edges]
        if not possible_adds:
            break
        e_to_add = random.choice(possible_adds)
        
        new_edges = set(edges)
        new_edges.remove(e_to_remove)
        new_edges.add(e_to_add)
        
        new_score = get_degree_sum(new_edges)
        if new_score < current_score:
            edges = new_edges
            current_score = new_score
            if current_score <= 23:
                break

    # Convert to list of lists
    edge_list = [[u, v] for u, v in edges]
    return {'n': n, 'edges': edge_list}
