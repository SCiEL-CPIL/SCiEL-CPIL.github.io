import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 140

    # Initialize complete graph
    edges = []
    edge_index = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edges.append([u, v, random.randint(0, 19)])
            edge_index[(u, v)] = idx
            idx += 1
            
    def get_violating_cycles():
        # Returns list of lists of edge indices
        violations = []
        # Adjacency list of active edges
        adj = {i: set() for i in range(n)}
        for i, e in enumerate(edges):
            if e[2] is not None:
                u, v = e[0], e[1]
                adj[u].add(v)
                adj[v].add(u)
                
        for u in range(n):
            for w in range(u + 1, n):
                common = adj[u] & adj[w]
                common_list = list(common)
                if len(common_list) < 2:
                    continue
                for i in range(len(common_list)):
                    for j in range(i + 1, len(common_list)):
                        v = common_list[i]
                        x = common_list[j]
                        if x in adj[v]:
                            # Cycle u-v-w-x-u
                            def get_color(a, b):
                                key = (min(a,b), max(a,b))
                                if key in edge_index:
                                    idx = edge_index[key]
                                    if edges[idx][2] is not None:
                                        return edges[idx][2]
                                return None
                            
                            c1 = get_color(u, v)
                            c2 = get_color(v, w)
                            c3 = get_color(w, x)
                            c4 = get_color(x, u)
                            
                            if c1 is None or c2 is None or c3 is None or c4 is None:
                                continue
                                
                            if len({c1, c2, c3, c4}) < 4:
                                idx1 = edge_index[(min(u,v), max(u,v))]
                                idx2 = edge_index[(min(v,w), max(v,w))]
                                idx3 = edge_index[(min(w,x), max(w,x))]
                                idx4 = edge_index[(min(x,u), max(x,u))]
                                violations.append([idx1, idx2, idx3, idx4])
        return violations

    while time.time() < deadline:
        violations = get_violating_cycles()
        if not violations:
            break
            
        # Count edge frequencies in violations
        freq = {}
        for v in violations:
            for e_idx in v:
                freq[e_idx] = freq.get(e_idx, 0) + 1
                
        # Pick the most frequent edge
        max_count = -1
        target_idx = -1
        for e_idx, count in freq.items():
            if count > max_count:
                max_count = count
                target_idx = e_idx
                
        # Remove the edge
        edges[target_idx][2] = None

    valid_edges = [e for e in edges if e[2] is not None]
    return valid_edges
