import time
import random
from collections import defaultdict, deque
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def hopcroft_karp(adj, n_left, n_right):
        """
        Hopcroft-Karp for maximum bipartite matching.
        adj[u] = list of right nodes adjacent to left node u.
        Returns (match_l, match_r) where match_l[u] = right node or -1.
        """
        match_l = [-1] * n_left
        match_r = [-1] * n_right
        dist = [0] * n_left
        
        def bfs():
            queue = deque()
            for u in range(n_left):
                if match_l[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    mu = match_r[v]
                    if mu == -1:
                        found = True
                    elif dist[mu] == float('inf'):
                        dist[mu] = dist[u] + 1
                        queue.append(mu)
            return found
        
        def dfs(u):
            for v in adj[u]:
                mu = match_r[v]
                if mu == -1 or (dist[mu] == dist[u] + 1 and dfs(mu)):
                    match_l[u] = v
                    match_r[v] = u
                    return True
            dist[u] = float('inf')
            return False
        
        matching = 0
        while bfs():
            for u in range(n_left):
                if match_l[u] == -1:
                    if dfs(u):
                        matching += 1
        return match_l, match_r

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        # Build list of K-edges
        k_edges = [(u, w) for u in range(k) for w in range(u + 1, k)]
        num_k_edges = len(k_edges)
        num_i_vertices = n - k
        i_vertices = list(range(k, n))
        
        # Build bipartite graph: left = K-edges, right = I-vertices
        adj = [[] for _ in range(num_k_edges)]
        for ei, (u, w) in enumerate(k_edges):
            for ii, v in enumerate(i_vertices):
                if u in neighbors[v] and w in neighbors[v]:
                    adj[ei].append(ii)
        
        # Find maximum matching
        match_l, match_r = hopcroft_karp(adj, num_k_edges, num_i_vertices)
        
        # Build cliques from matching
        cliques = []
        used_k_edges = set()
        for ei in range(num_k_edges):
            if match_l[ei] != -1:
                ii = match_l[ei]
                v = i_vertices[ii]
                u, w = k_edges[ei]
                cliques.append([u, w, v])
                used_k_edges.add(ei)
        
        # For each I-vertex, add 2-cliques for unmatched neighbors
        covered_by_3clique = {v: set() for v in i_vertices}
        for ei in range(num_k_edges):
            if match_l[ei] != -1:
                ii = match_l[ei]
                v = i_vertices[ii]
                u, w = k_edges[ei]
                covered_by_3clique[v].add(u)
                covered_by_3clique[v].add(w)
        
        for v in i_vertices:
            for u in neighbors[v]:
                if u not in covered_by_3clique[v]:
                    cliques.append([u, v])
        
        # Cover remaining K-edges
        remaining_edges = [(u, w) for ei, (u, w) in enumerate(k_edges) if ei not in used_k_edges]
        if remaining_edges:
            rem_set = set(remaining_edges)
            while rem_set:
                e = next(iter(rem_set))
                clique = {e[0], e[1]}
                changed = True
                while changed:
                    changed = False
                    for x in range(k):
                        if x not in clique:
                            if all((min(x, y), max(x, y)) in rem_set for y in clique):
                                clique.add(x)
                                changed = True
                if len(clique) >= 2:
                    cliques.append(sorted(clique))
                    for a, b in combinations(clique, 2):
                        rem_set.discard((min(a, b), max(a, b)))
                else:
                    cliques.append(list(e))
                    rem_set.discard(e)
        
        # Validate
        clean = [[int(x) for x in c] for c in cliques]
        
        edge_count = defaultdict(int)
        for c in clean:
            for i in range(len(c)):
                for j in range(i + 1, len(c)):
                    u, w = min(c[i], c[j]), max(c[i], c[j])
                    edge_count[(u, w)] += 1
        
        all_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                all_edges.add((u, w))
        for v in range(k, n):
            for u in neighbors[v]:
                all_edges.add((min(u, v), max(u, v)))
        
        valid = True
        for e in all_edges:
            if edge_count[e] != 1:
                valid = False
                break
        for e, cnt in edge_count.items():
            if e not in all_edges:
                valid = False
                break
        
        if not valid:
            cliques = []
            for v in range(k, n):
                for u in neighbors[v]:
                    cliques.append([u, v])
            for u in range(k):
                for w in range(u + 1, k):
                    cliques.append([u, w])
            return [[int(x) for x in c] for c in cliques]
        
        return clean

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)

    while len(results) < 3:
        results.append([])

    return results
