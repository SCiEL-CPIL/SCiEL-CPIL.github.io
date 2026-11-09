import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 15
    
    def make_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def get_degrees(adj):
        return [len(adj[i]) for i in range(n)]
    
    def is_valid(adj):
        degs = get_degrees(adj)
        deg_groups = defaultdict(list)
        for i in range(n):
            deg_groups[degs[i]].append(i)
        
        for deg, vertices in deg_groups.items():
            if len(vertices) < 2:
                continue
            for i, s in enumerate(vertices):
                for t in vertices[i+1:]:
                    # Check for simple path s-u-v-t
                    # s, u, v, t must be distinct
                    found = False
                    for u in adj[s]:
                        if u == t:
                            continue
                        for v in adj[u]:
                            if v == s or v == t or v == u:
                                continue
                            if t in adj[v]:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        return False
        return True
    
    # Start with K_{7,8}
    A = list(range(7))   # 0..6
    B = list(range(7, 15))  # 7..14
    
    base_edges = []
    for a in A:
        for b in B:
            base_edges.append([a, b])
    
    adj = make_adj(base_edges)
    
    if not is_valid(adj):
        return [[0, v] for v in range(1, n)]
    
    best_edges = list(base_edges)
    
    # Candidate edges: within A and within B
    candidate_edges = []
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            candidate_edges.append([A[i], A[j]])
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            candidate_edges.append([B[i], B[j]])
    
    random.shuffle(candidate_edges)
    for edge in candidate_edges:
        if time.time() > deadline:
            break
        u, v = edge
        if v in adj[u]:
            continue
        adj[u].add(v)
        adj[v].add(u)
        if is_valid(adj):
            best_edges.append(edge)
        else:
            adj[u].remove(v)
            adj[v].remove(u)
    
    # Local search
    all_possible = []
    for i in range(n):
        for j in range(i+1, n):
            if j not in adj[i]:
                all_possible.append([i, j])
    
    random.shuffle(all_possible)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(all_possible)):
            if time.time() > deadline:
                break
            edge = all_possible[i]
            u, v = edge
            if v in adj[u]:
                continue
            adj[u].add(v)
            adj[v].add(u)
            if is_valid(adj):
                best_edges.append(edge)
                improved = True
                all_possible.pop(i)
                i -= 1
            else:
                adj[u].remove(v)
                adj[v].remove(u)
    
    result = [[int(u), int(v)] for u, v in best_edges]
    return result
