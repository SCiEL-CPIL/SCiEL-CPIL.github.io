import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 138  # Safety margin
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_edges_set = set(all_edges)
    
    def build_adj(edge_set):
        adj = set()
        for u, v in edge_set:
            adj.add((u, v))
            adj.add((v, u))
        return adj
    
    def contains_q3(A, B, adj):
        """Check if bipartite graph between A (4 verts) and B (4 verts) contains Q3.
        Q3 has 12 edges out of 16 possible. Non-edges must form a matching (<=4 edges, no shared vertices)."""
        non_edges = []
        for a in A:
            for b in B:
                if (a, b) not in adj:
                    non_edges.append((a, b))
        if len(non_edges) > 4:
            return False
        used_a = set()
        used_b = set()
        for a, b in non_edges:
            if a in used_a or b in used_b:
                return False
            used_a.add(a)
            used_b.add(b)
        return True
    
    def find_q3(edge_set, adj):
        """Find a Q3 subgraph. Returns list of edges in the Q3 or None."""
        for A in itertools.combinations(range(n), 4):
            A_set = set(A)
            remaining = [v for v in range(n) if v not in A_set]
            for B in itertools.combinations(remaining, 4):
                if contains_q3(A, B, adj):
                    # Collect all edges between A and B that are in the graph
                    edges = []
                    for a in A:
                        for b in B:
                            if (a, b) in adj:
                                u, v = (a, b) if a < b else (b, a)
                                edges.append((u, v))
                    return edges
        return None
    
    def is_q3_free(edge_set, adj):
        return find_q3(edge_set, adj) is None
    
    best_edges = set()
    best_count = 0
    
    # Pre-compute all 4-combinations for speed
    all_4combs = list(itertools.combinations(range(n), 4))
    
    for seed in range(500):
        if time.time() > deadline - 3:
            break
        
        random.seed(seed)
        
        # Try different densities
        p = 0.60 + random.random() * 0.15
        edge_set = set()
        for u, v in all_edges:
            if random.random() < p:
                edge_set.add((u, v))
        
        adj = build_adj(edge_set)
        
        # Repair phase: remove edges to destroy Q3s
        for _ in range(30):
            q3 = find_q3(edge_set, adj)
            if q3 is None:
                break
            # Remove a random edge from the Q3
            u, v = random.choice(q3)
            edge_set.discard((u, v))
            adj.discard((u, v))
            adj.discard((v, u))
        
        # Try to add back edges greedily
        missing = [(u, v) for u, v in all_edges if (u, v) not in edge_set]
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline - 3:
                break
            if (u, v) in edge_set:
                continue
            new_set = edge_set | {(u, v)}
            new_adj = adj | {(u, v), (v, u)}
            if is_q3_free(new_set, new_adj):
                edge_set = new_set
                adj = new_adj
        
        if len(edge_set) > best_count:
            best_count = len(edge_set)
            best_edges = set(edge_set)
    
    return [[u, v] for u, v in best_edges]
