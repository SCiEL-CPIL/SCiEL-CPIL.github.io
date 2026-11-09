"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
# EVOLVE-BLOCK-START
import time
import random

def construct():
    n = 15
    deadline = time.time() + 85.0
    random.seed(42)
    
    # adj[v] = set of neighbors
    adj = [set() for _ in range(n)]
    
    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
    
    def remove_edge(u, v):
        adj[u].discard(v)
        adj[v].discard(u)
    
    def deg(v):
        return len(adj[v])
    
    def check_edge_valid(u, v):
        """Check if adding edge (u,v) would create any violation.
        We need to check: for the new edge (u,v), for all s in N(u)\{v}, t in N(v)\{u}, s!=t: deg(s) != deg(t).
        Also, adding (u,v) changes deg(u) and deg(v), which may affect other edges.
        For simplicity, just check the new edge's constraint and assume degrees don't change other edges' validity.
        Actually, degrees DO change, so we need full check. Let's just do full check."""
        pass
    
    def count_violations():
        """Count violations efficiently."""
        violations = 0
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    # For edge (u,v), check all s in N(u)\{v}, t in N(v)\{u}
                    degs_u = set()
                    for s in adj[u]:
                        if s != v:
                            degs_u.add(deg(s))
                    degs_v = set()
                    for t in adj[v]:
                        if t != u:
                            degs_v.add(deg(t))
                    # Violation if there exist s in N(u)\{v}, t in N(v)\{u}, s!=t, deg(s)==deg(t)
                    # This is NOT simply degs_u & degs_v, because s and t could be the same vertex
                    # Actually if s==t, then s is adjacent to both u and v, and deg(s)==deg(s) trivially, but s!=t is required
                    # So we need: exists s in N(u)\{v}, t in N(v)\{u} with s!=t and deg(s)==deg(t)
                    # Group by degree
                    from collections import defaultdict
                    deg_map = defaultdict(set)
                    for s in adj[u]:
                        if s != v:
                            deg_map[deg(s)].add(s)
                    for t in adj[v]:
                        if t != u:
                            if deg(t) in deg_map:
                                # Check if there's an s in deg_map[deg(t)] with s != t
                                if len(deg_map[deg(t)]) > 0:
                                    if t not in deg_map[deg(t)] or len(deg_map[deg(t)]) > 1:
                                        violations += 1
                                        break
        return violations
    
    def edge_count():
        return sum(len(adj[i]) for i in range(n)) // 2
    
    # Start with star
    for v in range(1, n):
        add_edge(0, v)
    
    best_adj = [set(s) for s in adj]
    best_edges = edge_count()
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Greedy: try to add each edge if it keeps graph valid
    for u, v in all_edges:
        if time.time() > deadline - 5.0:
            break
        if v not in adj[u]:
            add_edge(u, v)
            if count_violations() == 0:
                if edge_count() > best_edges:
                    best_adj = [set(s) for s in adj]
                    best_edges = edge_count()
            else:
                remove_edge(u, v)
    
    # Simulated annealing phase
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    current_viol = count_violations()
    
    while time.time() < deadline - 3.0 and T > T_min:
        # Random move: add or remove an edge
        u, v = random.choice(all_edges)
        if v in adj[u]:
            # Remove
            remove_edge(u, v)
            new_viol = count_violations()
            if new_viol <= current_viol:
                current_viol = new_viol
                if new_viol == 0 and edge_count() > best_edges:
                    best_adj = [set(s) for s in adj]
                    best_edges = edge_count()
            else:
                # Accept with probability
                if random.random() < math.exp(-(new_viol - current_viol) / T):
                    current_viol = new_viol
                else:
                    add_edge(u, v)
        else:
            # Add
            add_edge(u, v)
            new_viol = count_violations()
            if new_viol <= current_viol:
                current_viol = new_viol
                if new_viol == 0 and edge_count() > best_edges:
                    best_adj = [set(s) for s in adj]
                    best_edges = edge_count()
            else:
                if random.random() < math.exp(-(new_viol - current_viol) / T):
                    current_viol = new_viol
                else:
                    remove_edge(u, v)
        
        T *= cooling
    
    # Final greedy pass
    for u, v in all_edges:
        if time.time() > deadline - 2.0:
            break
        if v not in adj[u]:
            add_edge(u, v)
            if count_violations() == 0:
                if edge_count() > best_edges:
                    best_adj = [set(s) for s in adj]
                    best_edges = edge_count()
            else:
                remove_edge(u, v)
    
    adj = [set(s) for s in best_adj]
    
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([int(u), int(v)])
    
    return edges

# EVOLVE-BLOCK-END
