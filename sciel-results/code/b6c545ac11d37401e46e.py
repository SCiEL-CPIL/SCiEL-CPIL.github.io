"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 140  # 140s safety margin before 150s limit
    n = 16
    
    # Start with K3,13
    edge_set = set()
    for a in range(3):
        for b in range(3, 16):
            edge_set.add((a, b))
    
    def creates_q3(u, v, adj):
        """Check if adding edge (u,v) creates a Q3 in the graph."""
        adj_new = adj | {(min(u, v), max(u, v))}
        
        def in_adj(a, b):
            return (min(a, b), max(a, b)) in adj_new
        
        neigh_u = [w for w in range(n) if w != v and in_adj(u, w)]
        neigh_v = [w for w in range(n) if w != u and in_adj(v, w)]
        
        for a2 in neigh_u:
            for a4 in neigh_u:
                if a4 == a2:
                    continue
                for a3 in neigh_v:
                    if a3 in (a2, a4):
                        continue
                    if not in_adj(a2, a3):
                        continue
                    for a5 in neigh_v:
                        if a5 in (a2, a3, a4):
                            continue
                        if not in_adj(a4, a5):
                            continue
                        used = {u, v, a2, a3, a4, a5}
                        common_a2_a4 = [w for w in range(n) if w not in used and in_adj(a2, w) and in_adj(a4, w)]
                        if not common_a2_a4:
                            continue
                        common_a3_a5 = [w for w in range(n) if w not in used and in_adj(a3, w) and in_adj(a5, w)]
                        if not common_a3_a5:
                            continue
                        for a6 in common_a2_a4:
                            for a7 in common_a3_a5:
                                if a7 == a6:
                                    continue
                                if in_adj(a6, a7):
                                    return True
        return False
    
    # Greedy phase with multiple random trials
    all_possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in edge_set]
    
    best_edge_set = set(edge_set)
    best_count = len(edge_set)
    
    for trial in range(30):
        if time.time() > deadline - 50:
            break
        
        edge_set = set(best_edge_set)
        random.shuffle(all_possible_edges)
        
        for u, v in all_possible_edges:
            if time.time() > deadline - 50:
                break
            if not creates_q3(u, v, edge_set):
                edge_set.add((u, v))
        
        if len(edge_set) > best_count:
            best_count = len(edge_set)
            best_edge_set = set(edge_set)
    
    # Local search: edge swaps
    edge_set = set(best_edge_set)
    
    for _ in range(200):
        if time.time() > deadline - 5:
            break
        
        edges_list = list(edge_set)
        if len(edges_list) < 2:
            break
        
        e1, e2 = random.sample(edges_list, 2)
        edge_set.discard(e1)
        edge_set.discard(e2)
        
        added = 0
        candidates = [(u, v) for u in range(n) for v in range(u + 1, n) 
                      if (u, v) not in edge_set and (u, v) != e1 and (u, v) != e2]
        random.shuffle(candidates)
        
        for cand in candidates:
            if time.time() > deadline - 5:
                break
            if not creates_q3(cand[0], cand[1], edge_set):
                edge_set.add(cand)
                added += 1
                if added == 2:
                    break
        
        if added < 2:
            edge_set.add(e1)
            edge_set.add(e2)
    
    if len(edge_set) > best_count:
        best_count = len(edge_set)
        best_edge_set = set(edge_set)
    
    best_edges = [[u, v] for u, v in best_edge_set]
    return best_edges
# EVOLVE-BLOCK-END
