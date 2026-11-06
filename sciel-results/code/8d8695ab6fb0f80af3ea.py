import time
import random
import numpy as np
from itertools import combinations

N = 20

# Precompute triples for each vertex
triples_for_vertex = [[] for _ in range(N)]
triples_indices = []
for i in range(N):
    for j in range(i+1, N):
        for k in range(j+1, N):
            t_idx = len(triples_indices)
            triples_indices.append((i,j,k))
            triples_for_vertex[i].append((t_idx, j, k))
            triples_for_vertex[j].append((t_idx, i, k))
            triples_for_vertex[k].append((t_idx, i, j))

total_triples = len(triples_indices)

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_edges = None
    best_count = 0
    
    all_edges = [(i,j) for i in range(N) for j in range(i+1, N)]
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        random.shuffle(all_edges)
        M = np.zeros((N, N), dtype=np.int8)
        C = np.zeros(total_triples, dtype=np.int8)
        edge_set = set()
        
        # Helper functions using closures or passing M, C, edge_set
        def check_add(u, v):
            if M[u, v] == 1:
                return None
            conflicts = []
            for t_idx, x, y in triples_for_vertex[u]:
                if M[x, v] == 1 and M[y, v] == 1:
                    if C[t_idx] == 2:
                        conflicts.append(t_idx)
            for t_idx, x, y in triples_for_vertex[v]:
                if M[x, u] == 1 and M[y, u] == 1:
                    if C[t_idx] == 2:
                        conflicts.append(t_idx)
            return conflicts if conflicts else None

        def add_edge(u, v):
            M[u, v] = 1
            M[v, u] = 1
            edge_set.add((u,v))
            for t_idx, x, y in triples_for_vertex[u]:
                if M[x, v] == 1 and M[y, v] == 1:
                    C[t_idx] += 1
            for t_idx, x, y in triples_for_vertex[v]:
                if M[x, u] == 1 and M[y, u] == 1:
                    C[t_idx] += 1

        def remove_edge(u, v):
            M[u, v] = 0
            M[v, u] = 0
            edge_set.remove((u,v))
            for t_idx, x, y in triples_for_vertex[u]:
                if M[x, v] == 1 and M[y, v] == 1:
                    C[t_idx] -= 1
            for t_idx, x, y in triples_for_vertex[v]:
                if M[x, u] == 1 and M[y, u] == 1:
                    C[t_idx] -= 1

        # Phase 1: Greedy
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if check_add(u, v) is None:
                add_edge(u, v)
                
        current_count = len(edge_set)
        
        # Phase 2: Local Search
        # We want to add missing edges.
        # If we can't add, we swap.
        
        # To avoid infinite loops, we limit the number of swaps per restart or use a counter.
        # Also, we should periodically re-evaluate missing edges.
        
        # Create a list of missing edges
        missing_edges = [e for e in all_edges if e not in edge_set]
        
        max_swaps = 2000 # Limit swaps to prevent stalling
        swaps_made = 0
        
        while swaps_made < max_swaps and time.time() < deadline:
            if not missing_edges:
                break
                
            # Pick a random missing edge
            idx = random.randint(0, len(missing_edges) - 1)
            u, v = missing_edges[idx]
            
            conflicts = check_add(u, v)
            if conflicts is None:
                add_edge(u, v)
                missing_edges.pop(idx)
                current_count += 1
                # Don't reset swaps_made, but we made progress
            else:
                # Try to swap
                candidates = set()
                for t_idx in conflicts:
                    i, j, k = triples_indices[t_idx]
                    if u in (i,j,k):
                        others = [x for x in (i,j,k) if x != u]
                        x, y = others
                        if M[v, x] == 1: candidates.add((min(v,x), max(v,x)))
                        if M[v, y] == 1: candidates.add((min(v,y), max(v,y)))
                    if v in (i,j,k):
                        others = [x for x in (i,j,k) if x != v]
                        x, y = others
                        if M[u, x] == 1: candidates.add((min(u,x), max(u,x)))
                        if M[u, y] == 1: candidates.add((min(u,y), max(u,y)))
                        
                if candidates:
                    # Try a few random candidates
                    cand_list = list(candidates)
                    random.shuffle(cand_list)
                    for a, b in cand_list[:5]:
                        # Save state
                        remove_edge(a, b)
                        new_conflicts = check_add(u, v)
                        if new_conflicts is None:
                            add_edge(u, v)
                            # Update missing_edges
                            # (a,b) is now missing, (u,v) is now present
                            # Remove (u,v) from missing_edges (it was at idx)
                            # Add (a,b) to missing_edges
                            missing_edges[idx] = (a,b) # Replace (u,v) with (a,b)
                            swaps_made += 1
                            break
                        else:
                            # Restore
                            add_edge(a, b)
                            
        if current_count > best_count:
            best_count = current_count
            best_edges = [[u, v] for u, v in edge_set]
            
    if best_edges is None:
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    return best_edges
