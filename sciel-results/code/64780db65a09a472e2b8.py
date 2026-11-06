import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 145  # safety margin before 150s limit
    n = 16
    
    # Precompute all 4-sets
    all_4sets = list(combinations(range(n), 4))
    
    # For each 4-set, store the 4 vertices
    set_verts = [list(s) for s in all_4sets]
    
    # Precompute disjoint pairs of 4-sets
    disjoint_pairs = []
    for i in range(len(all_4sets)):
        si = set(all_4sets[i])
        disjoint = []
        for j in range(i + 1, len(all_4sets)):
            if not (si & set(all_4sets[j])):
                disjoint.append(j)
        disjoint_pairs.append(disjoint)
    
    def is_q3_free(adj):
        """Check if graph is Q3-free using the H[A,B] matching criterion."""
        for i in range(len(all_4sets)):
            A = set_verts[i]
            for j in disjoint_pairs[i]:
                B = set_verts[j]
                non_edges = []
                for u in A:
                    for v in B:
                        if not (adj[u] & (1 << v)):
                            non_edges.append((u, v))
                if len(non_edges) > 4:
                    continue
                # Check if non_edges form a matching
                used = set()
                is_match = True
                for u, v in non_edges:
                    if u in used or v in used:
                        is_match = False
                        break
                    used.add(u)
                    used.add(v)
                if is_match:
                    return False
        return True
    
    def edge_count(adj):
        return sum(bin(a).count('1') for a in adj) // 2
    
    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    best_adj = [0] * n
    best_count = 0
    
    # Generate initial constructions
    constructions = []
    
    # Construction 1: K3,13 (seed)
    adj1 = [0] * n
    for a in range(3):
        for b in range(3, 16):
            adj1[a] |= (1 << b)
            adj1[b] |= (1 << a)
    constructions.append(adj1[:])
    
    # Construction 2: K8,8 minus perfect matching
    adj2 = [0] * n
    for i in range(8):
        for j in range(8, 16):
            if (i - j) % 8 != 0:
                adj2[i] |= (1 << j)
                adj2[j] |= (1 << i)
    constructions.append(adj2[:])
    
    # Construction 3-5: Random graphs with different densities
    for seed in [1, 42, 123]:
        random.seed(seed)
        adj3 = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj3[i] |= (1 << j)
                    adj3[j] |= (1 << i)
        constructions.append(adj3[:])
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for init_adj in constructions:
        if time.time() > deadline:
            break
        
        adj = init_adj[:]
        cur_count = edge_count(adj)
        
        if cur_count > best_count and is_q3_free(adj):
            best_count = cur_count
            best_adj = adj[:]
        
        # Local search: try adding edges
        missing = [(i, j) for i, j in all_edges if not (adj[i] & (1 << j))]
        random.seed(42)
        random.shuffle(missing)
        
        for _ in range(20):  # Up to 20 passes
            if time.time() > deadline:
                break
            
            improved = False
            for u, v in missing:
                if time.time() > deadline:
                    break
                if adj[u] & (1 << v):
                    continue
                
                # Try adding edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if is_q3_free(adj):
                    new_count = edge_count(adj)
                    if new_count > best_count:
                        best_count = new_count
                        best_adj = adj[:]
                    improved = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            
            if not improved:
                break
            
            missing = [(i, j) for i, j in all_edges if not (adj[i] & (1 << j))]
            random.shuffle(missing)
        
        # Edge swaps
        if time.time() < deadline:
            present = [(i, j) for i, j in all_edges if adj[i] & (1 << j)]
            missing = [(i, j) for i, j in all_edges if not (adj[i] & (1 << j))]
            random.shuffle(present)
            random.shuffle(missing)
            
            for _ in range(10):
                if time.time() > deadline:
                    break
                
                swap_improved = False
                for u, v in present:
                    if time.time() > deadline:
                        break
                    for x, y in missing:
                        if time.time() > deadline:
                            break
                        
                        # Try swap
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        adj[x] |= (1 << y)
                        adj[y] |= (1 << x)
                        
                        if is_q3_free(adj):
                            new_count = edge_count(adj)
                            if new_count > best_count:
                                best_count = new_count
                                best_adj = adj[:]
                            swap_improved = True
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            adj[x] &= ~(1 << y)
                            adj[y] &= ~(1 << x)
                
                if not swap_improved:
                    break
                
                present = [(i, j) for i, j in all_edges if adj[i] & (1 << j)]
                missing = [(i, j) for i, j in all_edges if not (adj[i] & (1 << j))]
                random.shuffle(present)
                random.shuffle(missing)
    
    # Final verification
    if not is_q3_free(best_adj):
        # Fallback to seed
        best_adj = [0] * n
        for a in range(3):
            for b in range(3, 16):
                best_adj[a] |= (1 << b)
                best_adj[b] |= (1 << a)
    
    return get_edges(best_adj)
