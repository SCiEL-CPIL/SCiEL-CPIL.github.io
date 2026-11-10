import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    n = 20
    vertices = list(range(n))
    
    # Precompute all triples (u, v, w) for K3,3 check
    triples = list(combinations(vertices, 3))
    num_triples = len(triples)
    
    # Precompute for each vertex, the list of triple indices containing it
    triples_containing = [[] for _ in range(n)]
    for idx, (a, b, c) in enumerate(triples):
        triples_containing[a].append(idx)
        triples_containing[b].append(idx)
        triples_containing[c].append(idx)
    
    # Precompute for each pair (u,v), the sorted list of triple indices containing u or v
    pair_triples = {}
    for u in range(n):
        for v in range(u + 1, n):
            # Union of triples containing u and triples containing v
            set_u = set(triples_containing[u])
            set_v = set(triples_containing[v])
            combined = sorted(set_u | set_v)
            pair_triples[(u, v)] = combined
            pair_triples[(v, u)] = combined
    
    def bit(i):
        return 1 << i
    
    def popcount(x):
        return x.bit_count()
    
    def edge_count(adj_mask):
        return sum(m.bit_count() for m in adj_mask) // 2
    
    def mask_to_edges(adj_mask):
        edges = []
        for i in range(n):
            m = adj_mask[i]
            for j in range(i + 1, n):
                if m & bit(j):
                    edges.append([i, j])
        return edges
    
    def can_add(u, v, adj_mask):
        """
        Check if adding edge (u, v) keeps the graph K3,3-free.
        Only checks triples containing u or v (precomputed).
        """
        bit_v = bit(v)
        bit_u = bit(u)
        new_u = adj_mask[u] | bit_v
        new_v = adj_mask[v] | bit_u
        
        for tidx in pair_triples[(u, v)]:
            a, b, c = triples[tidx]
            
            ma = new_u if a == u else (new_v if a == v else adj_mask[a])
            mb = new_u if b == u else (new_v if b == v else adj_mask[b])
            mc = new_u if c == u else (new_v if c == v else adj_mask[c])
            
            if popcount(ma & mb & mc) > 2:
                return False
        return True
    
    def greedy_add(adj_mask):
        """Add all possible edges greedily."""
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj_mask[i] & bit(j)):
                        if can_add(i, j, adj_mask):
                            adj_mask[i] |= bit(j)
                            adj_mask[j] |= bit(i)
                            changed = True
        return adj_mask
    
    def find_violation(adj_mask):
        """Find a K3,3 violation, return (u,v,w,common_neighbors) or None."""
        for u, v, w in triples:
            common = adj_mask[u] & adj_mask[v] & adj_mask[w]
            if popcount(common) > 2:
                return (u, v, w, common)
        return None
    
    def count_violations(adj_mask):
        """Count all K3,3 violations."""
        cnt = 0
        for u, v, w in triples:
            if popcount(adj_mask[u] & adj_mask[v] & adj_mask[w]) > 2:
                cnt += 1
        return cnt
    
    def repair(adj_mask):
        """Remove edges to fix violations, prioritizing high-degree vertices."""
        for _ in range(300):
            if time.time() > deadline:
                break
            viol = find_violation(adj_mask)
            if viol is None:
                break
            u, v, w, common = viol
            common_list = [x for x in range(n) if common & bit(x)]
            # Sort by degree descending to remove edges from high-degree vertices first
            common_list.sort(key=lambda x: adj_mask[x].bit_count(), reverse=True)
            removed = False
            for c in common_list:
                for a, b in [(u, c), (v, c), (w, c)]:
                    if adj_mask[a] & bit(b):
                        adj_mask[a] &= ~bit(b)
                        adj_mask[b] &= ~bit(a)
                        if count_violations(adj_mask) == 0:
                            removed = True
                            break
                        else:
                            adj_mask[a] |= bit(b)
                            adj_mask[b] |= bit(a)
                if removed:
                    break
            if not removed:
                if common_list:
                    c = common_list[0]
                    adj_mask[u] &= ~bit(c)
                    adj_mask[c] &= ~bit(u)
        return adj_mask
    
    def local_search(adj_mask, max_iters=300):
        """Local search with edge additions and swaps."""
        adj_mask = repair(adj_mask)
        adj_mask = greedy_add(adj_mask)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try to add missing edges
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) 
                      if not (adj_mask[i] & bit(j))]
            
            if not missing:
                break
            
            random.shuffle(missing)
            improved = False
            
            for u, v in missing:
                if can_add(u, v, adj_mask):
                    adj_mask[u] |= bit(v)
                    adj_mask[v] |= bit(u)
                    improved = True
                    break
            
            if improved:
                adj_mask = greedy_add(adj_mask)
                continue
            
            # Try swaps: remove one edge, add another
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n) 
                        if adj_mask[i] & bit(j)]
            
            if not all_edges:
                break
            
            random.shuffle(all_edges)
            random.shuffle(missing)
            
            swapped = False
            for i in range(min(50, len(all_edges), len(missing))):
                if time.time() > deadline:
                    break
                
                r_u, r_v = all_edges[i]
                a_u, a_v = missing[i]
                
                # Remove (r_u, r_v)
                adj_mask[r_u] &= ~bit(r_v)
                adj_mask[r_v] &= ~bit(r_u)
                # Add (a_u, a_v)
                adj_mask[a_u] |= bit(a_v)
                adj_mask[a_v] |= bit(a_u)
                
                if find_violation(adj_mask) is None:
                    adj_mask = greedy_add(adj_mask)
                    swapped = True
                    break
                else:
                    # Revert
                    adj_mask[r_u] |= bit(r_v)
                    adj_mask[r_v] |= bit(r_u)
                    adj_mask[a_u] &= ~bit(a_v)
                    adj_mask[a_v] &= ~bit(a_u)
            
            if not swapped:
                # Try removing a random edge to escape local optimum
                if all_edges:
                    r_u, r_v = all_edges[0]
                    adj_mask[r_u] &= ~bit(r_v)
                    adj_mask[r_v] &= ~bit(r_u)
                    adj_mask = greedy_add(adj_mask)
        
        return adj_mask
    
    # Spectral gap-based initialization: use random symmetric matrix eigenvectors
    # to create structured dense graphs near the K3,3-free boundary
    best_adj = None
    best_edges = 0
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    
    def spectral_init(seed_offset=0):
        """Generate a dense graph using spectral methods."""
        local_rng = np.random.default_rng(42 + seed_offset)
        # Create random symmetric matrix
        A = local_rng.random((n, n))
        A = (A + A.T) / 2.0
        # Get eigenvectors
        eigvals, eigvecs = np.linalg.eigh(A)
        # Use top k eigenvectors to create a structured graph
        k = 5  # number of eigenvectors to use
        top_eigvecs = eigvecs[:, -k:]  # top k (largest eigenvalues)
        # Create adjacency based on projection similarity
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                # Cosine similarity of projected vectors
                vi = top_eigvecs[i]
                vj = top_eigvecs[j]
                sim = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-10)
                # Threshold to create edge
                if sim > 0.3:
                    adj[i] |= bit(j)
                    adj[j] |= bit(i)
        return adj
    
    # Try multiple initialization strategies
    for restart in range(200):
        if time.time() > deadline:
            break
        
        if restart < 80:
            # Spectral initialization
            adj = spectral_init(restart)
        elif restart < 140:
            # Dense random initialization
            density = 0.5 + rng.random() * 0.3
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < density:
                        adj[i] |= bit(j)
                        adj[j] |= bit(i)
        else:
            # Complementary of sparse random (dense by construction)
            density = 0.2 + rng.random() * 0.2
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() >= density:
                        adj[i] |= bit(j)
                        adj[j] |= bit(i)
        
        # Repair and optimize
        adj = local_search(adj, max_iters=200)
        
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
        
        # Early termination if we find a very good solution
        if best_edges > 150:
            break
    
    # Final polish pass with more iterations if time allows
    if best_adj is not None and time.time() < deadline - 15:
        adj = list(best_adj)
        adj = local_search(adj, max_iters=400)
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
    
    if best_adj is None:
        # Fallback: empty graph
        best_adj = [0] * n
        best_edges = 0
    
    edges = mask_to_edges(best_adj)
    return edges
