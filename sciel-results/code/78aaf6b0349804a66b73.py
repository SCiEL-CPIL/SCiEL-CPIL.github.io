import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def find_k4(adj_masks, n):
        """Find a K4 and return its vertices, or None."""
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj_masks[i] >> j) & 1:
                    continue
                common = adj_masks[i] & adj_masks[j]
                common &= ~(1 << i)
                common &= ~(1 << j)
                # Check if any two vertices in common are adjacent
                temp = common
                while temp:
                    c = temp & (-temp)
                    temp ^= c
                    c_idx = c.bit_length() - 1
                    rest = common ^ c
                    if rest & adj_masks[c_idx]:
                        # Find a neighbor d of c_idx in rest
                        d_bit = (rest & adj_masks[c_idx]) & -(rest & adj_masks[c_idx])
                        d_idx = d_bit.bit_length() - 1
                        return [i, j, c_idx, d_idx]
        return None

    def find_ind6(adj_masks, n):
        """Find an independent set of size 6 and return its vertices, or None."""
        # Branch and bound search for independent set of size 6
        # candidates: bitmask of vertices that can still be added
        # current: list of vertices in the current independent set
        def search(candidates, current):
            if len(current) == 6:
                return current[:]
            
            # Pruning: if we can't reach 6 vertices
            if len(current) + candidates.bit_count() < 6:
                return None
            
            if candidates == 0:
                return None
            
            # Pick a vertex from candidates
            # Heuristic: pick the vertex with the highest degree in the induced subgraph of candidates?
            # Or just pick the first one. Let's pick the first one for speed.
            v_bit = candidates & -candidates
            v_idx = v_bit.bit_length() - 1
            
            # Branch 1: Include v
            new_candidates = candidates & ~adj_masks[v_idx]
            new_candidates &= ~v_bit
            res = search(new_candidates, current + [v_idx])
            if res:
                return res
            
            # Branch 2: Exclude v
            new_candidates = candidates & ~v_bit
            res = search(new_candidates, current)
            if res:
                return res
            
            return None
        
        full_mask = (1 << n) - 1
        return search(full_mask, [])

    def is_valid(adj_masks, n):
        if find_k4(adj_masks, n):
            return False
        if find_ind6(adj_masks, n):
            return False
        return True

    def edges_from_masks(n, adj_masks):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_masks[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def random_adj(n, p=0.5):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def repair(adj, n, max_steps=20000):
        """Repair an invalid graph by removing K4 edges or adding Ind6 edges."""
        for _ in range(max_steps):
            if time.time() > deadline - 5.0:
                return None
            
            k4 = find_k4(adj, n)
            if k4:
                # Remove a random edge from the K4
                # Edges in K4: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
                # We need to check which ones are actually present (they should be)
                edges_in_k4 = []
                for a in range(4):
                    for b in range(a+1, 4):
                        u, v = k4[a], k4[b]
                        if (adj[u] >> v) & 1:
                            edges_in_k4.append((u, v))
                
                if edges_in_k4:
                    u, v = random.choice(edges_in_k4)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    continue
                else:
                    # Inconsistent state, break
                    return None

            ind6 = find_ind6(adj, n)
            if ind6:
                # Add a random non-edge within the Ind6
                non_edges = []
                for a in range(6):
                    for b in range(a+1, 6):
                        u, v = ind6[a], ind6[b]
                        if not ((adj[u] >> v) & 1):
                            non_edges.append((u, v))
                
                if non_edges:
                    u, v = random.choice(non_edges)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    continue
                else:
                    # Inconsistent state, break
                    return None
            
            # If neither K4 nor Ind6 found, it's valid
            return adj
        
        return None

    best_n = 1
    best_adj = [0]
    
    # Try to find valid graphs for increasing n
    for target_n in range(2, 36):
        if time.time() > deadline - 10.0:
            break
        
        found = False
        # Try a few random restarts
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            
            # Random initialization
            adj = random_adj(target_n, p=0.5)
            
            # Check if valid
            if is_valid(adj, target_n):
                found = True
                best_n = target_n
                best_adj = adj
                break
            
            # Try to repair
            repaired = repair(adj, target_n, max_steps=5000)
            if repaired is not None:
                found = True
                best_n = target_n
                best_adj = repaired
                break
        
        if not found:
            # If we failed to find a valid graph for target_n, 
            # it's likely we won't find one for larger n either in this run.
            # But we can continue trying with more attempts if we have time.
            # However, if we fail for n, we probably failed for n+1.
            # Let's stop if we've failed for the current target.
            # Actually, we might find a valid graph for n+1 by luck, but it's unlikely.
            # Let's break to save time for verification.
            break

    # Final verification
    if not is_valid(best_adj, best_n):
        # Fallback to a smaller valid graph if needed
        for fallback_n in range(best_n - 1, 1, -1):
            if time.time() > deadline - 2.0:
                break
            # Try to find a valid graph of size fallback_n
            for _ in range(10):
                adj = random_adj(fallback_n, p=0.5)
                if is_valid(adj, fallback_n):
                    best_n = fallback_n
                    best_adj = adj
                    break
                repaired = repair(adj, fallback_n, max_steps=1000)
                if repaired:
                    best_n = fallback_n
                    best_adj = repaired
                    break
            if best_n == fallback_n:
                break

    edges = edges_from_masks(best_n, best_adj)
    
    return {'n': best_n, 'edges': edges}
