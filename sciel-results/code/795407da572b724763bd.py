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
                temp = common
                while temp:
                    c = temp & (-temp)
                    temp ^= c
                    c_idx = c.bit_length() - 1
                    rest = common ^ c
                    if rest & adj_masks[c_idx]:
                        d_bit = (rest & adj_masks[c_idx]) & -(rest & adj_masks[c_idx])
                        d_idx = d_bit.bit_length() - 1
                        return [i, j, c_idx, d_idx]
        return None

    def find_ind6(adj_masks, n):
        """Find an independent set of size 6 and return its vertices, or None."""
        def search(candidates, current):
            if len(current) == 6:
                return current[:]
            if len(current) + candidates.bit_count() < 6:
                return None
            if candidates == 0:
                return None
            
            v_bit = candidates & -candidates
            v_idx = v_bit.bit_length() - 1
            
            new_candidates = candidates & ~adj_masks[v_idx]
            new_candidates &= ~v_bit
            res = search(new_candidates, current + [v_idx])
            if res:
                return res
            
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

    def is_quadratic_residue(a, p):
        """Check if a is a quadratic residue mod p."""
        if a == 0:
            return False
        return pow(a, (p - 1) // 2, p) == 1

    def build_paley_graph(p):
        """Build Paley graph of order p (p ≡ 1 mod 4)."""
        adj = [0] * p
        for i in range(p):
            for j in range(i + 1, p):
                if is_quadratic_residue((j - i) % p, p):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def delete_vertex(adj, n, v):
        """Delete vertex v from graph, returning new adj list for n-1 vertices."""
        new_adj = [0] * (n - 1)
        for i in range(n):
            if i == v:
                continue
            new_i = i if i < v else i - 1
            mask = adj[i]
            # Clear bit v
            mask &= ~(1 << v)
            # Remap bits: for each bit j in mask, if j > v, shift down by 1
            for j in range(n):
                if j == v:
                    continue
                if (mask >> j) & 1:
                    new_j = j if j < v else j - 1
                    new_adj[new_i] |= (1 << new_j)
        return new_adj

    def max_clique_in_neighborhood(adj, n, v):
        """Find the size of the maximum clique in the neighborhood of v."""
        neighbors = []
        for u in range(n):
            if u != v and (adj[v] >> u) & 1:
                neighbors.append(u)
        
        if not neighbors:
            return 0
        
        # Simple backtracking for max clique in small neighborhood
        best = 0
        
        def backtrack(candidates, current_size):
            nonlocal best
            if current_size > best:
                best = current_size
            if not candidates:
                return
            # Upper bound: current_size + len(candidates)
            if current_size + len(candidates) <= best:
                return
            
            # Pick first vertex
            v_idx = candidates[0]
            rest = candidates[1:]
            
            # Include v_idx
            new_candidates = [u for u in rest if (adj[v_idx] >> u) & 1]
            backtrack(new_candidates, current_size + 1)
            
            # Exclude v_idx
            backtrack(rest, current_size)
        
        backtrack(neighbors, 0)
        return best

    def min_independent_set_in_nonneighbors(adj, n, v):
        """Find the minimum size of an independent set in the non-neighborhood of v (including v's non-neighbors)."""
        # Actually, we want to find the largest independent set in the non-neighborhood
        # to determine if adding v would create an independent 6-set
        non_neighbors = []
        for u in range(n):
            if u != v and not ((adj[v] >> u) & 1):
                non_neighbors.append(u)
        
        if not non_neighbors:
            return 0
        
        # Find maximum independent set in induced subgraph on non_neighbors
        best = 0
        
        def backtrack(candidates, current_size):
            nonlocal best
            if current_size > best:
                best = current_size
            if not candidates:
                return
            if current_size + len(candidates) <= best:
                return
            
            v_idx = candidates[0]
            rest = candidates[1:]
            
            # Include v_idx
            new_candidates = [u for u in rest if not ((adj[v_idx] >> u) & 1)]
            backtrack(new_candidates, current_size + 1)
            
            # Exclude v_idx
            backtrack(rest, current_size)
        
        backtrack(non_neighbors, 0)
        return best

    def greedy_vertex_deletion(adj, n):
        """Iteratively delete vertices to make graph valid (no K4, no Ind6)."""
        for iteration in range(n):
            if time.time() > deadline - 10.0:
                break
            
            k4 = find_k4(adj, n)
            ind6 = find_ind6(adj, n)
            
            if not k4 and not ind6:
                return adj, n
            
            # Choose which vertex to delete
            # Heuristic: delete vertex that appears in the most violations
            # or has the worst local properties
            
            candidates_to_delete = set()
            
            if k4:
                for v in k4:
                    candidates_to_delete.add(v)
            
            if ind6:
                for v in ind6:
                    candidates_to_delete.add(v)
            
            if not candidates_to_delete:
                # No violations found, but should be valid
                return adj, n
            
            # Score each candidate: prefer deleting vertices that are in more violations
            # or have higher "badness" score
            best_v = None
            best_score = -1
            
            for v in candidates_to_delete:
                score = 0
                
                # Count how many K4s contain v (approximate by checking if v is in a K4)
                # This is expensive, so use a simpler heuristic
                # Score based on degree and local structure
                
                deg = adj[v].bit_count()
                
                # Higher degree vertices in K4s are more "central"
                if k4 and v in k4:
                    score += 10 + deg
                
                if ind6 and v in ind6:
                    score += 10 + (n - 1 - deg)  # Lower degree in Ind6 is more central
                
                if score > best_score:
                    best_score = score
                    best_v = v
            
            if best_v is None:
                best_v = min(candidates_to_delete)
            
            # Delete the vertex
            adj = delete_vertex(adj, n, best_v)
            n = n - 1
        
        return adj, n

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
                    return None
            
            ind6 = find_ind6(adj, n)
            if ind6:
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
                    return None
            
            return adj
        
        return None

    best_n = 1
    best_adj = [0]

    # Strategy 1: Try Paley graph approach
    # Paley graphs of prime order p ≡ 1 mod 4 are strongly regular
    # We want p close to 40, so try p = 37, 41, 43
    # But we need n ≤ 40, so we can delete vertices from larger Paley graphs
    
    paley_primes = [43, 41, 37, 29, 13]
    
    for p in paley_primes:
        if time.time() > deadline - 20.0:
            break
        
        if p <= 40:
            # Use the Paley graph directly
            adj = build_paley_graph(p)
            if is_valid(adj, p):
                if p > best_n:
                    best_n = p
                    best_adj = adj[:]
                # Try to see if we can do better with vertex additions
                # But for now, accept this
            else:
                # Need to delete vertices
                adj_deleted, n_deleted = greedy_vertex_deletion(adj[:], p)
                if is_valid(adj_deleted, n_deleted):
                    if n_deleted > best_n:
                        best_n = n_deleted
                        best_adj = adj_deleted[:]
        else:
            # p > 40, build it and then delete down to ≤ 40
            adj = build_paley_graph(p)
            # Delete vertices until we have ≤ 40
            while len(adj) > 40 and time.time() < deadline - 15.0:
                # Delete a random vertex
                v = random.randint(0, len(adj) - 1)
                adj = delete_vertex(adj, len(adj), v)
            
            if len(adj) > 0:
                n_current = len(adj)
                # Now try to make it valid by further deletions
                adj_deleted, n_deleted = greedy_vertex_deletion(adj[:], n_current)
                if is_valid(adj_deleted, n_deleted):
                    if n_deleted > best_n:
                        best_n = n_deleted
                        best_adj = adj_deleted[:]

    # Strategy 2: Random search + repair for any n we haven't reached
    for target_n in range(best_n + 1, 36):
        if time.time() > deadline - 15.0:
            break
        
        found = False
        for _ in range(15):
            if time.time() > deadline - 5.0:
                break
            
            adj = random_adj(target_n, p=0.5)
            
            if is_valid(adj, target_n):
                found = True
                best_n = target_n
                best_adj = adj[:]
                break
            
            repaired = repair(adj, target_n, max_steps=3000)
            if repaired is not None:
                found = True
                best_n = target_n
                best_adj = repaired[:]
                break
        
        if not found:
            break

    # Final verification and fallback
    if not is_valid(best_adj, best_n):
        for fallback_n in range(best_n - 1, 1, -1):
            if time.time() > deadline - 2.0:
                break
            for _ in range(10):
                adj = random_adj(fallback_n, p=0.5)
                if is_valid(adj, fallback_n):
                    best_n = fallback_n
                    best_adj = adj[:]
                    break
                repaired = repair(adj, fallback_n, max_steps=1000)
                if repaired:
                    best_n = fallback_n
                    best_adj = repaired[:]
                    break
            if best_n == fallback_n:
                break

    edges = edges_from_masks(best_n, best_adj)
    
    return {'n': best_n, 'edges': edges}
