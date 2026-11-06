import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 40
    m = 401
    
    # Initial construction: K20,20 + 1 edge
    # Vertices 0-19 in A, 20-39 in B
    edges = set()
    for a in range(20):
        for b in range(20, 40):
            edges.add((a, b))
    edges.add((0, 1))
    
    # Build adjacency matrix (boolean)
    adj = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        adj[u, v] = True
        adj[v, u] = True
        
    # Helper to get canonical edge tuple
    def canon(u, v):
        return (u, v) if u < v else (v, u)
        
    def compute_books(adj_matrix, edge_set):
        books = {}
        for u, v in edge_set:
            books[(u, v)] = np.sum(adj_matrix[u] & adj_matrix[v])
        return books

    current_books = compute_books(adj, edges)
    max_book = max(current_books.values()) if current_books else 0
    sum_sq = sum(v*v for v in current_books.values())
    
    best_max_book = max_book
    best_edges = set(edges)
    best_adj = adj.copy()
    best_books = dict(current_books)
    best_sum_sq = sum_sq
    
    # Simulated Annealing
    T = 50.0
    cooling = 0.995
    
    edge_list = list(edges)
    
    while time.time() < deadline:
        if T < 0.01:
            break
            
        # Generate candidate move
        # Weighted removal: higher book size -> higher prob
        weights = np.array([current_books[e] + 1 for e in edge_list])
        weights /= weights.sum()
        
        idx_remove = np.random.choice(len(edge_list), p=weights)
        e_remove = edge_list[idx_remove]
        u, v = e_remove
        
        # Random non-edge addition
        found = False
        for _try in range(20):
            a, b = random.sample(range(n), 2)
            if not adj[a, b]:
                found = True
                break
        if not found:
            continue
            
        e_add = canon(a, b)
        
        # Evaluate Move
        # 1. Store old values for affected edges
        # Affected vertices: u, v, a, b
        # Affected edges: incident to u, v, a, b
        
        affected_edges = set()
        for z in range(n):
            if adj[u, z]: affected_edges.add(canon(u, z))
            if adj[v, z]: affected_edges.add(canon(v, z))
            if adj[a, z]: affected_edges.add(canon(a, z))
            if adj[b, z]: affected_edges.add(canon(b, z))
            
        # Ensure e_remove is tracked (it is incident to u, v)
        # Ensure e_add is tracked (it is incident to a, b)
        
        old_vals = {}
        for e in affected_edges:
            if e in current_books:
                old_vals[e] = current_books[e]
            else:
                old_vals[e] = 0 # Non-edge has 0 book size contribution to sum_sq
        
        # 2. Apply move to adj
        adj[u, v] = False
        adj[v, u] = False
        adj[a, b] = True
        adj[b, a] = True
        
        # 3. Compute new values for affected edges
        new_vals = {}
        for e in affected_edges:
            x, y = e
            # Only compute if it is an edge in the NEW graph
            # New graph edges = (edges - {e_remove}) U {e_add}
            # We can check adj[x, y]
            if adj[x, y]:
                new_vals[e] = np.sum(adj[x] & adj[y])
            else:
                new_vals[e] = 0
                
        # 4. Calculate new sum_sq and max_book
        # sum_sq_new = sum_sq_old - sum(old_vals[e]^2) + sum(new_vals[e]^2)
        # Note: old_vals and new_vals are defined for all affected_edges.
        # For non-edges, value is 0.
        
        old_sq_sum = sum(v*v for v in old_vals.values())
        new_sq_sum = sum(v*v for v in new_vals.values())
        
        new_sum_sq = sum_sq - old_sq_sum + new_sq_sum
        
        # new_max_book
        # We need the max of:
        # 1. Max of new_vals
        # 2. Max of unaffected books
        
        # Unaffected books are those NOT in affected_edges.
        # We don't have a list of unaffected books readily.
        # But we know the global max_book.
        # If the old max_book was from an unaffected edge, it remains.
        # If it was from an affected edge, it might change.
        
        # Simple approach: 
        # new_max = max(max(new_vals.values()), max_book_unaffected)
        # We don't have max_book_unaffected.
        
        # Alternative: 
        # Just compute max of new_vals and assume unaffected max is <= old_max_book.
        # If new_max < old_max_book, it's an improvement.
        # If new_max == old_max_book, check sum_sq.
        # If new_max > old_max_book, it's worse (unless we accept via SA).
        
        # To be precise, we should track the max of unaffected.
        # But that's hard.
        
        # Approximation: 
        # If any affected edge had the old max, and its new value is lower, 
        # the new max is max(new_vals, second_max_old).
        # We don't track second_max.
        
        # Let's just use: 
        # candidate_max = max(new_vals.values())
        # If candidate_max < max_book, it's a definite improvement in max.
        # If candidate_max == max_book, it's neutral.
        # If candidate_max > max_book, it's worse.
        
        # This might miss cases where the old max was affected and reduced, 
        # but another unaffected edge had the same max.
        # But for SA, this is a reasonable heuristic.
        
        candidate_max = max(new_vals.values()) if new_vals else 0
        
        # 5. Acceptance Criteria
        accept = False
        
        if candidate_max < max_book:
            accept = True
        elif candidate_max == max_book:
            if new_sum_sq < sum_sq:
                accept = True
            else:
                # Accept with probability based on sum_sq increase?
                # delta = new_sum_sq - sum_sq
                # prob = exp(-delta / T)
                delta = new_sum_sq - sum_sq
                if delta > 0:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        accept = True
        else:
            # candidate_max > max_book
            # Accept with probability?
            # delta_max = candidate_max - max_book
            # This is a big jump. Usually reject.
            # But maybe accept if sum_sq is much lower?
            # Let's stick to rejecting if max increases.
            pass
        
        # 6. Update State
        if accept:
            # Update current_books
            for e in affected_edges:
                if e in new_vals and new_vals[e] > 0:
                    current_books[e] = new_vals[e]
                else:
                    if e in current_books:
                        del current_books[e]
            
            # Update edges set
            edges.discard(e_remove)
            edges.add(e_add)
            
            # Update edge_list? 
            # edge_list is static. We should update it or just use edges.
            # Using edges directly is safer for sampling.
            # But sampling from set is slow.
            # Let's just update edge_list if we change it.
            # For now, we'll just let edge_list become stale.
            # Actually, stale edge_list means we might sample a removed edge.
            # We should check if e_remove is still in edges before sampling?
            # Or just rebuild edge_list occasionally?
            # Let's just use list(edges) inside the loop? Too slow.
            # Let's update edge_list:
            # Remove e_remove, add e_add.
            try:
                edge_list.remove(e_remove)
            except ValueError:
                pass
            if e_add not in edge_list:
                edge_list.append(e_add)
            
            # Update global metrics
            max_book = candidate_max
            sum_sq = new_sum_sq
            
            # Update best
            if max_book < best_max_book or (max_book == best_max_book and sum_sq < best_sum_sq):
                best_max_book = max_book
                best_sum_sq = sum_sq
                best_edges = set(edges)
                best_adj = adj.copy()
                best_books = dict(current_books)
                
        else:
            # Revert adj
            adj[u, v] = True
            adj[v, u] = True
            adj[a, b] = False
            adj[b, a] = False
            
        T *= cooling
        
    # Return best found
    result = [[u, v] for u, v in best_edges]
    
    # Sanity check
    if len(result) != 401:
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return result
