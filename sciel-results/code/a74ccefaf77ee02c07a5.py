import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 14
    m = 35
    
    values = list(range(1, n + 1))
    
    all_pairs = list(combinations(range(n), 2))
    pair_sum = {}
    pair_prod = {}
    for u, v in all_pairs:
        pair_sum[(u, v)] = values[u] + values[v]
        pair_prod[(u, v)] = values[u] * values[v]
        
    # Precompute sets for fast evaluation?
    # No, just recompute from scratch for the 35 edges.
    
    def evaluate(edge_indices):
        sums = set()
        prods = set()
        for idx in edge_indices:
            u, v = all_pairs[idx]
            sums.add(pair_sum[(u, v)])
            prods.add(pair_prod[(u, v)])
        return max(len(sums), len(prods))
    
    # Initialize
    random.seed(42)
    current_edges = set(random.sample(range(len(all_pairs)), m))
    current_score = evaluate(current_edges)
    
    best_edges = set(current_edges)
    best_score = current_score
    
    # Local Search: First-improvement or Best-improvement?
    # Best-improvement is slower per iteration but fewer iterations.
    # First-improvement is faster per iteration.
    # Given 91 pairs, 35 in, 56 out.
    # 35 * 56 = 1960 swaps.
    # Each eval ~ 35 * 2 set adds. Very fast.
    # Let's do first-improvement.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate through edges in current set
        # To avoid bias, shuffle the order of edges to check
        edges_in_list = list(current_edges)
        random.shuffle(edges_in_list)
        
        for e_in in edges_in_list:
            if time.time() > deadline:
                break
            # Iterate through edges not in set
            edges_out_list = [i for i in range(len(all_pairs)) if i not in current_edges]
            random.shuffle(edges_out_list)
            
            for e_out in edges_out_list:
                # Try swap
                new_edges = set(current_edges)
                new_edges.remove(e_in)
                new_edges.add(e_out)
                
                new_score = evaluate(new_edges)
                
                if new_score < current_score:
                    current_edges = new_edges
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = set(current_edges)
                    # After a successful swap, restart the loop for e_in?
                    # Or just continue?
                    # Usually restarts are better to escape local minima quickly.
                    break # Break inner loop, restart outer loop effectively via 'improved' logic?
                    # Actually, if we break inner, we go to next e_in.
                    # But the set changed. The other e_in's are still valid indices.
                    # However, the optimal e_out for the next e_in might have changed.
                    # So it's better to restart the whole sweep.
                    # Let's break both.
                    # Use a flag or exception?
                    # Simpler: just break inner and set a flag to break outer.
                    # But we already have 'improved'.
                    # Let's just break inner and let the outer loop continue.
                    # It might find more improvements in one sweep.
                    # Or we can break outer too.
                    # Let's break outer for aggressive local search.
                    break
            if improved:
                break
            # If no improvement found for this e_in, try next e_in?
            # Or restart?
            # Standard first-improvement: accept first improvement found, then restart search from beginning.
            # So if we found an improvement, we break outer.
            # If we didn't, we continue to next e_in.
            
    # If we exited without improvement, we are at a local minimum.
    # We can try a random restart if time permits.
    
    # Actually, the loop above breaks outer if improved.
    # So it finds one improvement, then restarts.
    # This is good.
    
    # Check time, if we have lots of time, do more restarts?
    # The current loop runs until no improvement is found in a full sweep.
    # This converges to a local minimum.
    # To escape, we can do random restarts.
    
    # Let's add a simple restart mechanism if we finish early.
    # But the while loop condition `improved` stops it.
    # So we are stuck at a local min.
    # Let's modify to do multiple restarts if time remains.
    
    # Rewrite search logic for multiple restarts
    
    # Reset
    best_score = 100
    best_edges = None
    
    while time.time() < deadline:
        # Random restart
        random_edges = set(random.sample(range(len(all_pairs)), m))
        current_edges = random_edges
        current_score = evaluate(current_edges)
        
        if current_score < best_score:
            best_score = current_score
            best_edges = set(current_edges)
            
        # Local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            edges_in_list = list(current_edges)
            random.shuffle(edges_in_list)
            
            for e_in in edges_in_list:
                if time.time() > deadline:
                    break
                edges_out_list = [i for i in range(len(all_pairs)) if i not in current_edges]
                random.shuffle(edges_out_list)
                
                for e_out in edges_out_list:
                    new_edges = set(current_edges)
                    new_edges.remove(e_in)
                    new_edges.add(e_out)
                    
                    new_score = evaluate(new_edges)
                    
                    if new_score < current_score:
                        current_edges = new_edges
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_edges = set(current_edges)
                        break
                if improved:
                    break
            if time.time() > deadline:
                break

    final_edges = [list(all_pairs[i]) for i in best_edges]
    final_values = [int(v) for v in values]
    
    return {'values': final_values, 'edges': final_edges}
