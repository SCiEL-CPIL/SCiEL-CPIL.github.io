"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 90s limit
    
    # Objective: Minimize max(distinct_sums, distinct_products)
    # We need 14 distinct positive integers <= 10000.
    # We need 35 edges.
    
    # Strategy:
    # 1. Generate candidate value sets that are NOT Sidon (to allow sum collisions).
    #    Small integers are good for product collisions, but might create too many sum collisions.
    #    We want a balance.
    # 2. For a given value set, find the 35 edges that minimize the objective.
    # 3. Use local search to refine the value set.
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    def get_best_edges(values, max_iter=2000):
        # Greedy construction of edges to minimize objective
        # Start with empty edges, add edge that minimizes increase in max(distincts)
        # Or start with all possible edges and remove? 
        # Greedy addition is better for small k.
        
        all_possible_edges = list(combinations(range(14), 2))
        random.shuffle(all_possible_edges)
        
        best_edges = []
        current_sums = set()
        current_prods = set()
        
        # We want to pick 35 edges.
        # A simple greedy approach: pick edge that adds fewest new sums/prods.
        # However, we need exactly 35.
        
        # Let's try a different heuristic: 
        # Randomly initialize 35 edges, then swap edges to improve.
        
        # Initialize with random 35 edges
        edges = random.sample(all_possible_edges, 35)
        edges = [list(e) for e in edges]
        
        current_score = evaluate(values, edges)
        
        # Local search: try swapping one edge with another to improve score
        # Or adding/removing? We must keep exactly 35.
        # So swap: remove edge A, add edge B (not in set).
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Shuffle edges to avoid bias
            edges_idx = list(range(35))
            random.shuffle(edges_idx)
            
            for i in edges_idx:
                e_i = tuple(sorted(edges[i]))
                # Try removing e_i and adding a random new edge
                # To be efficient, try a few random candidates for the new edge
                for _ in range(5):
                    new_edge = random.choice(all_possible_edges)
                    if new_edge in edge_set and new_edge != e_i:
                        continue
                    
                    # Remove e_i, add new_edge
                    edge_set.remove(e_i)
                    edge_set.add(new_edge)
                    
                    new_edges_list = [list(e) for e in edge_set]
                    new_score = evaluate(values, new_edges_list)
                    
                    if new_score < current_score:
                        # Accept
                        edges[i] = list(new_edge)
                        current_score = new_score
                        improved = True
                        break # restart outer loop? or continue?
                    else:
                        # Revert
                        edge_set.remove(new_edge)
                        edge_set.add(e_i)
                
                if improved:
                    break
        return current_score, [list(e) for e in edge_set]
    
    def improve_values(values, best_edges, best_score):
        # Try swapping one value with another to improve score
        # Values must be distinct positive integers <= 10000
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Try changing one value
            idx = random.randint(0, 13)
            old_val = values[idx]
            
            # Try a few random new values
            for _ in range(10):
                new_val = random.randint(1, 10000)
                if new_val in values:
                    continue
                
                # Apply change
                values[idx] = new_val
                
                # Re-optimize edges for this new value set?
                # That's expensive. Let's just evaluate with current edges first.
                # If it improves, we might do a quick edge re-optimization.
                new_score = evaluate(values, best_edges)
                
                if new_score < best_score:
                    # Potential improvement. Do a quick edge local search to see if we can do better.
                    s, e = get_best_edges(values, max_iter=500)
                    if s < best_score:
                        best_score = s
                        best_edges = e
                        improved = True
                        break
                    else:
                        # Revert value change
                        values[idx] = old_val
                else:
                    # Revert value change
                    values[idx] = old_val
            
            if improved:
                break
        return best_score, best_edges, values

    # Initial Value Generation Strategies
    # 1. Small integers (1-20): high product collision, high sum collision?
    # 2. Geometric progression (2, 4, 8...): low product collision (distinct), high sum collision? No, Sidon.
    # 3. Random small integers.
    
    # Let's start with a set of small integers to encourage collisions.
    # Distinct values from 1 to 10000.
    
    best_overall_score = float('inf')
    best_overall_values = None
    best_overall_edges = None
    
    # Try several random initializations
    for init_iter in range(20):
        if time.time() > deadline - 10.0:
            break
        
        # Generate distinct values
        # Mix of small numbers
        vals = set()
        while len(vals) < 14:
            # Bias towards smaller numbers
            if random.random() < 0.5:
                vals.add(random.randint(1, 50))
            else:
                vals.add(random.randint(1, 200))
        
        values = list(vals)
        
        # Initial edge selection
        score, edges = get_best_edges(values)
        
        # Refine values
        score, edges, values = improve_values(values, edges, score)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_values = values
            best_overall_edges = edges
            
            # If we hit the theoretical lower bound or a very good score, stop?
            # Lower bound for sums is 35? No, if sums collide, distinct sums < 35.
            # Can distinct sums be < 14? No, min sum is v1+v2, max is v13+v14.
            # With 14 numbers, how many distinct sums can we have?
            # If we use {1, 2, 4, 8...}, sums are unique (35).
            # If we use {1, 1, 1...} (not distinct), sums are few.
            # But values MUST be distinct.
            # Can we get distinct sums < 27?
            # Example: 1, 2, 3, 4, 5... 
            # Sums of 1..14: 1+2=3, 1+3=4, 2+3=5... 
            # Max sum 13+14=27. Min sum 1+2=3.
            # Range 3..27. That's 25 distinct sums!
            # Products: 1*2=2, 1*3=3, 2*3=6...
            # Products of 1..14: max 13*14=182.
            # How many distinct products?
            # 1*14=14, 2*7=14. Collision!
            # 1*13=13, 13*1=13.
            # 2*6=12, 3*4=12. Collision!
            # So small integers might have few distinct products too.
            # Let's check 1..14 explicitly.
            
            if best_overall_score <= 25:
                break

    # If we didn't find a good one, fallback to a known good heuristic
    # or just return the best found.
    
    if best_overall_values is None:
        # Fallback: 1..14
        values = list(range(1, 15))
        score, edges = get_best_edges(values)
        best_overall_values = values
        best_overall_edges = edges
        best_overall_score = score

    # Final verification
    assert len(best_overall_values) == 14
    assert len(set(best_overall_values)) == 14
    assert all(isinstance(v, int) for v in best_overall_values)
    assert all(1 <= v <= 10000 for v in best_overall_values)
    
    assert len(best_overall_edges) == 35
    assert all(isinstance(e, list) for e in best_overall_edges)
    assert all(len(e) == 2 for e in best_overall_edges)
    assert all(isinstance(i, int) for e in best_overall_edges for i in e)
    assert all(0 <= i < 14 for e in best_overall_edges for i in e)
    
    edge_tuples = set(tuple(sorted(e)) for e in best_overall_edges)
    assert len(edge_tuples) == 35
    
    return {'values': best_overall_values, 'edges': best_overall_edges}
# EVOLVE-BLOCK-END
