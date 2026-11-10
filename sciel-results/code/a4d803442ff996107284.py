"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 14
    m = 35
    
    def _raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            v = values[i]
            w = values[j]
            sums.add(v + w)
            prods.add(v * w)
        return max(len(sums), len(prods))

    def optimize_joint(values_init, edges, time_budget=10.0):
        """
        Simulated Annealing on the values for a FIXED edge set.
        This allows values to be specifically tuned to create collisions
        for the chosen edges, rather than picking generic values first.
        """
        current_values = list(values_init)
        current_score = _raw_value(current_values, edges)
        
        # Precompute edge indices for speed
        edge_indices = [(i, j) for i, j in edges]
        
        def calc_score(vals):
            sums = set()
            prods = set()
            for i, j in edge_indices:
                v = vals[i]
                w = vals[j]
                sums.add(v + w)
                prods.add(v * w)
            return max(len(sums), len(prods))

        best_values = list(current_values)
        best_score = current_score
        
        T = 50.0
        T_min = 0.1
        cooling = 0.995
        
        while T > T_min and (time.time() - start_time) < deadline - 5:
            # Perturb one random value
            idx = random.randint(0, n - 1)
            old_val = current_values[idx]
            
            # Try a new value
            # Bias towards small numbers as they are more likely to collide
            # but allow exploration up to 10000
            if random.random() < 0.7:
                new_val = random.randint(1, 50)
            else:
                new_val = random.randint(1, 1000)
                
            if new_val == old_val:
                continue
                
            # Check uniqueness constraint
            if new_val in current_values and current_values.count(new_val) > 0 and current_values[idx] != new_val:
                 # Need to check if new_val is already present at another index
                 is_dup = False
                 for k in range(n):
                     if k != idx and current_values[k] == new_val:
                         is_dup = True
                         break
                 if is_dup:
                     continue

            current_values[idx] = new_val
            
            new_score = calc_score(current_values)
            
            if new_score < current_score or random.random() < math.exp(-(new_score - current_score) / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_values = list(current_values)
            else:
                current_values[idx] = old_val
                
            T *= cooling
            
        return best_values, best_score

    # Strategy: 
    # 1. Define a few high-density edge structures (cliques + extras).
    #    A clique of size k has k(k-1)/2 edges.
    #    k=9 -> 36 edges (too many, need 35). So take a 9-clique and remove 1 edge.
    #    k=8 -> 28 edges. Need 7 more.
    #    k=10 -> 45 edges. Too many.
    #    
    #    Structure A: 9-clique minus 1 edge. Vertices 0..8 involved.
    #    Structure B: 8-clique (0..7) + 7 edges connecting vertex 8 to 0..7? 
    #                 Actually, let's just generate random dense graphs and anneal values.
    
    # We will test a few specific graph topologies that are "collision-friendly".
    # Topology 1: The complete graph K9 minus one edge.
    # This forces 35 edges among 9 vertices. The remaining 5 vertices are isolated.
    # This maximizes local density.
    
    # Topology 2: Complete graph K8 (28 edges) plus a star or path from vertex 8 to others?
    # Let's stick to K9-1 as the primary candidate because it's the most constrained.
    
    # Generate the edge list for K9 minus (0,1)
    edges_k9 = []
    for i in range(9):
        for j in range(i+1, 9):
            if i == 0 and j == 1:
                continue
            edges_k9.append((i, j))
    # We have 36 edges in K9. Removing 1 gives 35.
    assert len(edges_k9) == 35
    
    # We need 14 values. The last 5 (indices 9-13) are not in any edge.
    # They can be any distinct integers. We just need them to be distinct from each other and the first 9.
    
    best_overall_score = float('inf')
    best_overall_values = None
    best_overall_edges = None
    
    # Initial value guesses for the 9 active vertices
    # Small integers are good for collisions.
    # e.g., 1, 2, 3, 4, 5, 6, 7, 8, 9
    # Sums: 3..17 (many collisions if values are close)
    # Products: 2..72
    
    # We will run SA on the first 9 values, and assign the last 5 to be large distinct numbers
    # to avoid interfering (though they don't interfere since they have no edges).
    
    random.seed(42)
    
    # Try multiple restarts with different initial seeds for the active 9 vertices
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        # Initialize active values
        # Mix of small integers
        active_vals = random.sample(range(1, 30), 9)
        # Ensure distinct
        # Fill inactive values
        inactive_vals = random.sample(range(100, 1000), 5)
        
        # Combine
        full_vals = active_vals + inactive_vals
        
        # Check distinctness
        if len(set(full_vals)) != 14:
            continue
            
        # Optimize
        opt_vals, score = optimize_joint(full_vals, edges_k9, time_budget=3.0)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_values = opt_vals
            best_overall_edges = edges_k9
            
        # Early stop if we get a very good score
        if best_overall_score <= 10:
            break

    # If the K9-1 strategy didn't yield a great result, try a different topology?
    # For now, K9-1 is a strong structural bet.
    
    # Fallback if optimization failed or score is bad
    if best_overall_values is None or best_overall_score > 20:
        # Simple fallback: 1..14 and first 35 edges
        best_overall_values = list(range(1, 15))
        best_overall_edges = [list(e) for e in combinations(range(14), 2)[:35]]
        best_overall_score = _raw_value(best_overall_values, best_overall_edges)

    # Final Validation
    assert len(best_overall_values) == 14
    assert len(set(best_overall_values)) == 14
    assert all(isinstance(v, int) and v > 0 and v <= 10000 for v in best_overall_values)
    assert len(best_overall_edges) == 35
    assert all(len(e) == 2 for e in best_overall_edges)
    assert all(e[0] != e[1] for e in best_overall_edges)
    assert len(set(tuple(sorted(e)) for e in best_overall_edges)) == 35
    
    # Convert to lists for return
    return {
        'values': list(best_overall_values),
        'edges': [list(e) for e in best_overall_edges]
    }
# EVOLVE-BLOCK-END
