"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 90s limit
    
    # Strategy:
    # 1. Use values from an arithmetic progression to maximize sum collisions.
    #    If values are v_i = a + i*d, then v_i + v_j = 2a + (i+j)d.
    #    Distinct sums depend only on i+j.
    #    With 14 vertices (indices 0..13), possible sums i+j range from 1 to 25.
    #    So max distinct sums is 25 (if all index sums are covered by edges).
    #    Actually, min distinct sums is determined by the specific edges chosen.
    #    
    # 2. Use a modulus to force product collisions.
    #    If v_i = M * k_i + c, then v_i * v_j = M^2 k_i k_j + M c (k_i + k_j) + c^2.
    #    This is complex. Simpler: choose values such that many products are equal.
    #    Example: values {2, 3, 4, 6, 8, 9, ...}. 
    #    Actually, let's just search over a few structured value sets and optimize edges.
    
    # Candidate value sets:
    # Set A: Arithmetic progression 1..14. Sums: 3..27 (25 distinct). Products: many collisions?
    # Set B: Arithmetic progression 1, 3, 5, ..., 27. Sums: 4..54 step 2 (25 distinct).
    # Set C: Powers of 2? No, that's Sidon (35 sums).
    # Set D: Small numbers with high collision potential.
    
    # Let's try to construct a set where both sums and products have low distinct counts.
    # Consider values: 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48
    # This is a multiplicative Sidon set? No.
    # Let's use a heuristic search for values and edges jointly.
    
    # Given the time constraint, I'll try a few promising structured value sets
    # and for each, find the best 35 edges using a greedy/heuristic approach.
    
    # Value Set 1: 1..14
    # Value Set 2: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 (Sidon - bad for sums)
    # Value Set 3: Multiples of 2: 2, 4, 6, ..., 28
    # Value Set 4: Multiples of 3: 3, 6, 9, ..., 42
    
    # Let's evaluate a few sets and pick the one that allows the lowest max(distinct_sums, distinct_products).
    
    def evaluate_set(values):
        """Find best 35 edges for given values."""
        n = len(values)
        all_edges = list(combinations(range(n), 2))
        
        # Precompute sum and product for all edges
        edge_props = []
        for u, v in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            edge_props.append((u, v, s, p))
        
        # We want to pick 35 edges to minimize max(|unique_sums|, |unique_products|).
        # Greedy approach: 
        # 1. Start with empty set.
        # 2. Add edge that minimizes the increase in max(distinct_sums, distinct_products).
        #    This is a heuristic.
        
        # Better heuristic:
        # Group edges by sum. Group edges by product.
        # We want to pick edges such that the number of distinct sums used is small 
        # AND the number of distinct products used is small.
        
        # Let's try a local search / simulated annealing on the edge set.
        # Initial state: first 35 edges.
        
        def get_score(edges):
            sums = set()
            prods = set()
            for u, v in edges:
                sums.add(values[u] + values[v])
                prods.add(values[u] * values[v])
            return max(len(sums), len(prods))
        
        # Initial edges: first 35
        current_edges = set(all_edges[:35])
        current_score = get_score(current_edges)
        
        # Simple local search: swap one edge in with one edge out
        best_edges = current_edges
        best_score = current_score
        
        # Time limit for this sub-problem
        sub_deadline = start_time + 45.0
        
        # Precompute properties for fast lookup
        edge_to_props = { (u,v): (s, p) for u, v, s, p in edge_props }
        
        # To speed up get_score, we can maintain counts
        sum_counts = {}
        prod_counts = {}
        for u, v in current_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
            
        distinct_sums = len([s for s, c in sum_counts.items() if c > 0])
        distinct_prods = len([p for p, c in prod_counts.items() if c > 0])
        
        def try_swap(remove_edge, add_edge):
            nonlocal sum_counts, prod_counts, distinct_sums, distinct_prods, best_score, best_edges
            
            u1, v1 = remove_edge
            s1 = values[u1] + values[v1]
            p1 = values[u1] * values[v1]
            
            u2, v2 = add_edge
            s2 = values[u2] + values[v2]
            p2 = values[u2] * values[v2]
            
            # Check if add_edge is already in current_edges
            if add_edge in current_edges:
                return False
            
            # Remove remove_edge
            sum_counts[s1] -= 1
            if sum_counts[s1] == 0:
                distinct_sums -= 1
            prod_counts[p1] -= 1
            if prod_counts[p1] == 0:
                distinct_prods -= 1
                
            # Add add_edge
            if s2 not in sum_counts:
                sum_counts[s2] = 1
                distinct_sums += 1
            else:
                sum_counts[s2] += 1
                
            if p2 not in prod_counts:
                prod_counts[p2] = 1
                distinct_prods += 1
            else:
                prod_counts[p2] += 1
                
            new_score = max(distinct_sums, distinct_prods)
            
            if new_score < best_score:
                best_score = new_score
                # Commit the change
                current_edges.discard(remove_edge)
                current_edges.add(add_edge)
                best_edges = set(current_edges)
                return True
            else:
                # Revert
                sum_counts[s1] += 1
                if sum_counts[s1] == 1:
                    distinct_sums += 1
                prod_counts[p1] += 1
                if prod_counts[p1] == 1:
                    distinct_prods += 1
                    
                sum_counts[s2] -= 1
                if sum_counts[s2] == 0:
                    distinct_sums -= 1
                prod_counts[p2] -= 1
                if prod_counts[p2] == 0:
                    distinct_prods -= 1
                return False

        # Iterate swaps
        # To be efficient, we can just try random swaps
        random.seed(42)
        iterations = 0
        while time.time() < sub_deadline:
            # Pick a random edge to remove from current set
            if not current_edges: break
            remove_edge = random.choice(list(current_edges))
            # Pick a random edge to add from all edges not in current set
            # To avoid O(N) lookup, we can pick randomly from all_edges and check
            add_edge = random.choice(all_edges)
            if add_edge == remove_edge or add_edge in current_edges:
                continue
                
            try_swap(remove_edge, add_edge)
            iterations += 1
            if iterations % 1000 == 0:
                pass # Just a marker
        
        return list(best_edges), best_score

    # Test a few value sets
    value_sets = [
        [i+1 for i in range(14)],          # 1..14
        [2*(i+1) for i in range(14)],      # 2..28
        [3*(i+1) for i in range(14)],      # 3..42
        [i*2 for i in range(1, 15)],       # 2, 4, ..., 28
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        # Try a set with more product structure
        [2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48],
        # Try a geometric-ish progression but small
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192] # Sidon, expect 35
    ]
    
    best_global_score = float('inf')
    best_global_result = None
    
    for vals in value_sets:
        # Ensure distinct and positive
        if len(set(vals)) != 14 or any(v <= 0 or v > 10000 for v in vals):
            continue
            
        edges, score = evaluate_set(vals)
        
        if score < best_global_score:
            best_global_score = score
            best_global_result = {'values': vals, 'edges': [list(e) for e in edges]}
            
        if time.time() > deadline - 5:
            break
            
    # If no result found (should not happen), fallback
    if best_global_result is None:
        best_global_result = {'values': [i+1 for i in range(14)], 'edges': [list(e) for e in list(combinations(range(14), 2))[:35]]}
        
    # Final verification
    values = best_global_result['values']
    edges = best_global_result['edges']
    
    assert len(values) == 14
    assert len(set(values)) == 14
    assert all(isinstance(v, int) for v in values)
    assert all(1 <= v <= 10000 for v in values)
    assert len(edges) == 35
    assert all(isinstance(e, list) for e in edges)
    assert all(len(e) == 2 for e in edges)
    assert all(isinstance(i, int) for e in edges for i in e)
    assert all(0 <= i < 14 for e in edges for i in e)
    edge_tuples = set(tuple(sorted(e)) for e in edges)
    assert len(edge_tuples) == 35
    
    return best_global_result
# EVOLVE-BLOCK-END
