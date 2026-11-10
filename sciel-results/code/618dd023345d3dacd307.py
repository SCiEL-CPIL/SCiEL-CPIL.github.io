"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    n = 14
    m = 35
    maximum = 10000
    
    def _raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            v = values[i]
            w = values[j]
            sums.add(v + w)
            prods.add(v * w)
        return max(len(sums), len(prods))
    
    # Strategy:
    # 1. The graph structure matters. We need 35 edges on 14 vertices.
    #    Complete graph K14 has 14*13/2 = 91 edges.
    #    We want to minimize distinct sums and products.
    #    If we choose values from an arithmetic progression (AP), sums collide frequently.
    #    If we choose values from a geometric progression (GP), products collide frequently.
    #    A mix or specific structure might help.
    #    
    #    Key Insight: To minimize the number of distinct sums, we want many pairs to have the same sum.
    #    This is achieved if the values are symmetric around a center, e.g., {1, 2, ..., k}.
    #    For products, if values are powers of 2, products are unique (binary representation).
    #    So we need a balance.
    #    
    #    Let's try a specific graph structure:
    #    A complete bipartite graph? Or a dense subgraph?
    #    If we use K9 (36 edges) minus 1 edge, we have 35 edges.
    #    Vertices 0..8 form a clique (minus one edge). Vertices 9..13 are isolated.
    #    This concentrates all sums/products on 9 values.
    #    If we spread edges out, we involve more values, potentially increasing distinct counts.
    #    So concentrating on a small subset of values is likely better.
    #    
    #    Let's assume the optimal structure is a clique of size 9 (or similar dense subgraph)
    #    using 9 specific values, and the other 5 values are arbitrary distinct integers.
    #    
    #    We need to find 9 values that minimize max(distinct_sums, distinct_products)
    #    over the edges of a K9 (or K9 minus one edge).
    #    
    #    Let's search for 9 values from a small range.
    #    Candidate sets:
    #    - Consecutive integers: 1..9
    #    - Arithmetic progressions: a, a+d, ..., a+8d
    #    - Geometric progressions: a, ar, ..., ar^8 (integers)
    #    - Random small integers
    
    # We will search over value sets and a fixed graph structure (K9 minus 1 edge on first 9 vertices).
    # Why K9 minus 1? K9 has 36 edges. We need 35. Removing one edge reduces distinct sums/products by at most 1.
    # Removing an edge with a "unique" sum/product is best.
    
    # Graph: Vertices 0..8. Edges: all pairs (i,j) with 0<=i<j<=8 except (7,8).
    # This is symmetric enough that we can treat the 9 values as a multiset?
    # No, the assignment to vertices matters if the graph is not vertex-transitive.
    # K9 is vertex-transitive. K9 minus one edge is NOT vertex-transitive.
    # Vertices 7 and 8 have degree 7. Others have degree 8.
    # So we should assign specific values to 7 and 8.
    
    # Let's iterate over possible sets of 9 values.
    # To keep it fast, we'll sample from small ranges.
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Fixed edges for the active subgraph (indices 0..8)
    active_edges = []
    for i in range(9):
        for j in range(i+1, 9):
            if i == 7 and j == 8:
                continue
            active_edges.append([i, j])
    
    def evaluate_set_and_perm(vals):
        # vals is a list of 9 distinct ints
        # We need to assign them to indices 0..8.
        # Due to symmetry of 0..6 and symmetry of 7,8:
        # We can sort the 7 values for 0..6 and assign the remaining 2 to 7,8.
        # But which 2 go to 7,8? And in what order? (Order between 7 and 8 doesn't matter for the set of sums/products? 
        # Actually, edge (7,0) vs (8,0). Since 0 is connected to both, the sums are v0+v7 and v0+v8.
        # The set of sums depends on the multiset of values at 7 and 8.
        # The set of products depends on the multiset of values at 7 and 8.
        # So the order of 7 and 8 doesn't matter.
        # However, the assignment of the other 7 values to 0..6 doesn't matter either, because 0..6 form a clique among themselves
        # and are all connected to 7 and 8.
        # Wait, 0..6 are a clique. 7 is connected to 0..6. 8 is connected to 0..6. 7 and 8 are NOT connected.
        # So the graph is:
        # - Clique C of size 7 (indices 0..6)
        # - Two vertices A, B (indices 7,8)
        # - A and B connected to all in C.
        # - A and B NOT connected to each other.
        # 
        # Sums:
        # - Pairs in C: sum of two values from C.
        # - Pairs (A, c): value_A + value_c for all c in C.
        # - Pairs (B, c): value_B + value_c for all c in C.
        # 
        # Products:
        # - Pairs in C: product of two values from C.
        # - Pairs (A, c): value_A * value_c for all c in C.
        # - Pairs (B, c): value_B * value_c for all c in C.
        # 
        # So the objective depends on:
        # 1. The set of 7 values for C.
        # 2. The two values for A and B.
        # 
        # We can iterate over all ways to choose 2 values for {A,B} from the 9, and the remaining 7 for C.
        # There are C(9,2) = 36 ways.
        # For each way, we compute the score.
        
        sums = set()
        prods = set()
        
        # Let C_vals be the 7 values, A_val, B_val be the other two.
        # We don't need to permute C_vals among themselves because the internal edges of C form a complete graph.
        # The set of sums for C is just {c_i + c_j for i<j in C}.
        # The set of products for C is just {c_i * c_j for i<j in C}.
        
        # We need to try all 36 splits.
        best_local = float('inf')
        for a_idx in range(9):
            for b_idx in range(a_idx+1, 9):
                A_val = vals[a_idx]
                B_val = vals[b_idx]
                C_vals = [v for k, v in enumerate(vals) if k != a_idx and k != b_idx]
                
                # Compute sums and products for C
                for i in range(7):
                    for j in range(i+1, 7):
                        sums.add(C_vals[i] + C_vals[j])
                        prods.add(C_vals[i] * C_vals[j])
                
                # Add edges to A
                for c in C_vals:
                    sums.add(A_val + c)
                    prods.add(A_val * c)
                
                # Add edges to B
                for c in C_vals:
                    sums.add(B_val + c)
                    prods.add(B_val * c)
                
                score = max(len(sums), len(prods))
                if score < best_local:
                    best_local = score
        
        return best_local

    # Search for good value sets
    # We'll try random sets of 9 distinct integers from 1..N for various N.
    # Also try APs and GPs.
    
    # 1. Try consecutive integers 1..9, 2..10, etc.
    for start in range(1, 100):
        if time.time() > deadline - 5:
            break
        vals = list(range(start, start+9))
        score = evaluate_set_and_perm(vals)
        if score < best_score:
            best_score = score
            best_values = vals[:]
    
    # 2. Try APs
    for start in range(1, 50):
        for d in range(1, 20):
            if time.time() > deadline - 5:
                break
            vals = [start + i*d for i in range(9)]
            if max(vals) > 10000:
                continue
            score = evaluate_set_and_perm(vals)
            if score < best_score:
                best_score = score
                best_values = vals[:]
    
    # 3. Try Random Sets
    random.seed(42)
    for _ in range(5000):
        if time.time() > deadline - 5:
            break
        # Sample from 1..100
        vals = random.sample(range(1, 101), 9)
        score = evaluate_set_and_perm(vals)
        if score < best_score:
            best_score = score
            best_values = vals[:]
    
    # 4. Try specific "collision-prone" sets
    # e.g., {1, 2, 3, 4, 5, 6, 7, 8, 9}
    # {1, 2, 4, 8, 16, 32, 64, 128, 256} (powers of 2 - bad for products)
    # {1, 3, 5, 7, 9, 11, 13, 15, 17} (odd AP)
    
    # If we found a good set, construct the full solution.
    if best_values is None:
        # Fallback to 1..9
        best_values = list(range(1, 10))
        best_score = evaluate_set_and_perm(best_values)
    
    # Now we need to determine the actual assignment to indices 0..8 that achieves best_score.
    # And fill in the remaining 5 values.
    
    # Re-evaluate to find the best split (A, B, C)
    best_local_score = float('inf')
    best_A = None
    best_B = None
    best_C = None
    
    for a_idx in range(9):
        for b_idx in range(a_idx+1, 9):
            A_val = best_values[a_idx]
            B_val = best_values[b_idx]
            C_vals = [v for k, v in enumerate(best_values) if k != a_idx and k != b_idx]
            
            sums = set()
            prods = set()
            for i in range(7):
                for j in range(i+1, 7):
                    sums.add(C_vals[i] + C_vals[j])
                    prods.add(C_vals[i] * C_vals[j])
            for c in C_vals:
                sums.add(A_val + c)
                prods.add(A_val * c)
            for c in C_vals:
                sums.add(B_val + c)
                prods.add(B_val * c)
            
            score = max(len(sums), len(prods))
            if score < best_local_score:
                best_local_score = score
                best_A = A_val
                best_B = B_val
                best_C = C_vals
    
    # Assign:
    # Indices 0..6 get best_C (order doesn't matter for the set, but we must assign distinct indices)
    # Indices 7, 8 get best_A, best_B
    final_values = [0] * 14
    for i in range(7):
        final_values[i] = best_C[i]
    final_values[7] = best_A
    final_values[8] = best_B
    
    # Fill remaining 5 values (9..13) with distinct integers not used
    used = set(final_values[:9])
    for i in range(9, 14):
        # Find a small integer not in used
        for candidate in range(1, 10001):
            if candidate not in used:
                final_values[i] = candidate
                used.add(candidate)
                break
    
    # Construct edges
    final_edges = active_edges[:] # This is already the list of [i,j] for the 9-vertex subgraph
    
    # Verify
    actual_score = _raw_value(final_values, final_edges)
    # It should match best_local_score
    
    # Ensure types
    final_values = [int(v) for v in final_values]
    final_edges = [[int(i), int(j)] for i, j in final_edges]
    
    return {'values': final_values, 'edges': final_edges}

# EVOLVE-BLOCK-END
