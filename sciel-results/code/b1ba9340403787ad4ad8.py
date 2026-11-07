"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations

def construct():
    """
    Construct 14 distinct positive integers <= 10000 and exactly 35 edges
    to minimize max(number of distinct edge sums, number of distinct edge products).
    
    Strategy:
    1. Use a set of candidate values (small integers favor collisions).
    2. For each value set, use a greedy algorithm to select 35 edges that minimize
       the introduction of new distinct sums and products.
    3. Refine the edge selection using local search (edge swaps) to further minimize
       the max distinct count.
    4. Track the best solution found.
    """
    
    n = 14
    m = 35
    max_val = 10000
    
    def score(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            s = values[i] + values[j]
            p = values[i] * values[j]
            sums.add(s)
            prods.add(p)
        return max(len(sums), len(prods))

    def greedy_edge_selection(values, seed=0):
        """
        Greedily select 35 edges minimizing new distinct sums/products.
        """
        rng = random.Random(seed)
        all_pairs = list(combinations(range(n), 2))
        selected_edges = []
        used_sums = set()
        used_prods = set()
        
        # Precompute sums and products for all pairs
        pair_info = {}
        for i, j in all_pairs:
            s = values[i] + values[j]
            p = values[i] * values[j]
            pair_info[(i, j)] = (s, p)
        
        available = list(all_pairs)
        
        for _ in range(m):
            if not available:
                break
                
            best_pair = None
            best_cost = float('inf')
            best_new_sums = 0
            best_new_prods = 0
            
            # To avoid O(M*N^2) per step which is slow, we can sample or sort once?
            # M=35, N=91. 35*91 is small.
            for pair in available:
                s, p = pair_info[pair]
                new_s = 0 if s in used_sums else 1
                new_p = 0 if p in used_prods else 1
                # Cost function: prioritize reducing max distinct
                # If current max is determined by sums, adding a new sum is bad.
                # Simple heuristic: minimize (new_s + new_p), tie-break by random
                cost = new_s + new_p
                if cost < best_cost:
                    best_cost = cost
                    best_pair = pair
                    best_new_sums = new_s
                    best_new_prods = new_p
                elif cost == best_cost:
                    # Tie-breaking: prefer edges that share sums/prods with existing?
                    # Or just random to vary restarts
                    if rng.random() < 0.5:
                        best_pair = pair
                        best_new_sums = new_s
                        best_new_prods = new_p
            
            if best_pair is None:
                break
                
            s, p = pair_info[best_pair]
            selected_edges.append(list(best_pair))
            used_sums.add(s)
            used_prods.add(p)
            available.remove(best_pair)
            
        return selected_edges, len(used_sums), len(used_prods)

    def local_search_edges(values, edges, iterations=500, seed=0):
        """
        Refine edge set via swaps to minimize max distinct sums/prods.
        """
        rng = random.Random(seed)
        current_edges = [list(e) for e in edges]
        current_sums = set()
        current_prods = set()
        for i, j in current_edges:
            current_sums.add(values[i] + values[j])
            current_prods.add(values[i] * values[j])
        
        current_score = max(len(current_sums), len(current_prods))
        best_score = current_score
        best_edges = [list(e) for e in current_edges]
        
        edge_set = set(tuple(sorted(e)) for e in current_edges)
        all_pairs = list(combinations(range(n), 2))
        
        for _ in range(iterations):
            if best_score <= 12: # Theoretical lower bound is higher, but 12 is very good
                break
                
            # Pick a random edge to remove
            rem_idx = rng.randrange(len(current_edges))
            rem_edge = current_edges[rem_idx]
            rem_i, rem_j = rem_edge
            
            # Pick a random non-edge to add
            # Efficiently pick from all_pairs not in edge_set
            # Since n=14, total pairs 91. We can just filter.
            non_edges = [p for p in all_pairs if p not in edge_set]
            if not non_edges:
                continue
            add_edge = rng.choice(non_edges)
            add_i, add_j = add_edge
            
            # Calculate new score
            # Remove old contributions
            old_sum = values[rem_i] + values[rem_j]
            old_prod = values[rem_i] * values[rem_j]
            new_sum = values[add_i] + values[add_j]
            new_prod = values[add_i] * values[add_j]
            
            # We need to know if old_sum/old_prod become unique (removed from set)
            # and if new_sum/new_prod are already in set.
            
            # Count occurrences in current edges for old_sum/old_prod
            # This is expensive if done naively. 
            # Optimization: Maintain counts?
            # For n=14, m=35, we can just recompute sets for the candidate swap.
            # 500 iterations * 35 edges = 17500 ops. Very fast.
            
            candidate_edges = current_edges[:]
            candidate_edges[rem_idx] = list(add_edge)
            
            cand_sums = set()
            cand_prods = set()
            for i, j in candidate_edges:
                cand_sums.add(values[i] + values[j])
                cand_prods.add(values[i] * values[j])
                
            cand_score = max(len(cand_sums), len(cand_prods))
            
            if cand_score < current_score:
                current_edges = candidate_edges
                current_sums = cand_sums
                current_prods = cand_prods
                current_score = cand_score
                edge_set.remove(tuple(sorted(rem_edge)))
                edge_set.add(tuple(sorted(add_edge)))
                
                if current_score < best_score:
                    best_score = current_score
                    best_edges = [list(e) for e in current_edges]
            # Accept equal score moves with some probability to escape local minima?
            # Or just strict improvement for now.
            
        return best_edges, best_score

    # Candidate value sets
    # 1. 1..14
    # 2. Powers of 2? No, products distinct.
    # 3. Small primes?
    # 4. Random small ints
    
    value_candidates = [
        list(range(1, 15)),
        list(range(2, 16)),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], # Same as 1
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], # Large spread, bad for sums/prods collisions?
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    ]
    
    # Add some random small value sets
    for _ in range(5):
        vals = random.sample(range(1, 21), 14)
        vals.sort()
        value_candidates.append(vals)

    best_overall_score = float('inf')
    best_overall_values = None
    best_overall_edges = None

    for vals in value_candidates:
        # Ensure distinct and positive
        if len(set(vals)) != 14 or any(v < 1 or v > 10000 for v in vals):
            continue
            
        # Try multiple greedy starts
        for seed in range(5):
            edges, s_cnt, p_cnt = greedy_edge_selection(vals, seed)
            if len(edges) < m:
                continue
            # Refine
            refined_edges, ref_score = local_search_edges(vals, edges, iterations=1000, seed=seed)
            
            if ref_score < best_overall_score:
                best_overall_score = ref_score
                best_overall_values = vals[:]
                best_overall_edges = refined_edges
                
            # Early termination if we hit a very low score
            if best_overall_score <= 14:
                break
        if best_overall_score <= 14:
            break

    return {'values': best_overall_values, 'edges': best_overall_edges}

# EVOLVE-BLOCK-END
