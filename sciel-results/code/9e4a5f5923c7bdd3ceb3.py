import time
import random
from itertools import combinations
import sys

def _score(values, edges):
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _local_search_edges(values, n, m, rng, max_iters=5000):
    """
    Optimizes the set of m edges for given values using local search.
    Returns the best edge list found.
    """
    all_edges = list(combinations(range(n), 2))
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    # Precompute sums and products for all edges
    edge_sums = {}
    edge_prods = {}
    for u, v in all_edges:
        edge_sums[(u, v)] = values[u] + values[v]
        edge_prods[(u, v)] = values[u] * values[v]

    # Initialize with a greedy set or random set
    # Greedy: pick edges that minimize new distinct sums/prods
    # But local search can fix it. Let's start with random m edges.
    initial_edges = rng.sample(all_edges, m)
    
    # To make it deterministic-ish but varied, we can sort by some heuristic
    # Let's just start random and let local search do the work.
    
    current_edges = set(initial_edges)
    current_edge_list = list(current_edges)
    
    # Maintain counts
    sum_counts = {}
    prod_counts = {}
    
    def update_counts(edges, add=True):
        for e in edges:
            s = edge_sums[e]
            p = edge_prods[e]
            if add:
                sum_counts[s] = sum_counts.get(s, 0) + 1
                prod_counts[p] = prod_counts.get(p, 0) + 1
            else:
                sum_counts[s] -= 1
                if sum_counts[s] == 0:
                    del sum_counts[s]
                prod_counts[p] -= 1
                if prod_counts[p] == 0:
                    del prod_counts[p]

    update_counts(current_edges, True)
    
    current_distinct_sums = len(sum_counts)
    current_distinct_prods = len(prod_counts)
    current_score = max(current_distinct_sums, current_distinct_prods)
    
    best_edges = current_edges.copy()
    best_score = current_score
    
    # Precompute list of edges not in current set for faster selection?
    # Just filter each time or maintain a set.
    not_in_current = set(all_edges) - current_edges
    
    for _ in range(max_iters):
        # Pick a random edge to remove
        rem_edge = rng.choice(list(current_edges))
        
        # Pick a random edge to add from not_in_current
        if not not_in_current:
            break
        add_edge = rng.choice(list(not_in_current))
        
        # Calculate delta
        s_rem = edge_sums[rem_edge]
        p_rem = edge_prods[rem_edge]
        s_add = edge_sums[add_edge]
        p_add = edge_prods[add_edge]
        
        # Simulate removal
        # distinct sums change:
        # if sum_counts[s_rem] == 1, removing it reduces distinct sums by 1
        # if sum_counts[s_rem] > 1, no change
        
        new_distinct_sums = current_distinct_sums
        new_distinct_prods = current_distinct_prods
        
        # Remove rem_edge
        if sum_counts[s_rem] == 1:
            new_distinct_sums -= 1
        if prod_counts[p_rem] == 1:
            new_distinct_prods -= 1
            
        # Add add_edge
        # Note: if s_add == s_rem, the count logic is tricky.
        # Let's just calculate the new counts explicitly for the two affected sums/prods.
        
        # It's safer to just compute the new distinct counts if the sets are small.
        # But we can do it incrementally.
        
        # Let's trace carefully.
        # Old state: sum_counts, prod_counts.
        
        # 1. Remove rem_edge
        sum_counts[s_rem] -= 1
        if sum_counts[s_rem] == 0:
            del sum_counts[s_rem]
        prod_counts[p_rem] -= 1
        if prod_counts[p_rem] == 0:
            del prod_counts[p_rem]
            
        # 2. Add add_edge
        if s_add not in sum_counts:
            sum_counts[s_add] = 1
        else:
            sum_counts[s_add] += 1
            
        if p_add not in prod_counts:
            prod_counts[p_add] = 1
        else:
            prod_counts[p_add] += 1
            
        new_distinct_sums = len(sum_counts)
        new_distinct_prods = len(prod_counts)
        new_score = max(new_distinct_sums, new_distinct_prods)
        
        if new_score < current_score:
            # Accept move
            current_edges.remove(rem_edge)
            current_edges.add(add_edge)
            not_in_current.remove(add_edge)
            not_in_current.add(rem_edge)
            
            current_distinct_sums = new_distinct_sums
            current_distinct_prods = new_distinct_prods
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_edges = current_edges.copy()
                
        else:
            # Reject move, revert counts
            # Undo Add
            sum_counts[s_add] -= 1
            if sum_counts[s_add] == 0:
                del sum_counts[s_add]
            prod_counts[p_add] -= 1
            if prod_counts[p_add] == 0:
                del prod_counts[p_add]
                
            # Undo Remove
            if s_rem not in sum_counts:
                sum_counts[s_rem] = 1
            else:
                sum_counts[s_rem] += 1
            if p_rem not in prod_counts:
                prod_counts[p_rem] = 1
            else:
                prod_counts[p_rem] += 1
                
            # Restore current distinct counts (they should be back to original)
            # current_distinct_sums and prods were not updated in reject branch
            # So they are still valid.

    return list(best_edges), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 14
    m = 35
    max_val = 10000
    
    rng = random.Random(42)
    
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    # Value candidates
    # 1. Consecutive 1..14
    # 2. Consecutive 2..15
    # 3. Random small
    # 4. Perturbed 1..14
    
    value_candidates = []
    value_candidates.append(list(range(1, n + 1)))
    value_candidates.append(list(range(2, n + 2)))
    value_candidates.append(list(range(1, n + 1, 2)) + list(range(n+1, n+7))) # Mix of odds and higher evens?
    
    # Generate random candidates
    for _ in range(20):
        # Sample from 1..30
        vals = rng.sample(range(1, 31), n)
        value_candidates.append(vals)
        
    # Perturbed 1..14
    for _ in range(20):
        vals = list(range(1, n + 1))
        for _ in range(rng.randint(1, 3)):
            i = rng.randrange(n)
            # Swap with a random value in 1..30
            j = rng.randint(1, 30)
            # Ensure distinct
            if j not in vals:
                vals[i] = j
            else:
                # Try to find a distinct one
                while j in vals:
                    j = rng.randint(1, 30)
                vals[i] = j
        value_candidates.append(vals)
        
    # Run local search for each candidate
    for vals in value_candidates:
        if time.time() > deadline:
            break
            
        # Validate
        if len(set(vals)) != n or any(v < 1 or v > max_val for v in vals):
            continue
            
        # Run local search
        edges, score = _local_search_edges(vals, n, m, rng, max_iters=2000)
        
        if score < best_score:
            best_score = score
            best_values = vals[:]
            best_edges = [list(e) for e in edges]
            
            # If we find a very good score, we can stop early
            if best_score <= 20:
                break
                
    # If no valid solution found, use fallback
    if best_values is None:
        best_values = list(range(1, n + 1))
        best_edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
        
    # Final validation
    assert len(best_values) == n
    assert len(set(best_values)) == n
    assert all(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= max_val for v in best_values)
    assert len(best_edges) == m
    edge_set = set()
    for e in best_edges:
        assert len(e) == 2
        i, j = e
        assert isinstance(i, int) and isinstance(j, int)
        assert i != j
        assert 0 <= i < n and 0 <= j < n
        key = (min(i, j), max(i, j))
        assert key not in edge_set
        edge_set.add(key)
        
    return {'values': best_values, 'edges': best_edges}
