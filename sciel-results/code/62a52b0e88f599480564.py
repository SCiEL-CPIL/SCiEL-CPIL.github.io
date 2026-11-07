"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _raw_score(values, edges):
    """Compute max(distinct sums, distinct products) over edges."""
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return max(len(sums), len(prods))

def _all_pairs(n):
    """Return all possible undirected edges as (i, j) tuples with i < j."""
    return list(combinations(range(n), 2))

def _greedy_select_edges(values, n, m, rng, deadline):
    """
    Greedily select m edges that minimize max(distinct sums, distinct products).
    At each step, pick the edge that adds the least new distinct values.
    """
    all_edges = _all_pairs(n)
    # Precompute sum and product for each edge
    edge_data = {}
    for i, j in all_edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        edge_data[(i, j)] = (s, p)
    
    selected = []
    current_sums = set()
    current_prods = set()
    remaining = all_edges[:]
    rng.shuffle(remaining)
    
    while len(selected) < m and time.time() < deadline:
        best_edge = None
        best_new_score = float('inf')
        best_new_sums = None
        best_new_prods = None
        
        for e in remaining:
            s, p = edge_data[e]
            new_sums = current_sums | {s}
            new_prods = current_prods | {p}
            score = max(len(new_sums), len(new_prods))
            if score < best_new_score:
                best_new_score = score
                best_edge = e
                best_new_sums = new_sums
                best_new_prods = new_prods
        
        if best_edge is None:
            break
        
        selected.append(best_edge)
        current_sums = best_new_sums
        current_prods = best_new_prods
        remaining.remove(best_edge)
    
    return selected

def _sa_optimize(init_vals, init_edges, n, m, max_val, rng, deadline, max_iter=5000):
    """
    Simulated annealing that jointly optimizes values and edges.
    Moves: change one value, swap one edge, add/remove edge (maintaining m edges).
    """
    current_vals = init_vals[:]
    current_edges = init_edges[:]
    current_score = _raw_score(current_vals, current_edges)
    
    best_vals = current_vals[:]
    best_edges = current_edges[:]
    best_score = current_score
    
    all_edges_set = set(_all_pairs(n))
    current_edge_set = set(tuple(sorted(e)) for e in current_edges)
    
    T = 10.0
    cooling = 0.9995
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.3:
            # Change one value
            idx = rng.randint(0, n - 1)
            old_val = current_vals[idx]
            # Try a random new value
            new_val = rng.randint(1, max_val)
            if new_val == old_val:
                continue
            # Check distinctness
            if new_val in current_vals:
                continue
            current_vals[idx] = new_val
            
            new_score = _raw_score(current_vals, current_edges)
            delta = new_score - current_score
            
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_vals = current_vals[:]
                    best_edges = current_edges[:]
            else:
                current_vals[idx] = old_val
                
        elif move_type < 0.6:
            # Swap one edge: remove a random edge, add a random different edge
            if len(current_edges) < 2:
                continue
            remove_idx = rng.randint(0, len(current_edges) - 1)
            removed_edge = tuple(sorted(current_edges[remove_idx]))
            
            # Find a random edge not currently in the set
            candidates = [e for e in all_edges_set if e not in current_edge_set]
            if not candidates:
                continue
            added_edge = rng.choice(candidates)
            
            current_edges[remove_idx] = list(added_edge)
            current_edge_set.discard(removed_edge)
            current_edge_set.add(added_edge)
            
            new_score = _raw_score(current_vals, current_edges)
            delta = new_score - current_score
            
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_vals = current_vals[:]
                    best_edges = current_edges[:]
            else:
                # Revert
                current_edges[remove_idx] = list(removed_edge)
                current_edge_set.discard(added_edge)
                current_edge_set.add(removed_edge)
                
        else:
            # Change two values
            idx1 = rng.randint(0, n - 1)
            idx2 = rng.randint(0, n - 1)
            if idx1 == idx2:
                continue
            old_v1 = current_vals[idx1]
            old_v2 = current_vals[idx2]
            new_v1 = rng.randint(1, max_val)
            new_v2 = rng.randint(1, max_val)
            if new_v1 == old_v1 and new_v2 == old_v2:
                continue
            if new_v1 in current_vals or new_v2 in current_vals:
                continue
            if new_v1 == new_v2:
                continue
            
            current_vals[idx1] = new_v1
            current_vals[idx2] = new_v2
            
            new_score = _raw_score(current_vals, current_edges)
            delta = new_score - current_score
            
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_vals = current_vals[:]
                    best_edges = current_edges[:]
            else:
                current_vals[idx1] = old_v1
                current_vals[idx2] = old_v2
        
        T *= cooling
    
    return best_vals, best_edges, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline, 10 second safety margin
    
    n = 14
    m = 35
    max_val = 10000
    
    best_values = list(range(1, n + 1))
    best_edges = [list(e) for e in _all_pairs(n)[:m]]
    best_score = _raw_score(best_values, best_edges)
    
    rng = random.Random(42)
    
    # Multi-start: generate diverse initial solutions, run SA on each
    num_starts = 30
    
    for start_i in range(num_starts):
        if time.time() > deadline:
            break
        
        # Generate initial values
        if start_i == 0:
            init_vals = list(range(1, n + 1))
        elif start_i == 1:
            init_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        elif start_i == 2:
            # Small values to encourage collisions
            init_vals = rng.sample(range(1, 30), n)
        elif start_i == 3:
            # Arithmetic progression
            step = rng.randint(1, 5)
            init_vals = [1 + i * step for i in range(n)]
            if len(set(init_vals)) != n or any(v > max_val for v in init_vals):
                init_vals = rng.sample(range(1, max_val + 1), n)
        elif start_i == 4:
            # Geometric-ish
            base = rng.randint(2, 5)
            init_vals = []
            for i in range(n):
                v = base ** i
                if v > max_val:
                    v = rng.randint(1, max_val)
                init_vals.append(v)
            if len(set(init_vals)) != n:
                init_vals = rng.sample(range(1, max_val + 1), n)
        else:
            # Random
            init_vals = rng.sample(range(1, max_val + 1), n)
        
        # Greedy edge selection for this value set
        init_edges = _greedy_select_edges(init_vals, n, m, rng, deadline)
        if len(init_edges) < m:
            continue
        
        init_score = _raw_score(init_vals, init_edges)
        
        # Run simulated annealing
        sa_vals, sa_edges, sa_score = _sa_optimize(
            init_vals, init_edges, n, m, max_val, rng, deadline, max_iter=3000
        )
        
        if sa_score < best_score:
            best_score = sa_score
            best_values = sa_vals[:]
            best_edges = sa_edges[:]
    
    # Final validation
    assert len(best_values) == n
    assert len(set(best_values)) == n
    assert all(isinstance(v, int) for v in best_values)
    assert all(1 <= v <= max_val for v in best_values)
    assert len(best_edges) == m
    for e in best_edges:
        assert len(e) == 2
        assert e[0] != e[1]
        assert 0 <= e[0] < n
        assert 0 <= e[1] < n
    
    # Check for duplicate edges
    edge_set = set()
    for e in best_edges:
        key = tuple(sorted(e))
        assert key not in edge_set
        edge_set.add(key)
    
    # Ensure types are correct
    best_values = [int(v) for v in best_values]
    best_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': best_values, 'edges': best_edges}
# EVOLVE-BLOCK-END
