"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _score(values, edges):
    """Compute the raw score: max(distinct sums, distinct products)."""
    sums = set()
    products = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        products.add(p)
    return max(len(sums), len(products))

def _all_edges(n):
    """Generate all possible edges for n vertices."""
    return list(combinations(range(n), 2))

def _random_initial_values(n, max_val=10000):
    """Generate initial values with structure: multiples of a base plus small offsets."""
    # Use a base multiplier to create shared factors
    base = random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18, 20])
    values = set()
    while len(values) < n:
        # Generate values as base * k + offset where offset is small
        k = random.randint(1, max_val // base)
        offset = random.choice([0, 0, 0, 1, -1, 2, -2])  # bias toward exact multiples
        v = base * k + offset
        if 1 <= v <= max_val and v not in values:
            values.add(v)
    return sorted(list(values))

def _random_edges(n, m):
    """Generate m random edges."""
    all_e = _all_edges(n)
    return [list(e) for e in random.sample(all_e, m)]

def _local_search(values, edges, n, m, max_time=55):
    """Local search to improve the solution."""
    start_time = time.time()
    best_score = _score(values, edges)
    best_values = list(values)
    best_edges = [list(e) for e in edges]
    
    all_possible_edges = _all_edges(n)
    current_edges = set(tuple(e) for e in edges)
    
    # Precompute current edge set for quick lookup
    edge_set = set(tuple(e) for e in best_edges)
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < max_time:
        if no_improve_count > max_no_improve:
            break
            
        # Randomly choose a move: modify a value or modify an edge
        move_type = random.random()
        
        if move_type < 0.4:
            # Modify a value
            idx = random.randint(0, n - 1)
            old_val = best_values[idx]
            new_val = old_val + random.choice([-3, -2, -1, 1, 2, 3, -5, 5, -10, 10, -20, 20])
            if new_val < 1 or new_val > 10000:
                new_val = random.randint(1, 10000)
            
            # Check distinctness
            if new_val in best_values and new_val != old_val:
                continue
            
            trial_values = list(best_values)
            trial_values[idx] = new_val
            trial_score = _score(trial_values, best_edges)
            
            if trial_score < best_score:
                best_values = trial_values
                best_score = trial_score
                no_improve_count = 0
            else:
                no_improve_count += 1
                
        elif move_type < 0.7:
            # Add an edge
            if len(best_edges) >= m:
                continue
            # Try adding a random edge not already present
            for _ in range(20):
                e = random.choice(all_possible_edges)
                e_tuple = tuple(e)
                if e_tuple not in edge_set:
                    trial_edges = [list(x) for x in best_edges] + [list(e)]
                    trial_score = _score(best_values, trial_edges)
                    if trial_score < best_score:
                        best_edges = trial_edges
                        edge_set.add(e_tuple)
                        best_score = trial_score
                        no_improve_count = 0
                        break
                    else:
                        no_improve_count += 1
                        break
            else:
                no_improve_count += 1
                
        else:
            # Remove an edge and add another
            if len(best_edges) < 2:
                continue
            remove_idx = random.randint(0, len(best_edges) - 1)
            removed_edge = tuple(best_edges[remove_idx])
            
            trial_edges = [list(e) for i, e in enumerate(best_edges) if i != remove_idx]
            edge_set.discard(removed_edge)
            
            # Try adding a different edge
            for _ in range(20):
                e = random.choice(all_possible_edges)
                e_tuple = tuple(e)
                if e_tuple not in edge_set and e_tuple != removed_edge:
                    trial_edges_with_new = trial_edges + [list(e)]
                    trial_score = _score(best_values, trial_edges_with_new)
                    if trial_score < best_score:
                        best_edges = trial_edges_with_new
                        edge_set.add(e_tuple)
                        best_score = trial_score
                        no_improve_count = 0
                        break
                    else:
                        # Restore removed edge
                        trial_edges.append(list(removed_edge))
                        edge_set.add(removed_edge)
                        no_improve_count += 1
                        break
            else:
                # Restore removed edge
                trial_edges.append(list(removed_edge))
                edge_set.add(removed_edge)
                no_improve_count += 1
                
    return best_values, best_edges, best_score

def _structured_initial_solution():
    """Create a structured initial solution using arithmetic progressions."""
    n = 14
    m = 35
    
    # Try different base structures
    best_solution = None
    best_score = float('inf')
    
    for base in [2, 3, 4, 5, 6, 7, 8, 10, 12]:
        for offset_pattern in range(0, 5):
            values = []
            for i in range(n):
                # Create values with shared structure
                if offset_pattern == 0:
                    v = base * (i + 1)
                elif offset_pattern == 1:
                    v = base * (i + 1) + (i % 2)
                elif offset_pattern == 2:
                    v = base * (i + 1) + ((i // 2) % 3)
                elif offset_pattern == 3:
                    v = base * (i + 1) + (i % 3)
                else:
                    v = base * (i + 1) + (i % 4)
                
                if 1 <= v <= 10000:
                    values.append(v)
            
            # Check distinctness
            if len(set(values)) != n:
                continue
            
            # Select edges: prefer edges that create product overlaps
            all_e = _all_edges(n)
            
            # Score edges by how "good" they are (create more overlaps)
            edge_scores = []
            for e in all_e:
                i, j = e
                prod = values[i] * values[j]
                s = values[i] + values[j]
                # Prefer edges with smaller products (more likely to overlap)
                edge_scores.append((prod, s, e))
            
            edge_scores.sort()
            edges = [list(e) for _, _, e in edge_scores[:m]]
            
            score = _score(values, edges)
            if score < best_score:
                best_score = score
                best_solution = (values, edges, score)
    
    return best_solution

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 second deadline for safety
    
    n = 14
    m = 35
    
    # Start with structured initial solution
    structured = _structured_initial_solution()
    if structured:
        best_values, best_edges, best_score = structured
    else:
        best_values = _random_initial_values(n)
        best_edges = _random_edges(n, m)
        best_score = _score(best_values, best_edges)
    
    # Run local search with multiple restarts
    restarts = 0
    max_restarts = 5
    
    while time.time() < deadline and restarts < max_restarts:
        # Run local search
        vals, edges, score = _local_search(best_values, best_edges, n, m, max_time=10)
        
        if score < best_score:
            best_values = vals
            best_edges = edges
            best_score = score
        
        # If not improved, try a new random start
        if time.time() < deadline:
            new_values = _random_initial_values(n)
            new_edges = _random_edges(n, m)
            new_score = _score(new_values, new_edges)
            
            if new_score < best_score:
                best_values = new_values
                best_edges = new_edges
                best_score = new_score
            
            restarts += 1
    
    # Final check and cleanup
    # Ensure all values are Python ints
    best_values = [int(v) for v in best_values]
    # Ensure edges are lists of Python ints
    best_edges = [[int(i), int(j)] for i, j in best_edges]
    
    # Verify constraints
    assert len(best_values) == n, f"Expected {n} values, got {len(best_values)}"
    assert len(set(best_values)) == n, "Values must be distinct"
    assert all(1 <= v <= 10000 for v in best_values), "Values must be in [1, 10000]"
    assert len(best_edges) == m, f"Expected {m} edges, got {len(best_edges)}"
    
    edge_set = set()
    for i, j in best_edges:
        assert i != j, "No loops allowed"
        assert 0 <= i < n and 0 <= j < n, "Edge indices out of range"
        e = tuple(sorted((i, j)))
        assert e not in edge_set, "Duplicate edge"
        edge_set.add(e)
    
    return {'values': best_values, 'edges': best_edges}
# EVOLVE-BLOCK-END
