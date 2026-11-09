"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def _score(values, edges):
    """Compute the raw value: max(distinct sums, distinct products)."""
    sums = set()
    products = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        products.add(p)
    return max(len(sums), len(products))

def _generate_divisor_structured_values():
    """Generate 14 distinct positive integers <= 10000 with high multiplicative redundancy.
    
    Strategy: Pick a highly composite base number, then pick values that are divisors
    or multiples of it to encourage product collisions.
    """
    # Highly composite numbers: 12, 24, 36, 48, 60, 120, 240, 360, 720, 840, 1260, 1680, 2520, 5040
    # Let's use multiples of a common factor to create product structure
    
    # Approach: Pick values from a set where many pairs share common factors
    # Use divisors of a highly composite number
    
    # 2520 = 2^3 * 3^2 * 5 * 7
    # Its divisors: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18, 20, 21, 24, 28, 30, 35, 36, 40, 42, 45, 56, 60, 63, 70, 72, 84, 90, 105, 120, 126, 140, 168, 180, 210, 252, 280, 315, 360, 420, 504, 630, 840, 1260, 2520
    
    # Pick 14 divisors of 2520, scaled to stay <= 10000
    divisors_2520 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18, 20, 21, 24, 28, 30, 35, 36, 40, 42, 45, 56, 60, 63, 70, 72, 84, 90, 105, 120, 126, 140, 168, 180, 210, 252, 280, 315, 360, 420, 504, 630, 840, 1260, 2520]
    
    # We want 14 distinct values. Let's pick a mix of small divisors to keep values small
    # but also include some larger ones for variety
    # Actually, let's try a different approach: use values that are all multiples of a common base
    # or use a geometric-ish progression with common factors
    
    # Better: Use values that are divisors of 2520, but pick 14 of them
    # To maximize product collisions, we want values where a*b = c*d for many pairs
    # This happens when values share prime factors
    
    # Let's pick: 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30
    # These are all divisors of 2520 and have lots of shared factors
    
    candidate_sets = [
        [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30],
        [2, 3, 4, 6, 8, 9, 12, 15, 18, 20, 24, 30, 36, 40],
        [3, 4, 6, 8, 9, 12, 15, 18, 20, 24, 30, 36, 40, 45],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32],
        [2, 3, 4, 6, 9, 10, 12, 15, 18, 20, 25, 27, 30, 36],
    ]
    
    # Shuffle and pick one, or try to find the best
    return random.choice(candidate_sets)

def _generate_edges_for_values(values):
    """Generate 35 edges that maximize sum/product collisions for the given values."""
    n = len(values)
    all_pairs = list(combinations(range(n), 2))
    
    # Score each edge by how "redundant" it is
    # An edge (i,j) is good if values[i]+values[j] or values[i]*values[j] 
    # matches many other pairs' sums/products
    
    # Precompute all pair sums and products
    pair_sums = {}
    pair_prods = {}
    for i, j in all_pairs:
        s = values[i] + values[j]
        p = values[i] * values[j]
        if s not in pair_sums:
            pair_sums[s] = []
        pair_sums[s].append((i, j))
        if p not in pair_prods:
            pair_prods[p] = []
        pair_prods[p].append((i, j))
    
    # Score each pair: higher score = more collisions
    def edge_score(i, j):
        s = values[i] + values[j]
        p = values[i] * values[j]
        sum_collisions = len(pair_sums.get(s, []))
        prod_collisions = len(pair_prods.get(p, []))
        # We want to pick edges that contribute to fewer distinct sums/products
        # So we prefer edges where the sum/product is shared by many pairs
        # But we also need to pick exactly 35 edges
        # Let's score by: how many other pairs share this sum + how many share this product
        return sum_collisions + prod_collisions
    
    # Sort all pairs by score (descending) and pick top 35
    scored_pairs = [(edge_score(i, j), i, j) for i, j in all_pairs]
    scored_pairs.sort(key=lambda x: -x[0])
    
    edges = []
    used = set()
    for score, i, j in scored_pairs:
        if len(edges) >= 35:
            break
        key = (min(i, j), max(i, j))
        if key not in used:
            edges.append([i, j])
            used.add(key)
    
    # If we don't have 35 (shouldn't happen since there are 91 pairs), add random ones
    while len(edges) < 35:
        i, j = random.choice(all_pairs)
        key = (min(i, j), max(i, j))
        if key not in used:
            edges.append([i, j])
            used.add(key)
    
    return edges

def _local_search(values, edges, max_iters=5000, time_limit=45):
    """Perform local search by mutating values and edges."""
    start_time = time.time()
    best_score = _score(values, edges)
    best_values = values[:]
    best_edges = [e[:] for e in edges]
    
    n = len(values)
    all_pairs = list(combinations(range(n), 2))
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        current_score = _score(values, edges)
        if current_score < best_score:
            best_score = current_score
            best_values = values[:]
            best_edges = [e[:] for e in edges]
        
        # Randomly mutate: either change a value or change an edge
        if random.random() < 0.5:
            # Mutate a value
            idx = random.randint(0, n - 1)
            old_val = values[idx]
            # Try a new value
            new_val = random.randint(1, 10000)
            # Ensure distinctness
            attempts = 0
            while new_val in values and attempts < 100:
                new_val = random.randint(1, 10000)
                attempts += 1
            if new_val not in values:
                values[idx] = new_val
                new_score = _score(values, edges)
                if new_score > current_score:
                    # Revert
                    values[idx] = old_val
                # Accept with some probability (simulated annealing style)
                elif random.random() < 0.1:
                    pass  # Keep the worse solution occasionally
                else:
                    values[idx] = old_val
        else:
            # Mutate an edge: swap one edge for another
            edge_idx = random.randint(0, len(edges) - 1)
            old_edge = edges[edge_idx][:]
            # Pick a new edge
            new_edge = list(random.choice(all_pairs))
            if new_edge != old_edge and new_edge != [old_edge[1], old_edge[0]]:
                # Check no duplicate
                dup = False
                for e in edges:
                    if (e[0] == new_edge[0] and e[1] == new_edge[1]) or (e[0] == new_edge[1] and e[1] == new_edge[0]):
                        dup = True
                        break
                if not dup:
                    edges[edge_idx] = new_edge
                    new_score = _score(values, edges)
                    if new_score > current_score:
                        # Revert
                        edges[edge_idx] = old_edge
                    elif random.random() < 0.1:
                        pass
                    else:
                        edges[edge_idx] = old_edge
    
    return best_values, best_edges, best_score

def construct():
    """Construct the solution."""
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    # Try multiple random restarts with divisor-structured values
    random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Generate values
        if restart < 5:
            # Use divisor-structured sets
            values = _generate_divisor_structured_values()
        else:
            # Use random values but biased toward small numbers
            values = random.sample(range(1, 200), 14)
            # Ensure distinct (sample already does this)
        
        # Generate edges
        edges = _generate_edges_for_values(values)
        
        # Local search
        v, e, s = _local_search(values, edges, max_iters=2000, time_limit=2)
        
        if s < best_score:
            best_score = s
            best_values = v
            best_edges = e
        
        if time.time() > deadline:
            break
    
    # Final check
    if best_values is None:
        # Fallback
        values = list(range(1, 15))
        edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
        best_values = values
        best_edges = edges
    
    # Ensure types are correct
    result_values = [int(v) for v in best_values]
    result_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': result_values, 'edges': result_edges}
# EVOLVE-BLOCK-END
