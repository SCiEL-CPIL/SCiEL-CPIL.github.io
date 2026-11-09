"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations
from collections import defaultdict, Counter

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

def _generate_divisor_values(N, k=14):
    """Generate k distinct divisors of N as values."""
    divisors = []
    for i in range(1, int(N**0.5) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    divisors.sort()
    if len(divisors) >= k:
        # Pick k divisors that maximize pair-product collisions
        # Prefer divisors that form pairs multiplying to N
        best_combo = None
        best_collision = 0
        # Try different subsets
        for _ in range(50):
            subset = random.sample(divisors, k)
            subset.sort()
            # Count product collisions
            prods = Counter()
            for i in range(k):
                for j in range(i+1, k):
                    prods[subset[i] * subset[j]] += 1
            collisions = sum(c*(c-1)//2 for c in prods.values())
            if collisions > best_collision:
                best_collision = collisions
                best_combo = subset
        return best_combo
    return divisors[:k]

def _greedy_edge_selection(values, m=35, deadline=None):
    """Select m edges greedily to maximize sum/product collisions."""
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    # Precompute sum and product for each edge
    edge_data = {}
    for i, j in all_edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        edge_data[(i, j)] = (s, p)
    
    selected_edges = []
    sum_counts = Counter()
    prod_counts = Counter()
    used_vertices = set()
    
    # First pass: select edges that create the most collisions
    # Sort edges by potential to create collisions
    # An edge (i,j) with sum s and product p has marginal value:
    # - If s already in sum_counts, it adds a collision
    # - If p already in prod_counts, it adds a collision
    
    remaining = list(all_edges)
    
    for _ in range(m):
        if deadline is not None and time.time() > deadline:
            break
        
        best_edge = None
        best_gain = -1
        
        for edge in remaining:
            i, j = edge
            s, p = edge_data[edge]
            
            # Gain from adding this edge
            gain = 0
            if s in sum_counts:
                gain += sum_counts[s]  # Creates more sum collisions
            if p in prod_counts:
                gain += prod_counts[p]  # Creates more product collisions
            
            # Also consider future potential: edges sharing vertices
            # with this edge might benefit
            vertex_bonus = 0
            for other in remaining:
                if other == edge:
                    continue
                os, op = edge_data[other]
                # If other edge shares a vertex, its sum/product might collide
                if (other[0] in (i, j) or other[1] in (i, j)):
                    if os == s:
                        vertex_bonus += 1
                    if op == p:
                        vertex_bonus += 1
            
            total_gain = gain + vertex_bonus * 0.1  # Slight preference for vertex sharing
            
            if total_gain > best_gain:
                best_gain = total_gain
                best_edge = edge
        
        if best_edge is None:
            break
        
        selected_edges.append(best_edge)
        s, p = edge_data[best_edge]
        sum_counts[s] += 1
        prod_counts[p] += 1
        remaining.remove(best_edge)
    
    return [list(e) for e in selected_edges]

def _local_search_edges(values, edges, deadline=None, max_iters=2000):
    """Local search to improve edge selection."""
    n = len(values)
    m = len(edges)
    all_edges = list(combinations(range(n), 2))
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    current_score = _score(values, edges)
    best_edges = [list(e) for e in edges]
    best_score = current_score
    
    # Precompute all edge data
    edge_data = {}
    for i, j in all_edges:
        edge_data[(i, j)] = (values[i] + values[j], values[i] * values[j])
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try swapping an edge
        if not edges:
            break
        
        idx = random.randint(0, len(edges) - 1)
        old_edge = tuple(sorted(edges[idx]))
        
        # Find a candidate replacement
        candidates = [e for e in all_edges if e not in edge_set]
        if not candidates:
            break
        
        new_edge = random.choice(candidates)
        
        # Tentatively swap
        edges[idx] = list(new_edge)
        edge_set.discard(old_edge)
        edge_set.add(new_edge)
        
        new_score = _score(values, edges)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_edges = [list(e) for e in edges]
        else:
            # Revert
            edges[idx] = list(old_edge)
            edge_set.discard(new_edge)
            edge_set.add(old_edge)
    
    return best_edges, best_score

def construct():
    n = 14
    m = 35
    deadline = time.time() + 50  # Safety margin
    
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    # Strategy 1: Divisor-based value sets
    # Try several highly composite numbers
    highly_composite = [5040, 7560, 10080, 15120, 20160, 25200, 27720, 45360, 50400, 55440]
    
    for N in highly_composite:
        if time.time() > deadline:
            break
        
        values = _generate_divisor_values(N, 14)
        if values is None or len(set(values)) != 14 or max(values) > 10000:
            continue
        
        # Greedy edge selection
        edges = _greedy_edge_selection(values, m, deadline)
        
        if len(edges) < m:
            # Fill remaining with random edges
            all_edges = list(combinations(range(n), 2))
            edge_set = set(tuple(sorted(e)) for e in edges)
            for e in all_edges:
                if len(edges) >= m:
                    break
                if e not in edge_set:
                    edges.append(list(e))
                    edge_set.add(e)
        
        score = _score(values, edges)
        
        # Local search to improve
        if time.time() < deadline - 5:
            improved_edges, improved_score = _local_search_edges(values, edges, deadline)
            if improved_score < score:
                score = improved_score
                edges = improved_edges
        
        if score < best_score:
            best_score = score
            best_values = values[:]
            best_edges = edges[:]
        
        # Early exit if very good
        if best_score <= 12:
            break
    
    # Strategy 2: Small arithmetic progression
    if time.time() < deadline - 10:
        for start in range(1, 20):
            for step in range(1, 5):
                if time.time() > deadline:
                    break
                values = [start + i * step for i in range(14)]
                if max(values) > 10000:
                    continue
                if len(set(values)) != 14:
                    continue
                
                edges = _greedy_edge_selection(values, m, deadline)
                if len(edges) < m:
                    all_edges = list(combinations(range(n), 2))
                    edge_set = set(tuple(sorted(e)) for e in edges)
                    for e in all_edges:
                        if len(edges) >= m:
                            break
                        if e not in edge_set:
                            edges.append(list(e))
                            edge_set.add(e)
                
                score = _score(values, edges)
                
                if time.time() < deadline - 3:
                    improved_edges, improved_score = _local_search_edges(values, edges, deadline)
                    if improved_score < score:
                        score = improved_score
                        edges = improved_edges
                
                if score < best_score:
                    best_score = score
                    best_values = values[:]
                    best_edges = edges[:]
                
                if best_score <= 10:
                    break
            if best_score <= 10:
                break
    
    # Final fallback
    if best_values is None:
        best_values = list(range(1, 15))
        best_edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
    
    return {'values': best_values, 'edges': best_edges}
# EVOLVE-BLOCK-END
