"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

N = 14
M = 35
MAX_VAL = 10000
ALL_EDGES = list(combinations(range(N), 2))

def _score(values, edges):
    """Compute the raw value: max(distinct sums, distinct products)."""
    sums = set()
    products = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        products.add(values[i] * values[j])
    return max(len(sums), len(products))

def _greedy_edge_selection(values, k=M):
    """Select k edges greedily to minimize growth of distinct sums/products."""
    sums = set()
    products = set()
    selected = []
    remaining = list(ALL_EDGES)
    random.shuffle(remaining)
    
    for _ in range(k):
        best_edge = None
        best_score = float('inf')
        for e in remaining:
            i, j = e
            s = values[i] + values[j]
            p = values[i] * values[j]
            new_sums = len(sums | {s})
            new_products = len(products | {p})
            score = max(new_sums, new_products)
            if score < best_score:
                best_score = score
                best_edge = e
        if best_edge is None:
            break
        i, j = best_edge
        sums.add(values[i] + values[j])
        products.add(values[i] * values[j])
        selected.append(list(best_edge))
        remaining.remove(best_edge)
    
    return selected

def _local_search(values, edges, max_iters=500, deadline=None):
    """Perform local search by mutating values and edges."""
    best_score = _score(values, edges)
    current_values = list(values)
    current_edges = [list(e) for e in edges]
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try to improve by changing one value
        improved = False
        for idx in range(N):
            old_val = current_values[idx]
            # Try a few random new values
            for _ in range(3):
                new_val = random.randint(1, MAX_VAL)
                if new_val == old_val:
                    continue
                if any(new_val == v for j, v in enumerate(current_values) if j != idx):
                    continue
                current_values[idx] = new_val
                new_score = _score(current_values, current_edges)
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                    break
                current_values[idx] = old_val
        
        # Try to improve by swapping an edge
        if not improved:
            for _ in range(5):
                if deadline is not None and time.time() > deadline:
                    break
                # Pick a random edge to remove and a random non-edge to add
                rem_idx = random.randint(0, len(current_edges) - 1)
                rem_edge = tuple(current_edges[rem_idx])
                
                # Find a non-edge to add
                edge_set = set(tuple(e) for e in current_edges)
                candidates = [e for e in ALL_EDGES if e not in edge_set]
                if not candidates:
                    continue
                add_edge = random.choice(candidates)
                
                current_edges[rem_idx] = list(add_edge)
                new_score = _score(current_values, current_edges)
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                else:
                    current_edges[rem_idx] = list(rem_edge)
        
        if not improved and it > 100:
            # Try a larger perturbation
            for _ in range(2):
                if deadline is not None and time.time() > deadline:
                    break
                # Replace 2 random values
                idx1, idx2 = random.sample(range(N), 2)
                old1, old2 = current_values[idx1], current_values[idx2]
                current_values[idx1] = random.randint(1, MAX_VAL)
                current_values[idx2] = random.randint(1, MAX_VAL)
                # Ensure distinctness
                if current_values[idx1] == current_values[idx2]:
                    current_values[idx1] = old1
                    current_values[idx2] = old2
                    continue
                if any(current_values[idx1] == v for j, v in enumerate(current_values) if j not in (idx1, idx2)):
                    current_values[idx1] = old1
                    continue
                if any(current_values[idx2] == v for j, v in enumerate(current_values) if j not in (idx1, idx2)):
                    current_values[idx2] = old2
                    continue
                new_score = _score(current_values, current_edges)
                if new_score < best_score:
                    best_score = new_score
                else:
                    current_values[idx1] = old1
                    current_values[idx2] = old2
    
    return current_values, current_edges, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s deadline, 10s safety margin
    
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    # Multiple random restarts
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Generate random values
        values = random.sample(range(1, MAX_VAL + 1), N)
        
        # Greedy edge selection
        edges = _greedy_edge_selection(values)
        score = _score(values, edges)
        
        if score < best_score:
            best_score = score
            best_values = list(values)
            best_edges = [list(e) for e in edges]
        
        # Local search if time permits
        if time.time() < deadline - 5:
            values_ls, edges_ls, score_ls = _local_search(values, edges, max_iters=300, deadline=deadline)
            if score_ls < best_score:
                best_score = score_ls
                best_values = list(values_ls)
                best_edges = [list(e) for e in edges_ls]
    
    # Final fallback
    if best_values is None:
        best_values = list(range(1, N + 1))
        best_edges = [list(e) for e in ALL_EDGES[:M]]
    
    # Ensure types are correct
    best_values = [int(v) for v in best_values]
    best_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': best_values, 'edges': best_edges}
# EVOLVE-BLOCK-END
