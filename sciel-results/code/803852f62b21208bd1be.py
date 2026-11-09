"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

N = 14
M = 35
MAXV = 10000
ALL_EDGES = list(combinations(range(N), 2))

def _score(values, edges):
    sums = set()
    prods = set()
    for i, j in edges:
        a = values[i]
        b = values[j]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _edge_score(values, edges):
    # returns (distinct_sums, distinct_prods)
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return len(sums), len(prods)

def _precompute_edge_data(values):
    """Precompute sums and products for all edges."""
    edge_data = {}
    for i, j in ALL_EDGES:
        key = (i, j) if i < j else (j, i)
        s = values[i] + values[j]
        p = values[i] * values[j]
        edge_data[key] = (s, p)
    return edge_data

def _greedy_edges(values):
    """Pick M edges greedily to minimize distinct sums/prods."""
    edge_data = _precompute_edge_data(values)
    
    # Count how many edges share each sum/product
    sum_counts = {}
    prod_counts = {}
    for key, (s, p) in edge_data.items():
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1
    
    # Sort edges by popularity (high count = more collisions)
    sorted_edges = sorted(edge_data.keys(), 
                          key=lambda k: -(sum_counts[edge_data[k][0]] + prod_counts[edge_data[k][1]]))
    
    chosen = []
    chosen_sums = set()
    chosen_prods = set()
    chosen_set = set()
    
    for key in sorted_edges:
        if len(chosen) >= M:
            break
        if key in chosen_set:
            continue
        s, p = edge_data[key]
        chosen.append(list(key))
        chosen_sums.add(s)
        chosen_prods.add(p)
        chosen_set.add(key)
    
    # If we didn't get enough, fill randomly
    if len(chosen) < M:
        for key in edge_data.keys():
            if len(chosen) >= M:
                break
            if key not in chosen_set:
                chosen.append(list(key))
                chosen_set.add(key)
    
    return chosen

def _local_search_edges_fast(values, edges, max_iters=200, deadline=None):
    """Fast local search with incremental evaluation."""
    edge_data = _precompute_edge_data(values)
    
    # Build current state
    edge_set = set(tuple(sorted(e)) for e in edges)
    sums = set()
    prods = set()
    for e in edge_set:
        s, p = edge_data[e]
        sums.add(s)
        prods.add(p)
    
    current = max(len(sums), len(prods))
    edge_list = list(edge_set)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        random.shuffle(edge_list)
        for e_out in edge_list:
            if e_out not in edge_set:
                continue
            s_out, p_out = edge_data[e_out]
            
            # Try all possible edges to add
            for e_in in ALL_EDGES:
                if e_in in edge_set or e_in == e_out:
                    continue
                s_in, p_in = edge_data[e_in]
                
                # Compute new sums/prods counts incrementally
                new_sums_count = len(sums)
                new_prods_count = len(prods)
                
                # Remove e_out
                if sums[s_out] == 1 if isinstance(sums, dict) else (s_out not in sums):
                    pass
                # Simpler: just compute the delta
                # New sum set would be: sums - {s_out if count==1} + {s_in if not in sums}
                # New prod set would be: prods - {p_out if count==1} + {p_in if not in prods}
                
                # Count occurrences
                s_out_count = sum(1 for e in edge_set if e != e_out and edge_data[e][0] == s_out)
                p_out_count = sum(1 for e in edge_set if e != e_out and edge_data[e][1] == p_out)
                
                s_in_new = s_in not in sums or (s_in == s_out and s_out_count == 0)
                p_in_new = p_in not in prods or (p_in == p_out and p_out_count == 0)
                
                # After removing e_out:
                # s_out stays if s_out_count > 0, else removed
                # p_out stays if p_out_count > 0, else removed
                # After adding e_in:
                # s_in added if not present (considering removal)
                # p_in added if not present (considering removal)
                
                effective_sums = len(sums)
                effective_prods = len(prods)
                
                # If s_out was unique, removing it decreases sum count
                if s_out_count == 0:
                    effective_sums -= 1
                # If s_in is new (not in sums, or was s_out which is being removed)
                if s_in not in sums or (s_in == s_out and s_out_count == 0):
                    effective_sums += 1
                    
                if p_out_count == 0:
                    effective_prods -= 1
                if p_in not in prods or (p_in == p_out and p_out_count == 0):
                    effective_prods += 1
                
                new_score = max(effective_sums, effective_prods)
                
                if new_score < current:
                    # Accept the move
                    edge_set.discard(e_out)
                    edge_set.add(e_in)
                    
                    # Update sums
                    if s_out_count == 0:
                        sums.discard(s_out)
                    sums.add(s_in)
                    
                    # Update prods
                    if p_out_count == 0:
                        prods.discard(p_out)
                    prods.add(p_in)
                    
                    current = new_score
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return [[i, j] for i, j in edge_set]

def _polish_edges(values, edges, deadline):
    """Final polish: more iterations on the best solution."""
    return _local_search_edges_fast(values, edges, max_iters=500, deadline=deadline)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin (90s limit)
    
    random.seed(42)
    
    best_values = None
    best_edges = None
    best_score = 10**9
    
    # Strategy 1: Arithmetic progression values
    for d in [1, 2, 3, 4, 5, 7, 10, 15, 20]:
        if time.time() > deadline:
            break
        base = 1
        vals = [base + k * d for k in range(N)]
        if vals[-1] > MAXV:
            continue
        if len(set(vals)) != N:
            continue
        edges = _greedy_edges(vals)
        edges = _local_search_edges_fast(vals, edges, max_iters=100, deadline=deadline)
        s = _score(vals, edges)
        if s < best_score:
            best_score = s
            best_values = vals[:]
            best_edges = [e[:] for e in edges]
    
    # Strategy 2: Structured sets with many multiplicative/additive relations
    candidate_sets = [
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 24, 30, 40, 48],
        [2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
        [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
    ]
    for vals in candidate_sets:
        if time.time() > deadline:
            break
        if len(set(vals)) != N or any(v < 1 or v > MAXV for v in vals):
            continue
        edges = _greedy_edges(vals)
        edges = _local_search_edges_fast(vals, edges, max_iters=100, deadline=deadline)
        s = _score(vals, edges)
        if s < best_score:
            best_score = s
            best_values = vals[:]
            best_edges = [e[:] for e in edges]
    
    # Strategy 3: Random search with structured pairs
    if time.time() < deadline - 10:
        for attempt in range(150):
            if time.time() > deadline - 5:
                break
            # Try different structures
            strategy = attempt % 3
            
            if strategy == 0:
                # Pairs summing to same value
                S = random.randint(10, 50)
                vals = []
                for _ in range(7):
                    a = random.randint(1, S - 1)
                    b = S - a
                    if a != b and a not in vals and b not in vals and 1 <= a <= MAXV and 1 <= b <= MAXV:
                        vals.append(a)
                        vals.append(b)
                if len(vals) != N:
                    while len(vals) < N:
                        v = random.randint(1, MAXV)
                        if v not in vals:
                            vals.append(v)
                    vals = vals[:N]
            
            elif strategy == 1:
                # Small highly composite-like numbers
                base_candidates = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30, 32, 40, 48, 60, 64, 80]
                random.shuffle(base_candidates)
                vals = base_candidates[:N]
                if len(set(vals)) != N:
                    continue
            
            else:
                # Random small values
                vals = random.sample(range(1, 100), N)
            
            edges = _greedy_edges(vals)
            edges = _local_search_edges_fast(vals, edges, max_iters=50, deadline=deadline)
            s = _score(vals, edges)
            if s < best_score:
                best_score = s
                best_values = vals[:]
                best_edges = [e[:] for e in edges]
    
    # Final polish on best
    if best_values is not None and time.time() < deadline - 2:
        best_edges = _polish_edges(best_values, best_edges, deadline)
        best_score = _score(best_values, best_edges)
    
    # Ensure types are correct
    values = [int(v) for v in best_values]
    edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
