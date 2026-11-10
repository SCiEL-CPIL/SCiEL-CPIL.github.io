"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import Counter

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _raw_value(values, edges):
    """Compute the objective: max(#distinct sums, #distinct products)."""
    sums = set()
    prods = set()
    for i, j in edges:
        v = values[i]
        w = values[j]
        sums.add(v + w)
        prods.add(v * w)
    return max(len(sums), len(prods))

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill
    n = P['n']
    m = P['m']
    maximum = P['maximum']
    
    best_raw = float('inf')
    best_values = None
    best_edges = None
    
    random.seed(42)
    
    def try_value_set(values):
        """Given 14 values, find 35 edges minimizing the objective using improved greedy + local search."""
        # Precompute all pair (i,j) with i<j, and their sum and product
        all_pairs = []
        for i in range(n):
            for j in range(i+1, n):
                s = values[i] + values[j]
                p = values[i] * values[j]
                all_pairs.append((i, j, s, p))
        
        # Count sum and product frequencies
        sum_counts = Counter()
        prod_counts = Counter()
        for _, _, s, p in all_pairs:
            sum_counts[s] += 1
            prod_counts[p] += 1
        
        # Greedy: pick edge that minimizes max(|used_sums|, |used_prods|)
        # Use a scoring heuristic to avoid full O(remaining) scan each time
        used_sums = set()
        used_prods = set()
        selected = []
        remaining_indices = list(range(len(all_pairs)))
        
        # Sort remaining by popularity (higher count = more likely to collide)
        remaining_indices.sort(key=lambda k: (sum_counts[all_pairs[k][2]], prod_counts[all_pairs[k][3]]), reverse=True)
        
        for idx in range(m):
            best_idx = -1
            best_score = float('inf')
            # Check top candidates first for speed
            check_limit = min(len(remaining_indices), 200)
            for pos in range(check_limit):
                k = remaining_indices[pos]
                i, j, s, p = all_pairs[k]
                new_sums = len(used_sums) + (0 if s in used_sums else 1)
                new_prods = len(used_prods) + (0 if p in used_prods else 1)
                score = max(new_sums, new_prods)
                if score < best_score:
                    best_score = score
                    best_idx = pos
                    if score == max(len(used_sums), len(used_prods)):
                        break  # Can't do better than current max
            if best_idx >= 0:
                k = remaining_indices.pop(best_idx)
                i, j, s, p = all_pairs[k]
                selected.append((i, j))
                used_sums.add(s)
                used_prods.add(p)
        
        raw = max(len(used_sums), len(used_prods))
        
        # Local search: try swapping one selected edge with one unselected edge
        selected_set = set(selected)
        unselected = [p for p in all_pairs if (p[0], p[1]) not in selected_set]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for si in range(len(selected)):
                if time.time() > deadline:
                    break
                si_edge = selected[si]
                si_s = values[si_edge[0]] + values[si_edge[1]]
                si_p = values[si_edge[0]] * values[si_edge[1]]
                
                # Try swapping with unselected edges
                for ui in range(min(len(unselected), 50)):
                    ui_edge = unselected[ui]
                    ui_s = values[ui_edge[0]] + values[ui_edge[1]]
                    ui_p = values[ui_edge[0]] * values[ui_edge[1]]
                    
                    # Compute new sets if we swap
                    new_sums = set(used_sums)
                    new_prods = set(used_prods)
                    # Remove old
                    if len([x for x in selected if x != si_edge and values[x[0]] + values[x[1]] == si_s]) == 0:
                        new_sums.discard(si_s)
                    if len([x for x in selected if x != si_edge and values[x[0]] * values[x[1]] == si_p]) == 0:
                        new_prods.discard(si_p)
                    # Add new
                    new_sums.add(ui_s)
                    new_prods.add(ui_p)
                    
                    new_raw = max(len(new_sums), len(new_prods))
                    if new_raw < raw:
                        # Apply swap
                        selected[si] = ui_edge
                        used_sums = new_sums
                        used_prods = new_prods
                        raw = new_raw
                        improved = True
                        break
                if improved:
                    break
        
        return raw, selected, used_sums, used_prods
    
    # Generate diverse value sets and try them
    num_attempts = 300
    
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
        
        # Generate a value set with emphasis on collision structures
        if attempt % 6 == 0:
            # Pairs that sum to same constant: a, b, c, d, ... where a+d=b+c=const
            # Use symmetric pairs around a center
            center = random.randint(20, 500)
            offsets = random.sample(range(1, center), 7)
            values = []
            for off in offsets:
                values.append(center - off)
                values.append(center + off)
            values = list(set(values))
            while len(values) < n:
                values.append(random.randint(1, maximum))
            values = list(set(values))[:n]
        elif attempt % 6 == 1:
            # Arithmetic progression
            a = random.randint(1, 50)
            d = random.randint(1, 30)
            values = [a + i*d for i in range(n)]
            if max(values) > maximum or min(values) < 1:
                values = list(range(1, n+1))
        elif attempt % 6 == 2:
            # Small clustered numbers
            base = random.randint(1, 20)
            values = [base + random.randint(0, 30) for _ in range(n)]
            values = list(set(values))
            while len(values) < n:
                values.append(random.randint(1, maximum))
            values = list(set(values))[:n]
        elif attempt % 6 == 3:
            # Random distinct small
            values = random.sample(range(1, 200), n)
        elif attempt % 6 == 4:
            # Mix small and medium
            small = random.sample(range(1, 50), n//2)
            medium = random.sample(range(50, 500), n - n//2)
            values = small + medium
            if len(set(values)) < n:
                values = random.sample(range(1, maximum+1), n)
        else:
            # Random
            values = random.sample(range(1, maximum+1), n)
        
        # Ensure valid
        values = list(set(values))
        if len(values) < n:
            continue
        values = values[:n]
        if any(v < 1 or v > maximum for v in values):
            continue
        
        raw, edges, _, _ = try_value_set(values)
        
        if raw < best_raw:
            best_raw = raw
            best_values = values
            best_edges = edges
    
    # Fallback: always valid solution
    if best_values is None:
        best_values = list(range(1, n+1))
        best_edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    
    # Ensure types are correct
    best_values = [int(v) for v in best_values]
    best_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': best_values, 'edges': best_edges}

# EVOLVE-BLOCK-END
