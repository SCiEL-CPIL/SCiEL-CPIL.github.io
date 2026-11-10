import time
import random
from itertools import combinations

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
    deadline = start_time + 50.0  # 10s safety margin
    
    n = P['n']
    m = P['m']
    maximum = P['maximum']
    
    def greedy_select_edges(values):
        """
        Given 14 values, greedily select 35 edges to minimize 
        max(#distinct sums, #distinct products).
        """
        # Precompute all 91 pairs
        all_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                s = values[i] + values[j]
                p = values[i] * values[j]
                all_pairs.append((i, j, s, p))
        
        # We need to pick m edges from all_pairs
        used_sums = set()
        used_prods = set()
        selected_edges = []
        remaining = list(all_pairs)
        
        # To speed up, we can sort remaining initially by some heuristic,
        # but the greedy step re-evaluates everything anyway.
        # However, we can maintain a list of "active" candidates.
        
        for _ in range(m):
            best_idx = -1
            best_score = float('inf')
            # Tie-breaker: prefer edges that add 0 new sums/prods, then 1, etc.
            # Actually, the score is max(len(S), len(P)).
            # Adding an edge (s, p):
            # new_S = len(used_sums) + (1 if s not in used_sums else 0)
            # new_P = len(used_prods) + (1 if p not in used_prods else 0)
            # score = max(new_S, new_P)
            
            # We can optimize the inner loop by pre-computing 
            # whether s/p are already used, but since we update sets, 
            # we must check.
            
            for k in range(len(remaining)):
                i, j, s, p = remaining[k]
                new_sums = len(used_sums) + (0 if s in used_sums else 1)
                new_prods = len(used_prods) + (0 if p in used_prods else 1)
                score = max(new_sums, new_prods)
                
                # We want to minimize score.
                # If scores are equal, we might prefer edges that have more "potential"
                # i.e., s and p are common in the remaining set?
                # For now, simple min score.
                if score < best_score:
                    best_score = score
                    best_idx = k
            
            if best_idx >= 0:
                i, j, s, p = remaining[best_idx]
                selected_edges.append((i, j))
                used_sums.add(s)
                used_prods.add(p)
                remaining.pop(best_idx)
            else:
                # Should not happen if m <= 91
                break
        
        return selected_edges, max(len(used_sums), len(used_prods))

    def try_values(values):
        # Ensure valid values
        if len(set(values)) != n:
            return None, float('inf')
        if any(v < 1 or v > maximum for v in values):
            return None, float('inf')
        
        edges, score = greedy_select_edges(values)
        if len(edges) != m:
            return None, float('inf')
            
        # Convert to list of lists for output format consistency
        edges_list = [[i, j] for i, j in edges]
        return edges_list, score

    best_score = float('inf')
    best_values = None
    best_edges = None
    
    random.seed(42)
    
    # Initial candidates
    candidates = []
    # 1. Small consecutive
    candidates.append(list(range(1, 15)))
    # 2. Small random
    for _ in range(20):
        candidates.append(random.sample(range(1, 50), n))
    # 3. Structured small
    for _ in range(10):
        base = random.randint(1, 5)
        vals = []
        for i in range(n):
            v = base * (i + 1) + random.randint(0, 5)
            vals.append(v)
        if len(set(vals)) == n:
            candidates.append(vals)
            
    # Evaluate initial candidates
    for vals in candidates:
        edges, score = try_values(vals)
        if score < best_score:
            best_score = score
            best_values = vals[:]
            best_edges = edges
    
    # Local Search / Hill Climbing
    current_values = best_values[:]
    current_score = best_score
    
    while time.time() < deadline:
        # Generate a neighbor
        new_values = current_values[:]
        idx_to_change = random.randint(0, n - 1)
        
        # Try to replace with a small number to encourage collisions
        # Or a number that creates a relation
        # Strategy: random small integer
        new_val = random.randint(1, 100)
        
        # Ensure distinctness
        if new_val in new_values:
            continue
        new_values[idx_to_change] = new_val
        
        # Check bounds
        if new_val > maximum:
            continue
            
        edges, score = try_values(new_values)
        
        if score < best_score:
            best_score = score
            best_values = new_values[:]
            best_edges = edges
            current_values = new_values[:]
            current_score = score
        elif score < current_score and random.random() < 0.1:
            # Simulated annealing step? Or just accept better local moves
            # Let's stick to strict hill climbing for stability, 
            # but occasionally jump to a random new basin if stuck.
            pass
            
        # If stuck for a while, try a bigger jump
        # Simple: every 100 iterations, if no improvement, randomize one value more aggressively
        # For now, the loop runs fast.
        
    # Final validation and type conversion
    if best_values is None:
        # Fallback
        best_values = list(range(1, 15))
        best_edges, _ = greedy_select_edges(best_values)
        best_edges = [[i, j] for i, j in best_edges]
        
    # Ensure types are Python int and list
    final_values = [int(v) for v in best_values]
    final_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': final_values, 'edges': final_edges}
