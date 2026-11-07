"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import Counter

P = {'n': 14, 'm': 35, 'maximum': 10000}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Conservative safety margin
    
    n = P['n']
    m = P['m']
    max_val = P['maximum']
    
    all_pairs = list(itertools.combinations(range(n), 2))
    all_pairs_set = set(frozenset(p) for p in all_pairs)
    
    def get_score(vals, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(vals[i] + vals[j])
            prods.add(vals[i] * vals[j])
        return max(len(sums), len(prods))
    
    def find_best_edges(vals):
        # Precompute sums and products for all pairs
        pair_data = []
        for i, j in all_pairs:
            s = vals[i] + vals[j]
            p = vals[i] * vals[j]
            pair_data.append((s, p, i, j))
        
        sum_freq = Counter(s for s, p, i, j in pair_data)
        prod_freq = Counter(p for s, p, i, j in pair_data)
        
        # Score each edge by how "popular" its sum and product are
        scored = []
        for s, p, i, j in pair_data:
            score = sum_freq[s] * 2 + prod_freq[p]  # Weight sums slightly more
            scored.append((score, s, p, i, j))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Greedy selection with incremental tracking
        selected = []
        selected_set = set()
        seen_sums = set()
        seen_prods = set()
        
        for score, s, p, i, j in scored:
            if len(selected) >= m:
                break
            key = frozenset((i, j))
            if key in selected_set:
                continue
            
            # Count how many new distinct sums/prods this adds
            new_sums = 0 if s in seen_sums else 1
            new_prods = 0 if p in seen_prods else 1
            
            # Accept if it doesn't increase the max too much
            # Or simply accept top-m by frequency
            selected.append([i, j])
            selected_set.add(key)
            seen_sums.add(s)
            seen_prods.add(p)
        
        return selected
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Candidate pool: small numbers with high collision potential
    # Include numbers with many divisors and small values
    candidates = list(range(1, 100))
    random.shuffle(candidates)
    
    # Multiple restarts
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Initialize with a random set of 14 small values
        current_values = random.sample(range(1, 60), n)
        current_values.sort()
        
        # Find optimal edges for these values
        current_edges = find_best_edges(current_values)
        current_score = get_score(current_values, current_edges)
        
        if current_score < best_score:
            best_score = current_score
            best_values = current_values[:]
            best_edges = [e[:] for e in current_edges]
        
        # Local search on values
        max_val_iters = 300
        for _ in range(max_val_iters):
            if time.time() > deadline:
                break
            
            # Try replacing one value
            idx = random.randint(0, n - 1)
            old_val = current_values[idx]
            
            # Try a few replacements
            found_improvement = False
            for _ in range(15):
                new_val = random.choice(candidates)
                if new_val == old_val or new_val in current_values:
                    continue
                if new_val < 1 or new_val > max_val:
                    continue
                
                temp_vals = current_values[:]
                temp_vals[idx] = new_val
                temp_vals.sort()
                
                # Re-find best edges
                temp_edges = find_best_edges(temp_vals)
                temp_score = get_score(temp_vals, temp_edges)
                
                if temp_score < current_score:
                    current_values = temp_vals
                    current_edges = temp_edges
                    current_score = temp_score
                    found_improvement = True
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_values = current_values[:]
                        best_edges = [e[:] for e in current_edges]
                    break
            
            if not found_improvement:
                # Try edge local search
                edge_set = set(frozenset(e) for e in current_edges)
                non_edges = [p for p in all_pairs if frozenset(p) not in edge_set]
                
                if non_edges:
                    for _ in range(8):
                        e_idx = random.randint(0, m - 1)
                        new_edge = random.choice(non_edges)
                        
                        temp_edges = current_edges[:]
                        temp_edges[e_idx] = list(new_edge)
                        
                        temp_score = get_score(current_values, temp_edges)
                        
                        if temp_score < current_score:
                            current_edges = temp_edges
                            current_score = temp_score
                            
                            if current_score < best_score:
                                best_score = current_score
                                best_values = current_values[:]
                                best_edges = [e[:] for e in current_edges]
                            break
    
    # Final type safety
    if best_values is None:
        best_values = list(range(1, n + 1))
        best_edges = find_best_edges(best_values)
    
    best_values = [int(v) for v in best_values]
    best_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': best_values, 'edges': best_edges}
# EVOLVE-BLOCK-END
