import time
import random
import math
from itertools import combinations

N = 14
M = 35
MAX_VAL = 10000

def score_of(values, edges):
    """Return max(distinct sums, distinct products) for the given edges."""
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return max(len(sums), len(prods))

def greedy_edges(values, m=M):
    """Pick m edges minimizing max(distinct sums, distinct products) greedily."""
    n = len(values)
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            s = values[i] + values[j]
            p = values[i] * values[j]
            all_edges.append((i, j, s, p))
    
    chosen = []
    sums = set()
    prods = set()
    remaining = list(all_edges)
    
    for _ in range(m):
        best_idx = -1
        best_max = float('inf')
        best_new_sums = float('inf')
        best_new_prods = float('inf')
        
        for idx in range(len(remaining)):
            i, j, s, p = remaining[idx]
            ns = len(sums) + (0 if s in sums else 1)
            np_ = len(prods) + (0 if p in prods else 1)
            nm = max(ns, np_)
            
            if nm < best_max or (nm == best_max and (ns + np_) < best_new_sums + best_new_prods):
                best_max = nm
                best_idx = idx
                best_new_sums = ns
                best_new_prods = np_
        
        if best_idx >= 0:
            i, j, s, p = remaining.pop(best_idx)
            chosen.append((i, j))
            sums.add(s)
            prods.add(p)
    
    return chosen

def local_search(values, edges, deadline, rng, max_steps=50000):
    """
    Local search that jointly perturbs values and edges.
    Uses acceptance criterion that always accepts improvements and sometimes accepts worse.
    """
    n = len(values)
    cur_vals = list(values)
    cur_edges = list(edges)
    cur_score = score_of(cur_vals, cur_edges)
    
    best_vals = list(cur_vals)
    best_edges = list(cur_edges)
    best_score = cur_score
    
    all_edges = list(combinations(range(n), 2))
    
    T = 20.0
    step = 0
    no_improve = 0
    max_no_improve = 8000
    
    while time.time() < deadline and no_improve < max_no_improve:
        step += 1
        T = max(0.01, 20.0 * (0.9998 ** step))
        
        r = rng.random()
        
        if r < 0.35:
            # Perturb one value
            idx = rng.randrange(n)
            old = cur_vals[idx]
            step_size = rng.choice([1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100])
            direction = rng.choice([-1, 1])
            new_v = old + direction * step_size
            
            if new_v < 1 or new_v > MAX_VAL:
                continue
            if new_v in cur_vals:
                continue
            
            cur_vals[idx] = new_v
            new_score = score_of(cur_vals, cur_edges)
            delta = new_score - cur_score
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_vals = list(cur_vals)
                    best_edges = list(cur_edges)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                cur_vals[idx] = old
                no_improve += 1
                
        elif r < 0.55:
            # Swap two values
            i, j = rng.sample(range(n), 2)
            cur_vals[i], cur_vals[j] = cur_vals[j], cur_vals[i]
            
            new_score = score_of(cur_vals, cur_edges)
            delta = new_score - cur_score
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_vals = list(cur_vals)
                    best_edges = list(cur_edges)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                cur_vals[i], cur_vals[j] = cur_vals[j], cur_vals[i]
                no_improve += 1
                
        elif r < 0.80:
            # Swap one edge: remove random edge, add random edge not in set
            if len(cur_edges) < M:
                candidates = [e for e in all_edges if e not in set(tuple(x) for x in cur_edges)]
                if not candidates:
                    continue
                new_e = rng.choice(candidates)
                cur_edges.append(new_e)
                new_score = score_of(cur_vals, cur_edges)
                delta = new_score - cur_score
                
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    cur_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_vals = list(cur_vals)
                        best_edges = list(cur_edges)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    cur_edges.remove(new_e)
                    no_improve += 1
            else:
                edge_set = set(tuple(e) for e in cur_edges)
                # Remove a random edge
                rem_idx = rng.randrange(len(cur_edges))
                rem_edge = cur_edges.pop(rem_idx)
                # Add a random edge not in set
                candidates = [e for e in all_edges if e not in edge_set]
                if not candidates:
                    cur_edges.append(rem_edge)
                    continue
                new_e = rng.choice(candidates)
                cur_edges.append(new_e)
                new_score = score_of(cur_vals, cur_edges)
                delta = new_score - cur_score
                
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    cur_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_vals = list(cur_vals)
                        best_edges = list(cur_edges)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    cur_edges.remove(new_e)
                    cur_edges.append(rem_edge)
                    no_improve += 1
                    
        else:
            # Perturb two values
            i, j = rng.sample(range(n), 2)
            old_i, old_j = cur_vals[i], cur_vals[j]
            step_i = rng.choice([1, 2, 3, 5, 10]) * rng.choice([-1, 1])
            step_j = rng.choice([1, 2, 3, 5, 10]) * rng.choice([-1, 1])
            new_i = old_i + step_i
            new_j = old_j + step_j
            
            if new_i < 1 or new_i > MAX_VAL or new_j < 1 or new_j > MAX_VAL:
                continue
            if new_i in cur_vals or new_j in cur_vals:
                continue
            if new_i == new_j:
                continue
            
            cur_vals[i] = new_i
            cur_vals[j] = new_j
            new_score = score_of(cur_vals, cur_edges)
            delta = new_score - cur_score
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_vals = list(cur_vals)
                    best_edges = list(cur_edges)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                cur_vals[i] = old_i
                cur_vals[j] = old_j
                no_improve += 1
    
    return best_vals, best_edges, best_score

def multi_start(deadline):
    """Run multiple restarts with different initial value sets and edge configurations."""
    start = time.time()
    rng = random.Random(2024)
    
    initial_sets = [
        list(range(1, 15)),
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41],
        sorted(rng.sample(range(1, 100), 14)),
        sorted(rng.sample(range(1, 500), 14)),
        sorted(rng.sample(range(1, 1000), 14)),
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
        [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15],
        [1, 2, 4, 7, 12, 20, 33, 54, 88, 143, 232, 376, 609, 986],
        [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384],
    ]
    
    # Validate and filter
    valid_sets = []
    for s in initial_sets:
        if (len(s) == N and len(set(s)) == N and 
            all(isinstance(x, int) for x in s) and
            all(1 <= x <= MAX_VAL for x in s)):
            valid_sets.append(s)
    
    if not valid_sets:
        valid_sets = [list(range(1, 15))]
    
    best_vals = None
    best_edges = None
    best_score = float('inf')
    
    for vals in valid_sets:
        if time.time() > deadline - 3.0:
            break
        
        # Get initial edges
        edges = greedy_edges(vals)
        
        # Run local search with remaining time
        remaining = deadline - 2.0
        v, e, s = local_search(vals, edges, remaining, rng)
        
        if s < best_score:
            best_score = s
            best_vals = v
            best_edges = e
    
    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = greedy_edges(best_vals)
        best_score = score_of(best_vals, best_edges)
    
    return best_vals, best_edges, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8s safety margin
    
    vals, edges, score = multi_start(deadline)
    
    # Ensure correct types
    vals_out = [int(v) for v in vals]
    edges_out = [[int(a), int(b)] for a, b in edges]
    
    return {'values': vals_out, 'edges': edges_out}
