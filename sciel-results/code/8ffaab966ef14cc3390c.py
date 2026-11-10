import time
import random
from itertools import combinations

ALL_TRIPLES = list(combinations(range(16), 3))
TRIPLE_PAIRS = {t: [(t[0],t[1]), (t[1],t[0]), (t[0],t[2]), (t[2],t[0]), (t[1],t[2]), (t[2],t[1])] for t in ALL_TRIPLES}

def find_max_independent_set(edges):
    """Returns the size and the set of a maximum independent set."""
    for k in range(8, 0, -1):
        for subset in combinations(range(16), k):
            subset_set = set(subset)
            is_ind = True
            for e in edges:
                if e[0] in subset_set and e[1] in subset_set and e[2] in subset_set:
                    is_ind = False
                    break
            if is_ind:
                return k, subset_set
    return 0, set()

def get_used_pairs(edges):
    pairs = set()
    for e in edges:
        for p in TRIPLE_PAIRS[e]:
            pairs.add(p)
    return pairs

def is_valid_edge(edge, used_pairs, current_edges_set):
    if edge in current_edges_set:
        return False
    for p in TRIPLE_PAIRS[edge]:
        if p in used_pairs:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_edges = []
    best_alpha = 16
    best_edges_set = set()
    
    # Initial Greedy
    for _ in range(20):
        if time.time() > deadline - 30: break
        
        random.shuffle(ALL_TRIPLES)
        edges = []
        edges_set = set()
        used_pairs = set()
        
        for t in ALL_TRIPLES:
            if len(edges) >= 37: break
            valid = True
            for p in TRIPLE_PAIRS[t]:
                if p in used_pairs:
                    valid = False
                    break
            if valid:
                edges.append(t)
                edges_set.add(t)
                for p in TRIPLE_PAIRS[t]:
                    used_pairs.add(p)
        
        alpha, _ = find_max_independent_set(edges)
        if alpha < best_alpha:
            best_alpha = alpha
            best_edges = list(edges)
            best_edges_set = set(edges)
            
        if best_alpha <= 5:
            break

    # Targeted Local Search
    current_edges = list(best_edges)
    current_edges_set = set(best_edges)
    current_alpha = best_alpha
    
    while time.time() < deadline - 5:
        if current_alpha <= 5:
            break
            
        alpha, ind_set = find_max_independent_set(current_edges)
        
        if alpha < current_alpha:
            current_alpha = alpha
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_edges = list(current_edges)
                best_edges_set = set(current_edges)
            continue
        
        if alpha > current_alpha:
            current_alpha = alpha
            continue
            
        # Try to add an edge contained in ind_set
        used_pairs = get_used_pairs(current_edges)
        valid_candidates = []
        for t in combinations(ind_set, 3):
            t_tuple = tuple(t)
            if t_tuple in current_edges_set:
                continue
            if is_valid_edge(t_tuple, used_pairs, current_edges_set):
                valid_candidates.append(t_tuple)
                
        if valid_candidates:
            new_edge = random.choice(valid_candidates)
            current_edges.append(new_edge)
            current_edges_set.add(new_edge)
        else:
            # Stuck: try a random swap
            if current_edges:
                idx = random.randint(0, len(current_edges) - 1)
                edge_to_remove = current_edges[idx]
                temp_edges = current_edges[:idx] + current_edges[idx+1:]
                temp_edges_set = set(temp_edges)
                temp_used_pairs = get_used_pairs(temp_edges)
                
                candidates = []
                for t in ALL_TRIPLES:
                    if t in current_edges_set:
                        continue
                    if is_valid_edge(t, temp_used_pairs, temp_edges_set):
                        candidates.append(t)
                        
                if candidates:
                    new_edge = random.choice(candidates)
                    current_edges = temp_edges + [new_edge]
                    current_edges_set = set(current_edges)
                else:
                    break

    return [list(e) for e in best_edges]
