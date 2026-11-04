import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    random.seed(42)
    
    subsets = []
    for comb in itertools.combinations(range(D), 4):
        mask = 0
        for i in comb:
            mask |= (1 << i)
        subsets.append(mask)
    
    n_subsets = len(subsets)
    
    inter = np.zeros((n_subsets, n_subsets), dtype=np.int8)
    for i in range(n_subsets):
        for j in range(i+1, n_subsets):
            inter[i, j] = (subsets[i] & subsets[j]).bit_count()
            inter[j, i] = inter[i, j]
    
    conflict_with = [set() for _ in range(n_subsets)]
    for i in range(n_subsets):
        for j in range(i+1, n_subsets):
            if inter[i, j] > 2:
                conflict_with[i].add(j)
                conflict_with[j].add(i)
    
    def greedy_initial():
        order = list(range(n_subsets))
        random.shuffle(order)
        selected = set()
        for idx in order:
            if not any(idx in conflict_with[j] for j in selected):
                selected.add(idx)
        return selected
    
    def try_grow(selected, max_removals):
        selected_list = list(selected)
        unselected = [j for j in range(n_subsets) if j not in selected]
        
        for to_add in unselected:
            conflicts = [r for r in selected_list if to_add in conflict_with[r]]
            if len(conflicts) <= max_removals:
                new_selected = selected - set(conflicts)
                new_selected.add(to_add)
                if len(new_selected) > len(selected):
                    return new_selected
        return None
    
    def local_search(selected, time_limit):
        while time.time() < time_limit:
            improved = False
            # Try grow with 1 removal
            new_sel = try_grow(selected, 1)
            if new_sel is not None:
                selected = new_sel
                improved = True
                continue
            # Try grow with 2 removals
            new_sel = try_grow(selected, 2)
            if new_sel is not None:
                selected = new_sel
                improved = True
                continue
            # No growth possible, perturb
            if selected:
                to_remove = random.choice(list(selected))
                selected = selected - {to_remove}
                improved = True
            if not improved:
                break
        return selected
    
    best_overall = set()
    best_overall_size = 0
    
    while time.time() < deadline - 5:
        initial = greedy_initial()
        if len(initial) > best_overall_size:
            best_overall = set(initial)
            best_overall_size = len(initial)
        
        time_limit = min(deadline - 5, time.time() + 2.5)
        if time.time() >= time_limit:
            break
        result = local_search(initial, time_limit)
        if len(result) > best_overall_size:
            best_overall = result
            best_overall_size = len(result)
    
    selected = best_overall
    
    vectors = []
    for idx in selected:
        mask = subsets[idx]
        indices = []
        for i in range(D):
            if mask & (1 << i):
                indices.append(i)
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx2, s in zip(indices, signs):
                v[idx2] = s
            vectors.append(v)
            
    if not vectors:
        mask = subsets[0]
        indices = []
        for i in range(D):
            if mask & (1 << i):
                indices.append(i)
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx2, s in zip(indices, signs):
                v[idx2] = s
            vectors.append(v)
                    
    return np.array(vectors, dtype=np.int64)
