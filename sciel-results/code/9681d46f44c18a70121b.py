import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n = len(all_triples)  # 560
    
    # Precompute pair sets and triple masks
    triple_pairs = []
    triple_masks = []
    for t in all_triples:
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        triple_pairs.append(pairs)
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(mask)
    
    # Precompute conflict pairs: for each triple, which other triples conflict (share >= 2 points)
    # Actually, for SA we need: given a set of selected triples, can we add triple i?
    # Triple i can be added iff none of its 3 pairs are already used.
    
    # Precompute for each pair, which triples contain it
    all_pairs = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    pair_to_idx = {p: idx for idx, p in enumerate(all_pairs)}
    n_pairs = len(all_pairs)  # 120
    
    # pair_to_triples[pair] = list of triple indices containing that pair
    pair_to_triples = {p: [] for p in all_pairs}
    for i, t in enumerate(all_triples):
        for p in triple_pairs[i]:
            pair_to_triples[p].append(i)
    
    # Precompute: for each triple i, the list of pairs it uses (as indices into all_pairs)
    triple_pair_indices = []
    for i in range(n):
        pidx = [pair_to_idx[p] for p in triple_pairs[i]]
        triple_pair_indices.append(pidx)
    
    # Precompute triple masks as numpy-friendly integers
    triple_masks_list = triple_masks
    
    def get_alpha_from_selection(selected_set):
        """
        Compute independence number given a set of selected triple indices.
        Uses the precomputed approach: mark all supersets of each selected triple as 'bad'.
        A subset S is independent iff it contains no selected triple.
        """
        if not selected_set:
            return 16
        
        size = 1 << 16
        is_bad = bytearray(size)  # bytearray for speed
        
        for t_idx in selected_set:
            tm = triple_masks_list[t_idx]
            free = (~tm) & (size - 1)
            sub = free
            while True:
                sup = tm | sub
                is_bad[sup] = 1
                if sub == 0:
                    break
                sub = (sub - 1) & free
        
        # Find max popcount among non-bad masks
        # Iterate by decreasing size for early exit
        max_alpha = 0
        # For efficiency, iterate all masks and track max
        for m in range(size):
            if not is_bad[m]:
                bc = m.bit_count()
                if bc > max_alpha:
                    max_alpha = bc
                    if max_alpha == 16:
                        break
        return max_alpha
    
    def get_mis_mask(selected_set):
        """Return the actual mask of a maximum independent set."""
        if not selected_set:
            return (1 << 16) - 1
        
        size = 1 << 16
        is_bad = bytearray(size)
        
        for t_idx in selected_set:
            tm = triple_masks_list[t_idx]
            free = (~tm) & (size - 1)
            sub = free
            while True:
                sup = tm | sub
                is_bad[sup] = 1
                if sub == 0:
                    break
                sub = (sub - 1) & free
        
        max_alpha = 0
        best_mask = 0
        for m in range(size):
            if not is_bad[m]:
                bc = m.bit_count()
                if bc > max_alpha:
                    max_alpha = bc
                    best_mask = m
                    if max_alpha == 16:
                        break
        return best_mask
    
    best_alpha = 16
    best_selected = set()
    
    def update_best(selected_set):
        nonlocal best_alpha, best_selected
        if not selected_set:
            return
        alpha = get_alpha_from_selection(selected_set)
        if alpha < best_alpha:
            best_alpha = alpha
            best_selected = set(selected_set)
    
    # === Simulated Annealing with QUBO-inspired objective ===
    # Objective: minimize alpha, but use a smooth surrogate during SA
    # Surrogate: f(x) = alpha(x) + lambda * (n_max - |x|) where n_max is max possible
    # Actually, let's just minimize alpha directly with SA moves.
    
    # SA state:
    # - selected: set of triple indices
    # - used_pair_count: array of length n_pairs, count of how many selected triples use each pair
    #   (should be 0 or 1 for valid packings)
    
    def greedy_init():
        """Random greedy construction."""
        indices = list(range(n))
        random.shuffle(indices)
        selected = set()
        used_pairs = set()
        for idx in indices:
            pairs = triple_pairs[idx]
            conflict = False
            for p in pairs:
                if p in used_pairs:
                    conflict = True
                    break
            if not conflict:
                selected.add(idx)
                for p in pairs:
                    used_pairs.add(p)
        return selected
    
    def can_add(selected, used_pairs, t_idx):
        """Check if triple t_idx can be added without violating linearity."""
        for p in triple_pairs[t_idx]:
            if p in used_pairs:
                return False
        return True
    
    def simulate_annealing(initial_selected, steps, t_start, t_end):
        """
        SA loop. State: (selected_set, used_pairs_set)
        Moves: 
          1. Add a random valid triple
          2. Remove a random triple
          3. Swap: remove one, add another
        Objective: alpha (minimize)
        """
        selected = set(initial_selected)
        used_pairs = set()
        for t_idx in selected:
            for p in triple_pairs[t_idx]:
                used_pairs.add(p)
        
        curr_alpha = get_alpha_from_selection(selected)
        
        best_local_alpha = curr_alpha
        best_local_selected = set(selected)
        
        for step in range(steps):
            if time.time() > deadline - 3:
                break
            
            # Temperature
            progress = step / max(steps - 1, 1)
            T = t_start * (t_end / t_start) ** progress
            
            # Choose move type
            move_type = random.random()
            
            if move_type < 0.3 and len(selected) > 0:
                # Remove a random triple
                t_out = random.choice(list(selected))
                new_selected = set(selected)
                new_selected.discard(t_out)
                new_used_pairs = set(used_pairs)
                for p in triple_pairs[t_out]:
                    new_used_pairs.discard(p)
                
                new_alpha = get_alpha_from_selection(new_selected)
                delta = new_alpha - curr_alpha
                
                if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    selected = new_selected
                    used_pairs = new_used_pairs
                    curr_alpha = new_alpha
                    if curr_alpha < best_local_alpha:
                        best_local_alpha = curr_alpha
                        best_local_selected = set(selected)
                        
            elif move_type < 0.6:
                # Add a random valid triple
                # Find all valid triples
                valid_triples = []
                for t_idx in range(n):
                    if t_idx in selected:
                        continue
                    pairs = triple_pairs[t_idx]
                    conflict = False
                    for p in pairs:
                        if p in used_pairs:
                            conflict = True
                            break
                    if not conflict:
                        valid_triples.append(t_idx)
                
                if not valid_triples:
                    continue
                    
                t_in = random.choice(valid_triples)
                new_selected = set(selected)
                new_selected.add(t_in)
                new_used_pairs = set(used_pairs)
                for p in triple_pairs[t_in]:
                    new_used_pairs.add(p)
                
                new_alpha = get_alpha_from_selection(new_selected)
                delta = new_alpha - curr_alpha
                
                if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    selected = new_selected
                    used_pairs = new_used_pairs
                    curr_alpha = new_alpha
                    if curr_alpha < best_local_alpha:
                        best_local_alpha = curr_alpha
                        best_local_selected = set(selected)
                        
            else:
                # Swap: remove one, add another
                if len(selected) == 0:
                    continue
                    
                t_out = random.choice(list(selected))
                # Remove t_out
                temp_selected = set(selected)
                temp_selected.discard(t_out)
                temp_used_pairs = set(used_pairs)
                for p in triple_pairs[t_out]:
                    temp_used_pairs.discard(p)
                
                # Find valid additions
                valid_triples = []
                for t_idx in range(n):
                    if t_idx in temp_selected:
                        continue
                    pairs = triple_pairs[t_idx]
                    conflict = False
                    for p in pairs:
                        if p in temp_used_pairs:
                            conflict = True
                            break
                    if not conflict:
                        valid_triples.append(t_idx)
                
                if not valid_triples:
                    continue
                    
                t_in = random.choice(valid_triples)
                new_selected = set(temp_selected)
                new_selected.add(t_in)
                new_used_pairs = set(temp_used_pairs)
                for p in triple_pairs[t_in]:
                    new_used_pairs.add(p)
                
                new_alpha = get_alpha_from_selection(new_selected)
                delta = new_alpha - curr_alpha
                
                if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    selected = new_selected
                    used_pairs = new_used_pairs
                    curr_alpha = new_alpha
                    if curr_alpha < best_local_alpha:
                        best_local_alpha = curr_alpha
                        best_local_selected = set(selected)
        
        return best_local_selected, best_local_alpha
    
    # Main SA loop with multiple restarts
    num_restarts = 200
    sa_steps = 300
    t_start = 5.0
    t_end = 0.1
    
    for restart in range(num_restarts):
        if time.time() > deadline - 3:
            break
        
        # Greedy initialization
        init_selected = greedy_init()
        
        # Run SA
        sa_selected, sa_alpha = simulate_annealing(init_selected, sa_steps, t_start, t_end)
        
        if sa_alpha < best_alpha:
            best_alpha = sa_alpha
            best_selected = sa_selected
    
    # Final polish: local search on best solution
    # Try all single swaps and additions/removals to see if we can improve
    if best_selected:
        polish_selected = set(best_selected)
        polish_alpha = best_alpha
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 1:
                break
            
            # Try removing each triple
            for t_out in list(polish_selected):
                trial = set(polish_selected)
                trial.discard(t_out)
                trial_alpha = get_alpha_from_selection(trial)
                if trial_alpha < polish_alpha:
                    polish_selected = trial
                    polish_alpha = trial_alpha
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try adding each valid triple
            used_pairs = set()
            for t_idx in polish_selected:
                for p in triple_pairs[t_idx]:
                    used_pairs.add(p)
            
            for t_in in range(n):
                if t_in in polish_selected:
                    continue
                pairs = triple_pairs[t_in]
                conflict = False
                for p in pairs:
                    if p in used_pairs:
                        conflict = True
                        break
                if conflict:
                    continue
                trial = set(polish_selected)
                trial.add(t_in)
                trial_alpha = get_alpha_from_selection(trial)
                if trial_alpha < polish_alpha:
                    polish_selected = trial
                    polish_alpha = trial_alpha
                    improved = True
                    break
    
    if best_selected:
        result = [list(all_triples[i]) for i in sorted(best_selected)]
    else:
        result = []
    
    return result
