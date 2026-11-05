import math
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute pairs for each triple
    triple_pairs = [ [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])] for t in all_triples ]
    
    # Precompute all 8-subsets as bitmasks for fast independent set checking
    # Actually, let's use a different approach: track which 8-subsets are "covered"
    # An 8-subset is covered if it contains at least one triple from our system
    
    # For efficiency, let's work with bitmasks
    # Each triple has a 3-bit mask. An 8-subset S contains triple T if T's mask is a subset of S's mask.
    
    # Let's precompute all 8-subsets
    all_8subsets = list(combinations(points, 8))
    n_8subsets = len(all_8subsets)
    
    # For each 8-subset, precompute which triples are contained in it
    # triple_in_8subset[j][i] = 1 if triple i is contained in 8-subset j
    # This is a 1287 x 560 matrix, which is manageable
    
    # Actually, let's use a different representation for speed
    # For each triple, store which 8-subsets it's contained in
    triple_to_8subsets = [[] for _ in range(n_triples)]
    for j, s8 in enumerate(all_8subsets):
        s8_set = set(s8)
        for i, t in enumerate(all_triples):
            if t[0] in s8_set and t[1] in s8_set and t[2] in s8_set:
                triple_to_8subsets[i].append(j)
    
    # State
    used_pairs = set()
    current_triples = []
    # covered_8subsets[j] = count of triples in current system that are contained in 8-subset j
    covered_8subsets = [0] * n_8subsets
    current_alpha = 16
    
    def get_alpha_from_covered():
        """Check if all 8-subsets are covered. If not, find the largest independent set."""
        # First check if alpha <= 7 (all 8-subsets covered)
        for j in range(n_8subsets):
            if covered_8subsets[j] == 0:
                # This 8-subset is independent, so alpha >= 8
                return 8  # We know alpha is at least 8, but could be larger
        # All 8-subsets are covered, so alpha <= 7
        # Check if alpha <= 6
        # This is more expensive, but we can do it later
        return 7  # Conservative: we know alpha <= 7
    
    def get_exact_alpha():
        """Compute the exact independence number by checking all subsets."""
        # Check from largest to smallest
        for k in range(16, -1, -1):
            for s in combinations(points, k):
                s_set = set(s)
                independent = True
                for t in current_triples:
                    if t[0] in s_set and t[1] in s_set and t[2] in s_set:
                        independent = False
                        break
                if independent:
                    return k
        return 0
    
    def is_valid_add(idx):
        for p in triple_pairs[idx]:
            if p in used_pairs:
                return False
        return True
    
    def add_triple(idx):
        t = all_triples[idx]
        current_triples.append(t)
        for p in triple_pairs[idx]:
            used_pairs.add(p)
        for j in triple_to_8subsets[idx]:
            covered_8subsets[j] += 1
    
    def remove_triple(idx):
        t = all_triples[idx]
        for i, ct in enumerate(current_triples):
            if ct == t:
                current_triples.pop(i)
                break
        for p in triple_pairs[idx]:
            used_pairs.discard(p)
        for j in triple_to_8subsets[idx]:
            covered_8subsets[j] -= 1
    
    def init_greedy():
        nonlocal used_pairs, current_triples, covered_8subsets, current_alpha
        used_pairs.clear()
        current_triples.clear()
        covered_8subsets = [0] * n_8subsets
        
        indices = list(range(n_triples))
        random.shuffle(indices)
        for idx in indices:
            if is_valid_add(idx):
                add_triple(idx)
        current_alpha = get_alpha_from_covered()
    
    # Initial greedy
    init_greedy()
    best_alpha = current_alpha
    best_triples = list(current_triples)
    
    # Local search: try to cover all 8-subsets
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    steps_since_improvement = 0
    max_stuck = 3000
    
    while time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.5:
            # Add a triple
            candidates = []
            for idx in range(n_triples):
                if is_valid_add(idx):
                    candidates.append(idx)
            if not candidates:
                continue
                
            idx = random.choice(candidates)
            old_alpha = current_alpha
            
            add_triple(idx)
            new_alpha = get_alpha_from_covered()
            current_alpha = new_alpha
            
            if new_alpha <= old_alpha or (T > T_min and random.random() < min(1.0, math.exp(-(new_alpha - old_alpha) / T))):
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = list(current_triples)
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                remove_triple(idx)
                current_alpha = old_alpha
                steps_since_improvement += 1

        elif move_type < 0.8:
            # Remove a triple
            if not current_triples:
                continue
            rem_pos = random.randrange(len(current_triples))
            t_rem = current_triples[rem_pos]
            idx = triple_to_idx[t_rem]
            
            old_alpha = current_alpha
            remove_triple(idx)
            new_alpha = get_alpha_from_covered()
            current_alpha = new_alpha
            
            if new_alpha <= old_alpha or (T > T_min and random.random() < min(1.0, math.exp(-(new_alpha - old_alpha) / T))):
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = list(current_triples)
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                add_triple(idx)
                current_alpha = old_alpha
                steps_since_improvement += 1

        else:
            # Swap: remove one, add another
            if not current_triples:
                continue
            rem_pos = random.randrange(len(current_triples))
            t_rem = current_triples[rem_pos]
            idx_rem = triple_to_idx[t_rem]
            
            remove_triple(idx_rem)
            
            candidates = []
            for idx_add in range(n_triples):
                if idx_add != idx_rem and is_valid_add(idx_add):
                    candidates.append(idx_add)
                    
            if not candidates:
                add_triple(idx_rem)
                continue
                
            idx_add = random.choice(candidates)
            
            old_alpha = current_alpha
            add_triple(idx_add)
            new_alpha = get_alpha_from_covered()
            current_alpha = new_alpha
            
            if new_alpha <= old_alpha or (T > T_min and random.random() < min(1.0, math.exp(-(new_alpha - old_alpha) / T))):
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = list(current_triples)
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                remove_triple(idx_add)
                add_triple(idx_rem)
                current_alpha = old_alpha
                steps_since_improvement += 1

        T *= cooling
        if T < T_min:
            T = T_min
            
        if steps_since_improvement > max_stuck:
            init_greedy()
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_triples = list(current_triples)
            steps_since_improvement = 0
            T = 1.0
    
    # Final: compute exact alpha for best_triples and verify
    # Reconstruct state for best_triples
    used_pairs.clear()
    current_triples.clear()
    for t in best_triples:
        idx = triple_to_idx[t]
        add_triple(idx)
    
    exact_alpha = get_exact_alpha()
    
    return best_triples
