"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from math import gcd
from fractions import Fraction
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Candidate moduli in [20, 200]
    candidates = list(range(20, 201))
    
    # Key structural insight: if we select pairwise coprime moduli,
    # then gcd(m_i, m_j) = 1 for all pairs, so the disjointness constraint
    # r_i != r_j (mod 1) is automatically satisfied for any residues.
    # Thus, we just need to find max-weight set of pairwise coprime moduli.
    
    # Precompute coprimality
    n = len(candidates)
    coprime = [[True] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            coprime[i][j] = coprime[j][i] = (gcd(candidates[i], candidates[j]) == 1)
    
    # Degree in non-coprime graph (lower degree = more flexible)
    non_coprime_degree = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and not coprime[i][j]:
                non_coprime_degree[i] += 1
    
    # Greedy: pick moduli in order of highest 1/m, skip if conflicts
    # Try multiple orderings
    best_set = []
    best_score = 0.0
    
    def compute_score(idx_set):
        return sum(1.0 / candidates[i] for i in idx_set)
    
    # Strategy 1: Greedy by descending 1/m (ascending m)
    def greedy_by_weight():
        selected = []
        used = set()
        for i in range(n):  # candidates sorted ascending, so 1/m descending
            if time.time() > deadline - 60:
                break
            # Check if coprime to all selected
            compatible = True
            for j in selected:
                if not coprime[i][j]:
                    compatible = False
                    break
            if compatible:
                selected.append(i)
        return selected
    
    # Strategy 2: Greedy by ascending non-coprime degree (most flexible first), tie-break by 1/m
    def greedy_by_flexibility():
        order = sorted(range(n), key=lambda i: (non_coprime_degree[i], candidates[i]))
        selected = []
        for i in order:
            if time.time() > deadline - 60:
                break
            compatible = True
            for j in selected:
                if not coprime[i][j]:
                    compatible = False
                    break
            if compatible:
                selected.append(i)
        return selected
    
    # Strategy 3: Multiple randomized greedy restarts
    def greedy_randomized(seed_val):
        rng = random.Random(seed_val)
        # Create weighted random order biased toward small moduli and low degree
        weights = [1.0 / candidates[i] * (1.0 / (1 + non_coprime_degree[i])) for i in range(n)]
        total_w = sum(weights)
        order = []
        available = list(range(n))
        for _ in range(n):
            if time.time() > deadline - 60:
                break
            w = [weights[i] for i in available]
            tw = sum(w)
            r = rng.random() * tw
            cum = 0
            idx_in_avail = 0
            for k, wv in enumerate(w):
                cum += wv
                if r <= cum:
                    idx_in_avail = k
                    break
            chosen = available[idx_in_avail]
            order.append(chosen)
            available.pop(idx_in_avail)
        
        selected = []
        for i in order:
            compatible = True
            for j in selected:
                if not coprime[i][j]:
                    compatible = False
                    break
            if compatible:
                selected.append(i)
        return selected
    
    # Run multiple strategies
    states = []
    
    s1 = greedy_by_weight()
    sc1 = compute_score(s1)
    states.append((s1, sc1))
    
    s2 = greedy_by_flexibility()
    sc2 = compute_score(s2)
    states.append((s2, sc2))
    
    for seed in range(50):
        if time.time() > deadline - 60:
            break
        s = greedy_randomized(seed)
        sc = compute_score(s)
        states.append((s, sc))
    
    for s, sc in states:
        if sc > best_score:
            best_set = s
            best_score = sc
    
    # Local search: try to swap out a modulus for a better combination
    def local_search_improve(idx_set, max_iter=5000):
        idx_set = list(idx_set)
        score = compute_score(idx_set)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() > deadline - 30:
                break
            
            # Try adding any missing modulus
            used_set = set(idx_set)
            for i in range(n):
                if i in used_set:
                    continue
                compatible = True
                for j in idx_set:
                    if not coprime[i][j]:
                        compatible = False
                        break
                if compatible:
                    new_set = idx_set + [i]
                    new_score = score + 1.0 / candidates[i]
                    if new_score > score:
                        idx_set = new_set
                        score = new_score
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try removing one and adding one or two
            for k in range(len(idx_set)):
                if time.time() > deadline - 30:
                    break
                removed = idx_set[k]
                without = [x for j, x in enumerate(idx_set) if j != k]
                score_without = score - 1.0 / candidates[removed]
                
                # Try adding one modulus
                for i in range(n):
                    if i in used_set or i == removed:
                        continue
                    compatible = True
                    for j in without:
                        if not coprime[i][j]:
                            compatible = False
                            break
                    if compatible:
                        new_score = score_without + 1.0 / candidates[i]
                        if new_score > score:
                            idx_set = without + [i]
                            score = new_score
                            improved = True
                            break
                if improved:
                    break
                
                # Try adding two moduli
                for i in range(n):
                    if time.time() > deadline - 30:
                        break
                    if i in used_set or i == removed:
                        continue
                    comp_i = True
                    for j in without:
                        if not coprime[i][j]:
                            comp_i = False
                            break
                    if not comp_i:
                        continue
                    for l in range(n):
                        if l <= i:
                            continue
                        if l in used_set or l == removed:
                            continue
                        comp_l = True
                        for j in without:
                            if not coprime[l][j]:
                                comp_l = False
                                break
                        if not comp_l:
                            continue
                        if not coprime[i][l]:
                            continue
                        new_score = score_without + 1.0 / candidates[i] + 1.0 / candidates[l]
                        if new_score > score:
                            idx_set = without + [i, l]
                            score = new_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return idx_set, score
    
    best_set, best_score = local_search_improve(best_set)
    
    # Simulated annealing for further improvement
    T = 0.1
    T_min = 0.001
    cooling = 0.999
    
    idx_set = list(best_set)
    score = best_score
    
    while T > T_min and time.time() < deadline - 15:
        move_type = random.choice(['add', 'remove', 'swap'])
        new_set = None
        new_score = None
        
        if move_type == 'add':
            used_set = set(idx_set)
            available = [i for i in range(n) if i not in used_set]
            if available:
                i = random.choice(available)
                compatible = True
                for j in idx_set:
                    if not coprime[i][j]:
                        compatible = False
                        break
                if compatible:
                    new_set = idx_set + [i]
                    new_score = score + 1.0 / candidates[i]
        elif move_type == 'remove':
            if len(idx_set) > 1:
                k = random.randint(0, len(idx_set) - 1)
                new_set = [x for j, x in enumerate(idx_set) if j != k]
                new_score = score - 1.0 / candidates[idx_set[k]]
        elif move_type == 'swap':
            if len(idx_set) >= 1:
                k = random.randint(0, len(idx_set) - 1)
                removed = idx_set[k]
                without = [x for j, x in enumerate(idx_set) if j != k]
                used_set = set(idx_set)
                available = [i for i in range(n) if i not in used_set]
                if available:
                    i = random.choice(available)
                    compatible = True
                    for j in without:
                        if not coprime[i][j]:
                            compatible = False
                            break
                    if compatible:
                        new_set = without + [i]
                        new_score = score - 1.0 / candidates[removed] + 1.0 / candidates[i]
        
        if new_set is not None and new_score is not None:
            delta = new_score - score
            if delta > 0 or random.random() < math.exp(delta / T):
                idx_set = new_set
                score = new_score
                if score > best_score:
                    best_set = list(idx_set)
                    best_score = score
        
        T *= cooling
    
    # Final greedy fill
    used_set = set(best_set)
    for i in range(n):
        if time.time() > deadline - 5:
            break
        if i in used_set:
            continue
        compatible = True
        for j in best_set:
            if not coprime[i][j]:
                compatible = False
                break
        if compatible:
            best_set.append(i)
            best_score += 1.0 / candidates[i]
            used_set.add(i)
    
    # Convert to required format: [[m, r], ...]
    # For pairwise coprime moduli, any residue works. Use r=0 for all.
    result = [[candidates[i], 0] for i in best_set]
    return result
# EVOLVE-BLOCK-END
