import numpy as np
import time
import random
from math import gcd

BASE = 51975

def _divisors_of(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted([d for d in divs if d > 1])

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    divs = _divisors_of(BASE)
    # Filter out 1, already done in _divisors_of
    # Sort moduli. We use all available divisors.
    # The problem states "distinct moduli m > 1 dividing 51975".
    # It does not say we must use ALL of them, but "With every divisor of 51975 available once"
    # implies we can choose a subset? 
    # "construct() -> classes [[m, r], ...] with distinct moduli m > 1 dividing 51975"
    # "score is the number of residues ... covered by no class"
    # Usually, in these tasks, using more moduli helps cover more.
    # However, adding a modulus with a large m might not help much if it doesn't cover new ground.
    # But since we can pick ANY subset, and the cost of evaluation is proportional to the number of classes,
    # we should probably stick to the most useful ones.
    # But wait, the seed program uses ALL divisors.
    # Let's check if using all divisors is mandatory or just available.
    # "classes [[m, r], ...] with distinct moduli m > 1 dividing 51975"
    # It doesn't explicitly say "using all". 
    # However, typically "covering system" problems imply a set of classes.
    # If I can choose the subset, I should probably choose the ones that cover the most.
    # But the seed uses all. Let's assume using all is a safe baseline, 
    # but maybe pruning is better?
    # Actually, if I add a class, I can always set it to cover 0 new residues (by picking a residue 
    # that is already fully covered? No, residues are specific).
    # Actually, if I have a modulus m, I MUST assign a residue r. 
    # If I don't use modulus m, I don't have that constraint.
    # Using more moduli gives more "tools" to cover gaps. 
    # It is unlikely that removing a modulus helps, unless it forces a bad configuration? 
    # No, the configuration is independent per modulus. 
    # So, using all available moduli should be at least as good as using a subset, 
    # because we can always "waste" a modulus by picking a residue that overlaps heavily 
    # with others, but we are optimizing, so we will pick the best residue for each.
    # So, we should use ALL divisors.
    
    all_moduli = divs
    
    def greedy_init(order='desc'):
        """
        Greedily assign residues to moduli.
        Order determines the sequence in which moduli are processed.
        """
        covered = np.zeros(BASE, dtype=np.uint8)
        classes = []
        
        if order == 'desc':
            mods_to_use = sorted(all_moduli, reverse=True)
        elif order == 'asc':
            mods_to_use = sorted(all_moduli)
        else:
            mods_to_use = all_moduli[:]
            random.shuffle(mods_to_use)
            
        for m in mods_to_use:
            best_r = 0
            best_gain = -1
            # For small m, checking all r is cheap (m is small).
            # For large m, checking all r is expensive (m is large, but number of steps is BASE/m).
            # Cost to evaluate r: O(BASE/m).
            # Total cost for this m: O(m * BASE/m) = O(BASE).
            # So total init cost is O(K * BASE) which is ~ 40 * 50k = 2M ops. Very fast.
            for r in range(m):
                # Count how many currently uncovered residues this class would cover
                gain = np.sum(covered[r::m] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            classes.append([m, best_r])
            covered[best_r::m] = 1
        return classes

    def local_search(classes, time_limit):
        """
        Local search with incremental updates.
        """
        # ref_count[i] = number of classes covering residue i
        ref_count = np.zeros(BASE, dtype=np.int32)
        for m, r in classes:
            ref_count[r::m] += 1
        
        uncovered = int(np.sum(ref_count == 0))
        best_classes = [c[:] for c in classes]
        best_uncovered = uncovered
        
        n_classes = len(classes)
        indices = list(range(n_classes))
        
        # Precompute strides and lengths for fast access?
        # We will just use slicing which is fast in numpy.
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            random.shuffle(indices)
            
            for idx in indices:
                m, r_old = classes[idx]
                
                # Find the best new residue for this modulus
                best_r = r_old
                best_uncovered_local = uncovered
                
                # Iterate over possible residues
                # Optimization: If m is very large, the number of residues is large,
                # but the arithmetic progression is short.
                # We can sample if m is huge? 
                # Max m is 51975.
                # If m=51975, there is only 1 residue (0).
                # If m=25987, there are 2 residues.
                # So iterating all r is always feasible?
                # Sum of m over all divisors?
                # No, we iterate r from 0 to m-1.
                # For m=51975, 51975 iterations.
                # For each iteration, we do O(BASE/m) work.
                # Total work for one modulus m: m * (BASE/m) = BASE.
                # So evaluating all residues for one modulus takes O(BASE) time.
                # With 40 moduli, one full pass is O(40 * BASE) ~ 2M operations.
                # This is very fast in Python/Numpy.
                
                for r_new in range(m):
                    if r_new == r_old:
                        continue
                    
                    # Calculate change in uncovered count
                    # We need to simulate the swap.
                    # Old contribution: ref_count[x] includes 1 for x in r_old::m
                    # New contribution: ref_count[x] includes 1 for x in r_new::m
                    
                    # Gain: residues that were 0 and become 1
                    # Loss: residues that were 1 and become 0 (if ref_count was 1, it becomes 0)
                    
                    # Efficient way:
                    # 1. Identify positions in old progression: pos_old = np.arange(r_old, BASE, m)
                    # 2. Identify positions in new progression: pos_new = np.arange(r_new, BASE, m)
                    # 3. For pos_new: if ref_count[pos] == 0, it becomes 1 (Gain)
                    #    BUT we must account for the fact that we are REMOVING old coverage first?
                    #    Actually, the set of positions covered by old and new might overlap.
                    
                    # Let's do it rigorously:
                    # Let S_old be the set of indices covered by (m, r_old)
                    # Let S_new be the set of indices covered by (m, r_new)
                    
                    # Current ref_count includes the class (m, r_old).
                    # If we swap to (m, r_new):
                    # For i in S_new \ S_old: ref_count[i] increases by 1.
                    #   If ref_count[i] was 0, uncovered decreases by 1.
                    # For i in S_old \ S_new: ref_count[i] decreases by 1.
                    #   If ref_count[i] was 1, uncovered increases by 1.
                    # For i in S_old \cap S_new: ref_count[i] unchanged.
                    
                    # Since S_old and S_new are arithmetic progressions with same difference m,
                    # they either are disjoint (if r_old != r_new mod m? No, they are mod m).
                    # Wait, r_old and r_new are in [0, m).
                    # If r_old != r_new, the sets S_old and S_new are DISJOINT.
                    # Because if x is in both, x = r_old + k1*m = r_new + k2*m => r_old - r_new = (k2-k1)*m.
                    # Since |r_old - r_new| < m, this implies r_old = r_new.
                    # So, for r_old != r_new, S_old and S_new are disjoint.
                    
                    # This simplifies things greatly!
                    
                    # Gain:
                    # Count how many i in S_new have ref_count[i] == 0.
                    # Note: ref_count currently includes the old class.
                    # But S_new is disjoint from S_old, so ref_count[i] for i in S_new does NOT include the old class.
                    # So we can just check the current ref_count.
                    
                    pos_new = np.arange(r_new, BASE, m)
                    gain = np.sum(ref_count[pos_new] == 0)
                    
                    # Loss:
                    # Count how many i in S_old have ref_count[i] == 1.
                    # If ref_count[i] == 1, it means ONLY this class covers i.
                    # If we remove this class, i becomes uncovered.
                    
                    pos_old = np.arange(r_old, BASE, m)
                    loss = np.sum(ref_count[pos_old] == 1)
                    
                    delta_uncovered = loss - gain
                    new_uncovered = uncovered + delta_uncovered
                    
                    if new_uncovered < best_uncovered_local:
                        best_uncovered_local = new_uncovered
                        best_r = r_new
                        # We don't update ref_count yet, just remember the best move
                        # We need to store the gain/loss details? No, we just need to apply the best one later.
                        # But wait, if we find a better r, we should keep its stats?
                        # No, we just pick the r that gives the best final uncovered count.
                        # We don't need to track intermediate states.
                        
                if best_r != r_old:
                    # Apply the best move
                    pos_old = np.arange(r_old, BASE, m)
                    pos_new = np.arange(best_r, BASE, m)
                    
                    # Update ref_count
                    ref_count[pos_old] -= 1
                    ref_count[pos_new] += 1
                    
                    # Update uncovered count
                    # We already calculated best_uncovered_local
                    uncovered = best_uncovered_local
                    classes[idx][1] = best_r
                    improved = True
                    
                    if uncovered < best_uncovered:
                        best_uncovered = uncovered
                        best_classes = [c[:] for c in classes]
                        
        return best_classes, best_uncovered

    best_classes_global = None
    best_score_global = BASE + 1
    
    # Multi-start
    # Try different initializations
    # 1. Descending (large moduli first, cover big chunks)
    # 2. Ascending (small moduli first, fill in gaps)
    # 3. Random
    
    strategies = ['desc', 'asc', 'rand', 'rand', 'rand', 'rand']
    
    for strat in strategies:
        if time.time() >= deadline:
            break
            
        # Allocate remaining time to this run
        time_left = deadline - time.time()
        if time_left < 5: # Need at least 5s for init + one pass
            continue
            
        # Init
        classes = greedy_init(order=strat)
        
        # Local search
        # Give it a portion of the remaining time, e.g., 80%
        search_limit = time.time() + (time_left * 0.8)
        
        res_classes, res_score = local_search(classes, search_limit)
        
        if res_score < best_score_global:
            best_score_global = res_score
            best_classes_global = res_classes
            # If we hit 0, we can stop
            if best_score_global == 0:
                break
                
    if best_classes_global is None:
        # Fallback
        best_classes_global = greedy_init(order='desc')
        
    return best_classes_global
