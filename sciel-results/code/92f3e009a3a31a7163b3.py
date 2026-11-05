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
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    divs = _divisors_of(BASE)
    all_moduli = divs
    
    def greedy_init(order='desc'):
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
            # For greedy init, we just want to cover as many *new* spots as possible.
            # Gains for all r can be computed via bincount on the *uncovered* set.
            # Initially all are uncovered.
            # As we add classes, the uncovered set shrinks.
            
            # Optimization: Use the same bincount trick.
            # Current uncovered mask is (covered == 0).
            unc_mask = (covered == 0)
            indices_unc = np.where(unc_mask)[0]
            if len(indices_unc) == 0:
                # Already covered everything? Unlikely in init.
                classes.append([m, 0])
                continue
                
            gains = np.bincount(indices_unc % m, minlength=m)
            best_r = int(np.argmax(gains))
            classes.append([m, best_r])
            covered[best_r::m] = 1
        return classes

    def local_search(classes, time_limit):
        # ref_count[i] = number of classes covering residue i
        ref_count = np.zeros(BASE, dtype=np.int32)
        for m, r in classes:
            ref_count[r::m] += 1
        
        uncovered_mask = (ref_count == 0)
        uncovered = int(np.sum(uncovered_mask))
        best_classes = [c[:] for c in classes]
        best_uncovered = uncovered
        
        n_classes = len(classes)
        indices = list(range(n_classes))
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() >= time_limit:
                    break
                    
                m, r_old = classes[idx]
                
                # Calculate loss: residues that ONLY this class covers
                # These will become uncovered if we remove this class.
                pos_old = np.arange(r_old, BASE, m)
                loss = int(np.sum(ref_count[pos_old] == 1))
                
                # Calculate potential gains for all residues r_new
                # A residue r_new covers spots x = r_new + k*m.
                # Gain is the number of such spots that are currently UNCOVERED (ref_count == 0).
                # Note: Since r_new != r_old, the sets are disjoint.
                # So we just count how many currently uncovered spots fall into the new progression.
                
                unc_indices = np.where(uncovered_mask)[0]
                if len(unc_indices) == 0:
                    break # Fully covered
                
                # For each uncovered index x, it contributes to gain[r_new] if x % m == r_new.
                # So gain[r] = count of x in unc_indices such that x % m == r.
                gains = np.bincount(unc_indices % m, minlength=m)
                
                # We want to maximize (gain - loss).
                # If max(gain - loss) > 0, we have an improvement.
                # If max(gain - loss) == 0, it's neutral.
                # If max(gain - loss) < 0, it's a worsening.
                
                best_r = int(np.argmax(gains))
                best_gain = int(gains[best_r])
                
                # Only accept if it improves the uncovered count
                # new_uncovered = uncovered - best_gain + loss
                # We want new_uncovered < uncovered => best_gain > loss
                # If best_gain == loss, it's neutral. We might accept to escape local minima?
                # For now, let's accept strict improvements.
                if best_gain > loss:
                    # Apply move
                    pos_new = np.arange(best_r, BASE, m)
                    
                    # Update ref_count
                    ref_count[pos_old] -= 1
                    ref_count[pos_new] += 1
                    
                    # Update uncovered mask
                    # Instead of recalculating full mask, we can update locally?
                    # No, ref_count changed for pos_old and pos_new.
                    # It's safer to just recalculate or update the mask based on changes.
                    # But recalc is O(BASE). Update is O(len(pos_old) + len(pos_new)).
                    # Let's try to update the mask efficiently.
                    # Actually, just recalculating `uncovered_mask = (ref_count == 0)` is fine and robust.
                    uncovered_mask = (ref_count == 0)
                    uncovered = int(np.sum(uncovered_mask))
                    
                    classes[idx][1] = best_r
                    improved = True
                    
                    if uncovered < best_uncovered:
                        best_uncovered = uncovered
                        best_classes = [c[:] for c in classes]
                        
                        if uncovered == 0:
                            return best_classes, 0
                            
        return best_classes, best_uncovered

    best_classes_global = None
    best_score_global = BASE + 1
    
    # Multi-start
    strategies = ['desc', 'asc', 'rand', 'rand', 'rand', 'rand', 'rand', 'rand']
    
    for strat in strategies:
        if time.time() >= deadline:
            break
            
        time_left = deadline - time.time()
        if time_left < 2:
            continue
            
        classes = greedy_init(order=strat)
        
        search_limit = time.time() + (time_left * 0.9)
        
        res_classes, res_score = local_search(classes, search_limit)
        
        if res_score < best_score_global:
            best_score_global = res_score
            best_classes_global = res_classes
            if best_score_global == 0:
                break
                
    if best_classes_global is None:
        best_classes_global = greedy_init(order='desc')
        
    return best_classes_global
