import time
import numpy as np
import random

BASE = 51975

def get_divisors(n):
    divs = []
    for i in range(2, n + 1):
        if n % i == 0:
            divs.append(i)
    return divs

DIVISORS = get_divisors(BASE)
DIVISORS.sort()

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s margin for 120s limit
    
    best_classes = []
    best_score = BASE
    
    # Try multiple random restarts with different greedy initializations
    # or just one very good one.
    # Given the structure, a strong greedy + fast local search is best.
    
    # 1. Initial Greedy: Smallest moduli first is usually good for coverage density.
    # We will use a greedy that picks the best (m, r) from the *remaining* available moduli
    # that maximizes gain. This is O(K^2 * N) which might be too slow if K is large, 
    # but K is small (divisors of 51975).
    # Number of divisors of 51975 = 3^3 * 5^2 * 7 * 11 is (3+1)*(2+1)*(1+1)*(1+1) = 4*3*2*2 = 48 divisors.
    # So K <= 47 (excluding 1).
    # O(47^2 * 51975) is too slow for global best selection at each step.
    # So we stick to a fixed order greedy, but maybe try a few orders.
    
    # Let's try 3 different greedy orders:
    # 1. Ascending moduli
    # 2. Descending moduli
    # 3. Random order
    
    candidates = []
    
    for run_id in range(3):
        if time.time() > deadline - 10:
            break
            
        if run_id == 0:
            divs = DIVISORS[:]
        elif run_id == 1:
            divs = DIVISORS[::-1]
        else:
            divs = DIVISORS[:]
            random.shuffle(divs)
        
        covered_count = np.zeros(BASE, dtype=np.int8)
        current_classes = []
        
        # Greedy Phase
        for m in divs:
            if time.time() > deadline - 15:
                break
            
            # Find best r for this m
            # We want to maximize number of x in r::m where covered_count[x] == 0
            # Reshape covered_count? No, we need sum of zeros.
            # sum_zeros = (BASE // m) - sum(covered_count[r::m])
            # We can compute sums of covered_count for all r at once?
            # covered_count.reshape(-1, m).sum(axis=0) gives sum of counts for each residue.
            # This counts how many times each residue is covered.
            # We want residues with count 0.
            
            try:
                reshaped = covered_count.reshape(-1, m)
                counts = np.sum(reshaped, axis=0)
                # counts[r] is the number of times residue r is covered.
                # We want to pick r with max (BASE//m - counts[r])? 
                # No, that's not right. 
                # If covered_count[x] is the number of classes covering x,
                # then "newly covered" by class (m, r) is the number of x in r::m 
                # such that covered_count[x] == 0.
                
                # The reshape trick sums the *values* of covered_count.
                # It does NOT tell us how many are 0.
                # Example: covered_count = [0, 1, 0, 1]. Sum = 2.
                # If we pick r=0 (indices 0, 2), values are 0, 0. Both are 0. Gain = 2.
                # If we pick r=1 (indices 1, 3), values are 1, 1. Neither is 0. Gain = 0.
                
                # So we need to count zeros.
                # This is harder to vectorize directly with reshape sum.
                
                # Alternative:
                # For a specific m, iterate r? No, too slow.
                # We can create a boolean mask: is_zero = (covered_count == 0)
                # Then reshape is_zero and sum.
                
                is_zero = (covered_count == 0)
                reshaped_zero = is_zero.reshape(-1, m)
                zero_counts = np.sum(reshaped_zero, axis=0)
                
                best_r = np.argmax(zero_counts)
                best_gain = zero_counts[best_r]
                
            except:
                continue
            
            if best_gain > 0:
                # Update covered_count
                covered_count[best_r::m] += 1
                current_classes.append([m, int(best_r)])
        
        current_score = np.sum(covered_count == 0)
        
        if current_score < best_score:
            best_score = current_score
            best_classes = [c[:] for c in current_classes]
        
        # Local Search on this candidate
        # Use coverage_count for fast evaluation
        # Moves:
        # 1. Change residue of a class
        # 2. Remove a class (if it doesn't reduce score? No, removing a class increases score or keeps same. We want to minimize score. So we only remove if score doesn't increase, i.e., it was redundant. But if it's redundant, score stays same. We prefer fewer classes? The problem doesn't penalize number of classes, only score. So redundant classes are harmless but clutter. We can remove them to simplify.)
        # 3. Add a class (if it reduces score)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # A. Re-optimize residues
            # For each class, try changing its residue to the one that maximizes gain - loss.
            # Gain = # of x in new_r::m with count 0
            # Loss = # of x in old_r::m with count 1 (they become 0)
            
            for idx in range(len(current_classes)):
                if time.time() > deadline - 5:
                    break
                
                m, r = current_classes[idx]
                
                # Compute potential gains/losses for all r'
                # We need counts of 0s and 1s in each residue class of m, 
                # excluding the contribution of this class itself?
                # No, `covered_count` includes this class.
                # Let's temporarily remove this class from covered_count to see the "background".
                
                covered_count[r::m] -= 1
                
                # Now covered_count reflects coverage by OTHER classes.
                # We want to pick r' that maximizes:
                # Gain(r') = # of x in r'::m with covered_count[x] == 0
                # Loss is 0 because we are adding to the background.
                # Wait, if we change r to r', we add r' and remove r.
                # In the "background" (without this class), adding r' covers those with count 0.
                # So Gain = sum( (covered_count == 0)[r'::m] )
                
                is_zero_bg = (covered_count == 0)
                try:
                    reshaped_bg = is_zero_bg.reshape(-1, m)
                    gain_counts = np.sum(reshaped_bg, axis=0)
                    best_r_new = np.argmax(gain_counts)
                    best_gain_val = gain_counts[best_r_new]
                except:
                    best_r_new = r
                    best_gain_val = 0
                
                # If best_r_new != r, we have a potential improvement.
                # But we must ensure that changing r to best_r_new actually reduces total uncovered.
                # Total Uncovered = Sum(covered_count_final == 0)
                # Current Total Uncovered (with class at r) = Score_Current
                # New Total Uncovered (with class at best_r_new) = ?
                
                # Let's just calculate the score if we swap.
                # It's O(N). With K~20, this is fine.
                
                if best_r_new != r:
                    # Apply swap to covered_count
                    covered_count[best_r_new::m] += 1
                    
                    new_score = np.sum(covered_count == 0)
                    
                    if new_score < current_score:
                        current_score = new_score
                        current_classes[idx] = [m, int(best_r_new)]
                        improved = True
                        break # Restart loop to re-evaluate others
                    else:
                        # Revert
                        covered_count[best_r_new::m] -= 1
                
                # Revert the temporary removal of class idx
                covered_count[r::m] += 1
            
            # B. Add a new class
            if not improved:
                used_set = set(m for m, r in current_classes)
                unused = [m for m in DIVISORS if m not in used_set]
                
                if unused:
                    best_m = None
                    best_r = None
                    best_gain = 0
                    
                    for m in unused:
                        if time.time() > deadline - 5:
                            break
                        is_zero = (covered_count == 0)
                        try:
                            reshaped_zero = is_zero.reshape(-1, m)
                            zero_counts = np.sum(reshaped_zero, axis=0)
                            max_gain = np.max(zero_counts)
                            if max_gain > best_gain:
                                best_gain = max_gain
                                best_m = m
                                best_r = np.argmax(zero_counts)
                        except:
                            pass
                    
                    if best_m is not None and best_gain > 0:
                        covered_count[best_r::best_m] += 1
                        current_classes.append([best_m, int(best_r)])
                        current_score = np.sum(covered_count == 0)
                        improved = True
            
            # C. Remove redundant classes
            # A class is redundant if removing it does not increase the score.
            # i.e., if it covers no unique residues.
            # Unique residues are those with covered_count == 1 and covered by this class.
            
            if not improved:
                for idx in range(len(current_classes)):
                    if time.time() > deadline - 5:
                        break
                    
                    m, r = current_classes[idx]
                    
                    # Check if any x in r::m has covered_count[x] == 1
                    # If yes, it's essential.
                    # If no, it's redundant.
                    
                    # We can check this by summing (covered_count == 1) over r::m
                    is_one = (covered_count == 1)
                    unique_count = np.sum(is_one[r::m])
                    
                    if unique_count == 0:
                        # Remove it
                        covered_count[r::m] -= 1
                        current_classes.pop(idx)
                        # Score doesn't change, but let's update just in case (it shouldn't)
                        current_score = np.sum(covered_count == 0)
                        improved = True
                        break
            
            if current_score < best_score:
                best_score = current_score
                best_classes = [c[:] for c in current_classes]
            
            if best_score == 0:
                break
    
    return best_classes
