import numpy as np
import time
import random

def _get_divisors(N):
    """Get all divisors of N greater than 1, sorted."""
    divisors = set()
    for i in range(2, int(np.sqrt(N)) + 1):
        if N % i == 0:
            divisors.add(i)
            divisors.add(N // i)
    divisors.add(N)
    return sorted(divisors)

def _build_covered(classes, N):
    """Build the covered array from a list of classes."""
    covered = np.zeros(N, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    return covered

def _score(classes, N):
    """Calculate the number of uncovered residues."""
    if not classes:
        return N
    covered = _build_covered(classes, N)
    return int(np.sum(1 - covered))

def _best_residue_for_modulus(m, covered, N):
    """Find the residue r that covers the most uncovered residues for modulus m."""
    best_gain = -1
    best_r = 0
    for r in range(m):
        uncovered = np.count_nonzero(covered[r::m] == 0)
        if uncovered > best_gain:
            best_gain = uncovered
            best_r = r
    return best_r, best_gain

def _greedy_build(N, divisors, time_limit=100.0, start_time=None):
    """Run greedy algorithm to build a covering system."""
    if start_time is None:
        start_time = time.time()
    
    covered = np.zeros(N, dtype=np.uint8)
    used_moduli = set()
    classes = []
    
    while time.time() - start_time < time_limit:
        best_gain = 0
        best_m = None
        best_r = None
        
        for m in divisors:
            if m in used_moduli:
                continue
            
            r, gain = _best_residue_for_modulus(m, covered, N)
            if gain > best_gain:
                best_gain = gain
                best_m = m
                best_r = r
        
        if best_gain == 0:
            break
        
        classes.append([best_m, best_r])
        used_moduli.add(best_m)
        covered[best_r::best_m] = 1
    
    return classes, covered

def _local_search_v2(classes, N, divisors, time_limit=10.0, start_time=None):
    """Perform local search by trying to swap classes, using coverage counts for efficiency."""
    if start_time is None:
        start_time = time.time()
    
    current_classes = list(classes)
    current_score = _score(current_classes, N)
    used_moduli = set(m for m, r in current_classes)
    
    # Precompute coverage count array: how many classes cover each residue
    cov_count = np.zeros(N, dtype=np.int32)
    for m, r in current_classes:
        cov_count[r::m] += 1
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        for i in range(len(current_classes)):
            if time.time() - start_time > time_limit:
                break
            
            m_old, r_old = current_classes[i]
            
            # Compute coverage without class i
            # cov_count - (1 where class i covers)
            # We need to find best residue for each unused m_new given coverage without class i
            
            for m_new in divisors:
                if m_new in used_moduli:
                    continue
                
                # For each residue r_new in 0..m_new-1, count uncovered residues
                # in positions r_new::m_new where cov_count == 1 (only covered by class i)
                # i.e., after removing class i, these become uncovered
                # gain = number of positions in r_new::m_new where cov_count[r_new::m_new] == 1
                # (these are positions that would become uncovered if we remove class i,
                #  but are re-covered by the new class)
                
                # Actually, the score change is:
                # - Removing class i: positions where cov_count==1 become uncovered (bad, +uncovered)
                # - Adding class m_new with residue r_new: positions in r_new::m_new that were 
                #   uncovered (cov_count==0 after removal) become covered (good, -uncovered)
                # - Positions in r_new::m_new that were already covered (cov_count>=1 after removal) 
                #   stay covered (no change)
                
                # Net change in uncovered = (positions where cov_count==1 and NOT in r_new::m_new) 
                #                           - (positions where cov_count==0 after removal AND in r_new::m_new)
                # Wait, let me think more carefully.
                
                # Original uncovered: positions where cov_count == 0
                # After removing class i: positions where cov_count == 1 become uncovered (if not in new class)
                # After adding new class: positions in r_new::m_new that are uncovered after removal become covered
                
                # Let's compute:
                # uncovered_after_removal = (cov_count == 1) for positions not in r_new::m_new, 
                #                          plus (cov_count == 0) for all positions
                # Then new class covers some of those.
                
                # Simpler: compute gain directly
                # For each r_new, the new class covers positions r_new::m_new
                # Among those, some were uncovered (cov_count==0) -> they become covered: gain
                # Some were covered only by class i (cov_count==1) -> they stay covered: no change
                # Some were covered by multiple (cov_count>=2) -> they stay covered: no change
                # 
                # But removing class i also makes positions with cov_count==1 uncovered,
                # UNLESS they're in the new class's range.
                
                # Net score change = (# positions where cov_count==1 and NOT in r_new::m_new) 
                #                   - (# positions where cov_count==0 and in r_new::m_new)
                
                # Let me compute this for each r_new:
                best_r_new = 0
                best_net = -10**9
                
                # Precompute: for each residue class mod m_new, count of cov_count==0 and cov_count==1
                # This is expensive if done naively. Let's just iterate over r_new.
                
                for r_new in range(m_new):
                    # Count positions in r_new::m_new where cov_count == 0 (become covered)
                    # Count positions in r_new::m_new where cov_count == 1 (stay covered)
                    slice_cov = cov_count[r_new::m_new]
                    n_zero = np.count_nonzero(slice_cov == 0)
                    n_one = np.count_nonzero(slice_cov == 1)
                    
                    # Total positions where cov_count==1 in the whole array
                    total_one = np.count_nonzero(cov_count == 1)
                    
                    # Positions where cov_count==1 and NOT in r_new::m_new:
                    # total_one - n_one
                    # These become uncovered when we remove class i
                    # Positions where cov_count==0 and in r_new::m_new: n_zero
                    # These become covered when we add new class
                    
                    net_change = (total_one - n_one) - n_zero
                    
                    if net_change < best_net:
                        best_net = net_change
                        best_r_new = r_new
                
                trial_score = current_score + best_net
                if trial_score < current_score:
                    # Apply the change
                    current_classes[i] = [m_new, best_r_new]
                    current_score = trial_score
                    used_moduli = set(m for m, r in current_classes)
                    
                    # Rebuild cov_count
                    cov_count = np.zeros(N, dtype=np.int32)
                    for m, r in current_classes:
                        cov_count[r::m] += 1
                    
                    improved = True
                    break  # Restart outer loop
    
    return current_classes, current_score

def construct():
    N = 51975
    divisors = _get_divisors(N)
    
    start_time = time.time()
    total_time_limit = 115.0  # Safety margin before 120s
    
    # Strategy 1: Standard greedy
    classes1, covered1 = _greedy_build(N, divisors, time_limit=50.0, start_time=start_time)
    score1 = _score(classes1, N)
    
    # Strategy 2: Greedy with different ordering (larger moduli first)
    elapsed = time.time() - start_time
    if elapsed < 80.0:
        divisors_desc = sorted(divisors, reverse=True)
        classes2, covered2 = _greedy_build(N, divisors_desc, time_limit=20.0, start_time=start_time)
        score2 = _score(classes2, N)
        if score2 < score1:
            classes1, score1 = classes2, score2
    
    # Strategy 3: Randomized greedy
    elapsed = time.time() - start_time
    if elapsed < 90.0:
        random.seed(42)
        divisors_shuffled = list(divisors)
        random.shuffle(divisors_shuffled)
        classes3, covered3 = _greedy_build(N, divisors_shuffled, time_limit=15.0, start_time=start_time)
        score3 = _score(classes3, N)
        if score3 < score1:
            classes1, score1 = classes3, score3
    
    # Local search on the best solution
    elapsed = time.time() - start_time
    if elapsed < total_time_limit - 2.0:
        local_time = min(20.0, total_time_limit - elapsed - 1.0)
        classes_final, score_final = _local_search_v2(classes1, N, divisors, time_limit=local_time, start_time=start_time)
        if score_final < score1:
            classes1, score1 = classes_final, score_final
    
    return classes1
