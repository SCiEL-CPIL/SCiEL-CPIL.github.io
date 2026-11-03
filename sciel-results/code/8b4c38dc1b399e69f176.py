import time
import math
import random
from itertools import combinations

def _score(U):
    """Compute the score for a given set U (as a sorted list)."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0  # degenerate case
    
    # Compute U+U
    sum_set = set()
    for a in U_set:
        for b in U_set:
            sum_set.add(a + b)
    
    # Compute U-U
    diff_set = set()
    for a in U_set:
        for b in U_set:
            diff_set.add(a - b)
    
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 1:
        return 1.0
    
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
    
    return 1.0 + log_ratio / log_denom

def _score_fast(U):
    """Faster score using numpy for larger sets."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    U_list = sorted(U_set)
    
    # Compute U+U using set operations
    sum_set = set()
    for i, a in enumerate(U_list):
        for b in U_list[i:]:  # Only need a<=b since addition is commutative
            sum_set.add(a + b)
    
    # Compute U-U
    diff_set = set()
    for a in U_list:
        for b in U_list:
            diff_set.add(a - b)
    
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 1:
        return 1.0
    
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
    
    return 1.0 + log_ratio / log_denom

def _local_search(U, deadline, max_iterations=5000):
    """Local search: try adding, removing, or swapping elements."""
    U_set = set(U)
    current_score = _score_fast(U)
    best_set = list(U_set)
    best_score = current_score
    max_u = max(U_set)
    
    # Define candidate elements to consider
    # Range around current max, plus some random ones
    candidates = set()
    for i in range(1, max_u + 10):
        candidates.add(i)
    # Add some larger candidates
    for i in range(max_u + 1, max_u + 50):
        candidates.add(i)
    
    candidates.discard(0)  # 0 must stay
    candidates_list = sorted(candidates)
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try adding an element
        for c in random.sample(candidates_list, min(20, len(candidates_list))):
            if c in U_set:
                continue
            new_set = U_set | {c}
            s = _score_fast(list(new_set))
            if s > best_score + 1e-9:
                best_score = s
                best_set = list(new_set)
                U_set = new_set
                improved = True
                # Update candidates to include neighbors
                for delta in [-2, -1, 1, 2]:
                    nc = c + delta
                    if nc > 0:
                        candidates.add(nc)
                break
        
        if improved:
            continue
        
        # Try removing an element (not 0)
        removable = [x for x in U_set if x != 0]
        if removable:
            for c in random.sample(removable, min(10, len(removable))):
                new_set = U_set - {c}
                s = _score_fast(list(new_set))
                if s > best_score + 1e-9:
                    best_score = s
                    best_set = list(new_set)
                    U_set = new_set
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try swapping
        if len(U_set) >= 2:
            removable = [x for x in U_set if x != 0]
            if removable:
                c_remove = random.choice(removable)
                c_add = random.choice(candidates_list)
                if c_add not in U_set:
                    new_set = (U_set - {c_remove}) | {c_add}
                    s = _score_fast(list(new_set))
                    if s > best_score + 1e-9:
                        best_score = s
                        best_set = list(new_set)
                        U_set = new_set
                        improved = True
        
        if not improved:
            # Occasionally add a random element to escape local optima
            if random.random() < 0.1:
                c = random.choice(candidates_list)
                if c not in U_set:
                    new_set = U_set | {c}
                    s = _score_fast(list(new_set))
                    if s > best_score * 0.95:  # Accept slightly worse to explore
                        best_score = max(best_score, s)
                        best_set = list(new_set)
                        U_set = new_set
    
    return best_set, best_score

def _golomb_rulers():
    """Generate some known Golomb ruler constructions."""
    rulers = []
    
    # Small optimal Golomb rulers
    # n=2: [0,1]
    rulers.append([0, 1])
    # n=3: [0,1,3]
    rulers.append([0, 1, 3])
    # n=4: [0,1,4,6]
    rulers.append([0, 1, 4, 6])
    # n=5: [0,1,4,9,11]
    rulers.append([0, 1, 4, 9, 11])
    # n=6: [0,1,4,10,12,17]
    rulers.append([0, 1, 4, 10, 12, 17])
    # n=7: [0,1,4,10,18,23,29]
    rulers.append([0, 1, 4, 10, 18, 23, 29])
    # n=8: [0,1,4,10,18,23,29,38]
    rulers.append([0, 1, 4, 10, 18, 23, 29, 38])
    
    # Construct longer rulers using greedy method
    for target_size in range(9, 20):
        ruler = [0]
        current_max = 0
        while len(ruler) < target_size:
            best_pos = None
            best_min_diff = 0
            for pos in range(1, current_max + target_size):
                # Check if pos creates a valid Golomb ruler
                valid = True
                for existing in ruler:
                    d = abs(pos - existing)
                    # Check if this difference already exists
                    for other in ruler:
                        if other != existing and abs(other - pos) == d:
                            valid = False
                            break
                        if other != existing and abs(other - existing) == d:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    # Compute minimum difference to existing elements
                    min_d = min(abs(pos - e) for e in ruler)
                    if min_d > best_min_diff:
                        best_min_diff = min_d
                        best_pos = pos
            if best_pos is None:
                break
            ruler.append(best_pos)
            current_max = max(ruler)
        if len(ruler) >= 5:
            rulers.append(sorted(ruler))
    
    return rulers

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    best_set = [0]
    best_score = 0.0
    
    # Try known Golomb rulers
    for ruler in _golomb_rulers():
        if time.time() > deadline:
            break
        s = _score_fast(ruler)
        if s > best_score:
            best_score = s
            best_set = ruler[:]
    
    # Try powers of 2
    for n in range(2, 15):
        if time.time() > deadline:
            break
        U = [2**i for i in range(n)]
        s = _score_fast(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Try geometric sequences with different bases
    for base in [2, 3, 4, 5]:
        for n in range(3, 12):
            if time.time() > deadline:
                break
            U = [int(base**i) for i in range(n)]
            s = _score_fast(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Local search from best found so far
    if time.time() < deadline - 10:
        best_set, best_score = _local_search(best_set, deadline)
    
    # Try a few random starting points and local search
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        num_random_starts = min(5, int(remaining_time / 15))
        for _ in range(num_random_starts):
            if time.time() > deadline:
                break
            # Random set
            size = random.randint(5, 15)
            max_val = random.randint(20, 100)
            U = [0] + random.sample(range(1, max_val + 1), min(size - 1, max_val))
            U.sort()
            s = _score_fast(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
            
            if time.time() < deadline - 5:
                best_set2, best_score2 = _local_search(U, deadline - 2)
                if best_score2 > best_score:
                    best_score = best_score2
                    best_set = best_set2
    
    return best_set
