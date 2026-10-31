"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

def _compute_score(U):
    """Compute the score for a set U."""
    U = sorted(set(U))
    if not U or U[0] != 0:
        return -1
    n = len(U)
    if n < 2:
        return -1
    max_u = max(U)
    if max_u == 0:
        return -1
    
    # Compute |U+U| and |U-U|
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    
    sum_size = len(sum_set)
    diff_size = len(diff_set)
    
    if sum_size == 0:
        return -1
    
    ratio = diff_size / sum_size
    if ratio <= 0:
        return -1
    
    log_term = math.log(ratio) / math.log(2 * max_u + 1)
    return 1.0 + log_term

def construct():
    """Construct a finite set U of nonnegative integers containing 0 to maximize the score."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_set = [0, 1, 3]
    best_score = _compute_score(best_set)
    
    # Try various structured families
    # Family 1: Geometric-like sequences
    for ratio in [2, 3, 4, 5, 7, 10]:
        for max_len in range(3, 15):
            if time.time() > deadline:
                break
            U = []
            val = 0
            U.append(val)
            val = 1
            for i in range(max_len - 1):
                if val > 100000:
                    break
                U.append(val)
                val *= ratio
            if len(U) >= 3:
                s = _compute_score(U)
                if s > best_score:
                    best_score = s
                    best_set = list(U)
    
    # Family 2: Sparse arithmetic + perturbation
    for base in [10, 20, 50, 100, 200]:
        for n in range(3, 20):
            if time.time() > deadline:
                break
            # U = {i * base + offset_i}
            U = [0]
            for i in range(1, n):
                U.append(i * base + (i % 3))
            if len(set(U)) == len(U):
                s = _compute_score(U)
                if s > best_score:
                    best_score = s
                    best_set = list(U)
    
    # Family 3: Random local search
    random.seed(42)
    
    def _local_search(start_U, max_iters=500):
        """Local search by adding, removing, or perturbing elements."""
        U = sorted(set(start_U))
        if 0 not in U:
            U = [0] + U
            U = sorted(U)
        current_score = _compute_score(U)
        if current_score < 0:
            return U, -1
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            # Try adding a random element
            max_u = max(U) if U else 0
            candidates = []
            
            # Try adding elements near existing ones or at new positions
            for _ in range(20):
                if random.random() < 0.5:
                    # Near existing
                    if U:
                        base = random.choice(U)
                        new_val = base + random.randint(1, max(1, max_u // 5))
                    else:
                        new_val = random.randint(1, 100)
                else:
                    # New position
                    new_val = random.randint(1, max(10, max_u * 2))
                
                if new_val > 0 and new_val not in U:
                    candidates.append(new_val)
            
            # Try removing an element (not 0)
            removable = [x for x in U if x != 0]
            
            # Try perturbing an element
            perturbable = [x for x in U if x != 0]
            
            best_candidate_score = current_score
            best_candidate_set = None
            
            # Try additions
            for cand in candidates[:10]:
                new_U = sorted(U + [cand])
                s = _compute_score(new_U)
                if s > best_candidate_score:
                    best_candidate_score = s
                    best_candidate_set = new_U
                    improved = True
            
            # Try removals
            if removable:
                for _ in range(min(5, len(removable))):
                    idx = random.choice(removable)
                    new_U = [x for x in U if x != idx]
                    if len(new_U) >= 2:
                        s = _compute_score(new_U)
                        if s > best_candidate_score:
                            best_candidate_score = s
                            best_candidate_set = new_U
                            improved = True
            
            # Try perturbations
            if perturbable:
                for _ in range(min(5, len(perturbable))):
                    old_val = random.choice(perturbable)
                    delta = random.randint(-3, 3)
                    new_val = old_val + delta
                    if new_val > 0 and new_val not in U:
                        new_U = sorted([x if x != old_val else new_val for x in U])
                        if len(set(new_U)) == len(new_U):
                            s = _compute_score(new_U)
                            if s > best_candidate_score:
                                best_candidate_score = s
                                best_candidate_set = new_U
                                improved = True
            
            if improved and best_candidate_set is not None:
                U = best_candidate_set
                current_score = best_candidate_score
            else:
                # No improvement, maybe restart with perturbation
                if random.random() < 0.1:
                    # Perturb randomly
                    new_U = list(U)
                    if len(new_U) > 2:
                        idx = random.randint(1, len(new_U) - 1)
                        new_U[idx] = max(1, new_U[idx] + random.randint(-5, 5))
                        new_U = sorted(set(new_U))
                        if 0 not in new_U:
                            new_U = [0] + new_U
                            new_U = sorted(new_U)
                        if len(new_U) >= 2:
                            U = new_U
                            current_score = _compute_score(U)
        
        return U, current_score
    
    # Run local search from different starting points
    start_points = [
        [0, 1, 3],
        [0, 1, 4],
        [0, 2, 5],
        [0, 1, 2, 5],
        [0, 1, 3, 7],
        [0, 1, 4, 9],
        [0, 2, 4, 8],
        [0, 1, 3, 6, 10],
        [0, 1, 2, 4, 8, 16],
        [0, 1, 3, 7, 15, 31],
    ]
    
    for sp in start_points:
        if time.time() > deadline:
            break
        U, s = _local_search(sp, max_iters=200)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Final polish: try small perturbations on best set
    if time.time() < deadline - 5:
        for _ in range(100):
            if time.time() > deadline:
                break
            U = list(best_set)
            # Try adding one element
            max_u = max(U) if U else 0
            for delta in [1, 2, 3, 5, 10, max_u // 2, max_u + 1]:
                new_val = max_u + delta
                if new_val > 0 and new_val not in U:
                    new_U = sorted(U + [new_val])
                    s = _compute_score(new_U)
                    if s > best_score:
                        best_score = s
                        best_set = new_U
                        break
            # Try removing one element
            removable = [x for x in U if x != 0]
            if removable:
                for x in removable[:5]:
                    new_U = [y for y in U if y != x]
                    if len(new_U) >= 2:
                        s = _compute_score(new_U)
                        if s > best_score:
                            best_score = s
                            best_set = new_U
    
    return sorted(set(best_set))
# EVOLVE-BLOCK-END
