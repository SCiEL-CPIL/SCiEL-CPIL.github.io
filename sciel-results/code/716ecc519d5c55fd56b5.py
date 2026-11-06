import time
import random
from itertools import combinations

def is_sidon_fast(subset):
    """Check if subset is Sidon using sorted sums with early exit."""
    sums = []
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            sums.append(subset[i] + subset[j])
    sums.sort()
    for k in range(1, len(sums)):
        if sums[k] == sums[k-1]:
            return False
    return True

def max_sidon_size_fast(S):
    """Find largest Sidon subset size of S using backtracking with pruning."""
    S = sorted(S)
    n = len(S)
    
    # Try k from n down to 1
    for k in range(n, 0, -1):
        # Use DFS with pruning to find any Sidon subset of size k
        found = _find_sidon_subset(S, k, 0, [], set())
        if found:
            return k
    return 0

def _find_sidon_subset(S, k, start, current, used_sums):
    """DFS to find a Sidon subset of size k starting from index 'start'."""
    if len(current) == k:
        return True
    
    # Pruning: not enough elements left
    if len(S) - start < k - len(current):
        return False
    
    for i in range(start, len(S)):
        x = S[i]
        # Check all pair sums involving x with existing elements
        valid = True
        new_sums = []
        for j in range(len(current)):
            s = x + current[j]
            if s in used_sums:
                valid = False
                break
            new_sums.append(s)
        if not valid:
            continue
        s = x + x
        if s in used_sums:
            continue
        new_sums.append(s)
        
        # Add x and its new sums
        for s in new_sums:
            used_sums.add(s)
        current.append(x)
        
        if _find_sidon_subset(S, k, i + 1, current, used_sums):
            return True
        
        # Backtrack
        current.pop()
        for s in new_sums:
            used_sums.discard(s)
    
    return False

def has_sidon_subset_of_size_k(S, k):
    """Check if S contains a Sidon subset of size k using collision graph."""
    S = sorted(S)
    n = len(S)
    
    # Build collision graph: edge between i,j if x_i+x_j == x_p+x_q for some p,q (including diagonals)
    # Two elements i,j are "incompatible" if their pair sum collides with any other pair sum
    # A Sidon subset must be an independent set in this incompatibility graph
    
    # For efficiency, check all C(n,k) subsets for small k
    if k <= 4:
        for subset in combinations(range(n), k):
            if is_sidon_fast([S[i] for i in subset]):
                return True
        return False
    
    # For k=5, use smarter approach: check if any 5-tuple is Sidon
    # Use the fact that in a Sidon set, all pair sums are unique
    # So we can check by building the set of pair sums and looking for collisions
    
    for subset in combinations(range(n), k):
        vals = [S[i] for i in subset]
        sums = set()
        valid = True
        for i in range(k):
            for j in range(i, k):
                s = vals[i] + vals[j]
                if s in sums:
                    valid = False
                    break
                sums.add(s)
            if not valid:
                break
        if valid:
            return True
    return False

def count_sidon_collisions(S):
    """Count the number of pair-sum collisions in S.
    Higher means harder to find large Sidon subsets."""
    S = sorted(S)
    n = len(S)
    sum_count = {}
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            sum_count[s] = sum_count.get(s, 0) + 1
    # Count collisions: for each sum value that appears c times, there are C(c,2) colliding pairs
    collisions = 0
    for c in sum_count.values():
        if c > 1:
            collisions += c * (c - 1) // 2
    return collisions

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin: evaluator kills at 90s
    
    # Initial candidates: various structured sets designed to have many collisions
    candidates = []
    
    # Pure AP - many collisions
    candidates.append(list(range(12)))
    
    # AP with different step sizes
    for step in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        cand = [i * step for i in range(12)]
        if max(cand) <= 128:
            candidates.append(cand)
    
    # Geometric-like sequences
    candidates.append([1, 2, 4, 8, 16, 32, 64, 5, 10, 20, 40, 80])
    candidates.append([1, 2, 4, 8, 16, 32, 64, 3, 6, 12, 24, 48])
    
    # Clustered sets (many close together)
    candidates.append(list(range(6)) + list(range(60, 66)))
    candidates.append(list(range(4)) + list(range(20, 24)) + list(range(40, 44)) + list(range(60, 64)))
    
    # Very dense clusters
    candidates.append(list(range(8)) + [100, 101, 102, 103])
    candidates.append(list(range(6)) + [10, 11, 12, 13, 14, 15])
    candidates.append(list(range(10)) + [100, 101])
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 101])
    
    # Sets with many arithmetic progressions
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    candidates.append([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 100, 101, 102])
    candidates.append([0, 1, 2, 3, 4, 5, 100, 101, 102, 103, 104, 105])
    
    # Evaluate initial candidates
    best_set = candidates[0]
    best_score = max_sidon_size_fast(best_set)
    best_collisions = count_sidon_collisions(best_set)
    
    for cand in candidates[1:]:
        if time.time() > deadline:
            break
        score = max_sidon_size_fast(cand)
        coll = count_sidon_collisions(cand)
        if score < best_score or (score == best_score and coll > best_collisions):
            best_score = score
            best_set = cand
            best_collisions = coll
    
    # Random search with local improvement focused on breaking Sidon subsets
    random.seed(42)
    
    # First phase: random search to find good starting points
    for i in range(20000):
        if time.time() > deadline:
            break
        
        # Generate random candidate with bias toward clustering
        if random.random() < 0.3:
            # Clustered: pick a base and add small offsets
            base = random.randint(0, 100)
            candidate = set()
            for _ in range(12):
                if random.random() < 0.7:
                    candidate.add(base + random.randint(0, 10))
                else:
                    candidate.add(random.randint(0, 128))
            candidate = list(candidate)
            while len(candidate) < 12:
                candidate.append(random.randint(0, 128))
            candidate = list(set(candidate))[:12]
        else:
            candidate = random.sample(range(129), 12)
        
        if len(candidate) != 12:
            continue
            
        score = max_sidon_size_fast(candidate)
        coll = count_sidon_collisions(candidate)
        
        if score < best_score or (score == best_score and coll > best_collisions):
            best_score = score
            best_set = candidate
            best_collisions = coll
            if best_score <= 3:
                break
    
    # Second phase: targeted local search to break Sidon subsets
    if best_score > 3:
        for i in range(30000):
            if time.time() > deadline:
                break
            
            perturbed = list(best_set)
            
            # Strategy 1: Move one element to create a collision with an existing pair sum
            if random.random() < 0.5:
                idx = random.randint(0, 11)
                old_val = perturbed[idx]
                
                # Find a target value that creates a collision
                # We want new_val + x_j == x_p + x_q for some j, p, q
                # So new_val = x_p + x_q - x_j
                targets = set()
                for p in range(12):
                    for q in range(p, 12):
                        for j in range(12):
                            if j != p and j != q:
                                target = perturbed[p] + perturbed[q] - perturbed[j]
                                if 0 <= target <= 128 and target != old_val:
                                    targets.add(target)
                
                if targets:
                    new_val = random.choice(list(targets))
                    if new_val not in perturbed:
                        perturbed[idx] = new_val
                        perturbed.sort()
                else:
                    # Fallback: random move
                    new_val = random.randint(0, 128)
                    if new_val not in perturbed:
                        perturbed[idx] = new_val
                        perturbed.sort()
            
            # Strategy 2: Swap two elements
            elif random.random() < 0.3:
                idx1, idx2 = random.sample(range(12), 2)
                perturbed[idx1], perturbed[idx2] = perturbed[idx2], perturbed[idx1]
            
            # Strategy 3: Random single element change
            else:
                idx = random.randint(0, 11)
                new_val = random.randint(0, 128)
                if new_val not in perturbed:
                    perturbed[idx] = new_val
                    perturbed.sort()
            
            if len(set(perturbed)) != 12:
                continue
                
            p_score = max_sidon_size_fast(perturbed)
            p_coll = count_sidon_collisions(perturbed)
            
            if p_score < best_score or (p_score == best_score and p_coll > best_collisions):
                best_score = p_score
                best_set = perturbed
                best_collisions = p_coll
                if best_score <= 3:
                    break
    
    # Third phase: If still > 3, try more aggressive moves
    if best_score > 3:
        for i in range(20000):
            if time.time() > deadline:
                break
            
            # Try to find a 5-element Sidon subset and break it
            S = sorted(best_set)
            broken = False
            for subset in combinations(range(12), 5):
                vals = [S[i] for i in subset]
                if is_sidon_fast(vals):
                    # Found a Sidon 5-subset, try to break it by moving one of its elements
                    elem_idx = subset[random.randint(0, 4)]
                    perturbed = list(best_set)
                    
                    # Try to move this element to create a collision within the subset
                    # For the 5-element subset, we want to create a pair sum collision
                    other_indices = [i for i in subset if i != elem_idx]
                    targets = set()
                    for p in other_indices:
                        for q in other_indices:
                            if p < q:
                                target = S[p] + S[q] - S[elem_idx]
                                if 0 <= target <= 128 and target != S[elem_idx]:
                                    targets.add(target)
                    
                    # Also try to create collision with an element outside the subset
                    for p in range(12):
                        if p not in subset:
                            for q in range(12):
                                if q not in subset and p < q:
                                    target = S[p] + S[q] - S[elem_idx]
                                    if 0 <= target <= 128 and target != S[elem_idx]:
                                        targets.add(target)
                    
                    if targets:
                        new_val = random.choice(list(targets))
                        if new_val not in perturbed:
                            perturbed[elem_idx] = new_val
                            perturbed.sort()
                            broken = True
                            break
                    
                    if broken:
                        break
            
            if not broken:
                # Fallback: random perturbation
                perturbed = list(best_set)
                idx = random.randint(0, 11)
                new_val = random.randint(0, 128)
                if new_val not in perturbed:
                    perturbed[idx] = new_val
                    perturbed.sort()
            
            if len(set(perturbed)) != 12:
                continue
                
            p_score = max_sidon_size_fast(perturbed)
            p_coll = count_sidon_collisions(perturbed)
            
            if p_score < best_score or (p_score == best_score and p_coll > best_collisions):
                best_score = p_score
                best_set = perturbed
                best_collisions = p_coll
                if best_score <= 3:
                    break
    
    return best_set
