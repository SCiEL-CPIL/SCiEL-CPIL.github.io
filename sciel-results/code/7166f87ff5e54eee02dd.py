"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import itertools
import time
from collections import defaultdict


def _is_sidon(subset):
    """Check if a list of distinct ints is a Sidon set (pair sums unique, incl. diagonals)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _has_sidon_subset(A, k):
    """Check if A contains a Sidon subset of size k. Returns the subset if found, else None."""
    for subset in itertools.combinations(A, k):
        if _is_sidon(list(subset)):
            return list(subset)
    return None


def _count_sidon_subsets(A, k):
    """Count the number of Sidon subsets of size k in A."""
    count = 0
    for subset in itertools.combinations(A, k):
        if _is_sidon(list(subset)):
            count += 1
    return count


def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill
    
    best_set = list(range(12))
    best_max_size = _compute_max_sidon_size(best_set)
    
    # Strategy: Use structured initial sets with many sum collisions
    # Then local search to eliminate larger Sidon subsets
    
    # Generate candidate initial sets with high collision density
    initial_candidates = []
    
    # Arithmetic progressions (many sum collisions)
    for start in range(0, 120, 3):
        for step in [2, 3, 4, 5]:
            A = [start + i * step for i in range(12)]
            if all(0 <= x <= 128 for x in A) and len(set(A)) == 12:
                initial_candidates.append(A)
    
    # Dense clusters in small ranges
    for base in range(0, 100, 5):
        A = list(range(base, base + 12))
        initial_candidates.append(A)
    
    # Random dense sets in small ranges
    for _ in range(50):
        low = random.randint(0, 40)
        high = random.randint(low + 12, min(low + 30, 128))
        A = random.sample(range(low, high + 1), 12)
        initial_candidates.append(A)
    
    # Evaluate initial candidates
    for A in initial_candidates:
        if time.time() > deadline:
            break
        A = list(set(A))
        if len(A) != 12:
            continue
        max_sz = _compute_max_sidon_size(A)
        if max_sz < best_max_size:
            best_max_size = max_sz
            best_set = A[:]
        if max_sz <= 3:
            return A
    
    # Local search from best candidate
    A = best_set[:]
    
    for restart in range(200):
        if time.time() > deadline - 5:
            break
        
        # If starting from scratch, pick a random dense set
        if restart > 0 and restart % 10 == 0:
            low = random.randint(0, 50)
            high = random.randint(low + 12, min(low + 35, 128))
            A = random.sample(range(low, high + 1), 12)
        
        # Local search: find and break Sidon subsets
        for step in range(500):
            if time.time() > deadline - 5:
                    break
            
            # Check for Sidon subset of size 5 first
            sidon5 = _has_sidon_subset(A, 5)
            if sidon5 is None:
                # Check for size 4
                sidon4 = _has_sidon_subset(A, 4)
                if sidon4 is None:
                    # Max Sidon size is at most 3
                    max_sz = _compute_max_sidon_size(A)
                    if max_sz < best_max_size:
                        best_max_size = max_sz
                        best_set = A[:]
                    if max_sz <= 3:
                        return A
                    # Try to improve further
                    break
                
                target_subset = sidon4
                target_k = 4
            else:
                target_subset = sidon5
                target_k = 5
            
            # Replace an element from the Sidon subset
            elem_to_replace = random.choice(target_subset)
            idx = A.index(elem_to_replace)
            
            # Try multiple replacement candidates
            best_new_val = None
            best_improvement = 0
            
            available = [x for x in range(129) if x not in A]
            random.shuffle(available)
            
            for new_val in available[:20]:
                A_test = A[:]
                A_test[idx] = new_val
                
                # Quick check: does this break the target Sidon subset?
                if not _is_sidon([x for x in target_subset if x != elem_to_replace] + [new_val]):
                    # Check if we reduced max Sidon size
                    # Count Sidon subsets of size target_k
                    old_count = _count_sidon_subsets(A, target_k)
                    new_count = _count_sidon_subsets(A_test, target_k)
                    if new_count < old_count:
                        improvement = old_count - new_count
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_new_val = new_val
            
            if best_new_val is not None:
                A[idx] = best_new_val
            else:
                # No improvement found, try random move
                if available:
                    A[idx] = random.choice(available)
        
        # After local search, evaluate
        max_sz = _compute_max_sidon_size(A)
        if max_sz < best_max_size:
            best_max_size = max_sz
            best_set = A[:]
            if max_sz <= 3:
                return A
    
    return best_set


def _compute_max_sidon_size(A):
    """Compute the maximum size of a Sidon subset of A."""
    # Check from largest possible down
    for k in range(len(A), 1, -1):
        if _has_sidon_subset(A, k) is not None:
            return k
    return 2  # Any two distinct elements form a Sidon set (sums: a+a, a+b, b+b are all different if a!=b)
# EVOLVE-BLOCK-END
