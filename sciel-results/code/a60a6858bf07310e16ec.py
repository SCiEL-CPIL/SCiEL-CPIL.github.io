"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _score_set(s):
    """
    Compute the number of distinct d > 0 such that there exist a, a+d, a+2d in s.
    """
    s_set = set(s)
    max_val = max(s_set) if s_set else 0
    arr = np.zeros(max_val + 1, dtype=bool)
    for x in s_set:
        arr[x] = True
    
    diffs_found = set()
    lst = sorted(s_set)
    n = len(lst)
    
    for i in range(n):
        for j in range(i+1, n):
            diff = lst[j] - lst[i]
            if diff % 2 == 0:
                mid = (lst[i] + lst[j]) // 2
                if arr[mid]:
                    diffs_found.add(diff // 2)
    
    return len(diffs_found)


def _score_set_bitset(s):
    """
    Fast scoring using bitset operations.
    """
    s_set = set(s)
    max_val = max(s_set)
    # Create a bitset where bit i is 1 if i in s
    # We'll use Python integers as arbitrary-precision bitsets
    bitset = 0
    for x in s_set:
        bitset |= (1 << x)
    
    diffs_found = set()
    for d in range(1, max_val // 2 + 1):
        # Check if there exists a such that a, a+d, a+2d are all in s
        # (bitset & (bitset >> d) & (bitset >> (2*d))) should have at least one bit set
        # But we need to be careful about the bit positions
        # If bitset has bit x set for x in s, then bitset >> d has bit x-d set for x in s with x >= d
        # So (bitset & (bitset >> d)) has bit a set if a and a+d are both in s
        # Then (bitset & (bitset >> d) & (bitset >> (2*d))) has bit a set if a, a+d, a+2d are all in s
        # We just need to check if this is non-zero
        if (bitset & (bitset >> d) & (bitset >> (2 * d))) != 0:
            diffs_found.add(d)
    
    return len(diffs_found)


def _construct_greedy_bitset():
    """
    Greedy construction: build the set element by element, choosing the element
    that maximizes the number of new progression differences.
    """
    # Start with a small seed set
    current = set()
    
    # Add elements one by one, choosing the one that adds the most new d's
    for _ in range(32):
        best_elem = None
        best_new_diffs = -1
        best_new_score = -1
        
        # Try all possible elements from 1 to 512
        for elem in range(1, 513):
            if elem in current:
                continue
            
            # Score the new set
            new_set = current | {elem}
            score = _score_set_bitset(new_set)
            
            if score > best_new_score:
                best_new_score = score
                best_elem = elem
        
        if best_elem is not None:
            current.add(best_elem)
    
    return list(current)


def _construct_iterated_local_search():
    """
    Iterated Local Search: repeatedly do local search, then perturb and repeat.
    """
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin
    
    random.seed(42)
    
    # Start with a good initial set
    # Dense interval gives many small differences
    current_set = list(range(1, 33))
    current_score = _score_set_bitset(current_set)
    
    best_set = list(current_set)
    best_score = current_score
    
    # Local search with multiple perturbations
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Local search: try all single-element swaps
        improved = True
        while improved:
            improved = False
            current_set_list = sorted(current_set)
            current_set = set(current_set_list)
            
            for i in range(len(current_set_list)):
                for new_elem in range(1, 513):
                    if new_elem in current_set:
                        continue
                    
                    new_set = current_set - {current_set_list[i]} | {new_elem}
                    new_score = _score_set_bitset(new_set)
                    
                    if new_score > current_score:
                        current_set = new_set
                        current_score = new_score
                        improved = True
                        if new_score > best_score:
                            best_score = new_score
                            best_set = list(current_set)
                        break  # Restart the outer loop after improvement
                if improved:
                    break
        
        # Perturbation: randomly change 3-5 elements
        num_perturb = random.randint(3, 5)
        current_set = set(best_set)
        for _ in range(num_perturb):
            to_remove = random.choice(list(current_set))
            current_set.remove(to_remove)
            new_elem = random.randint(1, 512)
            while new_elem in current_set:
                new_elem = random.randint(1, 512)
            current_set.add(new_elem)
        
        current_score = _score_set_bitset(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = list(current_set)
    
    return best_set


def _construct_hybrid():
    """
    Hybrid approach: combine greedy construction with local search refinement.
    """
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin
    
    random.seed(99)
    
    # Phase 1: Greedy construction
    greedy_set = _construct_greedy_bitset()
    greedy_score = _score_set_bitset(greedy_set)
    
    # Phase 2: Local search refinement
    current_set = set(greedy_set)
    current_score = greedy_score
    best_set = list(greedy_set)
    best_score = greedy_score
    
    # Also try starting from a dense interval
    interval_set = list(range(1, 33))
    interval_score = _score_set_bitset(interval_set)
    if interval_score > best_score:
        best_set = list(interval_set)
        best_score = interval_score
        current_set = set(interval_set)
        current_score = interval_score
    
    # Iterated local search
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Local search
        improved = True
        while improved:
            improved = False
            current_set_list = sorted(current_set)
            
            for i in range(len(current_set_list)):
                removed = current_set_list[i]
                for new_elem in range(1, 513):
                    if new_elem in current_set:
                        continue
                    
                    new_set = current_set - {removed} | {new_elem}
                    new_score = _score_set_bitset(new_set)
                    
                    if new_score > current_score:
                        current_set = new_set
                        current_score = new_score
                        improved = True
                        if new_score > best_score:
                            best_score = new_score
                            best_set = list(current_set)
                        break
                if improved:
                    break
        
        # Perturbation
        if time.time() > deadline:
            break
        
        num_perturb = random.randint(2, 4)
        current_set = set(best_set)
        for _ in range(num_perturb):
            to_remove = random.choice(list(current_set))
            current_set.remove(to_remove)
            new_elem = random.randint(1, 512)
            while new_elem in current_set:
                new_elem = random.randint(1, 512)
            current_set.add(new_elem)
        
        current_score = _score_set_bitset(current_set)
    
    return best_set


def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin
    
    # Try hybrid approach
    result = _construct_hybrid()
    
    return result

# EVOLVE-BLOCK-END
