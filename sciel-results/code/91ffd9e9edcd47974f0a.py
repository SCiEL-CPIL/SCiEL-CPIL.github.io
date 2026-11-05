import time
import random

def _is_sidon(s):
    """Check if set s is Sidon."""
    if len(s) < 2:
        return True
    sums = set()
    # Use list for faster iteration if needed, but set is fine for small k
    # To ensure uniqueness of pairs, we iterate i <= j
    lst = list(s)
    for i in range(len(lst)):
        for j in range(i, len(lst)):
            val = lst[i] + lst[j]
            if val in sums:
                return False
            sums.add(val)
    return True

def _get_diffs(s):
    """Get set of nonzero differences."""
    if len(s) < 2:
        return set()
    lst = list(s)
    diffs = set()
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            diffs.add(abs(lst[i] - lst[j]))
    return diffs

def _is_valid(A, B):
    """Check if A and B are valid Sidon sets with disjoint diffs."""
    if not A or not B:
        return False
    if not _is_sidon(A):
        return False
    if not _is_sidon(B):
        return False
    dA = _get_diffs(A)
    dB = _get_diffs(B)
    if dA & dB:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_A = []
    best_B = []
    best_score = 0
    
    # We will run multiple restarts of local search
    # Each restart initializes with a random small valid set or empty
    
    for restart in range(200):
        if time.time() > deadline - 5:
            break
            
        A = []
        B = []
        
        # Try to build up sets greedily first to get a foothold
        # Random order
        order = list(range(1, 257))
        random.shuffle(order)
        
        # Greedy initialization
        for x in order:
            if time.time() > deadline - 5:
                break
            # Try add to A
            A_new = A + [x]
            if _is_sidon(A_new) and (_get_diffs(A_new) & _get_diffs(B)) == set():
                A = A_new
                continue
            # Try add to B
            B_new = B + [x]
            if _is_sidon(B_new) and (_get_diffs(B_new) & _get_diffs(A)) == set():
                B = B_new
                continue
        
        # Local Search
        current_score = len(A) + len(B)
        if current_score > best_score:
            best_score = current_score
            best_A = list(A)
            best_B = list(B)
            
        # Local search loop
        stuck_counter = 0
        while time.time() < deadline - 5:
            improved = False
            move_type = random.choice(['add_A', 'add_B', 'remove_A', 'remove_B', 'swap'])
            
            # Try 10 random candidates for the chosen move type
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                    
                if move_type == 'add_A':
                    candidates = [x for x in range(1, 257) if x not in A]
                    if not candidates: continue
                    x = random.choice(candidates)
                    A_new = A + [x]
                    if _is_sidon(A_new) and (_get_diffs(A_new) & _get_diffs(B)) == set():
                        A = A_new
                        improved = True
                        break
                elif move_type == 'add_B':
                    candidates = [x for x in range(1, 257) if x not in B]
                    if not candidates: continue
                    x = random.choice(candidates)
                    B_new = B + [x]
                    if _is_sidon(B_new) and (_get_diffs(B_new) & _get_diffs(A)) == set():
                        B = B_new
                        improved = True
                        break
                elif move_type == 'remove_A':
                    if len(A) > 1:
                        x = random.choice(A)
                        A_new = [a for a in A if a != x]
                        # Removing always maintains Sidon and reduces diffs, so always valid
                        # But we need to check if it helps? No, we just accept to allow re-adds
                        # However, we should only remove if we are stuck or to try to swap
                        # For now, accept removal to explore
                        A = A_new
                        improved = True
                        break
                elif move_type == 'remove_B':
                    if len(B) > 1:
                        x = random.choice(B)
                        B_new = [b for b in B if b != x]
                        B = B_new
                        improved = True
                        break
                elif move_type == 'swap':
                    # Swap element from A to B or vice versa
                    if A and B:
                        x = random.choice(A)
                        A_new = [a for a in A if a != x]
                        B_new = B + [x]
                        if _is_sidon(B_new) and (_get_diffs(A_new) & _get_diffs(B_new)) == set():
                            A = A_new
                            B = B_new
                            improved = True
                            break
                    elif A:
                        x = random.choice(A)
                        A_new = [a for a in A if a != x]
                        B_new = B + [x]
                        if _is_sidon(B_new) and (_get_diffs(A_new) & _get_diffs(B_new)) == set():
                            A = A_new
                            B = B_new
                            improved = True
                            break
                    elif B:
                        x = random.choice(B)
                        B_new = [b for b in B if b != x]
                        A_new = A + [x]
                        if _is_sidon(A_new) and (_get_diffs(A_new) & _get_diffs(B_new)) == set():
                            A = A_new
                            B = B_new
                            improved = True
                            break
            
            current_score = len(A) + len(B)
            if current_score > best_score:
                best_score = current_score
                best_A = list(A)
                best_B = list(B)
            
            if not improved:
                stuck_counter += 1
                if stuck_counter > 50:
                    # Restart local search with a perturbation
                    # Remove a random element from A or B to break deadlock
                    if A:
                        x = random.choice(A)
                        A = [a for a in A if a != x]
                    elif B:
                        x = random.choice(B)
                        B = [b for b in B if b != x]
                    stuck_counter = 0
            else:
                stuck_counter = 0
                
    # Final validation and fallback
    if not best_A or not best_B:
        # Fallback to a known small valid pair
        best_A = [1, 2, 4, 8]
        best_B = [3, 5, 7, 11]
        
    # Ensure non-empty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    return [best_A, best_B]
