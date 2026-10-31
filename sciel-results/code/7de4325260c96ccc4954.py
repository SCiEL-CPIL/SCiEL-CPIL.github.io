"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline for 150s limit
    
    def compute_score(U):
        """Compute the score for set U."""
        if not U or 0 not in U:
            return -1.0
        max_u = max(U)
        if max_u == 0:
            return 1.0  # log(1/1)/log(1) is undefined, treat as 1.0
        U_set = set(U)
        # Compute U+U
        sumset = set()
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
        # Compute U-U
        diffset = set()
        for a in U_set:
            for b in U_set:
                diffset.add(a - b)
        n_sum = len(sumset)
        n_diff = len(diffset)
        if n_sum == 0:
            return -1.0
        ratio = n_diff / n_sum
        if ratio <= 0:
            return -1.0
        denom = math.log(2 * max_u + 1)
        if denom <= 0:
            return 1.0
        return 1.0 + math.log(ratio) / denom
    
    # Start with a reasonable initial set
    # Use a geometric-like progression which tends to have good diff/sum ratio
    best_set = [0, 1, 3, 7, 15, 31, 63, 127]
    best_score = compute_score(best_set)
    
    # Also try some other initial sets
    init_sets = [
        [0, 1, 3, 9, 27, 81, 243, 729],
        [0, 1, 2, 4, 8, 16, 32, 64],
        [0, 1, 3, 7, 15, 31, 63, 127, 255],
        [0, 2, 5, 12, 29, 71, 175],
    ]
    for s in init_sets:
        sc = compute_score(s)
        if sc > best_score:
            best_score = sc
            best_set = list(s)
    
    # Local search: try adding, removing, or replacing elements
    random.seed(42)
    
    def try_add(U, candidate):
        if candidate in U or candidate < 0:
            return None
        new_U = sorted(U + [candidate])
        return new_U
    
    def try_remove(U, idx):
        if len(U) <= 2:
            return None
        new_U = U[:idx] + U[idx+1:]
        return new_U
    
    def try_replace(U, idx, candidate):
        if candidate in U or candidate < 0:
            return None
        new_U = U[:idx] + [candidate] + U[idx+1:]
        return sorted(new_U)
    
    current_set = list(best_set)
    current_score = best_score
    no_improve_count = 0
    
    while time.time() < deadline:
        if no_improve_count > 200:
            # Reset with a new random perturbation
            current_set = list(best_set)
            current_score = best_score
            no_improve_count = 0
            # Perturb
            if random.random() < 0.5:
                # Add a random element
                max_u = max(current_set)
                candidate = random.randint(1, max_u * 2 + 10)
                new_set = try_add(current_set, candidate)
                if new_set:
                    current_set = new_set
                    current_score = compute_score(current_set)
            else:
                # Remove a random element
                idx = random.randint(1, len(current_set) - 1)  # Don't remove 0
                new_set = try_remove(current_set, idx)
                if new_set:
                    current_set = new_set
                    current_score = compute_score(current_set)
            continue
        
        # Try various moves
        max_u = max(current_set)
        improved = False
        
        # Try adding elements in a range
        # Focus on elements that could improve the ratio
        candidates_to_add = set()
        # Elements near max_u * 2 (to expand diffset without too much sumset growth)
        candidates_to_add.add(max_u * 2)
        candidates_to_add.add(max_u * 2 + 1)
        candidates_to_add.add(max_u * 2 + 2)
        # Elements in between
        for i in range(max(1, max_u // 2), max_u + 1):
            candidates_to_add.add(i)
        # Some random candidates
        for _ in range(5):
            candidates_to_add.add(random.randint(1, max_u * 3))
        
        for cand in candidates_to_add:
            if cand < 0 or cand in current_set:
                continue
            new_set = try_add(current_set, cand)
            if new_set is None:
                continue
            sc = compute_score(new_set)
            if sc > current_score:
                current_set = new_set
                current_score = sc
                improved = True
                if sc > best_score:
                    best_score = sc
                    best_set = list(current_set)
                break
        
        if improved:
            no_improve_count = 0
            continue
        
        # Try removing elements (except 0)
        for idx in range(1, len(current_set)):
            new_set = try_remove(current_set, idx)
            if new_set is None:
                continue
            sc = compute_score(new_set)
            if sc > current_score:
                current_set = new_set
                current_score = sc
                improved = True
                if sc > best_score:
                    best_score = sc
                    best_set = list(current_set)
                break
        
        if improved:
            no_improve_count = 0
            continue
        
        # Try replacing elements
        for idx in range(1, len(current_set)):
            old_val = current_set[idx]
            # Try nearby values
            for delta in [-3, -2, -1, 1, 2, 3]:
                new_val = old_val + delta
                if new_val < 0 or new_val in current_set:
                    continue
                new_set = try_replace(current_set, idx, new_val)
                if new_set is None:
                    continue
                sc = compute_score(new_set)
                if sc > current_score:
                    current_set = new_set
                    current_score = sc
                    improved = True
                    if sc > best_score:
                        best_score = sc
                        best_set = list(current_set)
                    break
            if improved:
                break
        
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    # Return the best set found
    return sorted(best_set)
# EVOLVE-BLOCK-END
