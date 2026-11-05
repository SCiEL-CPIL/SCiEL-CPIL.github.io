import time
import random
from math import gcd

def construct():
    """
    Returns a nonempty subset of [1, 128] with no three distinct elements 
    whose three pairwise gcds are equal. Maximizes cardinality.
    """
    deadline = time.time() + 85.0  # Leave 5s margin
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute gcd matrix for fast access
    # gcd_matrix[i][j] = gcd(i+1, j+1)
    gcd_matrix = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            gcd_matrix[i][j] = gcd(i + 1, j + 1)
            
    def is_bad_triple(i, j, k):
        """Check if elements at indices i, j, k form a bad triple."""
        g_ij = gcd_matrix[i][j]
        g_ik = gcd_matrix[i][k]
        g_jk = gcd_matrix[j][k]
        return g_ij == g_ik == g_jk

    def count_bad_triples_with_new(S_set, new_idx):
        """
        Count how many bad triples are formed by adding element at index new_idx 
        to the set S_set (which does not contain new_idx).
        S_set is a set of indices.
        """
        count = 0
        # We need pairs (a, b) in S_set such that {new_idx, a, b} is a bad triple
        # Iterate over all pairs in S_set? That's O(|S|^2).
        # |S| can be up to 128. 128^2 = 16384. This is acceptable per step.
        list_S = list(S_set)
        n = len(list_S)
        for a_idx in range(n):
            a = list_S[a_idx]
            for b_idx in range(a_idx + 1, n):
                b = list_S[b_idx]
                if is_bad_triple(new_idx, a, b):
                    count += 1
        return count

    def calculate_total_bad_triples(S_set):
        """Calculate total number of bad triples in S_set."""
        list_S = list(S_set)
        n = len(list_S)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if is_bad_triple(list_S[i], list_S[j], list_S[k]):
                        count += 1
        return count

    # Initialize with a random set or greedy
    # Let's start with a set of size ~60-80
    initial_size = 70
    current_S = set(random.sample(range(N), initial_size))
    
    # Precompute current bad triples count
    current_bad_count = calculate_total_bad_triples(current_S)
    
    # Penalty factor M. We want to prioritize eliminating bad triples.
    # Max possible bad triples is C(128,3) ~ 349k.
    # If we have 1 bad triple, P = M - |S|.
    # If we have 0 bad triples, P = -|S|.
    # We want 0 bad triples to be always better than 1 bad triple, even if |S| is max.
    # So M > max possible |S| difference. M=1000 is enough. Let's use 10000 to be safe.
    M = 10000
    
    def get_penalty(S_set, bad_count):
        return bad_count * M - len(S_set)

    current_penalty = get_penalty(current_S, current_bad_count)
    
    best_S = set(current_S)
    best_penalty = current_penalty
    if current_bad_count == 0:
        best_valid_size = len(current_S)
    else:
        best_valid_size = 0

    # Simulated Annealing
    T = 100.0  # Initial temperature
    cooling = 0.9995  # Cooling rate
    min_T = 1e-6
    
    # We will run until time limit or temperature is very low
    # Then restart with new random seed or perturbation if needed
    
    while time.time() < deadline:
        if T < min_T:
            # Restart with a new random set or perturbation of best found
            # Let's try to build a greedy set from scratch or random
            if random.random() < 0.5:
                # Random restart
                size = random.randint(50, 100)
                current_S = set(random.sample(range(N), size))
                current_bad_count = calculate_total_bad_triples(current_S)
            else:
                # Perturb best valid if we have one
                if best_valid_size > 0:
                    current_S = set(best_S)
                    # Randomly remove a few elements
                    to_remove = random.sample(list(current_S), min(5, len(current_S)))
                    for x in to_remove:
                        current_S.remove(x)
                    current_bad_count = calculate_total_bad_triples(current_S)
                else:
                    # No valid set found yet, random restart
                    size = random.randint(50, 100)
                    current_S = set(random.sample(range(N), size))
                    current_bad_count = calculate_total_bad_triples(current_S)
            
            current_penalty = get_penalty(current_S, current_bad_count)
            T = 100.0
            continue

        # Pick a random element to toggle
        idx = random.randint(0, N - 1)
        
        if idx in current_S:
            # Remove idx
            # Calculate delta in bad count
            # Bad triples removed are those containing idx
            # We need to count pairs (a,b) in current_S \ {idx} such that {idx, a, b} is bad
            # This is the same as count_bad_triples_with_new but on S without idx
            S_without_idx = current_S - {idx}
            delta_bad = -count_bad_triples_with_new(S_without_idx, idx)
            # Note: count_bad_triples_with_new counts triples formed by adding idx to S_without_idx.
            # These are exactly the triples containing idx in current_S.
            
            delta_size = -1
            delta_penalty = delta_bad * M - delta_size
            
            if delta_penalty < 0 or random.random() < pow(2.71828, -delta_penalty / T):
                current_S.remove(idx)
                current_bad_count += delta_bad
                current_penalty += delta_penalty
                
                if current_bad_count == 0:
                    if len(current_S) > best_valid_size:
                        best_valid_size = len(current_S)
                        best_S = set(current_S)
                # Update best penalty if lower (for tracking, though we care about valid sets)
                if current_penalty < best_penalty:
                    best_penalty = current_penalty
        else:
            # Add idx
            # Calculate delta in bad count
            # Bad triples added are those containing idx
            delta_bad = count_bad_triples_with_new(current_S, idx)
            
            delta_size = 1
            delta_penalty = delta_bad * M - delta_size
            
            if delta_penalty < 0 or random.random() < pow(2.71828, -delta_penalty / T):
                current_S.add(idx)
                current_bad_count += delta_bad
                current_penalty += delta_penalty
                
                if current_bad_count == 0:
                    if len(current_S) > best_valid_size:
                        best_valid_size = len(current_S)
                        best_S = set(current_S)
                if current_penalty < best_penalty:
                    best_penalty = current_penalty

        T *= cooling
        
        # Check time
        if time.time() > deadline:
            break

    # Final check: ensure best_S is valid and non-empty
    if best_valid_size == 0:
        # Fallback: try to construct a small valid set greedily if SA failed completely
        # This is unlikely but good for safety
        selected = []
        for x in range(1, N + 1):
            idx = x - 1
            # Check if adding idx to selected creates a bad triple
            # We can reuse the logic
            # But for simplicity and speed, just return a known small valid set if needed
            # e.g. {1, 2, 3, ...} might have bad triples.
            # Let's just do a quick greedy on the fly
            pass
        
        # Quick greedy fallback
        selected = []
        for x in range(1, N + 1):
            idx = x - 1
            ok = True
            # Check all pairs in selected
            sel_list = [y-1 for y in selected]
            for i in range(len(sel_list)):
                for j in range(i+1, len(sel_list)):
                    if is_bad_triple(idx, sel_list[i], sel_list[j]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                selected.append(x)
        
        if not selected:
            return [1]
        return selected

    return list(best_S)

# EVOLVE-BLOCK-END
