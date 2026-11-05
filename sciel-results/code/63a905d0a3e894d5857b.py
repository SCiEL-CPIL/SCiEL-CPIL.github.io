import random
import time

def construct():
    N = 256
    
    # Precomputation
    # bad_triplets[i] is a list of bitmasks. If A & mask == mask, adding i+1 creates a 4-AP.
    bad_triplets = [[] for _ in range(N)]
    # good_pairs[i] is a list of tuples (j, k). If j and k are in A, adding i+1 creates a 3-AP.
    good_pairs = [[] for _ in range(N)]
    
    # Generate 4-APs
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            # 4-AP: a, a+d, a+2d, a+3d
            idx_a = a - 1
            idx_b = a + d - 1
            idx_c = a + 2 * d - 1
            idx_d = a + 3 * d - 1
            
            # For each element, the other 3 form a bad triplet
            # If we add a, we need b, c, d to be present
            mask_a = (1 << idx_b) | (1 << idx_c) | (1 << idx_d)
            bad_triplets[idx_a].append(mask_a)
            
            # If we add b, we need a, c, d to be present
            mask_b = (1 << idx_a) | (1 << idx_c) | (1 << idx_d)
            bad_triplets[idx_b].append(mask_b)
            
            # If we add c, we need a, b, d to be present
            mask_c = (1 << idx_a) | (1 << idx_b) | (1 << idx_d)
            bad_triplets[idx_c].append(mask_c)
            
            # If we add d, we need a, b, c to be present
            mask_d = (1 << idx_a) | (1 << idx_b) | (1 << idx_c)
            bad_triplets[idx_d].append(mask_d)
            
    # Generate 3-APs
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            # 3-AP: a, a+d, a+2d
            idx_a = a - 1
            idx_b = a + d - 1
            idx_c = a + 2 * d - 1
            
            # If we add a, we need b, c
            good_pairs[idx_a].append((idx_b, idx_c))
            
            # If we add b, we need a, c
            good_pairs[idx_b].append((idx_a, idx_c))
            
            # If we add c, we need a, b
            good_pairs[idx_c].append((idx_a, idx_b))

    def is_valid_add(x_idx, A):
        """Check if adding x_idx (0-based) to A creates a 4-AP."""
        for mask in bad_triplets[x_idx]:
            if (A & mask) == mask:
                return False
        return True

    def count_3aps_delta(x_idx, A):
        """Count how many 3-APs are formed by adding x_idx to A."""
        count = 0
        for u, v in good_pairs[x_idx]:
            if (A & (1 << u)) and (A & (1 << v)):
                count += 1
        return count

    # Initial State: Greedy Random
    # We try a few random orderings and pick the one that yields the most 3-APs initially
    best_initial_A = 0
    best_initial_score = -1
    
    for _ in range(10):
        A = 0
        order = list(range(N))
        random.shuffle(order)
        score = 0
        for x_idx in order:
            if is_valid_add(x_idx, A):
                A |= (1 << x_idx)
                score += count_3aps_delta(x_idx, A)
        if score > best_initial_score:
            best_initial_score = score
            best_initial_A = A

    current_A = best_initial_A
    current_score = best_initial_score
    best_A = current_A
    best_score = current_score

    start_time = time.time()
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    
    while T > T_min and time.time() - start_time < 55:
        x_idx = random.randint(0, N - 1)
        
        if not (current_A & (1 << x_idx)):
            # Try adding
            if is_valid_add(x_idx, current_A):
                delta = count_3aps_delta(x_idx, current_A)
                new_score = current_score + delta
                if new_score > best_score:
                    best_score = new_score
                    best_A = current_A | (1 << x_idx)
                
                # Accept
                if delta >= 0:
                    current_A |= (1 << x_idx)
                    current_score = new_score
                else:
                    # Metropolis criterion
                    if random.random() < pow(2.718, delta / T):
                        current_A |= (1 << x_idx)
                        current_score = new_score
        else:
            # Try removing
            delta = count_3aps_delta(x_idx, current_A)
            new_score = current_score - delta
            
            if new_score > best_score:
                best_score = new_score
                best_A = current_A & ~(1 << x_idx)
            
            # Accept
            if delta <= 0:
                # delta is negative or 0, so new_score >= current_score
                current_A &= ~(1 << x_idx)
                current_score = new_score
            else:
                # delta > 0, new_score < current_score
                if random.random() < pow(2.718, -delta / T):
                    current_A &= ~(1 << x_idx)
                    current_score = new_score

        T *= cooling

    # Convert bitmask to sorted list
    result = []
    for i in range(N):
        if best_A & (1 << i):
            result.append(i + 1)
            
    return sorted(result)
