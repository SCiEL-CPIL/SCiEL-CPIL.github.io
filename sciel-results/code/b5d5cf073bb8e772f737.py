"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    timeout = 145.0  # Leave 5s margin for safety
    
    total_words = q ** n
    
    # Precompute powers of q for digit extraction
    powers = [q ** i for i in range(n)]
    
    # Convert between integer representation and digit lists
    def int_to_digits(x):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % q
            x //= q
        return digits
    
    # Precompute neighbors as bitmasks for fast operations
    # For radius=1, each word has 1 + n*(q-1) neighbors
    neighbor_masks = [0] * total_words
    for u in range(total_words):
        mask = 1 << u  # self
        du = int_to_digits(u)
        for pos in range(n):
            orig_val = du[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                v = u - orig_val * powers[pos] + new_val * powers[pos]
                mask |= (1 << v)
        neighbor_masks[u] = mask
    
    # Full mask representing all words
    full_mask = (1 << total_words) - 1
    
    # Greedy set cover initialization using bitmasks
    def greedy_cover(seed_offset=0):
        uncovered = full_mask
        C = []
        while uncovered:
            best_u = -1
            best_count = -1
            for u in range(total_words):
                cnt = bin(uncovered & neighbor_masks[u]).count('1')
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
                    if cnt == bin(neighbor_masks[u]).count('1'):
                        break
            if best_u == -1:
                break
            C.append(best_u)
            uncovered &= ~neighbor_masks[best_u]
        return C
    
    # Helper: check if a set of codewords covers all words using bitmask
    def is_covering(C_list):
        covered = 0
        for c in C_list:
            covered |= neighbor_masks[c]
        return covered == full_mask
    
    # Local search: removals using bitmask
    def local_remove(C):
        removed_any = True
        while removed_any and (time.time() - start_time) < timeout:
            removed_any = False
            random.shuffle(C)
            i = 0
            while i < len(C):
                # Check if removing C[i] still covers
                covered = 0
                for j, c in enumerate(C):
                    if j != i:
                        covered |= neighbor_masks[c]
                if covered == full_mask:
                    C.pop(i)
                    removed_any = True
                else:
                    i += 1
        return C
    
    # Local search: swaps to escape local minima
    def local_swap(C):
        for _ in range(30):
            if (time.time() - start_time) >= timeout:
                break
            
            # Try removals first
            C = local_remove(C)
            
            # If no removals worked, try a random swap
            if len(C) > 0:
                i = random.randint(0, len(C) - 1)
                w = random.randint(0, total_words - 1)
                if w != C[i]:
                    C_temp = C.copy()
                    C_temp[i] = w
                    if is_covering(C_temp):
                        C = C_temp
                        # After a swap, try removals again
                        C = local_remove(C)
        return C
    
    # Multiple random restarts to find best solution
    best_C = None
    best_size = float('inf')
    
    # First, do one greedy + local search
    C = greedy_cover()
    C = local_remove(C)
    C = local_swap(C)
    best_C = C.copy()
    best_size = len(C)
    
    # Try multiple restarts with different random seeds
    num_restarts = 8
    for r in range(num_restarts):
        if (time.time() - start_time) >= timeout - 5:
            break
        
        # Randomized greedy: shuffle order of candidates
        C = greedy_cover()
        
        # Add some random perturbation: replace a few codewords
        if len(C) > 3:
            for _ in range(min(3, len(C) // 5)):
                idx = random.randint(0, len(C) - 1)
                C[idx] = random.randint(0, total_words - 1)
        
        # Local search
        C = local_remove(C)
        C = local_swap(C)
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
    
    # Convert back to digit lists
    result = [int_to_digits(c) for c in best_C]
    return result
# EVOLVE-BLOCK-END
