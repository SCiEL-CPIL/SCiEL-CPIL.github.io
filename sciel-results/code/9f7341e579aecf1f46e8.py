"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    timeout = 140.0  # Leave 10s margin for safety
    
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
    
    def digits_to_int(digits):
        val = 0
        for i in range(n):
            val = val * q + digits[i]
        return val
    
    # Precompute neighbors for all words (distance <= radius)
    # For radius=1, each word has 1 + n*(q-1) neighbors
    neighbors = [[] for _ in range(total_words)]
    for u in range(total_words):
        neighbors[u].append(u)  # self
        du = int_to_digits(u)
        for pos in range(n):
            orig_val = du[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                v = u - orig_val * powers[pos] + new_val * powers[pos]
                neighbors[u].append(v)
    
    # Greedy set cover initialization
    uncovered = set(range(total_words))
    C = []
    
    while uncovered:
        best_u = -1
        best_count = -1
        for u in range(total_words):
            cnt = sum(1 for v in neighbors[u] if v in uncovered)
            if cnt > best_count:
                best_count = cnt
                best_u = u
                if cnt == len(neighbors[u]):
                    break
        if best_u == -1:
            break
        C.append(best_u)
        for v in neighbors[best_u]:
            if v in uncovered:
                uncovered.remove(v)
    
    # Helper: check if a set of codewords covers all words
    def is_covering(C_list):
        counts = [0] * total_words
        for c in C_list:
            for v in neighbors[c]:
                counts[v] += 1
        return min(counts) > 0
    
    # Local search: removals
    removed_any = True
    while removed_any and (time.time() - start_time) < timeout:
        removed_any = False
        random.shuffle(C)
        i = 0
        while i < len(C):
            C_temp = C[:i] + C[i+1:]
            if is_covering(C_temp):
                C = C_temp
                removed_any = True
            else:
                i += 1
    
    # Local search: swaps to escape local minima
    # Try random swaps if we're stuck
    max_swap_rounds = 50
    for _ in range(max_swap_rounds):
        if (time.time() - start_time) >= timeout:
            break
        
        # Try removals first
        removed_any = True
        while removed_any:
            removed_any = False
            random.shuffle(C)
            i = 0
            while i < len(C):
                C_temp = C[:i] + C[i+1:]
                if is_covering(C_temp):
                    C = C_temp
                    removed_any = True
                else:
                    i += 1
        
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
                    removed_any = True
                    while removed_any and (time.time() - start_time) < timeout:
                        removed_any = False
                        random.shuffle(C)
                        j = 0
                        while j < len(C):
                            C_temp2 = C[:j] + C[j+1:]
                            if is_covering(C_temp2):
                                C = C_temp2
                                removed_any = True
                            else:
                                j += 1
    
    # Convert back to digit lists
    result = [int_to_digits(c) for c in C]
    return result
# EVOLVE-BLOCK-END
