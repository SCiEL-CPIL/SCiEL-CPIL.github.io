"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n):
    if n <= 6:
        return _classic_construct(n)
    
    # For n=7, use optimized ordering of the final step
    word = _classic_construct(n - 1)
    return _optimized_step(word, n)

def _classic_construct(n):
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

def _optimized_step(word_str, n):
    word = [int(c) for c in word_str]
    m = n
    perms, seen = [], set()
    for i in range(len(word) - m + 2):
        window = tuple(word[i:i + m - 1])
        if len(set(window)) == m - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    
    # Create blocks
    blocks = [list(p) + [m] + list(p) for p in perms]
    num_blocks = len(blocks)
    
    # Precompute overlap matrix
    # M[i][j] = overlap between block i and block j
    # Overlap is max k such that suffix_k(block_i) == prefix_k(block_j)
    M = [[0]*num_blocks for _ in range(num_blocks)]
    for i in range(num_blocks):
        b1 = blocks[i]
        for j in range(num_blocks):
            b2 = blocks[j]
            max_k = min(len(b1), len(b2))
            for k in range(max_k, -1, -1):
                if k == 0 or b1[-k:] == b2[:k]:
                    M[i][j] = k
                    break
                    
    # Calculate total length given an order
    def calc_length(ord):
        total = len(blocks[ord[0]])
        for i in range(len(ord) - 1):
            total += len(blocks[ord[i+1]]) - M[ord[i]][ord[i+1]]
        return total
    
    # Simulated Annealing
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    order = list(range(num_blocks))
    current_length = calc_length(order)
    best_order = order[:]
    best_length = current_length
    
    # Annealing parameters
    T = 100.0
    T_min = 0.01
    cooling = 0.999
    
    random.seed(42)
    
    while time.time() < deadline and T > T_min:
        # Pick a random move
        if random.random() < 0.5:
            # Swap adjacent
            i = random.randint(0, num_blocks - 2)
            new_order = order[:]
            new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
        else:
            # Move element i to position j
            i = random.randint(0, num_blocks - 1)
            j = random.randint(0, num_blocks - 1)
            if i == j:
                continue
            new_order = order[:]
            item = new_order.pop(i)
            new_order.insert(j, item)
        
        # Calculate new length
        new_length = calc_length(new_order)
        delta = new_length - current_length
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            order = new_order
            current_length = new_length
            
            if current_length < best_length:
                best_length = current_length
                best_order = order[:]
        
        T *= cooling
        
    # Construct final word from best_order
    new_word = list(blocks[best_order[0]])
    for k in range(1, len(best_order)):
        prev_idx = best_order[k-1]
        curr_idx = best_order[k]
        b_curr = blocks[curr_idx]
        # Find overlap using precomputed matrix
        overlap = M[prev_idx][curr_idx]
        new_word += b_curr[overlap:]
        
    return "".join(map(str, new_word))

# EVOLVE-BLOCK-END
