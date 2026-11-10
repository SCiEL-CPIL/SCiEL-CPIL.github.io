import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # For small n, use the standard construction which is optimal
    if n <= 3:
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

    # For n >= 4, we try to improve the standard construction
    # Generate standard superpermutation for n-1 to get the order
    
    def get_standard_order(n_prev):
        word = [1]
        for m in range(2, n_prev + 1):
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
        return word

    # Get S_{n-1}
    s_prev = get_standard_order(n - 1)
    
    # Extract the order of permutations of 1..n-1 in s_prev
    order = []
    seen = set()
    for i in range(len(s_prev) - (n - 1) + 1):
        window = tuple(s_prev[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            order.append(window)
            
    # Generate blocks for n
    blocks = []
    for p in order:
        blocks.append(list(p) + [n] + list(p))
        
    M = len(blocks)
    block_len = len(blocks[0])
    
    # Precompute overlap matrix using string operations for speed
    # Convert blocks to strings for faster comparison
    block_strs = ["".join(map(str, b)) for b in blocks]
    
    overlap_matrix = [[0] * M for _ in range(M)]
    for i in range(M):
        for j in range(M):
            # Max overlap is min(len(block_i), len(block_j)) - 1
            max_ov = min(block_len, block_len) - 1
            for k in range(max_ov, -1, -1):
                if block_strs[i][-k:] == block_strs[j][:k]:
                    overlap_matrix[i][j] = k
                    break
    
    def evaluate(order_idx):
        # Calculate length
        length = block_len
        for i in range(1, M):
            length += block_len - overlap_matrix[order_idx[i-1]][order_idx[i]]
        return length

    def build_string(order_idx):
        s = block_strs[order_idx[0]]
        for i in range(1, M):
            ov = overlap_matrix[order_idx[i-1]][order_idx[i]]
            s += block_strs[order_idx[i]][ov:]
        return s

    def is_valid(s):
        # Check if all permutations of 1..n are in s
        seen_perms = set()
        for i in range(len(s) - n + 1):
            window = s[i:i+n]
            if len(set(window)) == n:
                seen_perms.add(window)
        return len(seen_perms) == n

    # Initial evaluation
    current_order = list(range(M))
    current_string = build_string(current_order)
    current_length = len(current_string)
    is_current_valid = is_valid(current_string)
    
    best_order = current_order[:]
    best_string = current_string
    best_length = current_length
    
    if not is_current_valid:
        return best_string

    # Local Search with multiple move types
    # Move types: swap, 2-opt (reverse segment), relocate
    
    while time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.4:
            # Swap two random positions
            i, j = random.sample(range(M), 2)
            new_order = current_order[:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
            
        elif move_type < 0.7:
            # 2-opt: reverse a segment
            i, j = random.sample(range(M), 2)
            if i > j:
                i, j = j, i
            new_order = current_order[:]
            new_order[i:j+1] = reversed(new_order[i:j+1])
            
        else:
            # Relocate: move one element to another position
            i = random.randint(0, M - 1)
            j = random.randint(0, M - 1)
            if i == j:
                continue
            new_order = current_order[:]
            elem = new_order.pop(i)
            new_order.insert(j, elem)
        
        new_length = evaluate(new_order)
        
        # Quick check: if length is not better, skip build
        if new_length >= best_length:
            continue
            
        new_string = build_string(new_order)
        
        if is_valid(new_string):
            best_order = new_order
            best_string = new_string
            best_length = new_length
            current_order = new_order
            current_string = new_string
        else:
            # Revert - keep current_order as is
            pass

    return best_string
