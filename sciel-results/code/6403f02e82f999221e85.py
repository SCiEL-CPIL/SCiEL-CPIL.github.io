import time
import itertools
import sys
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    if n == 1:
        return "1"
    if n == 2:
        return "121"

    # 1. Generate standard superpermutation for n-1
    # We need the sequence of (n-1)-permutations as they appear in the standard construction for n-1.
    # The standard construction for n is built by taking the (n-1)-permutations from S_{n-1},
    # and for each permutation p, appending "p n p" with maximum overlap.
    
    # First, generate S_{n-1} and extract the unique (n-1)-permutations in order of appearance.
    word_prev = [1]
    for m in range(2, n):
        if time.time() > deadline:
            break
        perms, seen = [], set()
        for i in range(len(word_prev) - m + 2):
            window = tuple(word_prev[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word_prev = new_word

    # Extract the (n-1)-permutations from word_prev in order
    perms, seen = [], set()
    for i in range(len(word_prev) - n + 2):
        window = tuple(word_prev[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
            
    num_perms = len(perms)
    # For n=7, num_perms should be 720.
    
    # 2. Precompute overlap matrix for the (n-1)-permutations
    # M[i][j] = longest suffix of perms[i] matching prefix of perms[j]
    # The block for permutation p is "p n p".
    # When concatenating block_i and block_j, the overlap is determined by:
    # suffix of "p_i n p_i" matching prefix of "p_j n p_j".
    # Since the blocks are length 2(n-1)+1, the max overlap is 2(n-1).
    # However, the structure is specific.
    # Let's just compute the overlap between the full blocks.
    
    blocks = []
    for p in perms:
        blocks.append(list(p) + [n] + list(p))
        
    M = [[0]*num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        b1 = blocks[i]
        for j in range(num_perms):
            b2 = blocks[j]
            max_ov = min(len(b1), len(b2))
            ov = 0
            for k in range(max_ov, -1, -1):
                if b1[-k:] == b2[:k]:
                    ov = k
                    break
            M[i][j] = ov

    # 3. Find a good ordering of permutations to minimize total length
    # Total length = sum(len(block)) - sum(M[order[i]][order[i+1]])
    # Since sum(len(block)) is constant, we want to maximize sum(M[...]).
    # This is a Traveling Salesman Problem variant (Max TSP path).
    
    # Initial order: identity
    order = list(range(num_perms))
    
    def calc_score(ord):
        score = 0
        for i in range(len(ord) - 1):
            score += M[ord[i]][ord[i+1]]
        return score

    best_order = order[:]
    best_score = calc_score(order)
    current_order = order[:]
    current_score = best_score

    # Local Search: Simulated Annealing or Hill Climbing
    # Given time constraints, a fast hill climbing with restarts or simple SA is good.
    # 720 items is small enough for O(N) score updates if we use delta, but O(N) full calc is ~720 ops.
    # 100,000 iterations * 720 ops = 72M ops. In Python, this might be too slow (~10s+).
    # Let's use delta calculation for swaps/moves to make it O(1) or O(n) per move.
    
    # Delta for swapping i, i+1:
    # Old: ... a, i, j, b ...
    # New: ... a, j, i, b ...
    # Change = (M[a][j] + M[j][i] + M[i][b]) - (M[a][i] + M[i][j] + M[j][b])
    # Handle boundaries.
    
    def get_delta_swap(ord, i):
        # Swap ord[i] and ord[i+1]
        x = ord[i]
        y = ord[i+1]
        
        left = ord[i-1] if i > 0 else -1
        right = ord[i+2] if i+2 < len(ord) else -1
        
        delta = 0
        
        # Remove old edges
        if i > 0:
            delta -= M[left][x]
        delta -= M[x][y]
        if i+1 < len(ord)-1:
            delta -= M[y][right]
            
        # Add new edges
        if i > 0:
            delta += M[left][y]
        delta += M[y][x]
        if i+1 < len(ord)-1:
            delta += M[x][right]
            
        return delta

    def get_delta_move(ord, i, j):
        # Move element at i to position j
        # Assume i < j for simplicity, or handle generally.
        # General move is harder for delta. Let's stick to swaps and maybe 2-opt.
        # Or just use full recalc if moves are sparse.
        # Given N=720, full recalc is fast enough if we don't do millions of iterations.
        # Let's try a simple hill climbing with swaps first.
        pass

    random.seed(42)
    
    # Simple Hill Climbing with random restarts or just one long run?
    # Let's do a few passes of hill climbing.
    
    max_iterations = 500000
    iters = 0
    while iters < max_iterations and time.time() < deadline - 10:
        iters += 1
        i = random.randint(0, num_perms - 2)
        delta = get_delta_swap(current_order, i)
        if delta >= 0:
            # Accept
            current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_order = current_order[:]
        else:
            # Random restart if stuck? Or just continue.
            # If we get stuck, we might loop. 
            # Let's add a simple restart mechanism if no improvement for X steps.
            pass
            
    # Try a few random restarts if time permits
    restarts = 0
    while time.time() < deadline - 10 and restarts < 5:
        restarts += 1
        current_order = list(range(num_perms))
        random.shuffle(current_order)
        current_score = calc_score(current_order)
        
        stuck_count = 0
        for _ in range(10000):
            if time.time() > deadline - 10:
                break
            i = random.randint(0, num_perms - 2)
            delta = get_delta_swap(current_order, i)
            if delta > 0:
                current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
                current_score += delta
                stuck_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_order = current_order[:]
            else:
                stuck_count += 1
                if stuck_count > 1000:
                    break

    # 4. Construct word from best_order
    word = []
    for idx, p_idx in enumerate(best_order):
        block = blocks[p_idx]
        if idx == 0:
            word = block[:]
        else:
            # Find overlap with current word
            max_ov = min(len(word), len(block))
            ov = 0
            for k in range(max_ov, -1, -1):
                if word[-k:] == block[:k]:
                    ov = k
                    break
            word += block[ov:]
            
    S = word

    # 5. Pruning Loop (from current program) to remove redundant symbols
    # This is crucial for the final score.
    all_perms = set(itertools.permutations(range(1, n + 1)))
    
    # Initialize counts
    perm_counts = {}
    for i in range(len(S) - n + 1):
        p = tuple(S[i:i + n])
        perm_counts[p] = perm_counts.get(p, 0) + 1
        
    def is_valid_incremental(S, i, perm_counts):
        L = len(S)
        if L <= n:
            return False, None, None
        
        start_min = max(0, i - n + 1)
        start_max = min(i, len(S) - n)
        
        deltas = {}
        
        # Old windows to remove
        for j in range(start_min, start_max + 1):
            w = tuple(S[j:j+n])
            deltas[w] = deltas.get(w, 0) - 1
            
        # New windows to add
        for j in range(max(0, i - n + 1), i):
            if j > len(S) - 1 - n:
                continue
            part1 = S[j:i]
            part2 = S[i+1:j+n+1]
            new_win = tuple(part1 + part2)
            deltas[new_win] = deltas.get(new_win, 0) + 1
            
        for p, delta in deltas.items():
            if p in all_perms:
                new_count = perm_counts.get(p, 0) + delta
                if new_count < 1:
                    return False, None, None
                    
        return True, S[:i] + S[i+1:], deltas

    # Pruning
    for _pass in range(100):
        if time.time() > deadline - 5:
            break
        removed = False
        # Iterate backwards
        for i in range(len(S) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            if len(S) <= n:
                break
            valid, new_S, deltas = is_valid_incremental(S, i, perm_counts)
            if valid:
                S = new_S
                for p, delta in deltas.items():
                    perm_counts[p] = perm_counts.get(p, 0) + delta
                    if perm_counts[p] == 0:
                        del perm_counts[p]
                removed = True
                break
        if not removed:
            break
            
    return "".join(map(str, S))
