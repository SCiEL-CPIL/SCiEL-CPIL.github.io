import time
import random
import heapq
from collections import deque

N = 12

def calculate_threshold(a_trans, b_trans):
    """
    Computes the exact shortest reset word length (threshold) for a binary automaton.
    Uses A* search on the subset automaton with bitmasks for efficiency.
    Heuristic: h(mask) = popcount(mask) - 1
    """
    all_states = (1 << N) - 1
    
    # Precompute transitions for all masks? 
    # 2^12 = 4096 masks. 
    # Precomputing all is fast and simplifies A* logic.
    trans_a = [0] * (1 << N)
    trans_b = [0] * (1 << N)
    
    for mask in range(1 << N):
        res_a = 0
        res_b = 0
        m = mask
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            res_a |= (1 << a_trans[i])
            res_b |= (1 << b_trans[i])
            m ^= lsb
        trans_a[mask] = res_a
        trans_b[mask] = res_b
        
    start = all_states
    
    # Check if already singleton
    if (start & (start - 1)) == 0:
        return 0

    # A* Search
    # g_score: cost from start
    # f_score: g_score + h_score
    # h_score: popcount - 1
    def h(mask):
        return mask.bit_count() - 1
        
    # Priority queue: (f, g, mask)
    open_set = [(h(start), 0, start)]
    g_score = {start: 0}
    
    while open_set:
        f, g, curr = heapq.heappop(open_set)
        
        # If we popped a state with g > g_score[curr], it's stale
        if g > g_score.get(curr, float('inf')):
            continue
            
        # Check if singleton
        if (curr & (curr - 1)) == 0:
            return g
            
        # Generate neighbors
        next_a = trans_a[curr]
        next_b = trans_b[curr]
        
        # Process next_a
        if next_a in g_score:
            if g + 1 < g_score[next_a]:
                g_score[next_a] = g + 1
                heapq.heappush(open_set, (g + 1 + h(next_a), g + 1, next_a))
        else:
            g_score[next_a] = g + 1
            heapq.heappush(open_set, (g + 1 + h(next_a), g + 1, next_a))
            
        # Process next_b
        if next_b in g_score:
            if g + 1 < g_score[next_b]:
                g_score[next_b] = g + 1
                heapq.heappush(open_set, (g + 1 + h(next_b), g + 1, next_b))
        else:
            g_score[next_b] = g + 1
            heapq.heappush(open_set, (g + 1 + h(next_b), g + 1, next_b))
            
    return -1  # Not synchronizing

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_score = -1
    best_a = None
    best_b = None
    
    # Helper to update best
    def update(a, b):
        nonlocal best_score, best_a, best_b
        score = calculate_threshold(a, b)
        if score > best_score:
            best_score = score
            best_a = list(a)
            best_b = list(b)
        return score

    # Strategy: Focus on structured families known to produce high thresholds,
    # then use extensive hill climbing from multiple starting points.
    
    # 1. 'a' is a full cycle (shift by 1)
    a_cycle = [(i + 1) % N for i in range(N)]
    
    # Sub-family 1a: b is identity with 1 defect (quick, N*(N-1) = 132 cases)
    for k in range(N):
        for j in range(N):
            if j == k:
                continue
            b = [i for i in range(N)]
            b[k] = j
            if time.time() > deadline:
                return [best_a, best_b]
            update(a_cycle, b)
            
    # Sub-family 1b: Pinsker-like structure
    # a: 0->1->...->N-1->0
    # b: 0->0, 1->2, 2->3, ..., N-1->1
    b_pinsker = [0] * N
    b_pinsker[0] = 0
    for i in range(1, N):
        if i == N - 1:
            b_pinsker[i] = 1
        else:
            b_pinsker[i] = i + 1
    update(a_cycle, b_pinsker)
    
    # Sub-family 1c: Random permutations for b (reduced count, more time for hill climbing)
    random.seed(42)
    for _ in range(500):
        if time.time() > deadline:
            return [best_a, best_b]
        b = list(range(N))
        random.shuffle(b)
        update(a_cycle, b)

    # 2. 'a' is a shift by d (try d=1,2,3,4,5,6 - half due to symmetry)
    for d in range(1, 7):
        a_shift = [(i + d) % N for i in range(N)]
        # Try b as Pinsker-like cycle fixing one state
        for fixed in range(N):
            b = [0] * N
            b[fixed] = fixed
            others = [i for i in range(N) if i != fixed]
            # Create a cycle on 'others'
            for i in range(len(others)):
                b[others[i]] = others[(i + 1) % len(others)]
            if time.time() > deadline:
                return [best_a, best_b]
            update(a_shift, b)
        
        # Try b as identity with 1 defect
        for k in range(N):
            for j in range(N):
                if j == k:
                    continue
                b = [i for i in range(N)]
                b[k] = j
                if time.time() > deadline:
                    return [best_a, best_b]
                update(a_shift, b)
                
        # Try a few random permutations
        for _ in range(50):
            if time.time() > deadline:
                return [best_a, best_b]
            b = list(range(N))
            random.shuffle(b)
            update(a_shift, b)

    # 3. Extensive Hill Climbing from multiple starting points
    # Start from best found so far, plus several other promising starting points
    
    # Ensure we have a valid starting point
    if best_a is None:
        best_a = a_cycle
        best_b = b_pinsker
        update(best_a, best_b)
    
    # Create a list of starting points for hill climbing
    starting_points = [(best_a, best_b)]
    
    # Add some other structured starting points
    for d in [1, 2, 3, 4, 5, 6]:
        a_shift = [(i + d) % N for i in range(N)]
        b_pinsker_d = [0] * N
        b_pinsker_d[0] = 0
        for i in range(1, N):
            if i == N - 1:
                b_pinsker_d[i] = 1
            else:
                b_pinsker_d[i] = i + 1
        starting_points.append((a_shift, b_pinsker_d))
    
    # Random starting points
    for _ in range(20):
        a_rand = [random.randint(0, N-1) for _ in range(N)]
        b_rand = [random.randint(0, N-1) for _ in range(N)]
        starting_points.append((a_rand, b_rand))
    
    # Hill climb from each starting point
    for start_a, start_b in starting_points:
        if time.time() > deadline - 5:  # Leave 5s buffer
            break
        
        current_a = list(start_a)
        current_b = list(start_b)
        current_score = calculate_threshold(current_a, current_b)
        
        # If this starting point is better than our best, update
        if current_score > best_score:
            best_score = current_score
            best_a = list(current_a)
            best_b = list(current_b)
        
        # Local search
        no_improve_count = 0
        max_no_improve = 100
        while no_improve_count < max_no_improve:
            if time.time() > deadline - 3:  # Leave 3s buffer
                break
            
            # Generate a neighbor
            a_new = list(current_a)
            b_new = list(current_b)
            
            # Mutate 1 or 2 positions in a or b
            num_mut = random.randint(1, 2)
            for _ in range(num_mut):
                if random.random() < 0.5:
                    idx = random.randint(0, N-1)
                    val = random.randint(0, N-1)
                    a_new[idx] = val
                else:
                    idx = random.randint(0, N-1)
                    val = random.randint(0, N-1)
                    b_new[idx] = val
            
            score = calculate_threshold(a_new, b_new)
            if score > current_score:
                current_a = a_new
                current_b = b_new
                current_score = score
                no_improve_count = 0
                if score > best_score:
                    best_score = score
                    best_a = list(current_a)
                    best_b = list(current_b)
            else:
                no_improve_count += 1
        
        # Try some targeted mutations: swap two entries in a or b
        for _ in range(50):
            if time.time() > deadline - 3:
                break
            
            a_new = list(current_a)
            b_new = list(current_b)
            
            # Swap two entries
            if random.random() < 0.5:
                i, j = random.sample(range(N), 2)
                a_new[i], a_new[j] = a_new[j], a_new[i]
            else:
                i, j = random.sample(range(N), 2)
                b_new[i], b_new[j] = b_new[j], b_new[i]
            
            score = calculate_threshold(a_new, b_new)
            if score > current_score:
                current_a = a_new
                current_b = b_new
                current_score = score
                if score > best_score:
                    best_score = score
                    best_a = list(current_a)
                    best_b = list(current_b)

    return [best_a, best_b]

# EVOLVE-BLOCK-START
# The function construct() is defined above.
# EVOLVE-BLOCK-END
