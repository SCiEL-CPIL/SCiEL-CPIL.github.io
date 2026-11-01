import time
import random
import math

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # 115s hard limit for internal logic to allow for return overhead

    # 1. Generate the initial word using the seed's logic
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
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    # 2. Extract distinct permutations in order of appearance
    seen_perms = set()
    seq = []
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        if len(set(p)) == n:
            if p not in seen_perms:
                seen_perms.add(p)
                seq.append(p)
                
    # Verify we have all permutations
    # For n=7, we should have 5040
    # If not, something is wrong, but the seed logic is standard.
    
    N = len(seq)
    perms = seq
    
    # 3. Build Overlap Matrix
    # overlap_matrix[i][j] = overlap between perms[i] and perms[j]
    # This is O(N^2 * n). N=5040, n=7.
    # 25M * 7 = 175M operations.
    # In Python, this might take ~10-20 seconds.
    # We have 115 seconds. This is acceptable.
    
    overlap_matrix = [[0]*N for _ in range(N)]
    for i, p1 in enumerate(perms):
        # Optimization: p1 is fixed.
        for j, p2 in enumerate(perms):
            if i == j:
                continue
            # Find max k such that p1[n-k:] == p2[:k]
            # k can be from 1 to n-1
            max_k = n - 1
            ov = 0
            # Check from largest k to smallest
            for k in range(max_k, 0, -1):
                if p1[n-k:] == p2[:k]:
                    ov = k
                    break
            overlap_matrix[i][j] = ov

    # 4. Local Search
    # Initial order: 0..N-1
    order = list(range(N))
    
    def get_ov(i, j):
        return overlap_matrix[i][j]

    def calc_total_cost(ord):
        cost = 0
        for k in range(len(ord) - 1):
            cost += n - get_ov(ord[k], ord[k+1])
        return cost

    current_cost = calc_total_cost(order)
    
    # 2-Opt Move
    def try_2opt(l, r):
        nonlocal order, current_cost
        if l >= r: return False
        if time.time() > deadline: return False
        
        cost_delta = 0
        
        # Boundary edges
        if l > 0:
            cost_delta -= (n - get_ov(order[l-1], order[l]))
            cost_delta += (n - get_ov(order[l-1], order[r]))
        if r < N - 1:
            cost_delta -= (n - get_ov(order[r], order[r+1]))
            cost_delta += (n - get_ov(order[l], order[r+1]))
            
        # Internal edges
        # Old: sum_{k=l}^{r-1} (n - ov(order[k], order[k+1]))
        # New: sum_{k=l}^{r-1} (n - ov(order[k+1], order[k]))
        
        for k in range(l, r):
            cost_delta -= (n - get_ov(order[k], order[k+1]))
            cost_delta += (n - get_ov(order[k+1], order[k]))
            
        if cost_delta < 0:
            order[l:r+1] = reversed(order[l:r+1])
            current_cost += cost_delta
            return True
        return False

    # Swap Move
    def try_swap(i, j):
        nonlocal order, current_cost
        if i == j: return False
        if time.time() > deadline: return False
        
        cost_delta = 0
        
        if abs(i - j) > 1:
            if i > j: i, j = j, i
            
            # Remove old costs
            if i > 0: cost_delta -= (n - get_ov(order[i-1], order[i]))
            if i < N - 1: cost_delta -= (n - get_ov(order[i], order[i+1]))
            if j > 0: cost_delta -= (n - get_ov(order[j-1], order[j]))
            if j < N - 1: cost_delta -= (n - get_ov(order[j], order[j+1]))
            
            # Add new costs
            if i > 0: cost_delta += (n - get_ov(order[i-1], order[j]))
            if i < N - 1: cost_delta += (n - get_ov(order[j], order[i+1]))
            if j > 0: cost_delta += (n - get_ov(order[j-1], order[i]))
            if j < N - 1: cost_delta += (n - get_ov(order[i], order[j+1]))
            
            if cost_delta < 0:
                order[i], order[j] = order[j], order[i]
                current_cost += cost_delta
                return True
            return False
        else:
            # Adjacent swap
            i, j = min(i, j), max(i, j)
            cost_delta = 0
            if i > 0:
                cost_delta -= (n - get_ov(order[i-1], order[i]))
                cost_delta += (n - get_ov(order[i-1], order[j]))
            if j < N - 1:
                cost_delta -= (n - get_ov(order[j], order[j+1]))
                cost_delta += (n - get_ov(order[i], order[j+1]))
            
            # Internal edge
            cost_delta -= (n - get_ov(order[i], order[j]))
            cost_delta += (n - get_ov(order[j], order[i]))
            
            if cost_delta < 0:
                order[i], order[j] = order[j], order[i]
                current_cost += cost_delta
                return True
            return False

    # Search Loop
    # We will run until deadline.
    # To avoid getting stuck in local minima, we can use random restarts or simulated annealing.
    # Given time constraint, simple hill climbing with random moves is a good start.
    # We can also try to increase the probability of 2-opt as it explores larger moves.
    
    while time.time() < deadline:
        move_type = random.random()
        if move_type < 0.8:
            # 2-opt
            # Bias towards shorter segments? 
            # Long segments are more likely to be beneficial if the current order is random-ish, 
            # but we start with a structured order.
            # Let's pick random l, r.
            l = random.randint(0, N - 2)
            r = random.randint(l + 1, N - 1)
            try_2opt(l, r)
        else:
            # Swap
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            try_swap(i, j)

    # Construct final word
    final_word = list(perms[order[0]])
    for k in range(1, N):
        prev_idx = order[k-1]
        curr_idx = order[k]
        ov = get_ov(prev_idx, curr_idx)
        curr_perm = perms[curr_idx]
        final_word.extend(curr_perm[ov:])
        
    return "".join(map(str, final_word))
