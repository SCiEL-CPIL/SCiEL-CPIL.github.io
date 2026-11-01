import random
import time
import math
import itertools

def construct(n):
    if n <= 4:
        # Standard construction is optimal for n <= 4
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

    # For n > 4, use Simulated Annealing on permutation ordering
    def get_standard_superperm(n):
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
        return word

    std_word = get_standard_superperm(n)
    
    # Extract standard order of permutations
    std_order = []
    seen = set()
    for i in range(len(std_word) - n + 1):
        w = tuple(std_word[i:i+n])
        if w not in seen:
            seen.add(w)
            std_order.append(w)
            
    if len(std_order) != math.factorial(n):
        # Fallback if extraction failed (shouldn't happen)
        return "".join(map(str, std_word))
        
    current_order = std_order[:]
    N = len(current_order)
    
    def get_overlap(p1, p2):
        # Max k such that p1[-k:] == p2[:k]
        # n is small (<=7 usually), so simple loop is fast
        max_k = min(len(p1), len(p2))
        for k in range(max_k, -1, -1):
            if p1[-k:] == p2[:k]:
                return k
        return 0

    def calc_total_length(order):
        length = n
        for i in range(len(order) - 1):
            length += n - get_overlap(order[i], order[i+1])
        return length

    # Precompute cost of each edge? No, we need dynamic updates.
    # But we can store the current edge costs to calculate Delta E quickly.
    # cost[i] = cost of edge (order[i], order[i+1]) = n - overlap(order[i], order[i+1])
    # Total length = n + sum(cost[i])
    
    current_costs = [0] * (N - 1)
    for i in range(N - 1):
        current_costs[i] = n - get_overlap(current_order[i], current_order[i+1])
        
    current_len = n + sum(current_costs)
    best_len = current_len
    best_order = current_order[:]
    
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer
    
    T = 10.0
    T_min = 0.01
    cooling = 0.999 # Tune this
    
    # Number of iterations to try
    # We can run until time limit
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Pick two random indices to swap
        # Swapping adjacent is a specific case, but random distant swaps help escape local minima
        i, j = random.sample(range(N), 2)
        if i > j:
            i, j = j, i
            
        # Calculate Delta E
        # Edges affected:
        # Left side: (i-1, i), (i, i+1)
        # Right side: (j-1, j), (j, j+1)
        # If i+1 == j, the middle edge (i, i+1) becomes (j, i) and is removed/changed differently.
        
        old_cost = 0
        new_cost = 0
        
        # Handle boundaries and adjacency
        
        # Left part
        if i > 0:
            old_cost += current_costs[i-1] # (i-1, i)
            new_cost += n - get_overlap(current_order[i-1], current_order[j]) # (i-1, j)
        
        # Middle/Right part
        if i + 1 < j:
            # Edge (i, i+1) becomes (j, i+1)
            old_cost += current_costs[i]
            new_cost += n - get_overlap(current_order[j], current_order[i+1])
            
            # Edge (j-1, j) becomes (j-1, i)
            old_cost += current_costs[j-1]
            new_cost += n - get_overlap(current_order[j-1], current_order[i])
            
            # Edge (j, j+1) becomes (i, j+1)
            if j < N - 1:
                old_cost += current_costs[j]
                new_cost += n - get_overlap(current_order[i], current_order[j+1])
        else: # i + 1 == j
            # Edge (i, i+1) i.e., (i, j) is removed/replaced by (j, i) ? No.
            # Sequence: ... A, Pi, Pj, B ...
            # Swap: ... A, Pj, Pi, B ...
            # Old edges: (A, Pi) cost[i-1], (Pi, Pj) cost[i], (Pj, B) cost[j]
            # New edges: (A, Pj), (Pj, Pi), (Pi, B)
            
            # (A, Pi) -> (A, Pj)
            if i > 0:
                old_cost += current_costs[i-1]
                new_cost += n - get_overlap(current_order[i-1], current_order[j])
            
            # (Pi, Pj) -> (Pj, Pi)
            old_cost += current_costs[i]
            new_cost += n - get_overlap(current_order[j], current_order[i])
            
            # (Pj, B) -> (Pi, B)
            if j < N - 1:
                old_cost += current_costs[j]
                new_cost += n - get_overlap(current_order[i], current_order[j+1])
            else:
                # If j is last, there is no edge (j, j+1).
                # So old_cost doesn't include cost[j].
                # new_cost doesn't include new edge.
                pass

        delta = new_cost - old_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept move
            # Update order
            current_order[i], current_order[j] = current_order[j], current_order[i]
            
            # Update costs
            if i + 1 < j:
                if i > 0:
                    current_costs[i-1] = n - get_overlap(current_order[i-1], current_order[i])
                current_costs[i] = n - get_overlap(current_order[i], current_order[i+1])
                current_costs[j-1] = n - get_overlap(current_order[j-1], current_order[j])
                if j < N - 1:
                    current_costs[j] = n - get_overlap(current_order[j], current_order[j+1])
            else: # i+1 == j
                if i > 0:
                    current_costs[i-1] = n - get_overlap(current_order[i-1], current_order[i])
                current_costs[i] = n - get_overlap(current_order[i], current_order[i+1])
                if j < N - 1:
                    current_costs[j] = n - get_overlap(current_order[j], current_order[j+1])
            
            current_len += delta
            
            if current_len < best_len:
                best_len = current_len
                best_order = current_order[:]
                
        T *= cooling
        
    # Reconstruct word from best_order
    word = list(best_order[0])
    for i in range(1, len(best_order)):
        p_prev = best_order[i-1]
        p_curr = best_order[i]
        ov = get_overlap(p_prev, p_curr)
        word.extend(p_curr[ov:])
        
    return "".join(map(str, word))
