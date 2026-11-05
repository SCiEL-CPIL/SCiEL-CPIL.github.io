import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0
    
    if n <= 1:
        return "1" if n == 1 else ""
    if n == 2:
        return "121" # 1,2,1 contains 12, 21
    
    # Generate all permutations of 1..n-1
    # For n=7, n-1=6, 720 perms.
    # For larger n, this might be too big, but task is likely n=7.
    # The contract says "given n". If n is large, this will timeout.
    # However, the task context implies n=7.
    # We should handle small n efficiently and assume n=7 for the heavy lifting.
    
    if n > 7:
        # Fallback to recursive construction for larger n to ensure validity
        # though it won't be optimal.
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            new_word = []
            for p in perms:
                block = list(p) + [m] + list(p)
                max_ov = min(len(new_word), len(block))
                overlap = 0
                for o in range(max_ov, -1, -1):
                    if new_word[len(new_word) - o:] == block[:o]:
                        overlap = o
                        break
                new_word += block[overlap:]
            word = new_word
        return "".join(map(str, word))

    # Specific optimization for n=7 (and n<=6)
    # We solve the TSP on the permutations of 1..n-1
    
    symbols = list(range(1, n))
    perms = []
    # Use itertools for permutations
    import itertools
    for p in itertools.permutations(symbols):
        perms.append(list(p))
        
    k = len(perms) # 720 for n=7
    
    # Precompute overlap matrix M
    # M[i][j] = longest suffix of perms[i] that is a prefix of perms[j]
    M = [[0] * k for _ in range(k)]
    for i in range(k):
        p1 = perms[i]
        for j in range(k):
            p2 = perms[j]
            # Max overlap is n-1
            for l in range(n - 1, -1, -1):
                if l == 0 or p1[-l:] == p2[:l]:
                    M[i][j] = l
                    break
                    
    # Initialize order using Greedy Nearest Neighbor
    # Start with a random node
    random.seed(42)
    start_node = random.randint(0, k - 1)
    order = [start_node]
    visited = [False] * k
    visited[start_node] = True
    
    for _ in range(k - 1):
        last = order[-1]
        best_next = -1
        best_score = -1
        for j in range(k):
            if not visited[j]:
                score = M[last][j]
                if score > best_score:
                    best_score = score
                    best_next = j
        order.append(best_next)
        visited[best_next] = True
        
    # Maintain cost arrays for fast 2-opt
    # cost_forward[i] = M[order[i]][order[i+1]]
    # cost_backward[i] = M[order[i+1]][order[i]]
    cost_forward = [0] * (k - 1)
    cost_backward = [0] * (k - 1)
    for i in range(k - 1):
        cost_forward[i] = M[order[i]][order[i+1]]
        cost_backward[i] = M[order[i+1]][order[i]]
        
    # Prefix sums for cost_forward and cost_backward
    # prefix_forward[i] = sum(cost_forward[0..i-1])
    prefix_forward = [0] * k
    prefix_backward = [0] * k
    for i in range(k - 1):
        prefix_forward[i+1] = prefix_forward[i] + cost_forward[i]
        prefix_backward[i+1] = prefix_backward[i] + cost_backward[i]
        
    total_score = prefix_forward[k-1]
    
    def update_costs(order, cost_forward, cost_backward, prefix_forward, prefix_backward, i, j):
        """
        Updates cost arrays and prefix sums after reversing order[i..j].
        Note: This is expensive if done naively. 
        For 2-opt, we reverse a segment.
        We need to update cost_forward and cost_backward for indices i..j-1.
        And update prefix sums from i to end.
        """
        # Reverse the segment in order
        order[i:j+1] = reversed(order[i:j+1])
        
        # Update cost_forward and cost_backward for the affected edges
        # The edges inside the segment are reversed in direction.
        # Edge (order[i], order[i+1]) becomes (order[i+1], order[i]) in the new path?
        # No, the path is ... order[i-1], order[j], order[j-1], ..., order[i], order[i+1] ...
        # The internal edges are (order[j], order[j-1]), ..., (order[i+1], order[i]).
        # The old internal edges were (order[i], order[i+1]), ..., (order[j-1], order[j]).
        # So cost_forward for index m in [i, j-1] becomes M[order[m+1]][order[m]] (old backward)
        # And cost_backward becomes M[order[m]][order[m+1]] (old forward)
        # So we just swap cost_forward and cost_backward in the range [i, j-1]?
        # Yes, because the physical nodes are the same, just traversed in reverse.
        
        # Swap values in cost_forward and cost_backward for range [i, j-1]
        for m in range(i, j):
            cost_forward[m], cost_backward[m] = cost_backward[m], cost_forward[m]
            
        # Update prefix sums from i to end
        # prefix_forward[i] is sum of 0..i-1, which is unchanged.
        # We need to recalculate prefix_forward[m] for m > i.
        # Actually, we can just recalculate from i to end.
        # prefix_forward[i+1] = prefix_forward[i] + cost_forward[i]
        for m in range(i, k - 1):
            prefix_forward[m+1] = prefix_forward[m] + cost_forward[m]
            prefix_backward[m+1] = prefix_backward[m] + cost_backward[m]
            
        # Return new total score
        return prefix_forward[k-1]

    # 2-Opt Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(k - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, k):
                # Calculate delta
                # Old Internal: sum(cost_forward[i..j-1]) = prefix_forward[j] - prefix_forward[i]
                old_int = prefix_forward[j] - prefix_forward[i]
                # New Internal: sum(cost_backward[i..j-1]) = prefix_backward[j] - prefix_backward[i]
                new_int = prefix_backward[j] - prefix_backward[i]
                
                # Boundary costs
                # Left boundary: (order[i-1], order[i]) becomes (order[i-1], order[j])
                # Right boundary: (order[j], order[j+1]) becomes (order[i], order[j+1])
                
                old_left = M[order[i-1]][order[i]] if i > 0 else 0
                new_left = M[order[i-1]][order[j]] if i > 0 else 0
                
                old_right = M[order[j]][order[j+1]] if j < k - 1 else 0
                new_right = M[order[i]][order[j+1]] if j < k - 1 else 0
                
                delta = (new_int + new_left + new_right) - (old_int + old_left + old_right)
                
                if delta > 0:
                    # Apply move
                    new_total = update_costs(order, cost_forward, cost_backward, prefix_forward, prefix_backward, i, j)
                    # Verify total score
                    # total_score should be new_total
                    total_score = new_total
                    improved = True
                    # Restart i loop or continue? Usually continue or restart.
                    # Breaking out to restart i loop is safer to avoid invalid indices if we don't update carefully.
                    # But we updated in place.
                    break
            if improved:
                # Restart i loop
                continue
                
    # Construct final word
    final_word = []
    for idx, p_idx in enumerate(order):
        p = perms[p_idx]
        block = list(p) + [n] + list(p)
        if idx == 0:
            final_word = block
        else:
            max_ov = min(len(final_word), len(block))
            ov = 0
            for k_ov in range(max_ov, -1, -1):
                if final_word[-k_ov:] == block[:k_ov]:
                    ov = k_ov
                    break
            final_word += block[ov:]
            
    return "".join(map(str, final_word))
