import time
import random
import sys
from heapq import heappush, heappop

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 90s limit

    def is_valid_incremental(sums_set, suffix_sums, x):
        """Check if adding x to the end of seq is valid."""
        if x in sums_set:
            return False
        for ss in suffix_sums:
            if x + ss in sums_set:
                return False
        return True

    def get_new_state(sums_set, suffix_sums, x):
        """Compute new sums_set and suffix_sums after adding x."""
        new_suffix_sums = [x]
        for ss in suffix_sums:
            new_suffix_sums.append(x + ss)
        new_sums_set = sums_set.union(new_suffix_sums)
        return new_sums_set, new_suffix_sums

    def heuristic(sums_set, last_val, current_len):
        """
        A* heuristic: estimate max remaining length based on sum density.
        Higher values of x create larger gaps in the sum space.
        We estimate how many more elements we can add by looking at
        the density of occupied sums in the range [last_val+1, 256].
        """
        # The maximum possible sum of any consecutive block is at most 256 * L
        # where L is the final length. With current_len elements, remaining room
        # is limited by how many new distinct sums each new element must create.
        
        # Key insight: each new element x creates (current_len + 1) new sums.
        # These sums must all be distinct from existing sums.
        # The total sum space up to 256 * (current_len + remaining) is bounded.
        
        # Simpler heuristic: count how many candidate values > last_val
        # would not immediately conflict, weighted by their "gap quality"
        # (larger values tend to create sums in less crowded regions).
        
        # Estimate: the number of unused sum values available
        # Total possible sums range from 1 to ~256*(current_len+10)
        max_possible_total = 256 * (current_len + 12)
        
        # Density of occupied sums
        occupied = len(sums_set)
        # Available sum space
        available_space = max_possible_total - occupied
        
        # Each new element needs (current_len + 1) new distinct sums
        # Rough estimate of remaining capacity
        if available_space <= 0:
            return 0
        
        # The number of remaining elements is bounded by available_space / (current_len + 1)
        # But also by how many values remain in [last_val+1, 256]
        remaining_values = 256 - last_val
        
        # Use a conservative estimate: min of the two bounds
        sum_bound = available_space // (current_len + 1)
        return min(remaining_values, sum_bound)

    def greedy_build(strategy, deadline):
        """Build a sequence using a specific greedy strategy."""
        seq = []
        sums_set = set()
        suffix_sums = []
        last_val = 0
        
        while time.time() < deadline:
            found = False
            if strategy == 'smallest':
                candidates = range(last_val + 1, 257)
            elif strategy == 'largest':
                candidates = range(256, last_val, -1)
            else:
                candidates = list(range(last_val + 1, 257))
                random.shuffle(candidates)
            
            for x in candidates:
                if is_valid_incremental(sums_set, suffix_sums, x):
                    seq.append(x)
                    sums_set, suffix_sums = get_new_state(sums_set, suffix_sums, x)
                    last_val = x
                    found = True
                    break
            
            if not found:
                break
        
        return seq

    def a_star_search(deadline, max_depth=20):
        """
        A* search over state space (last_val, frozenset_of_suffix_sums).
        State: (seq, sums_set, suffix_sums, last_val)
        Heuristic: estimate remaining length based on sum density.
        """
        # Priority queue: (-f, -g, counter, seq, sums_set, suffix_sums, last_val)
        # f = g + h, we maximize f so we negate for min-heap
        counter = 0
        best_seq = []
        best_len = 0
        priority_queue = []
        
        # Initial state
        init_seq = []
        init_sums = set()
        init_suffix = []
        init_last = 0
        
        h0 = heuristic(init_sums, init_last, 0)
        heappush(priority_queue, (-h0, 0, counter, init_seq, init_sums, init_suffix, init_last))
        counter += 1
        
        visited = set()
        
        while priority_queue and time.time() < deadline:
            neg_f, neg_g, _, seq, sums_set, suffix_sums, last_val = heappop(priority_queue)
            g = -neg_g
            
            # State key for visited check
            state_key = (last_val, tuple(sorted(sums_set)))
            if state_key in visited:
                continue
            visited.add(state_key)
            
            # Update best
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = list(seq)
                
            # If we've reached max depth or no more candidates, skip expansion
            if g >= max_depth:
                continue
                
            # Generate successors: try each candidate x > last_val
            candidates = []
            for x in range(last_val + 1, 257):
                if is_valid_incremental(sums_set, suffix_sums, x):
                    candidates.append(x)
            
            if not candidates:
                continue
                
            # For each candidate, compute new state and push to queue
            for x in candidates:
                new_seq = seq + [x]
                new_sums, new_suffix = get_new_state(sums_set, suffix_sums, x)
                new_last = x
                new_g = g + 1
                
                # Heuristic for new state
                h = heuristic(new_sums, new_last, new_g)
                f = new_g + h
                
                # Only push if this looks promising
                if f > best_len or new_g > best_len:
                    heappush(priority_queue, (-f, -new_g, counter, new_seq, new_sums, new_suffix, new_last))
                    counter += 1
                    
                    # Time check inside inner loop for large branching
                    if time.time() > deadline:
                        return best_seq
                    
                    # Limit queue size to prevent memory explosion
                    if len(priority_queue) > 5000:
                        break
            
            # Periodic time check
            if time.time() > deadline:
                break
        
        return best_seq

    def local_search(seq, deadline, max_iters=5000):
        """Improve the sequence using local search."""
        if not seq:
            return seq
            
        current_seq = list(seq)
        current_set = set(current_seq)
        
        def is_valid_full(s):
            if not s:
                return True
            L = len(s)
            P = [0] * (L + 1)
            for i in range(L):
                P[i + 1] = P[i] + s[i]
            sums = set()
            for i in range(L):
                for j in range(i + 1, L + 1):
                    val = P[j] - P[i]
                    if val in sums:
                        return False
                    sums.add(val)
            return True

        best_seq = list(current_seq)
        best_len = len(best_seq)
        
        iters = 0
        while time.time() < deadline and iters < max_iters:
            iters += 1
            r = random.random()
            
            if r < 0.6:
                available = [x for x in range(1, 257) if x not in current_set]
                if available:
                    x = random.choice(available)
                    new_seq = current_seq + [x]
                    new_seq.sort()
                    if is_valid_full(new_seq):
                        current_seq = new_seq
                        current_set = set(current_seq)
                        if len(current_seq) > best_len:
                            best_len = len(current_seq)
                            best_seq = list(current_seq)
            elif r < 0.8:
                if current_seq:
                    available = [x for x in range(1, 257) if x not in current_set]
                    if available:
                        x_rem = random.choice(current_seq)
                        x_add = random.choice(available)
                        new_seq = [x for x in current_seq if x != x_rem]
                        new_seq.append(x_add)
                        new_seq.sort()
                        if is_valid_full(new_seq):
                            current_seq = new_seq
                            current_set = set(current_seq)
                            if len(current_seq) > best_len:
                                best_len = len(current_seq)
                                best_seq = list(current_seq)
            else:
                if current_seq and len(current_seq) > 1:
                    x_rem = random.choice(current_seq)
                    current_seq.remove(x_rem)
                    current_set = set(current_seq)
                    
        return best_seq

    best_seq = []
    
    # Phase 1: Quick greedy to get a baseline
    for strat in ['smallest', 'largest']:
        if time.time() > deadline - 10:
            break
        seq = greedy_build(strat, deadline - 10)
        if len(seq) > len(best_seq):
            best_seq = seq

    # Phase 2: A* search with heuristic guidance
    if time.time() < deadline - 5:
        a_star_result = a_star_search(deadline - 5, max_depth=18)
        if len(a_star_result) > len(best_seq):
            best_seq = a_star_result

    # Phase 3: Local search to polish
    if time.time() < deadline - 2:
        improved = local_search(best_seq, deadline - 2, max_iters=10000)
        if len(improved) > len(best_seq):
            best_seq = improved

    # Phase 4: Random restarts with A* + local search
    while time.time() < deadline - 1:
        # Random greedy start
        seq = greedy_build('random', deadline - 3)
        # A* from this prefix
        if time.time() < deadline - 3:
            a_star_result = a_star_search(deadline - 3, max_depth=15)
            if len(a_star_result) > len(best_seq):
                best_seq = a_star_result
        # Local search polish
        if time.time() < deadline - 1:
            improved = local_search(best_seq, deadline - 1, max_iters=2000)
            if len(improved) > len(best_seq):
                best_seq = improved

    return sorted(best_seq)
