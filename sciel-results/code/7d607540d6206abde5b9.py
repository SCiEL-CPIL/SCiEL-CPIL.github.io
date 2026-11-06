import time
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s kill

    # Primes up to 127
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Precompute factor vectors
    def get_vec(n):
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
            if temp == 1:
                break
        return tuple(v)
        
    vecs = {i: get_vec(i) for i in range(2, 129)}
    zero_vec = (0,) * num_primes

    # State: (neg_score, length, current_index, last_prefix, diff_set)
    # diff_set is a frozenset of tuples for hashing/comparison
    
    # Initial state
    # length 0, index 1 (next number to consider is 2)
    # last_prefix = 0
    # diff_set = empty
    initial_state = (-1 * (0 * 10000 + 1), 0, 1, zero_vec, frozenset())
    
    # Priority Queue: Min-heap based on neg_score
    # We want to pop the state with the highest (length, -index) priority.
    # Score = length * 10000 - current_index
    # Neg Score = -length * 10000 + current_index
    
    pq = [initial_state]
    best_length = 0
    best_seq = [] # We need to reconstruct the sequence? Or just return length?
    # The task requires returning the sequence.
    # So we need to store the sequence or be able to reconstruct it.
    # Storing the sequence in the PQ state makes it heavy.
    # Alternative: Store parent pointers? Complex with PQ.
    # Given the length is small (~30-40), storing the list of ints is fine.
    
    # Let's redefine state to include the sequence for easy return.
    # State: (neg_score, length, current_index, last_prefix, diff_set, seq_list)
    
    pq = [(-1 * (0 * 10000 + 1), 0, 1, zero_vec, frozenset(), [])]
    
    # Beam width limit to prevent memory explosion
    max_queue_size = 5000
    
    while pq:
        if time.time() > deadline:
            break
            
        # Pop best state
        neg_score, length, curr_idx, last_pref, diff_set, seq = heapq.heappop(pq)
        
        # Update best
        if length > best_length:
            best_length = length
            best_seq = seq
            
        # If we've processed all numbers, continue to see if we can get longer?
        # No, if curr_idx > 128, we can't add more.
        if curr_idx > 128:
            continue
            
        n = curr_idx + 1
        v_n = vecs.get(n)
        if v_n is None:
            # Should not happen if curr_idx < 128
            continue
            
        # Option 1: Exclude n
        # Score: length * 10000 - (n)
        # Neg Score: -length * 10000 + n
        neg_score_excl = -length * 10000 + n
        # Heuristic: If we exclude, length stays same.
        # We push this state.
        # To avoid pushing too many "exclude" states that are identical in (last_pref, diff_set) but different index?
        # No, index is different.
        
        # Option 2: Include n
        # Check validity
        new_pref_list = [last_pref[i] + v_n[i] for i in range(num_primes)]
        new_pref = tuple(new_pref_list)
        
        valid = True
        new_diffs = []
        
        # Generate new differences ending at n
        # S_new - S_i for all previous prefixes S_i
        # We don't store all prefixes, only the diff set?
        # WAIT. To generate new diffs, we need the previous prefix vectors S_i.
        # The state `diff_set` only contains differences, not the prefixes themselves.
        # We CANNOT generate new diffs without the list of previous prefix vectors!
        
        # Correction: The state MUST include the list of prefix vectors.
        # State: (..., prefix_list)
        # prefix_list[0] = 0
        # prefix_list[k] = last_pref
        
        # This increases state size but is necessary.
        
        # Let's restart the state definition.
        pass

    # The previous logic was flawed because I forgot we need the prefix list to compute new diffs.
    # Let's implement it correctly.

    # State: (neg_score, length, current_index, prefix_list, diff_set, seq_list)
    # prefix_list is a list of tuples.
    # diff_set is a frozenset of tuples.
    
    initial_prefix_list = [zero_vec]
    initial_diff_set = frozenset()
    initial_seq = []
    
    # Score: length * 100000 - current_index
    # Neg Score: -length * 100000 + current_index
    
    pq = [(-0 * 100000 + 1, 0, 1, initial_prefix_list, initial_diff_set, initial_seq)]
    
    best_length = 0
    best_seq = []
    
    max_queue_size = 2000 # Keep it manageable
    
    while pq:
        if time.time() > deadline:
            break
            
        neg_score, length, curr_idx, pref_list, diff_set, seq = heapq.heappop(pq)
        
        if length > best_length:
            best_length = length
            best_seq = seq
            
        if curr_idx > 128:
            continue
            
        n = curr_idx + 1
        v_n = vecs[n]
        
        # Option 1: Exclude n
        # New state: same pref_list, diff_set, seq, index n
        # Score: length * 100000 - n
        # Neg: -length * 100000 + n
        
        # We only push if the queue isn't too full or if it's better than the worst in queue?
        # Simple approach: Push, then if len(pq) > max_queue_size, remove worst.
        # But heapq doesn't support efficient removal of worst.
        # We can use a "beam" logic:
        # Collect all children, sort by score, keep top K.
        # But this is DFS/BFS hybrid.
        
        # Let's stick to PQ and a counter.
        # If we exceed max_queue_size, we discard the lowest priority item?
        # No, PQ pops best. The "worst" are at the bottom.
        # We can periodically prune the bottom?
        # Or just limit the expansion: If len(pq) > max_queue_size, we don't push new states?
        # That would stop the search prematurely.
        
        # Better: Use a Beam Search approach on top of PQ?
        # Or just use a standard Beam Search.
        
        pass

    # Given the complexity of managing PQ size, let's revert to a simple Beam Search
    # but with a smarter scoring function:
    # Score = Length + Potential
    # Potential = Number of next 5 numbers that are compatible.
    
    # Let's implement a robust Beam Search.
    
    def get_vec_fast(n):
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
            if temp == 1:
                break
        return tuple(v)

    vecs = {i: get_vec_fast(i) for i in range(2, 129)}
    zero_vec = (0,) * num_primes

    # Beam State: (length, seq, prefix_list, diff_set)
    # We sort by length (desc) and then by potential (desc).
    
    beam = [(0, [], [zero_vec], frozenset())]
    
    # Dynamic beam width: wider early, narrower late
    def get_beam_width(step, total_steps):
        if step < total_steps * 0.3:
            return 150
        elif step < total_steps * 0.6:
            return 120
        elif step < total_steps * 0.85:
            return 80
        else:
            return 60
    
    total_steps = 127  # numbers 2 to 128
    
    for step, n in enumerate(range(2, 129)):
        if time.time() > deadline:
            break
            
        current_beam_width = get_beam_width(step, total_steps)
        
        new_beam = []
        v_n = vecs[n]
        
        for state in beam:
            length, seq, pref_list, diff_set = state
            last_pref = pref_list[-1]
            
            # Option 1: Exclude
            new_beam.append((length, seq, pref_list, diff_set))
            
            # Option 2: Include
            new_pref_list_val = [last_pref[i] + v_n[i] for i in range(num_primes)]
            new_pref = tuple(new_pref_list_val)
            
            valid = True
            new_diffs = []
            
            # Check all previous prefixes
            for p in pref_list:
                diff_list = [new_pref[i] - p[i] for i in range(num_primes)]
                diff_tup = tuple(diff_list)
                if diff_tup in diff_set:
                    valid = False
                    break
                new_diffs.append(diff_tup)
            
            if valid:
                new_pref_list = pref_list + [new_pref]
                new_diff_set = diff_set.union(new_diffs)
                new_seq = seq + [n]
                new_length = length + 1
                new_beam.append((new_length, new_seq, new_pref_list, new_diff_set))
        
        # Sort new_beam by Length (desc), then by "Flexibility"
        def calc_score(state):
            l, seq, pl, ds = state
            # Potential: count compatible next 5 numbers with weighting
            potential = 0.0
            last_p = pl[-1]
            # Next numbers are n+1, n+2, n+3, n+4, n+5
            for i, m in enumerate(range(n+1, min(n+6, 129))):
                weight = 1.0 / (i + 1)  # closer numbers matter more
                v_m = vecs[m]
                new_p_val = [last_p[i2] + v_m[i2] for i2 in range(num_primes)]
                new_p = tuple(new_p_val)
                ok = True
                for p in pl:
                    d_val = [new_p[i2] - p[i2] for i2 in range(num_primes)]
                    if tuple(d_val) in ds:
                        ok = False
                        break
                if ok:
                    potential += weight
            return (l, potential)

        new_beam.sort(key=calc_score, reverse=True)
        beam = new_beam[:current_beam_width]
        
        # Early stop if best possible is reached
        if beam and beam[0][0] == 127:
            return beam[0][1]

    if beam:
        return beam[0][1]
    return []
