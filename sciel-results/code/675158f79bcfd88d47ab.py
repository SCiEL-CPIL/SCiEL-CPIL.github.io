import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit, 10s margin
    
    # Sieve of Eratosthenes
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, flag in enumerate(is_prime) if flag]
    n = len(primes)
    
    # states[i] is a list of tuples (length, last_gap, prev_index)
    # prev_index is the index in primes of the previous prime in the sequence
    # -1 for the first element
    states = [[] for _ in range(n)]
    states[0] = [(1, 0, -1)]
    
    max_len = 1
    max_idx = 0
    
    for i in range(1, n):
        if time.time() > deadline:
            break
            
        new_states = []
        pi = primes[i]
        
        for j in range(i):
            if time.time() > deadline:
                break
                
            pj = primes[j]
            gap = pi - pj
            
            for l, g, prev_idx in states[j]:
                if gap >= g:
                    new_states.append((l + 1, gap, j))
        
        if not new_states:
            # Should not happen, as we can always start a new sequence of length 1?
            # Wait, the problem says "increasing prime integers". 
            # Does a sequence of length 1 count? 
            # "successive gaps NONDECREASING". 
            # If length 1, there are no gaps. Vacuously true.
            # But we can always append to a length 1 sequence?
            # If states[j] is empty, we can't extend.
            # But we can always start a new sequence at i?
            # The problem asks for a subsequence.
            # So (pi) is a valid sequence of length 1.
            # My DP initializes states[0] with length 1.
            # For i>0, if no extension is possible, we should have a state (1, 0, -1) for i?
            # No, (1, 0, -1) means the sequence is just [pi].
            # But if we start a new sequence at i, it's length 1.
            # However, we want to MAXIMIZE length.
            # If we can't extend any previous sequence, the best ending at i is length 1.
            # So we should add (1, 0, -1) to new_states if it's not dominated?
            # Actually, (1, 0, -1) is dominated by any (l>1, g, j) because l>1 > 1.
            # So if new_states is not empty, (1,0,-1) is dominated.
            # If new_states IS empty, we should add (1, 0, -1).
            new_states.append((1, 0, -1))

        # Reduce new_states to non-dominated
        # Sort by length descending, then gap ascending
        new_states.sort(key=lambda x: (-x[0], x[1]))
        
        reduced = []
        min_gap = float('inf')
        for l, g, prev_idx in new_states:
            if g < min_gap:
                reduced.append((l, g, prev_idx))
                min_gap = g
        
        states[i] = reduced
        
        # Update max
        if reduced:
            # The first element in reduced has the max length
            # Because we sorted by length desc
            curr_max_len = reduced[0][0]
            if curr_max_len > max_len:
                max_len = curr_max_len
                max_idx = i
    
    # Reconstruct
    # Find the state with max length
    # It's in states[max_idx]
    # Actually, max_len might be in any state. 
    # We tracked max_idx based on reduced[0][0].
    # But reduced[0][0] is the max length ending at i.
    # So max_len is correct, and max_idx is the index of the prime where this max length was found.
    # We need to trace back from states[max_idx].
    # But which state in states[max_idx] to pick?
    # The one with max length. If multiple, any.
    # Let's find the state with length == max_len in states[max_idx].
    
    best_state = None
    for s in states[max_idx]:
        if s[0] == max_len:
            best_state = s
            break
            
    if not best_state:
        return primes[:max_len] # Fallback, should not happen
    
    # Trace back
    seq = []
    curr_idx = max_idx
    while curr_idx != -1:
        seq.append(primes[curr_idx])
        # Find the prev_idx
        # We need to find the state in states[curr_idx] that we used.
        # Wait, best_state has prev_idx.
        # But after tracing one step, we are at prev_idx.
        # We need to find the state in states[prev_idx] that led to the current state.
        # This is tricky because we reduced states.
        # The state (l, g, prev_idx) in states[curr_idx] implies that the previous state was in states[prev_idx] with length l-1 and gap <= g.
        # Actually, the transition was from a specific state in states[prev_idx].
        # Did we store which one?
        # In the DP loop:
        # for l, g, prev_idx in states[j]:
        #    if gap >= g: new_states.append((l+1, gap, j))
        # So the new state's prev_idx is j.
        # But we don't know WHICH state in states[j] was used.
        # However, we know the length was l.
        # So in states[j], we need to find a state with length l.
        # And we need one that could produce the gap.
        # Actually, any state in states[j] with length l and gap <= (primes[i]-primes[j]) works.
        # Since we reduced states[j], the kept states are non-dominated.
        # We can just pick any state in states[j] with length l.
        # Is it guaranteed to be valid?
        # Yes, because if it was kept, it means it was a valid state.
        # And the transition condition was checked.
        # So we can just pick the first state in states[j] with length l.
        
        # Update curr_idx to prev_idx
        # But we need to update best_state to the one in states[curr_idx]?
        # No, we just need the indices.
        
        # Get prev_idx from best_state
        prev_idx = best_state[2]
        curr_len = best_state[0]
        
        if prev_idx == -1:
            break
            
        # Find state in states[prev_idx] with length curr_len - 1
        next_best_state = None
        for s in states[prev_idx]:
            if s[0] == curr_len - 1:
                next_best_state = s
                break
                
        if next_best_state is None:
            # Error in reconstruction logic
            break
            
        best_state = next_best_state
        curr_idx = prev_idx
        
    seq.reverse()
    return seq

# Run construct
# construct()
