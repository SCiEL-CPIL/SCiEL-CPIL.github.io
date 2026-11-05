"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import bisect

def construct():
    # Generate primes up to 10000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for p in range(2, int(n**0.5) + 1):
            if is_prime[p]:
                for multiple in range(p*p, n + 1, p):
                    is_prime[multiple] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(10000)
    N = len(primes)
    
    if N == 0:
        return []
    
    # L[j] is a dict: gap -> (length, prev_idx, prev_gap)
    L = [dict() for _ in range(N)]
    
    # Data structures for fast prefix max queries
    # sorted_gaps[i]: list of gaps for node i, sorted
    # prefix_max_len[i]: list of max lengths for gaps <= sorted_gaps[i][k]
    # prefix_max_state_idx[i]: list of indices into sorted_gaps[i] that achieve the prefix max
    # state_prev_idxs[i]: list of prev_idx for each state in sorted order
    # state_prev_gaps[i]: list of prev_gap for each state in sorted order
    sorted_gaps = [None] * N
    prefix_max_len = [None] * N
    prefix_max_state_idx = [None] * N
    state_prev_idxs = [None] * N
    state_prev_gaps = [None] * N
    
    global_max_len = 0
    global_max_idx = -1
    global_max_gap = -1
    
    br = bisect.bisect_right
    
    for j in range(N):
        p_j = primes[j]
        L_j = L[j]
        # Initialize with sequence starting at j (length 1, last gap 0)
        L_j[0] = (1, -1, 0)
        
        # Process transitions from all i < j
        for i in range(j):
            g_new = p_j - primes[i]
            
            # Query node i for max length with last gap <= g_new
            sg_i = sorted_gaps[i]
            if sg_i is None:
                continue
                
            # Binary search for largest gap <= g_new in sorted_gaps[i]
            idx = br(sg_i, g_new) - 1
            
            if idx < 0:
                continue
                
            best_len_prev = prefix_max_len[i][idx]
            state_idx = prefix_max_state_idx[i][idx]
            
            # Retrieve the specific state in i that gave this max length
            prev_gap_used = state_prev_gaps[i][state_idx]
            
            new_len = best_len_prev + 1
            
            # Update L[j][g_new] if new_len is better
            if g_new not in L_j:
                L_j[g_new] = (new_len, i, prev_gap_used)
            else:
                old_len = L_j[g_new][0]
                if new_len > old_len:
                    L_j[g_new] = (new_len, i, prev_gap_used)
                    
        # After processing all i, build data structures for j
        items = list(L_j.items())
        items.sort(key=lambda x: x[0])
        
        n_states = len(items)
        sg = [0] * n_states
        s_prev_idx = [0] * n_states
        s_prev_gap = [0] * n_states
        lengths = [0] * n_states
        
        for k, (gap, state) in enumerate(items):
            sg[k] = gap
            lengths[k] = state[0]
            s_prev_idx[k] = state[1]
            s_prev_gap[k] = state[2]
            
        sorted_gaps[j] = sg
        state_prev_idxs[j] = s_prev_idx
        state_prev_gaps[j] = s_prev_gap
        
        # Compute prefix maxes
        pml = [0] * n_states
        pmsi = [0] * n_states
        
        current_max = 0
        current_max_idx = 0
        for k in range(n_states):
            len_k = lengths[k]
            if len_k > current_max:
                current_max = len_k
                current_max_idx = k
            pml[k] = current_max
            pmsi[k] = current_max_idx
            
        prefix_max_len[j] = pml
        prefix_max_state_idx[j] = pmsi
        
        # Update global max
        if pml[-1] > global_max_len:
            global_max_len = pml[-1]
            global_max_idx = j
            global_max_gap = sg[pmsi[-1]]
            
    # Backtrack to find the sequence
    if global_max_idx == -1:
        return [primes[0]]
        
    seq = []
    curr_idx = global_max_idx
    curr_gap = global_max_gap
    
    while curr_idx != -1:
        seq.append(primes[curr_idx])
        state = L[curr_idx][curr_gap]
        curr_idx = state[1]
        curr_gap = state[2]
        
    seq.reverse()
    return seq

# EVOLVE-BLOCK-END
