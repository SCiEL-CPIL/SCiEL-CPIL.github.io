"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12
import time
import random
import math
from collections import deque

def _bfs_reset_length(trans):
    """Compute exact shortest reset word length via BFS on subset automaton."""
    # trans: list of two lists, trans[letter][state] = dest
    # Start from full set {0,1,...,N-1}
    start = tuple(range(N))
    if len(set(start)) == 1:
        return 0
    # BFS
    visited = {start: 0}
    queue = deque([start])
    while queue:
        s = queue.popleft()
        d = visited[s]
        for letter in range(2):
            # apply letter to each state in subset
            new_set = tuple(sorted(set(trans[letter][st] for st in s)))
            if len(new_set) == 1:
                return d + 1
            if new_set not in visited:
                visited[new_set] = d + 1
                queue.append(new_set)
    return -1  # not synchronizing

def _bitmask_reset_length(a, b, timeout=None):
    """
    Fast BFS on subset automaton using bitmasks.
    a, b: lists of length N, a[i] and b[i] are destinations for state i.
    Returns shortest reset word length, or -1 if not synchronizing.
    """
    full = (1 << N) - 1
    if full == 0:
        return 0
    
    # Precompute subset transitions for single states
    # For each state and each letter, what state does it map to
    # For a subset S (bitmask), applying letter L gives:
    #   {trans_L(s) for s in S}
    
    # BFS
    visited = {full: 0}
    queue = deque([full])
    
    while queue:
        s = queue.popleft()
        d = visited[s]
        
        for letter in range(2):
            trans = a if letter == 0 else b
            # Compute image of subset s under trans
            new_mask = 0
            temp = s
            while temp:
                # Get lowest set bit
                lb = temp & (-temp)
                idx = lb.bit_length() - 1
                new_mask |= (1 << trans[idx])
                temp ^= lb
            
            if new_mask & (new_mask - 1) == 0:  # single element
                return d + 1
            if new_mask not in visited:
                visited[new_mask] = d + 1
                queue.append(new_mask)
    
    return -1

def _quick_synchronizing_check(a, b):
    """
    Quick check: is there any chance this automaton synchronizes?
    For binary automata, a necessary condition is that for some letter,
    the map is not a permutation (has a collision).
    """
    # Check if a has a collision
    seen_a = set(a)
    has_collision_a = len(seen_a) < N
    
    # Check if b has a collision
    seen_b = set(b)
    has_collision_b = len(seen_b) < N
    
    # If both are permutations, they can't synchronize (group action on finite set)
    if not has_collision_a and not has_collision_b:
        return False
    return True

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    best_trans = None
    best_len = 0
    
    # --- Phase 1: Family-based search (warm start) ---
    
    # Family 1: Pinwheel variants
    # a = cycle, b breaks at position k
    for break_pos in range(N):
        a = [(i + 1) % N for i in range(N)]
        b = [i + 1 for i in range(N)]
        b[N-1] = break_pos
        trans = [a, b]
        try:
            L = _bitmask_reset_length(a, b)
            if L > best_len:
                best_len = L
                best_trans = trans
        except:
            pass
        if time.time() > deadline:
            break
    
    # Family 2: a = cycle, b = identity except one state maps to 0
    for break_pos in range(N):
        a = [(i + 1) % N for i in range(N)]
        b = [i for i in range(N)]
        b[break_pos] = 0
        trans = [a, b]
        try:
            L = _bitmask_reset_length(a, b)
            if L > best_len:
                best_len = L
                best_trans = trans
        except:
            pass
        if time.time() > deadline:
            break
    
    # Family 3: a = cycle, b maps i -> i+1 except last maps to some k
    for k in range(N):
        a = [(i + 1) % N for i in range(N)]
        b = [(i + 1) % N for i in range(N)]
        b[N-1] = k
        trans = [a, b]
        try:
            L = _bitmask_reset_length(a, b)
            if L > best_len:
                best_len = L
                best_trans = trans
        except:
            pass
        if time.time() > deadline:
            break
    
    # Family 4: a = shift by 1, b = shift by 1 except one "jump"
    for jump_src in range(N):
        for jump_dst in range(N):
            a = [(i + 1) % N for i in range(N)]
            b = [(i + 1) % N for i in range(N)]
            b[jump_src] = jump_dst
            trans = [a, b]
            try:
                L = _bitmask_reset_length(a, b)
                if L > best_len:
                    best_len = L
                    best_trans = trans
            except:
                pass
            if time.time() > deadline:
                break
    
    # Family 6: a = cycle, b = constant map to 0
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    trans = [a, b]
    try:
        L = _bitmask_reset_length(a, b)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 7: a = cycle, b = i -> max(i-1, 0) (shift left with clamp)
    a = [(i + 1) % N for i in range(N)]
    b = [max(i - 1, 0) for i in range(N)]
    trans = [a, b]
    try:
        L = _bitmask_reset_length(a, b)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 8: a = cycle, b = i -> (i * 2) % N
    a = [(i + 1) % N for i in range(N)]
    b = [(i * 2) % N for i in range(N)]
    trans = [a, b]
    try:
        L = _bitmask_reset_length(a, b)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 13: a = cycle, b = i -> i+1 for i<N-1, b(N-1)=N-2
    a = [(i + 1) % N for i in range(N)]
    b = [(i + 1) % N for i in range(N-1)]
    b.append(N - 2)
    trans = [a, b]
    try:
        L = _bitmask_reset_length(a, b)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 18: a = cycle, b = i -> (i+1)%N except b(0)=0 (one fixed point)
    a = [(i + 1) % N for i in range(N)]
    b = [(i + 1) % N for i in range(N)]
    b[0] = 0
    trans = [a, b]
    try:
        L = _bitmask_reset_length(a, b)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 19: a = cycle, b = i -> (i+1)%N except b(1)=0
    a = [(i + 1) % N for i in range(N)]
    b = [(i + 1) % N for i in range(N)]
    b[1] = 0
    trans = [a, b]
    try:
        L = _bitmask_reset_length(a, b)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 20: a = cycle, b = i -> (i+1)%N except b(2)=0
    a = [(i + 1) % N for i in range(N)]
    b = [(i + 1) % N for i in range(N)]
    b[2] = 0
    trans = [a, b]
    try:
        L = _bitmask_reset_length(a, b)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # --- Phase 2: Simulated Annealing Search ---
    if best_trans is None:
        # Fallback initial
        best_trans = [[(i + 1) % N for i in range(N)], [0] * N]
        best_len = _bitmask_reset_length(best_trans[0], best_trans[1])
    
    # Start SA from best found
    cur_a = list(best_trans[0])
    cur_b = list(best_trans[1])
    cur_len = best_len
    
    # SA parameters
    T0 = 5.0  # initial temperature
    T_min = 0.01
    cooling = 0.995  # geometric cooling
    T = T0
    
    # Track best
    best_a_sa = list(cur_a)
    best_b_sa = list(cur_b)
    best_len_sa = cur_len
    
    # Precompute: for fast evaluation, we'll use _bitmask_reset_length
    # But that's still BFS. Let's add a quick upper bound check.
    
    # To speed up, we can first check if the automaton is synchronizing
    # using a quick heuristic, then do full BFS only if promising.
    
    iterations = 0
    sa_start = time.time()
    
    while time.time() < deadline - 5:  # leave 5s margin
        T = max(T * cooling, T_min)
        
        # Generate a neighbor
        # Mutate one state in one transition function
        if random.random() < 0.5:
            # Mutate a
            new_a = list(cur_a)
            new_b = list(cur_b)
            idx = random.randint(0, N - 1)
            new_a[idx] = random.randint(0, N - 1)
        else:
            # Mutate b
            new_a = list(cur_a)
            new_b = list(cur_b)
            idx = random.randint(0, N - 1)
            new_b[idx] = random.randint(0, N - 1)
        
        # Quick check: is it synchronizing?
        if not _quick_synchronizing_check(new_a, new_b):
            # Both are permutations - skip
            continue
        
        # Evaluate
        new_len = _bitmask_reset_length(new_a, new_b)
        iterations += 1
        
        if new_len < 0:
            # Not synchronizing, treat as very bad
            new_len = -1000
        
        # Acceptance
        if new_len > cur_len:
            # Better
            cur_a = new_a
            cur_b = new_b
            cur_len = new_len
            if new_len > best_len_sa:
                best_len_sa = new_len
                best_a_sa = list(new_a)
                best_b_sa = list(new_b)
        else:
            # Worse or equal: accept with probability exp((new_len - cur_len) / T)
            delta = new_len - cur_len
            if delta >= 0:
                accept_prob = 1.0
            else:
                # delta is negative
                accept_prob = math.exp(delta / T)
            
            if random.random() < accept_prob:
                cur_a = new_a
                cur_b = new_b
                cur_len = new_len
            # else reject, keep current
    
        # Update global best
        if best_len_sa > best_len:
            best_len = best_len_sa
            best_trans = [list(best_a_sa), list(best_b_sa)]
    
    # Final verification with original BFS
    if best_trans is not None:
        final_len = _bfs_reset_length(best_trans)
        if final_len > 0:
            return best_trans
        # If verification fails, try to find a valid one
        # Fallback
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        return [a, b]
    else:
        # Fallback: simple synchronizing automaton
        return [[(i + 1) % N for i in range(N)], [0] * N]
