"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12
import time
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

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    best_trans = None
    best_len = 0
    
    # Family 1: Pinwheel variants
    # a = cycle, b breaks at position k
    for break_pos in range(N):
        a = [(i + 1) % N for i in range(N)]
        b = [i + 1 for i in range(N)]
        b[N-1] = break_pos
        trans = [a, b]
        try:
            L = _bfs_reset_length(trans)
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
            L = _bfs_reset_length(trans)
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
            L = _bfs_reset_length(trans)
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
                L = _bfs_reset_length(trans)
                if L > best_len:
                    best_len = L
                    best_trans = trans
            except:
                pass
            if time.time() > deadline:
                break
    
    # Family 5: a = cycle, b = i -> i+2 mod N (different stride)
    for stride in range(2, N):
        if __import__('math').gcd(stride, N) == 1:
            a = [(i + 1) % N for i in range(N)]
            b = [(i + stride) % N for i in range(N)]
            # This is two permutations, likely not synchronizing
            # Skip unless it has a fixed point structure
            pass
    
    # Family 6: a = cycle, b = constant map to 0
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    trans = [a, b]
    try:
        L = _bfs_reset_length(trans)
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
        L = _bfs_reset_length(trans)
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
        L = _bfs_reset_length(trans)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 9: a = cycle, b = i -> (i+1) % N for i < N-2, b(N-2)=0, b(N-1)=N-1
    a = [(i + 1) % N for i in range(N)]
    b = [(i + 1) % N for i in range(N)]
    b[N-2] = 0
    b[N-1] = N-1
    trans = [a, b]
    try:
        L = _bfs_reset_length(trans)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 10: a = cycle, b = i -> i for i even, b = i+1 for i odd
    a = [(i + 1) % N for i in range(N)]
    b = [i if i % 2 == 0 else (i + 1) % N for i in range(N)]
    trans = [a, b]
    try:
        L = _bfs_reset_length(trans)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 11: a = cycle, b = i -> (i + 1) % N except b(0) = N-1 (reverse one edge)
    a = [(i + 1) % N for i in range(N)]
    b = [(i + 1) % N for i in range(N)]
    b[0] = N - 1
    trans = [a, b]
    try:
        L = _bfs_reset_length(trans)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 12: a = cycle, b = i -> (i - 1) % N (reverse cycle)
    a = [(i + 1) % N for i in range(N)]
    b = [(i - 1) % N for i in range(N)]
    trans = [a, b]
    try:
        L = _bfs_reset_length(trans)
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
        L = _bfs_reset_length(trans)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 14: a = cycle, b = i -> (i+1)%N for i < N-3, b(N-3)=0, b(N-2)=N-2, b(N-1)=N-1
    a = [(i + 1) % N for i in range(N)]
    b = [(i + 1) % N for i in range(N)]
    b[N-3] = 0
    b[N-2] = N-2
    b[N-1] = N-1
    trans = [a, b]
    try:
        L = _bfs_reset_length(trans)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 15: a = cycle, b = i -> i+1 for i<N-1, b(N-1)=1
    a = [(i + 1) % N for i in range(N)]
    b = [(i + 1) % N for i in range(N-1)]
    b.append(1)
    trans = [a, b]
    try:
        L = _bfs_reset_length(trans)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 16: a = cycle, b = i -> i+1 for i<N-1, b(N-1)=2
    a = [(i + 1) % N for i in range(N)]
    b = [(i + 1) % N for i in range(N-1)]
    b.append(2)
    trans = [a, b]
    try:
        L = _bfs_reset_length(trans)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    # Family 17: a = cycle, b = i -> i+1 for i<N-1, b(N-1)=N-3
    a = [(i + 1) % N for i in range(N)]
    b = [(i + 1) % N for i in range(N-1)]
    b.append(N - 3)
    trans = [a, b]
    try:
        L = _bfs_reset_length(trans)
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
        L = _bfs_reset_length(trans)
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
        L = _bfs_reset_length(trans)
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
        L = _bfs_reset_length(trans)
        if L > best_len:
            best_len = L
            best_trans = trans
    except:
        pass
    
    if best_trans is not None:
        return best_trans
    else:
        # Fallback: simple synchronizing automaton
        return [[(i + 1) % N for i in range(N)], [0] * N]
