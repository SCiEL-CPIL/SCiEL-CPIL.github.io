import time
import random

_DEADLINE = 150.0
_MARGIN = 5.0

def _is_sidon(s):
    n = len(s)
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = s[j] - s[i]
            if d in seen:
                return False
            seen.add(d)
    return True

def _covers_1_30(s):
    n = len(s)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(s[j] - s[i])
    for d in range(1, 31):
        if d not in diffs:
            return False
    return True

def _valid(s):
    if 0 not in s:
        return False
    if len(s) > 32:
        return False
    if max(s) > 10**6:
        return False
    return _is_sidon(s) and _covers_1_30(s)

def _greedy_construct(n):
    s = [0]
    for _ in range(n - 1):
        found = False
        for x in range(1, 100):
            if x in s:
                continue
            new_s = sorted(s + [x])
            if _is_sidon(new_s):
                s = new_s
                found = True
                break
        if not found:
            break
    return s

def _local_search(s, rng, deadline):
    s = sorted(s)
    best_s = s
    best_max = max(s)
    
    # We only accept moves that maintain validity (Sidon + Cover)
    # This might get stuck if initial set is bad.
    # But we start with a valid set.
    
    t = 1.0
    while time.time() < deadline - _MARGIN and t > 0.01:
        improved = False
        for i in range(len(s)):
            if s[i] == 0:
                continue
            old_val = s[i]
            # Try small moves
            for delta in [-3, -2, -1, 1, 2, 3]:
                new_val = old_val + delta
                if new_val <= 0 or new_val in s:
                    continue
                new_s = sorted([x if x != old_val else new_val for x in s])
                if _valid(new_s):
                    new_max = max(new_s)
                    if new_max < best_max:
                        best_max = new_max
                        best_s = new_s
                        s = new_s
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # Random restart within bounds
            i = rng.randint(1, len(s) - 1)
            old_val = s[i]
            new_val = rng.randint(1, best_max + 5)
            if new_val not in s:
                new_s = sorted([x if x != old_val else new_val for x in s])
                if _valid(new_s):
                    s = new_s
                    if max(s) < best_max:
                        best_max = max(s)
                        best_s = s[:]
        
        t *= 0.95
        
    return best_s

def construct():
    start = time.time()
    deadline = start + _DEADLINE
    rng = random.Random(42)
    
    best_set = None
    best_max = 10**6
    
    # Try size 9
    for n in [9, 10]:
        if time.time() > deadline - _MARGIN:
            break
        
        # Greedy start
        s_greedy = _greedy_construct(n)
        if _valid(s_greedy):
            m = max(s_greedy)
            if m < best_max:
                best_max = m
                best_set = s_greedy[:]
            
            # Local search
            if time.time() < deadline - _MARGIN:
                opt_s = _local_search(s_greedy, rng, deadline)
                if _valid(opt_s):
                    m_opt = max(opt_s)
                    if m_opt < best_max:
                        best_max = m_opt
                        best_set = opt_s[:]
        
        # Random starts
        for _ in range(20):
            if time.time() > deadline - _MARGIN:
                break
            s_rand = [0]
            for _ in range(n - 1):
                x = rng.randint(1, 60)
                if x not in s_rand:
                    s_rand.append(x)
            s_rand = sorted(s_rand)
            if _valid(s_rand):
                m = max(s_rand)
                if m < best_max:
                    best_max = m
                    best_set = s_rand[:]
                
                if time.time() < deadline - _MARGIN:
                    opt_s = _local_search(s_rand, rng, deadline)
                    if _valid(opt_s):
                        m_opt = max(opt_s)
                        if m_opt < best_max:
                            best_max = m_opt
                            best_set = opt_s[:]
                            
    if best_set is None:
        # Fallback
        s = _greedy_construct(32)
        while not _valid(s):
            s.append(max(s) + 1)
        best_set = s
        
    return best_set
