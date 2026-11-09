"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START

def construct():
    N = 128
    deadline = time.time() + 80
    best = list(range(N//2 + 1, N + 1))
    best_size = len(best)
    
    def has_forbidden_triple(s):
        """Check if s has a forbidden triple: distinct a,b,c in s
        with a+b, a+c, b+c all in s."""
        s_set = set(s)
        n = len(s)
        # For each c in s, find all x in s\{c} with x+c in s
        # Then check if any pair (a,b) from that list has a+b in s
        for c in s:
            cands = []
            for x in s:
                if x != c and x + c <= N and (x + c) in s_set:
                    cands.append(x)
            if len(cands) < 2:
                continue
            for i in range(len(cands)):
                a = cands[i]
                for j in range(i + 1, len(cands)):
                    b = cands[j]
                    if (a + b) in s_set:
                        return True
        return False
    
    def greedy(order):
        s = []
        s_set = set()
        for x in order:
            if x in s_set:
                continue
            s.append(x)
            s_set.add(x)
            if has_forbidden_triple(s):
                s.pop()
                s_set.discard(x)
        return s
    
    def local_search(s, max_iters=300):
        s = list(s)
        s_set = set(s)
        for _ in range(max_iters):
            improved = False
            # Try adding
            for x in range(1, N + 1):
                if x in s_set:
                    continue
                s.append(x)
                s_set.add(x)
                if not has_forbidden_triple(s):
                    improved = True
                    break
                s.pop()
                s_set.discard(x)
            if improved:
                continue
            # Try remove+add
            for i in range(len(s)):
                rem = s[i]
                s.pop(i)
                s_set.discard(rem)
                for x in range(1, N + 1):
                    if x in s_set:
                        continue
                    s.append(x)
                    s_set.add(x)
                    if not has_forbidden_triple(s):
                        improved = True
                        break
                    s.pop()
                    s_set.discard(x)
                if improved:
                    break
                s.insert(i, rem)
                s_set.add(rem)
            if not improved:
                break
        return s
    
    rng = random.Random(12345)
    
    # Try many random greedy constructions
    for trial in range(500):
        if time.time() > deadline - 10:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        s = greedy(order)
        if len(s) > best_size:
            best = s
            best_size = len(s)
        # Local search on promising ones
        if len(s) >= best_size - 2:
            if time.time() > deadline - 5:
                break
            s2 = local_search(s, max_iters=100)
            if len(s2) > best_size:
                best = s2
                best_size = len(s2)
    
    # Final polish
    if time.time() < deadline - 2:
        s2 = local_search(best, max_iters=500)
        if len(s2) > best_size:
            best = s2
    
    return [int(x) for x in best]

# EVOLVE-BLOCK-END
