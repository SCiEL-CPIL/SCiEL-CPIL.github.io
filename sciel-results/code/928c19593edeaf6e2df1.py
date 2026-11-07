import time
import random

P = {'n': 128}

def construct():
    n = P['n']
    deadline = time.time() + 85.0
    
    # Base set: upper half
    base = set(range(65, 129))
    best_size = len(base)
    best_set = base.copy()
    
    def is_valid(S):
        """
        Verify if S is valid.
        Forbid a,b,c distinct in S such that a+b, a+c, b+c in S.
        """
        S_list = sorted(S)
        S_set = set(S_list)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                ab = a + b
                if ab not in S_set:
                    continue
                for k in range(j + 1, len(S_list)):
                    c = S_list[k]
                    ac = a + c
                    bc = b + c
                    if ac in S_set and bc in S_set:
                        return False
        return True

    def can_add(S, x):
        """
        Check if x can be added to S without creating a violation.
        S is a set.
        """
        if x in S:
            return True
        S_set = S # already a set
        # A new violation must involve x.
        # Triple {x, u, v} with u, v in S.
        # Need x+u in S_new, x+v in S_new, u+v in S_new.
        # S_new = S U {x}.
        # x+u in S_new => x+u in S (since x+u != x for u>0).
        # x+v in S_new => x+v in S.
        # u+v in S_new => u+v in S or u+v == x.
        
        S_list = list(S)
        for i in range(len(S_list)):
            u = S_list[i]
            xu = x + u
            if xu not in S_set:
                continue
            for j in range(i + 1, len(S_list)):
                v = S_list[j]
                xv = x + v
                if xv not in S_set:
                    continue
                uv = u + v
                if uv in S_set or uv == x:
                    return False
        return True

    # Greedy addition
    def greedy_add(S, candidates, rng):
        S = S.copy()
        candidates = list(candidates)
        rng.shuffle(candidates)
        for x in candidates:
            if x in S:
                continue
            if can_add(S, x):
                S.add(x)
        return S

    # Local Search: try to remove one element to allow adding others
    def local_search(S, rng):
        S = S.copy()
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try removing each element
            for x in list(S):
                if time.time() > deadline - 5:
                    break
                S.remove(x)
                # Try to add as many as possible from candidates
                candidates = [i for i in range(1, 129) if i not in S]
                rng.shuffle(candidates)
                new_S = S.copy()
                added = []
                for y in candidates:
                    if can_add(new_S, y):
                        new_S.add(y)
                        added.append(y)
                if len(new_S) > len(S):
                    S = new_S
                    improved = True
                    break
                # If no improvement from this removal, revert
                # (S is already reverted by copy, but we need to ensure we don't lose the original if we don't update)
                # Actually, we copied S at start of loop, so S is intact.
        return S

    rng = random.Random(42)
    
    # Phase 1: Greedy from base
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        S = greedy_add(base, range(1, 65), rng)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()

    # Phase 2: Local Search on best found
    if time.time() < deadline - 10:
        S = local_search(best_set, rng)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()

    # Phase 3: More random restarts with local search
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        # Random subset of base? No, keep base as it's strong.
        # Maybe perturb base?
        S = base.copy()
        # Randomly remove a few from base to see if we can add more smalls?
        # Or just greedy add smalls.
        S = greedy_add(S, range(1, 65), rng)
        # Quick local search
        S = local_search(S, rng)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()

    # Final verification
    if not is_valid(best_set):
        # Fallback to known valid set
        result = list(range(65, 129))
    else:
        result = list(best_set)
        
    return result
