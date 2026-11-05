import time
import random
import math

def construct():
    deadline = time.time() + 50  # Safety margin: return well before 60s
    
    N = 180
    
    # Precompute all forbidden triples (a, b, c) with a < b < c
    # where 1/a = 1/b + 1/c
    # Equivalent to: (b-a)(c-a) = a^2
    # For each a from 1 to N//3:
    #   For each divisor d of a^2 where d > 0 and d != a:
    #     b = a + d, c = a + a^2/d
    #     if b != c and b <= N and c <= N and b < c:
    #       add triple (a, b, c)
    
    triples = []
    triple_map = {}  # maps element to list of (other1, other2, triple_index)
    
    for a in range(1, N + 1):
        a2 = a * a
        # Find all divisors of a^2
        divs = []
        i = 1
        while i * i <= a2:
            if a2 % i == 0:
                divs.append(i)
                if i != a2 // i:
                    divs.append(a2 // i)
            i += 1
        
        for d in divs:
            if d == a:
                continue
            b = a + d
            c = a + a2 // d
            if b > N or c > N:
                continue
            if b == c:
                continue
            if b < c:
                triple = (a, b, c)
            else:
                triple = (a, c, b)
            triples.append(triple)
    
    # Build conflict map: for each element x, store which triples contain x
    # triple_conflicts[x] = list of (other1, other2) pairs
    triple_conflicts = {x: [] for x in range(1, N + 1)}
    for idx, (a, b, c) in enumerate(triples):
        triple_conflicts[a].append((b, c))
        triple_conflicts[b].append((a, c))
        triple_conflicts[c].append((a, b))
    
    def find_conflict_fast(a, S):
        """
        Check if adding 'a' to valid set 'S' creates a conflict.
        Returns (b, c) if conflict exists, else None.
        """
        for (b, c) in triple_conflicts[a]:
            if b in S and c in S:
                return (b, c)
        return None
    
    # Greedy initialization: try to add numbers in order of "difficulty"
    # Numbers with fewer conflicts are easier to add
    conflict_counts = {x: len(triple_conflicts[x]) for x in range(1, N + 1)}
    
    S = set()
    
    # Sort by number of conflicts (ascending) - easier numbers first
    ordered = sorted(range(1, N + 1), key=lambda x: conflict_counts[x])
    
    for a in ordered:
        if time.time() > deadline:
            break
        if find_conflict_fast(a, S) is None:
            S.add(a)
    
    best_S = set(S)
    best_size = len(S)
    
    all_nums = set(range(1, N + 1))
    
    random.seed(42)
    
    # Local search with targeted swaps
    stuck_count = 0
    max_stuck = 3000
    
    while time.time() < deadline:
        excluded = all_nums - S
        if not excluded:
            break
        
        # Try to add a random excluded element
        a = random.choice(list(excluded))
        conflict = find_conflict_fast(a, S)
        
        if conflict is None:
            S.add(a)
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
            stuck_count = 0
        else:
            b, c = conflict
            # Try to remove b and add a
            S.remove(b)
            new_conflict = find_conflict_fast(a, S)
            if new_conflict is None:
                S.add(a)
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
                stuck_count = 0
            else:
                # Revert b
                S.add(b)
                # Try to remove c and add a
                S.remove(c)
                new_conflict2 = find_conflict_fast(a, S)
                if new_conflict2 is None:
                    S.add(a)
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
                    stuck_count = 0
                else:
                    # Revert c
                    S.add(c)
                    # Try 2-swap: remove b and c, add a and some other excluded element
                    if len(excluded) > 1:
                        other_excluded = list(excluded - {a})
                        if other_excluded:
                            d = random.choice(other_excluded)
                            S.remove(b)
                            S.remove(c)
                            new_conflict3 = find_conflict_fast(a, S)
                            if new_conflict3 is None:
                                S.add(a)
                                new_conflict4 = find_conflict_fast(d, S)
                                if new_conflict4 is None:
                                    S.add(d)
                                    if len(S) > best_size:
                                        best_S = set(S)
                                        best_size = len(S)
                                    stuck_count = 0
                                else:
                                    S.add(b)
                                    S.add(c)
                                    stuck_count += 1
                            else:
                                S.add(b)
                                S.add(c)
                                stuck_count += 1
                    else:
                        # Fallback: random swap
                        if S:
                            b_rand = random.choice(list(S))
                            S.remove(b_rand)
                            new_conflict3 = find_conflict_fast(a, S)
                            if new_conflict3 is None:
                                S.add(a)
                                if len(S) > best_size:
                                    best_S = set(S)
                                    best_size = len(S)
                                stuck_count = 0
                            else:
                                S.add(b_rand)
                                stuck_count += 1
                        else:
                            stuck_count += 1
                            
        # If stuck for too long, restart from best state with perturbation
        if stuck_count > max_stuck:
            S = set(best_S)
            if S:
                for _ in range(3):
                    if not S: break
                    S.remove(random.choice(list(S)))
            stuck_count = 0

    return sorted(best_S)
