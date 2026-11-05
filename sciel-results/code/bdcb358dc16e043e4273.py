import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin

    def is_valid_set(S):
        """Check if set S is valid."""
        S_list = sorted(S)
        L = len(S_list)
        for i in range(L):
            a = S_list[i]
            for j in range(i + 1, L):
                b = S_list[j]
                for k in range(j, L):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add(x, S):
        """Check if x can be added to S. Assumes x < min(S) for optimization? No, general case."""
        # For efficiency in our specific greedy (adding smaller to larger), x is min.
        # But for local search, x might be anything.
        # General check:
        # For all a < b <= c in S_new, a !| b+c.
        # S_new = S U {x}.
        # New triples involve x.
        # 1. x is a: x < b <= c in S. Check x !| b+c.
        # 2. x is b: a < x <= c. a in S, c in S, a < x, c >= x. Check a !| x+c.
        # 3. x is c: a < b <= x. a in S, b in S, a < b, b <= x. Check a !| b+x.
        
        S_list = list(S)
        S_list.append(x)
        S_list.sort()
        
        L = len(S_list)
        # We only need to check triples involving x?
        # Triples not involving x were valid in S.
        # So yes, only check triples with x.
        
        for i in range(L):
            a = S_list[i]
            if a == x:
                # x is a. b, c must be > x.
                # b, c are in S_list[i+1:]
                for j in range(i + 1, L):
                    b = S_list[j]
                    for k in range(j, L):
                        c = S_list[k]
                        if (b + c) % x == 0:
                            return False
            else:
                # a != x.
                # If b or c is x.
                # Case 2: b = x. Then a < x <= c.
                # Case 3: c = x. Then a < b <= x.
                
                # If b == x:
                # j must be index of x.
                # If c == x:
                # k must be index of x.
                
                # Simpler: just iterate all triples with x?
                # Since L is small, we can just iterate all triples in S_list and skip those without x?
                # Or just check the specific conditions.
                
                # Let's find index of x
                # This is inefficient if done inside loop.
                pass
        
        # Better approach for can_add:
        # 1. Check x as 'a':
        #    For all b, c in S with b > x, c >= b: x !| b+c
        # 2. Check x as 'b':
        #    For all a, c in S with a < x, c >= x: a !| x+c
        # 3. Check x as 'c':
        #    For all a, b in S with a < b <= x: a !| b+x
        
        valid = True
        
        # 1. x is a
        # b, c in S, b > x, c >= b
        for b in S:
            if b <= x:
                continue
            for c in S:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # 2. x is b
        # a, c in S, a < x, c >= x
        for a in S:
            if a >= x:
                continue
            for c in S:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # 3. x is c
        # a, b in S, a < b <= x
        for a in S:
            if a >= x:
                continue
            for b in S:
                if b <= a or b > x:
                    continue
                if (b + x) % a == 0:
                    return False
        
        return True

    # 1. Greedy Initialization
    # Start with safe set [171, 256]
    S = set(range(171, n + 1))
    candidates = list(range(170, 0, -1))
    
    for x in candidates:
        if time.time() > deadline - 5:
            break
        if can_add(x, S):
            S.add(x)

    best_S = set(S)
    best_size = len(S)

    # 2. Local Search
    # R = rejected candidates
    R = [x for x in range(170, 0, -1) if x not in S]
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for r in R:
            if time.time() > deadline - 10:
                break
            
            # Find blockers for r
            # Blockers are elements s in S such that removing s allows adding r?
            # Actually, we need to identify which elements in S cause conflicts with r.
            # Conflicts:
            # 1. r is a: pairs (b,c) in S with b>r, c>=b and r | b+c.
            # 2. r is b: pairs (a,c) in S with a<r, c>=r and a | r+c.
            # 3. r is c: pairs (a,b) in S with a<b<=r and a | b+r.
            
            blocker_candidates = set()
            
            # 1. r is a
            for b in S:
                if b <= r:
                    continue
                for c in S:
                    if c < b:
                        continue
                    if (b + c) % r == 0:
                        blocker_candidates.add(b)
                        blocker_candidates.add(c)
            
            # 2. r is b
            for a in S:
                if a >= r:
                    continue
                for c in S:
                    if c < r:
                        continue
                    if (r + c) % a == 0:
                        blocker_candidates.add(a)
                        blocker_candidates.add(c)
                        # a is in S, c is in S. r is new.
                        # The conflict is between a and the pair (r, c) or (c, r)?
                        # Triple is (a, r, c) or (a, c, r)?
                        # Condition: a < b <= c.
                        # If r is b, then a < r <= c. Triple (a, r, c).
                        # Conflict if a | r+c.
                        # To resolve, we must remove a, r, or c.
                        # We are trying to add r. So we must remove a or c.
                        blocker_candidates.add(a)
                        blocker_candidates.add(c)
            
            # 3. r is c
            for a in S:
                if a >= r:
                    continue
                for b in S:
                    if b <= a or b > r:
                        continue
                    if (b + r) % a == 0:
                        # Triple (a, b, r).
                        # Conflict if a | b+r.
                        # Must remove a, b, or r.
                        # Remove a or b.
                        blocker_candidates.add(a)
                        blocker_candidates.add(b)

            if not blocker_candidates:
                # Should not happen if greedy was correct, but maybe S changed?
                S.add(r)
                R.remove(r)
                improved = True
                break

            # Try removing each blocker s
            for s in blocker_candidates:
                if s not in S:
                    continue
                
                S_temp = S - {s}
                S_temp.add(r)
                
                if not can_add(r, S - {s}): # Wait, can_add checks if r can be added to S_temp?
                    # No, can_add(x, S) checks if x can be added to S.
                    # Here we want to check if S_temp is valid.
                    # S_temp = (S - {s}) U {r}.
                    # We know S - {s} is valid.
                    # We just need to check if r can be added to S - {s}.
                    pass
                
                # Actually, if we remove s, does r become addable?
                # The blockers were pairs involving s.
                # If we remove s, those specific conflicts are gone.
                # But removing s might not be enough if there are other conflicts?
                # No, the blocker_candidates are derived from ALL conflicts.
                # If we remove one element from EVERY conflicting pair, r should be addable.
                # But we are only removing ONE element s.
                # So we must ensure that s is part of ALL conflicting pairs?
                # Or just that removing s resolves ALL conflicts?
                
                # Let's just check validity of S_temp.
                if can_add(r, S - {s}):
                    # r can be added to S - {s}
                    # Now, can we add any other element from R to S_temp?
                    added = False
                    for r2 in R:
                        if r2 == r:
                            continue
                        if can_add(r2, S_temp):
                            S_temp.add(r2)
                            R.remove(r2)
                            R.remove(r)
                            S = S_temp
                            improved = True
                            added = True
                            break
                    if added:
                        break
                    # If we couldn't add r2, should we accept the swap S -> S_temp?
                    # Size is same.
                    # Maybe it allows future improvements?
                    # For now, let's only accept if size increases.
                    # Or accept swap if it seems to open up more options?
                    # Let's stick to size increase for safety.
            
            if improved:
                break

    return sorted(best_S if len(best_S) > len(S) else S)

# Test harness
if __name__ == "__main__":
    t0 = time.time()
    S = construct()
    t1 = time.time()
    print(f"Size: {len(S)}, Time: {t1-t0:.4f}s")
    # Verify
    valid = True
    L = len(S)
    for i in range(L):
        a = S[i]
        for j in range(i+1, L):
            b = S[j]
            for k in range(j, L):
                c = S[k]
                if (b+c) % a == 0:
                    valid = False
                    print(f"Invalid: {a} divides {b}+{c}")
                    break
            if not valid: break
        if not valid: break
    print("Valid" if valid else "Invalid")
