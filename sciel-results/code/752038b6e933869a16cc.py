import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    # Precompute conflict masks for speed
    # conflict_mask[a] = set of (b,c) pairs where a | b+c
    # But we need to check if adding x creates conflicts.
    # Let's use a different approach: for each x, precompute which pairs in [1..n] conflict with x.
    
    # For efficiency, let's precompute for each a in 1..n:
    # divisors_of_sums[a] = set of s such that a | s
    # Then for a given a, b, c: conflict if b+c is in divisors_of_sums[a]
    
    # Actually, let's just use direct computation but with bitmasks for the set.
    
    def can_add_fast(x, S_list):
        """Check if x can be added to sorted list S_list. Returns True if valid."""
        # S_list is sorted and does not contain x.
        # We need to check all triples (a,b,c) with a<b<=c in S_list ∪ {x}
        # that involve x.
        
        # Insert x into sorted position
        import bisect
        pos = bisect.bisect_left(S_list, x)
        
        # Case 1: x is the smallest element (a = x)
        # For all b, c in S_list with b > x, c >= b: check x !| b+c
        # b and c are in S_list[pos:] (elements > x)
        for i in range(pos, len(S_list)):
            b = S_list[i]
            for j in range(i, len(S_list)):
                c = S_list[j]
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is the middle element (b = x)
        # For all a, c in S_list with a < x, c >= x: check a !| x+c
        # a is in S_list[:pos], c is in S_list[pos:] (but c >= x, so c in S_list[pos:] since S_list[pos-1] < x <= S_list[pos])
        # Wait, c >= x. Elements in S_list that are >= x start at pos.
        for i in range(pos):
            a = S_list[i]
            for j in range(pos, len(S_list)):
                c = S_list[j]
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is the largest element (c = x)
        # For all a, b in S_list with a < b <= x: check a !| b+x
        # a and b are in S_list[:pos] (elements < x), with a < b
        for i in range(pos):
            a = S_list[i]
            for j in range(i+1, pos):
                b = S_list[j]
                if (b + x) % a == 0:
                    return False
        
        return True

    def get_blockers(x, S_list):
        """Return set of elements in S_list that block adding x."""
        import bisect
        pos = bisect.bisect_left(S_list, x)
        blockers = set()
        
        # Case 1: x is a
        for i in range(pos, len(S_list)):
            b = S_list[i]
            for j in range(i, len(S_list)):
                c = S_list[j]
                if (b + c) % x == 0:
                    blockers.add(b)
                    blockers.add(c)
        
        # Case 2: x is b
        for i in range(pos):
            a = S_list[i]
            for j in range(pos, len(S_list)):
                c = S_list[j]
                if (x + c) % a == 0:
                    blockers.add(a)
                    blockers.add(c)
        
        # Case 3: x is c
        for i in range(pos):
            a = S_list[i]
            for j in range(i+1, pos):
                b = S_list[j]
                if (b + x) % a == 0:
                    blockers.add(a)
                    blockers.add(b)
        
        return blockers

    # Greedy initialization with multiple starting points
    best_S = set()
    best_size = 0
    
    # Try different starting ranges
    start_points = [171, 160, 150, 140, 130, 120, 100, 80]
    
    for start_val in start_points:
        if time.time() > deadline - 30:
            break
        
        S = set(range(start_val, n + 1))
        S_list = sorted(S)
        
        # Greedy: try to add elements from start_val-1 down to 1
        for x in range(start_val - 1, 0, -1):
            if time.time() > deadline - 25:
                break
            if can_add_fast(x, S_list):
                S.add(x)
                S_list = sorted(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    # Local search with swap moves
    S = best_S
    S_list = sorted(S)
    R = [x for x in range(1, n + 1) if x not in S]
    
    improved = True
    max_iterations = 1000
    iteration = 0
    
    while improved and time.time() < deadline - 15 and iteration < max_iterations:
        improved = False
        iteration += 1
        S_list = sorted(S)
        
        # Try adding each rejected element
        for r in R[:]:
            if time.time() > deadline - 15:
                break
            
            if can_add_fast(r, S_list):
                S.add(r)
                R.remove(r)
                improved = True
                continue
            
            # Try swap: remove one blocker, add r
            blockers = get_blockers(r, S_list)
            if not blockers:
                continue
            
            for s in sorted(blockers):
                if s not in S:
                    continue
                
                # Try removing s and adding r
                S_temp = S - {s}
                S_temp_list = sorted(S_temp)
                
                if can_add_fast(r, S_temp_list):
                    # Check if we can also add another element
                    S_new = S_temp | {r}
                    S_new_list = sorted(S_new)
                    
                    # Try to add more elements from R
                    added_more = False
                    for r2 in R:
                        if r2 == r:
                            continue
                        if can_add_fast(r2, S_new_list):
                            S_new.add(r2)
                            S_new_list = sorted(S_new)
                            R.remove(r2)
                            added_more = True
                            break
                    
                    if added_more or len(S_new) > len(S):
                        S = S_new
                        R.remove(r)
                        improved = True
                        break
        
        if improved:
            continue
        
        # Try random swaps
        if time.time() < deadline - 15 and len(R) > 0 and len(S) > 0:
            for _ in range(50):
                if time.time() > deadline - 15:
                    break
                
                r = random.choice(R)
                s = random.choice(list(S))
                
                S_temp = S - {s}
                S_temp_list = sorted(S_temp)
                
                if can_add_fast(r, S_temp_list):
                    S_new = S_temp | {r}
                    S_new_list = sorted(S_new)
                    
                    # Try to add more
                    for r2 in R:
                        if r2 == r:
                            continue
                        if can_add_fast(r2, S_new_list):
                            S_new.add(r2)
                            S_new_list = sorted(S_new)
                            R.remove(r2)
                            break
                    
                    if len(S_new) >= len(S):
                        S = S_new
                        R.remove(r)
                        improved = True
                        break
    
    # Final polish: try to add any remaining elements
    S_list = sorted(S)
    for x in range(1, n + 1):
        if x in S:
            continue
        if can_add_fast(x, S_list):
            S.add(x)
            S_list = sorted(S)
    
    return sorted(S)

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
