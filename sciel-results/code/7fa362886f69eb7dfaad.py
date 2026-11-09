"""Bit-parallel maximum independent set for the no-triple-sums problem."""
import time
import random

N = 128
DEADLINE = 85.0  # safety margin before 90s kill

def construct():
    start = time.time()
    
    # Bit i (0-indexed) represents element (i+1), so bit 0 = element 1, bit 127 = element 128
    # Precompute for each pair (a,b) with a<b, the set of c values that would form a forbidden triple
    # A forbidden triple is {a,b,c} distinct such that a+b, a+c, b+c are all in S.
    # For a given pair (a,b), c is "bad" if a+c in S AND b+c in S (and a+b in S).
    # But we're building the conflict graph: two elements u,v conflict if there exists w
    # such that {u,v,w} forms a forbidden triple.
    
    # Actually, let me reconsider. The constraint is: no three DISTINCT members a,b,c 
    # such that a+b, a+c, b+c are all also selected.
    # This is a 3-uniform hypergraph constraint, not a 2-uniform one.
    
    # Let me think differently. For a set S to be valid, for every triple of distinct 
    # elements {a,b,c} in S, it's NOT the case that all of {a+b, a+c, b+c} are in S.
    
    # Bit-parallel approach: represent S as a bitmask.
    # For checking validity: for each pair (a,b) in S with a+b in S, check if there exists
    # c in S, c != a,b, such that (a+c) in S and (b+c) in S.
    
    # Let me precompute: for each element x, the bitmask of elements y such that x+y <= 128.
    # And for each pair (a,b), the bitmask of c such that a+c <= 128 and b+c <= 128.
    
    # Precompute sum_masks: sum_mask[x] = bitmask of all y in 1..128 such that x+y <= 128
    # Actually, we need: for a given a, which elements b have a+b in range?
    # Let's precompute: for each a (1..128), the set of b values where a+b <= 128.
    
    # Key insight for bit-parallel: 
    # For a set S (as bitmask), to check if adding x creates a violation:
    # A violation from adding x occurs if there exist distinct a,b in S (a,b != x) such that:
    #   a+x in S, b+x in S, and a+b in S (or a+b == x)
    # 
    # Let's define for each x: 
    #   partners[x] = bitmask of all a in 1..128, a != x, such that a+x <= 128
    #   Then if we add x, the "x-partners" in S are: S & partners[x]
    #   For two such partners a,b in S, we need a+b in S or a+b == x.
    
    # Precompute:
    #   for each x, partners_mask[x]: bitmask of a where a != x and a+x <= N
    #   for each pair (a,b) with a<b, the value a+b (for checking if it's in S)
    
    # Let me use a different approach: greedy with bit-parallel can_add.
    
    # Precompute:
    #   add_mask[x] = bitmask of all elements that are "compatible" in some sense? No.
    
    # Let's just do: for each x, precompute the list of (a,b) pairs where a+b involves x in a forbidden way.
    # Actually, let me just implement a fast _can_add using bitmasks.
    
    # For _can_add(S_mask, x):
    #   S_mask is the current set as a bitmask (bit i set means element i+1 is in S)
    #   x is the element to add (1..128)
    #   A violation: exist distinct a,b in S (a,b != x, a != b) such that:
    #     (a+x) in S AND (b+x) in S AND (a+b in S OR a+b == x)
    #   
    #   Let P = S_mask & partners_of_x  (elements in S that can pair with x to form a sum <= 128)
    #   For each pair a,b in P: check if a+x in S, b+x in S, and a+b in S or a+b==x
    #   
    #   But P could be large. Let's precompute for each x:
    #     x_partners[x] = bitmask of a where a != x and a+x <= N
    #     Then P = S_mask & x_partners[x]
    #     For each a in P: a+x must be in S (i.e., bit (a+x-1) set in S_mask)
    #     So valid_a = P & {a : a+x in S} = P & (S_mask shifted?) 
    #     Hmm, this is getting complex. Let me just iterate over P.
    
    # Precompute x_partners
    x_partners = [0] * (N + 1)
    for x in range(1, N + 1):
        mask = 0
        for a in range(1, N + 1):
            if a != x and a + x <= N:
                mask |= (1 << (a - 1))
        x_partners[x] = mask
    
    # For fast lookup: is element e in a bitmask?
    # bit(e) = 1 << (e-1)
    
    def bit(e):
        return 1 << (e - 1)
    
    def can_add(S_mask, x):
        """Check if adding element x to set S_mask keeps it valid."""
        # Get elements in S that can pair with x
        P = S_mask & x_partners[x]
        # For each a in P, check if a+x is in S
        # valid_a: a in P such that (a+x) is in S
        # Then for each pair a,b in valid_a, check if a+b in S or a+b == x
        # 
        # First, find valid_a:
        valid_a_mask = 0
        # Iterate over set bits in P
        p = P
        while p:
            # Get lowest set bit
            lb = p & (-p)
            a = lb.bit_length()  # a is the 1-based index
            p ^= lb
            # Check if a+x is in S
            if a + x <= N:
                if S_mask & bit(a + x):
                    valid_a_mask |= lb
        
        # Now check all pairs in valid_a_mask
        va = valid_a_mask
        while va:
            lb1 = va & (-va)
            a = lb1.bit_length()
            va ^= lb1
            # For each b in va (b > a to avoid duplicates, but we need all pairs)
            # Actually we need all unordered pairs {a,b} with a != b
            vb = va
            while vb:
                lb2 = vb & (-vb)
                b = lb2.bit_length()
                vb ^= lb2
                # Check a+b in S or a+b == x
                s = a + b
                if s == x or (s <= N and (S_mask & bit(s))):
                    return False
        return True
    
    # Greedy construction with random ordering
    def greedy(random_order=None):
        S_mask = 0
        if random_order is None:
            random_order = list(range(1, N + 1))
            random.shuffle(random_order)
        for x in random_order:
            if can_add(S_mask, x):
                S_mask |= bit(x)
        return S_mask
    
    # Local search: try to improve by removing one and adding one or more
    def local_search(S_mask, time_limit):
        start_ls = time.time()
        best = S_mask
        best_size = bin(S_mask).count('1')
        cur = S_mask
        cur_size = best_size
        
        no_improve = 0
        max_no_improve = 100
        random.seed(42)
        
        while time.time() - start_ls < time_limit and no_improve < max_no_improve:
            improved = False
            
            # Try adding any element
            for x in range(1, N + 1):
                if cur & bit(x):
                    continue
                if can_add(cur, x):
                    cur |= bit(x)
                    cur_size += 1
                    improved = True
                    break
            
            if improved:
                if cur_size > best_size:
                    best_size = cur_size
                    best = cur
                no_improve = 0
                continue
            
            # Try swap: remove one element, then add as many as possible
            # Pick a random element to remove
            removed = False
            for _ in range(20):  # try a few random removals
                if time.time() - start_ls > time_limit:
                    break
                # Pick random element in cur
                if cur_size == 0:
                    break
                # Random selection from cur
                idx = random.randint(0, cur_size - 1)
                # Find the (idx+1)-th set bit
                tmp = cur
                count = 0
                target_bit = 0
                while tmp:
                    lb = tmp & (-tmp)
                    if count == idx:
                        target_bit = lb
                        break
                    tmp ^= lb
                    count += 1
                if not target_bit:
                    continue
                x_remove = target_bit.bit_length()
                
                trial = cur & ~target_bit
                trial_size = cur_size - 1
                
                # Try to add elements
                for x in range(1, N + 1):
                    if trial & bit(x):
                        continue
                    if can_add(trial, x):
                        trial |= bit(x)
                        trial_size += 1
                
                if trial_size > cur_size:
                    cur = trial
                    cur_size = trial_size
                    removed = True
                    if cur_size > best_size:
                        best_size = cur_size
                        best = cur
                    break
            
            if removed:
                no_improve = 0
                continue
            
            # Try removing one element and adding multiple
            no_improve += 1
        
        return best
    
    # Run greedy with different random seeds, keep best
    best_mask = 0
    best_size = 0
    time_left = DEADLINE - time.time() - 2.0
    
    # Run multiple greedy trials
    max_trials = 20
    trial_time = time_left / max_trials if time_left > 0 else 0.5
    
    for t in range(max_trials):
        if time.time() - start > DEADLINE - 2.0:
            break
        random.seed(42 + t)
        order = list(range(1, N + 1))
        random.shuffle(order)
        mask = greedy(order)
        size = bin(mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = mask
    
    # Local search on best
    if time.time() - start < DEADLINE - 3.0:
        remaining = DEADLINE - time.time() - 3.0
        if remaining > 1.0:
            best_mask = local_search(best_mask, remaining)
            best_size = bin(best_mask).count('1')
    
    # Convert to list
    result = []
    for i in range(1, N + 1):
        if best_mask & bit(i):
            result.append(i)
    
    # Final validity check
    if not _is_valid(set(result)):
        # Fallback
        return list(range(N // 2 + 1, N + 1))
    
    return result

def _is_valid(S_set):
    """Check no distinct a,b,c in S_set with a+b, a+c, b+c all in S_set."""
    lst = list(S_set)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i + 1, n):
            b = lst[j]
            s = a + b
            if s not in S_set:
                continue
            for k in range(n):
                c = lst[k]
                if c == a or c == b:
                    continue
                if (a + c) in S_set and (b + c) in S_set:
                    return False
    return True

if __name__ == "__main__":
    t0 = time.time()
    r = construct()
    t1 = time.time()
    print(f"Size: {len(r)}, Time: {t1-t0:.2f}s")
    print(f"Valid: {_is_valid(set(r))}")
    print(r)
