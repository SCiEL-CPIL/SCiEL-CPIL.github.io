import sys
import time

def construct():
    n = 192
    max_val = 48
    
    # Precompute multiples for fast check
    # multiples[x] = list of multiples of x in range [3, 3*n]
    # We only care about sums of 3 distinct positive integers <= n.
    # Min sum: 1+2+3 = 6. Max sum: n+(n-1)+(n-2) = 3n-3.
    # So multiples in [6, 3n-3].
    
    multiples = {x: [] for x in range(1, n + 1)}
    for x in range(1, n + 1):
        for m in range(6, 3 * n - 2, x):
            multiples[x].append(m)
            
    # Candidates sorted descending (larger numbers are 'safer' as they have fewer small divisors)
    candidates = list(range(n, 0, -1))
    
    def can_add(x, S_list, S_set):
        K = len(S_list)
        
        # Check 1: x divides sum of any 3 in S_list
        # We need to check if any triple in S_list sums to a multiple of x.
        # Instead of iterating all triples, iterate multiples of x and check if any triple sums to it.
        # For each multiple M of x:
        #   Check if there exist a, b, c in S_list such that a+b+c = M.
        #   This is a 3-sum problem on S_list.
        #   Since K is small, we can do O(K^2) for each M? No, too many M.
        #   Better: Iterate pairs (a,b) in S_list, compute s = a+b.
        #   Check if M - s is in S_set and distinct from a,b.
        #   We can precompute pair_sums? No, S changes.
        
        # Optimization:
        # For each pair (a,b) in S_list, let s = a+b.
        # We need c in S_list such that s+c is a multiple of x.
        # c = k*x - s.
        # Check if c is in S_set and c != a and c != b.
        
        if K >= 3:
            for i in range(K):
                a = S_list[i]
                for j in range(i + 1, K):
                    b = S_list[j]
                    s = a + b
                    # c must be in S_list, c != a, c != b
                    # (s + c) % x == 0 => c % x == (-s) % x
                    # But we need exact c.
                    # c = k*x - s.
                    # Since c <= n, k*x - s <= n => k*x <= n + s.
                    # Also c >= 1 => k*x > s.
                    # So k is in range [floor(s/x) + 1, floor((n+s)/x)].
                    # Iterate possible k.
                    k_min = s // x + 1
                    k_max = (n + s) // x
                    for k in range(k_min, k_max + 1):
                        c = k * x - s
                        if c == a or c == b:
                            continue
                        if c in S_set:
                            return False
        
        # Check 2: Any y in S_list divides sum of x and 2 others
        # For each y in S_list:
        #   We need a, b in S_list \ {y} such that x + a + b is a multiple of y.
        #   Let M be a multiple of y.
        #   a + b = M - x.
        #   Check if any pair (a,b) in S_list \ {y} sums to M - x.
        
        for y in S_list:
            # Multiples of y in range [x + 1 + 2, x + n + n-1] = [x+3, x+2n-1]
            # Actually a,b >= 1, so min sum a+b = 1+2=3. Max a+b = n + n-1 = 2n-1.
            # So M - x in [3, 2n-1] => M in [x+3, x+2n-1].
            # Iterate multiples of y in this range.
            m_min = ((x + 3) + y - 1) // y * y
            m_max = x + 2 * n - 1
            
            if m_min > m_max:
                continue
                
            for M in range(m_min, m_max + 1, y):
                target_sum = M - x
                # Check if any pair (a,b) in S_list \ {y} sums to target_sum.
                # Iterate a in S_list \ {y}:
                #   b = target_sum - a
                #   Check if b in S_set, b != a, b != y.
                for a in S_list:
                    if a == y:
                        continue
                    b = target_sum - a
                    if b == a or b == y:
                        continue
                    if b in S_set:
                        return False
                        
        return True

    def greedy(order):
        S = []
        S_set = set()
        for x in order:
            if can_add(x, S, S_set):
                S.append(x)
                S_set.add(x)
        return S

    # Initial Greedy
    best_set = greedy(candidates)
    best_size = len(best_set)
    
    # B&B
    # We search for a better set.
    # To make B&B effective, we use the greedy solution as a lower bound.
    # We also use the fact that larger numbers are easier to place.
    
    global_best_size = best_size
    global_best_set = best_set[:]
    
    start_time = time.time()
    time_limit = 55
    
    def bnb(S, S_set, start_idx):
        nonlocal global_best_size, global_best_set
        
        if time.time() - start_time > time_limit:
            return
            
        # Pruning: If we can't beat the best, return.
        if len(S) + (len(candidates) - start_idx) <= global_best_size:
            return
            
        if len(S) >= max_val:
            if len(S) > global_best_size:
                global_best_size = len(S)
                global_best_set = S[:]
            return

        for i in range(start_idx, len(candidates)):
            if time.time() - start_time > time_limit:
                return
            
            x = candidates[i]
            
            # Simple heuristic: If x is in global_best_set, maybe prioritize?
            # No, just try to add it.
            
            if can_add(x, S, S_set):
                new_S = S + [x]
                new_S_set = S_set | {x}
                
                if len(new_S) > global_best_size:
                    global_best_size = len(new_S)
                    global_best_set = new_S[:]
                    
                bnb(new_S, new_S_set, i + 1)

    # Start B&B from empty set, but we can use a warm start?
    # B&B from empty set might be slow.
    # Let's try to extend the greedy solution?
    # Or just run B&B from scratch with good ordering.
    
    bnb([], set(), 0)
    
    return global_best_set
