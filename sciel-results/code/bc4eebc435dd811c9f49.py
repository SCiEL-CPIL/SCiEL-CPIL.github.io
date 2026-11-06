import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    N = 4096
    best_set = []
    best_len = 0
    
    def check_and_add(S, sums, x):
        """
        Check if x can be added to S without violating unique triple sums.
        S is a list of integers.
        sums is a set of all valid triple sums of S.
        Returns (True, new_sums_set) if valid, (False, None) otherwise.
        """
        new_sums = set()
        
        # 1. Triple (x, x, x) -> 3x
        s3 = 3 * x
        if s3 in sums:
            return False, None
        new_sums.add(s3)
        
        # 2. Triples (x, x, s) for s in S -> 2x + s
        for s in S:
            val = 2 * x + s
            if val in sums or val in new_sums:
                return False, None
            new_sums.add(val)
            
        # 3. Triples (x, s_i, s_j) for s_i <= s_j in S -> x + s_i + s_j
        n = len(S)
        for i in range(n):
            si = S[i]
            for j in range(i, n):
                sj = S[j]
                val = x + si + sj
                if val in sums or val in new_sums:
                    return False, None
                new_sums.add(val)
                
        return True, new_sums

    def run_greedy(order):
        """
        Run a greedy search with a given ordering of candidates.
        Returns the constructed set.
        """
        S = []
        sums = set()
        
        for x in order:
            if time.time() > deadline - 2.0:
                break
            valid, new_sums = check_and_add(S, sums, x)
            if valid:
                S.append(x)
                sums.update(new_sums)
                
        return S

    # Strategy 1: Deterministic Greedy (Ascending)
    # This provides a quick baseline.
    S_asc = run_greedy(range(1, N + 1))
    if len(S_asc) > best_len:
        best_set = S_asc
        best_len = len(S_asc)
        
    # Strategy 2: Deterministic Greedy (Descending)
    # Sometimes starting with larger numbers helps spread out the sums.
    S_desc = run_greedy(range(N, 0, -1))
    if len(S_desc) > best_len:
        best_set = S_desc
        best_len = len(S_desc)

    # Strategy 3: Randomized Greedy Restarts
    # Use multiple random seeds to explore different orderings.
    random.seed(42)
    
    # We can run as many restarts as possible within the time limit.
    # Each restart takes roughly O(N * K^2) where K is the set size.
    attempts = 0
    while time.time() < deadline - 5.0:
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S_try = run_greedy(order)
        
        if len(S_try) > best_len:
            best_set = S_try
            best_len = len(S_try)
            
        attempts += 1
        
    # Final Improvement: Try to add any missing element to the best set found
    # This is a single pass over all numbers, checking if they can be added.
    if best_len > 0 and time.time() < deadline - 1.0:
        S_cur = list(best_set)
        sums_cur = set()
        n = len(S_cur)
        # Rebuild sums for the current best set
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    sums_cur.add(S_cur[i] + S_cur[j] + S_cur[k])
                    
        S_cur_set = set(S_cur)
        
        # Try to add any number not in S_cur
        # We iterate through 1..N. If we add one, we continue with the updated set.
        for x in range(1, N + 1):
            if time.time() > deadline - 0.5:
                break
            if x in S_cur_set:
                continue
            valid, new_sums = check_and_add(S_cur, sums_cur, x)
            if valid:
                S_cur.append(x)
                sums_cur.update(new_sums)
                S_cur_set.add(x)
                best_len += 1
        
        if len(S_cur) > best_len:
            best_set = S_cur
            best_len = len(S_cur)

    return best_set
