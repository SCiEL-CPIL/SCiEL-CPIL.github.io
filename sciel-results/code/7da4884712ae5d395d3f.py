import time

def is_valid_set(S):
    lst = list(S)
    n = len(lst)
    if n < 4:
        return True
    
    for idx_a, a in enumerate(lst):
        others = lst[:idx_a] + lst[idx_a+1:]
        m = len(others)
        if m < 3:
            continue
        
        # Check if any 3 distinct others sum to a multiple of a
        # Use pair sums and a set of available values?
        # For each pair (i, j), we need k such that (others[i] + others[j] + others[k]) % a == 0
        # => others[k] % a == (-others[i] - others[j]) % a
        # And k must be distinct from i, j.
        
        # Precompute residues of others
        # But we need to ensure distinctness.
        
        # Simple triple loop is O(m^3). m <= 47. 47^3 = 103823.
        # 48 * 103823 = 5M operations.
        # In Python, 5M operations might take ~0.5s.
        # If we call this 48 times, that's ~24s. Too slow.
        
        # We need a faster check.
        
        # For fixed a, let's collect counts of residues mod a?
        # No, because we need distinct indices.
        
        # Let's try:
        # For each i in range(m):
        #   For each j in range(i+1, m):
        #     target_res = (-others[i] - others[j]) % a
        #     We need to know if there exists k > j such that others[k] % a == target_res.
        #     We can precompute a suffix array of sets of residues?
        #     Or just a simple check.
        
        # Let's use the suffix set approach.
        # For each a:
        #   Create a list of (value, residue) for others.
        #   Iterate i from 0 to m-3.
        #   Maintain a set of residues for elements at indices > j.
        #   This is getting complicated.
        
        # Alternative: 
        # Since a is part of the set, and we are picking large numbers, 
        # a is large. The number of multiples of a in the range [3, 576] is small.
        # Max sum = 576.
        # Multiples of a: a, 2a, 3a, ...
        # For a > 192, only 1 multiple? No, a is in [1, 192].
        # If a is large (e.g. 145), multiples are 145, 290, 435.
        # We check if any triple sums to 145, 290, or 435.
        # Number of triples summing to a specific value is small.
        
        # Let's stick to the triple loop but optimize with early exit and local variables.
        
        found = False
        for i in range(m):
            oi = others[i]
            for j in range(i+1, m):
                oj = others[j]
                s2 = oi + oj
                for k in range(j+1, m):
                    if (s2 + others[k]) % a == 0:
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if found:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    selected = []
    
    for x in range(192, 0, -1):
        if len(selected) == 48:
            break
        if time.time() > deadline:
            break
            
        candidate_set = set(selected)
        candidate_set.add(x)
        
        if is_valid_set(candidate_set):
            selected.append(x)
            
    return selected
