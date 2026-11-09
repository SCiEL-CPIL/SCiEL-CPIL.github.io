import itertools

def construct():
    # 1. Generate smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    M = len(smooth)
    
    # 2. Precompute divisors
    # divisors[j] contains indices i < j such that smooth[i] divides smooth[j]
    divisors = [set() for _ in range(M)]
    for j in range(M):
        for i in range(j):
            if smooth[j] % smooth[i] == 0:
                divisors[j].add(i)
                
    # 3. Find minimal representations
    best_rep = {}
    targets = set(range(1, 129))
    
    # Max k is 30, but likely much smaller
    for k in range(1, 31):
        if not targets:
            break
            
        # Iterate over combinations of indices of size k
        for combo in itertools.combinations(range(M), k):
            # Check antichain property
            is_anti = True
            # combo is sorted: c0 < c1 < ... < c_{k-1}
            # For each element c_j (j>0), check if any c_i (i<j) divides c_j
            for j_idx in range(1, k):
                j = combo[j_idx]
                # Check if any previous index in combo is in divisors[j]
                for i_idx in range(j_idx):
                    i = combo[i_idx]
                    if i in divisors[j]:
                        is_anti = False
                        break
                if not is_anti:
                    break
                    
            if not is_anti:
                continue
                
            s = sum(smooth[i] for i in combo)
            if s <= 128 and s in targets:
                best_rep[s] = [smooth[i] for i in combo]
                targets.remove(s)
                
    # 4. Construct output
    answers = []
    for n in range(1, 129):
        if n in best_rep:
            answers.append(best_rep[n])
        else:
            # This should not happen if the problem is well-posed
            # But to be safe, we might need a fallback or raise
            # Given the density of smooth numbers and antichains, k<=30 should suffice
            # If not, we raise an error as per original logic
            raise ValueError(f"Failed to find representation for {n}")
            
    return answers
