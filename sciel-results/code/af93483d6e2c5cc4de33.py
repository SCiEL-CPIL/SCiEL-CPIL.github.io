import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    def can_add(S, x):
        """Check if x can be added to set S (S as list, x not in S).
        Returns True if valid.
        
        Two conditions:
        1. x does not divide sum of any 3 distinct elements in S
        2. No a in S divides x + y + z for any distinct y,z in S\{a}
        """
        n = len(S)
        
        # Condition 1: x | (a+b+c) for some triple in S
        if n >= 3:
            # Use pair-sum residue: for each pair (i,j), compute (S[i]+S[j]) % x
            # Then check if there exists k such that (S[i]+S[j]+S[k]) % x == 0
            # i.e., (S[i]+S[j]) % x == (-S[k]) % x
            # For efficiency, count residues of pair sums, then for each k check
            # if the needed residue exists among pairs not involving k.
            # Simpler: just iterate triples if n is small, or use residue counting.
            # For n up to ~50, O(n^3) per check is ~125k ops. Acceptable if not too many checks.
            # But let's do O(n^2) with residue counting.
            
            # Count pair-sum residues
            pair_residue_count = {}
            for i in range(n):
                for j in range(i + 1, n):
                    r = (S[i] + S[j]) % x
                    pair_residue_count[r] = pair_residue_count.get(r, 0) + 1
            
            # For each k, check if there's a pair (i,j) with i,j != k such that
            # (S[i]+S[j]+S[k]) % x == 0, i.e., (S[i]+S[j]) % x == (-S[k]) % x
            for k in range(n):
                target = (-S[k]) % x
                if target in pair_residue_count:
                    # Need to verify there's a pair not involving k
                    # If pair_residue_count[target] >= 1, check if any such pair excludes k
                    # In worst case, all pairs with this residue involve k.
                    # Number of pairs involving k is n-1.
                    # So if pair_residue_count[target] > (n-1), definitely safe.
                    # Otherwise, we need to check.
                    if pair_residue_count[target] > (n - 1):
                        return False
                    else:
                        # Check if there's a pair (i,j) with i,j != k and (S[i]+S[j])%x == target
                        found = False
                        for i in range(n):
                            if i == k:
                                continue
                            for j in range(i + 1, n):
                                if j == k:
                                    continue
                                if (S[i] + S[j]) % x == target:
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            return False
        # Condition 1 passed
        
        # Condition 2: For each a in S, check if exists y,z in S\{a} distinct such that
        # (x + y + z) % a == 0, i.e., (y+z) % a == (-x) % a
        for idx_a, a in enumerate(S):
            target = (-x) % a
            # Need pair (y,z) from S\{a} with (y+z) % a == target
            # Count pair residues among S\{a}
            # S\{a} has n-1 elements
            # Build list of indices excluding idx_a
            others = []
            for idx in range(n):
                if idx != idx_a:
                    others.append(S[idx])
            m = len(others)
            if m < 2:
                continue
            # Count pair residues
            pair_res = {}
            for i in range(m):
                for j in range(i + 1, m):
                    r = (others[i] + others[j]) % a
                    pair_res[r] = pair_res.get(r, 0) + 1
            if target in pair_res and pair_res[target] > 0:
                return False
        
        return True

    def greedy(order):
        S = []
        S_set = set()
        for x in order:
            if x in S_set:
                continue
            if can_add(S, x):
                S.append(x)
                S_set.add(x)
        return S

    best_S = []
    best_len = 0

    # 1. Descending greedy (large numbers are safer)
    S_desc = greedy(range(192, 0, -1))
    if len(S_desc) > best_len:
        best_S = S_desc
        best_len = len(S_desc)

    # 2. Try a few structured orders
    # Large odds, then large evens, etc.
    S2 = greedy([x for x in range(192, 0, -2)] + [x for x in range(191, 0, -2)])
    if len(S2) > best_len:
        best_S = S2
        best_len = len(S2)

    # 3. Random restarts with local search
    random.seed(42)
    
    while time.time() - start_time < deadline:
        # Generate random order, biased toward larger numbers
        order = list(range(1, 193))
        # Weighted shuffle: larger numbers more likely earlier
        random.shuffle(order)
        S_rand = greedy(order)
        
        # Local search: try to add elements, then swaps
        S_set = set(S_rand)
        improved = True
        while improved and time.time() - start_time < deadline:
            improved = False
            
            # Try to add any missing element
            candidates = [x for x in range(1, 193) if x not in S_set]
            random.shuffle(candidates)
            for x in candidates:
                if can_add(S_rand, x):
                    S_rand.append(x)
                    S_set.add(x)
                    improved = True
                    break
            
            if not improved:
                # Try swaps: remove one, add one
                # Try a few random swaps
                for _ in range(20):
                    if time.time() - start_time > deadline:
                        break
                    if len(S_rand) == 0:
                        break
                    y = random.choice(S_rand)
                    S_temp = [v for v in S_rand if v != y]
                    S_temp_set = set(S_temp)
                    candidates = [x for x in range(1, 193) if x not in S_temp_set]
                    random.shuffle(candidates)
                    for x in candidates:
                        if can_add(S_temp, x):
                            S_rand = S_temp + [x]
                            S_set = set(S_rand)
                            improved = True
                            break
                    if improved:
                        break
        
        if len(S_rand) > best_len:
            best_S = S_rand
            best_len = len(S_rand)
            if best_len == 48:
                break

    return best_S
