import time
import random
import numpy as np

P = {'n': 128}

def construct():
    n = P['n']
    # Safety margin: evaluator kills at 90s. Use 85s for computation.
    deadline = time.time() + 85.0
    
    # Precompute all triples (a, b, c) with 1 <= a < b < c <= n
    # such that a+b, a+c, b+c are all <= n.
    # If any of the sums > n, they cannot be in the set S (subset of 1..n),
    # so that triple is automatically "safe" regarding the sum condition.
    # We only care about triples where all three sums are <= n.
    
    # Vectorized generation of triples
    # We need a < b < c
    # a+b <= n, a+c <= n, b+c <= n
    # Since a < b < c, b+c is the largest sum. So if b+c <= n, then a+c <= n and a+b <= n.
    # So the condition is simply b + c <= n.
    
    # Let's generate all (a, b, c) such that 1 <= a < b < c <= n and b + c <= n.
    # We can store these as a matrix of shape (M, 3) where M is the number of such triples.
    # Then for a given set S (boolean array of size n+1), we can check:
    # valid = ~np.any(S[triples[:, 0]] & S[triples[:, 1]] & S[triples[:, 2]] & 
    #                 S[triples[:, 0] + triples[:, 1]] & 
    #                 S[triples[:, 0] + triples[:, 2]] & 
    #                 S[triples[:, 1] + triples[:, 2]])
    
    # Actually, we can precompute the indices for the sums as well.
    # Let T be the list of (a, b, c, a+b, a+c, b+c).
    # We can store indices: idx_a, idx_b, idx_c, idx_ab, idx_ac, idx_bc.
    
    triples_data = []
    for c in range(3, n + 1):
        # b < c and b + c <= n => b <= n - c
        # Also b > a >= 1 => b >= 2
        max_b = min(c - 1, n - c)
        if max_b < 2:
            continue
        for b in range(2, max_b + 1):
            # a < b and a >= 1
            # a can be 1 to b-1
            for a in range(1, b):
                ab = a + b
                ac = a + c
                bc = b + c
                # All sums are <= n by construction of b
                triples_data.append((a, b, c, ab, ac, bc))
    
    if not triples_data:
        # No constraints, take all
        return list(range(1, n + 1))
        
    triples_arr = np.array(triples_data, dtype=np.int32)
    # Columns: 0:a, 1:b, 2:c, 3:a+b, 4:a+c, 5:b+c
    
    # Function to check validity of a boolean array S (size n+1, index 0 unused)
    def is_valid(S):
        # S is a numpy bool array of size n+1
        # We need to check if there exists any row in triples_arr such that
        # S[a] and S[b] and S[c] and S[a+b] and S[a+c] and S[b+c] are all True.
        # If any such row exists, it's invalid.
        
        # Vectorized lookup
        # Note: S indices are 0..n. Our values are 1..n.
        mask = (
            S[triples_arr[:, 0]] & 
            S[triples_arr[:, 1]] & 
            S[triples_arr[:, 2]] & 
            S[triples_arr[:, 3]] & 
            S[triples_arr[:, 4]] & 
            S[triples_arr[:, 5]]
        )
        return not np.any(mask)
    
    best_set = None
    best_size = 0
    
    # Strategy: Random Restart + Greedy Addition
    # 1. Generate a random initial set (e.g., random subset of size ~60-70)
    # 2. Check if valid. If not, remove elements causing conflicts? 
    #    Easier: Start with empty set, add elements greedily in random order.
    #    This guarantees validity at each step.
    #    To maximize size, we want to add as many as possible.
    #    The order matters. Random restarts with different orders can find larger sets.
    
    # Optimization: 
    # Instead of full re-check O(M) for each addition (M~10k-50k), 
    # we can maintain the set and only check conflicts involving the new element.
    # However, the vectorized check is quite fast. 
    # Let's estimate M.
    # n=128. b+c <= 128.
    # c ranges 3..128. b ranges 2..128-c. a ranges 1..b-1.
    # Roughly sum_{c=3}^{64} sum_{b=2}^{128-c} (b-1).
    # Let k = 128-c. k ranges 125 down to 64.
    # sum_{k=64}^{125} sum_{b=2}^{k} (b-1) = sum_{k=64}^{125} k(k-1)/2.
    # ~ 60 * (100^2)/2 = 300,000.
    # So M is around 300k.
    # 300k boolean ops in NumPy is very fast (~microseconds).
    # We can do thousands of checks per second.
    
    # Let's try Random Greedy Construction:
    # 1. Shuffle 1..n
    # 2. Iterate through shuffled list
    # 3. If adding x keeps set valid, add it.
    # 4. Repeat many times, keep best.
    
    # To speed up, we can try to start with a "good" base set.
    # The set {65..128} is size 64. 
    # Is it valid?
    # Min sum 65+66=131 > 128. So no sums are in the set. Valid.
    # Can we do better?
    # If we take {1..n}, invalid.
    # What if we take numbers > n/2? Size n/2 = 64.
    # What if we take numbers in [n/3, n]?
    # Min sum n/3 + n/3 = 2n/3 = 85.
    # Sums can be in the set.
    
    # Let's run random greedy restarts.
    
    rng = np.random.default_rng()
    
    for i in range(5000): # Max iterations, controlled by time
        if time.time() > deadline:
            break
            
        # Random permutation of 1..n
        perm = rng.permutation(np.arange(1, n + 1))
        
        current_set_bool = np.zeros(n + 1, dtype=bool)
        current_size = 0
        
        # Greedy add
        for x in perm:
            # Tentatively add x
            current_set_bool[x] = True
            
            # Check validity
            if not is_valid(current_set_bool):
                # Revert
                current_set_bool[x] = False
            else:
                current_size += 1
        
        if current_size > best_size:
            best_size = current_size
            best_set = np.where(current_set_bool)[1].tolist()
            
        # If we found a very large set, we might stop early?
        # Theoretical max?
        # If we can find 70+, that's a big jump from 64.
        # Let's see how far we can go.
        
        # Optimization: If best_size reaches a high threshold, we can try to optimize further?
        # For now, just keep searching.
        
        # Also, we can try to "repair" invalid sets? No, greedy is safe.
        
        # Let's also try a variant: Start with the top half, and try to swap in smaller numbers?
        # If we have S = {65..128}, can we replace 128 with 1?
        # 1+65=66 (in S), 1+66=67 (in S), 65+66=131 (not in S).
        # Triple (1, 65, 66): sums 66, 67, 131. 131 not in S. So valid.
        # Triple (1, 65, 67): sums 66, 68, 132. 132 not in S. Valid.
        # It seems replacing large elements with small ones might be possible if the sums fall out of range.
        
        # The greedy random restart should find these if the random order allows it.
        # However, greedy is order-dependent.
        # To improve, we can try "Simulated Annealing" or "Local Search" on the boolean array?
        # But vectorized check makes local search steps expensive if we check full validity.
        # We can check only conflicts involving the changed element.
        
        # Let's stick to Random Greedy for now, but increase iterations if time permits.
        # The loop runs until deadline.
        
        # If we have time, we can try to "polish" the best set found.
        # Try to add any remaining element? Greedy already tried all in random order.
        # But maybe a different order would have allowed more?
        
        # Let's add a second phase: Local Search on the best set found so far.
        # But only if time allows and we have a candidate.
        # Actually, let's just keep doing random restarts. They are independent and safe.
        
    if best_set is None:
        # Fallback: top half
        return list(range(65, n + 1))
        
    return best_set

